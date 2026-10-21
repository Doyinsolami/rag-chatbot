import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from sparse_core import (
    health_terms,
    identity_diagonal,
    is_informative,
    mahalanobis_noise,
    mask_spread,
    neighbor_overlap,
    remove_health_information,
    removal_pattern,
    shape_diagonal,
)

MODEL = "all-MiniLM-L6-v2"
SPLITS = ("train", "validation", "test")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def records(root, split):
    p = root / "data/geia" / (split + ".jsonl")
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def prepare(a):
    from sentence_transformers import SentenceTransformer

    out = a.workdir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a new --workdir; {out} already contains files")
    train, val, test = [records(a.project, s) for s in SPLITS]
    groups = [{r["patient_id"] for r in rows} for rows in (train, val, test)]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("Patient overlap across existing splits")
    all_rows = train + val + test
    if len({r["encounter_id"] for r in all_rows}) != len(all_rows):
        raise ValueError("Duplicate encounter IDs")
    gt = json.loads((a.project / "data/ground_truth.json").read_text())
    if any(gt[r["encounter_id"]]["patient_id"] != r["patient_id"] for r in all_rows):
        raise ValueError("Split patient IDs disagree with ground truth")
    fields = tuple(a.fields.split(","))
    if not set(fields) <= {"conditions", "medications", "procedures", "allergies"}:
        raise ValueError("Unsupported health-information field")
    terms = health_terms(gt, [r["encounter_id"] for r in train], fields)
    pattern = removal_pattern(terms)
    model = SentenceTransformer(MODEL, device=a.device)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "encoder": MODEL,
        "fields": fields,
        "concept_source": "training_patients_only",
        "scope": "exact full phrases from training diagnoses, medications, procedures and allergies",
        "source_hashes": {s: digest(a.project / "data/geia" / f"{s}.jsonl") for s in SPLITS},
        "ground_truth_sha256": digest(a.project / "data/ground_truth.json"),
        "splits": {},
    }
    for name, rows in [("train", train), ("validation", val)]:
        pairs = []
        for r in rows:
            note = (a.project / "data/notes" / f"{r['encounter_id']}.txt").read_text().strip()
            if note != r["text"].strip():
                raise ValueError("Split text differs from current note: " + r["encounter_id"])
            removed, changed = remove_health_information(r["text"], pattern)
            if changed:
                pairs.append((r, removed))
        report["splits"][name] = {"total": len(rows), "matched": len(pairs)}
        if a.limit_pairs:
            pairs = pairs[:a.limit_pairs]
        if len(pairs) < 2:
            raise ValueError(f"Too few {name} pairs")
        pos = model.encode([r["text"] for r, _ in pairs], batch_size=64, convert_to_numpy=True,
                           normalize_embeddings=False, show_progress_bar=True)
        neg = model.encode([t for _, t in pairs], batch_size=64, convert_to_numpy=True,
                           normalize_embeddings=False, show_progress_bar=True)
        visible = np.linalg.norm(pos - neg, axis=1) > 1e-7
        report["splits"][name].update(used_pairs=int(visible.sum()),
                                      unchanged_embedding_pairs=int((~visible).sum()))
        if visible.sum() < 2:
            raise ValueError(f"Too few visible {name} pairs after encoder truncation")
        np.savez_compressed(out / f"{name}_pairs.npz", positive=pos[visible], negative=neg[visible])
        dump(out / f"{name}_pair_ids.json",
             [r["encounter_id"] for (r, _), keep in zip(pairs, visible) if keep])
    dump(out / "health_information.json", terms)
    dump(out / "prepare.json", report)
    print(json.dumps(report, indent=2))


def train(a):
    import torch
    from torch import nn

    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    def load(name):
        with np.load(a.workdir / f"{name}_pairs.npz") as f:
            p, n = torch.tensor(f["positive"]), torch.tensor(f["negative"])
        return p.to(device), n.to(device)

    pos, neg = load("train")
    vp, vn = load("validation")
    d = pos.shape[1]
    if d != 384:
        raise ValueError("Expected 384-dimensional MiniLM representations")
    output = a.workdir / f"mask_{a.regularizer}_lam{a.lam:g}_lr{a.lr:g}_seed{a.seed}"
    if output.exists():
        raise FileExistsError(f"{output} already exists")
    output.mkdir()
    log_alpha = nn.Parameter(torch.zeros(d, device=device))
    log_beta = nn.Parameter(torch.full((d,), float(np.log(2 / 3)), device=device))
    predictor = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                              nn.Linear(128, 1)).to(device)
    optimizer = torch.optim.Adam([log_alpha, log_beta, *predictor.parameters()], lr=a.lr)
    sign = -1.0 if a.regularizer == "paper_literal" else 1.0
    steps_per_epoch = (len(pos) + a.batch_size - 1) // a.batch_size
    total_steps = steps_per_epoch * a.epochs
    print(json.dumps({"train_pairs": len(pos), "validation_pairs": len(vp),
                      "steps_per_epoch": steps_per_epoch, "total_steps": total_steps}))
    history, best, best_epoch = [], float("inf"), None
    for epoch in range(a.epochs):
        predictor.train()
        permutation = torch.randperm(len(pos), device=device)
        total = 0.0
        for indices in permutation.split(a.batch_size):
            beta = log_beta.exp()
            u = torch.rand(d, device=device).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.logit(u) + log_alpha) / beta)
            mask = (s * 1.2 - 0.1).clamp(0, 1)
            logits = predictor(torch.cat((pos[indices], neg[indices])) * mask).squeeze(-1)
            labels = torch.cat((torch.ones(len(indices), device=device),
                                torch.zeros(len(indices), device=device)))
            classification = 2 * nn.functional.binary_cross_entropy_with_logits(logits, labels)
            active = torch.sigmoid(log_alpha - beta * np.log(0.1 / 1.1)).mean()
            loss = classification + a.lam * sign * active
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                log_beta.clamp_(-5, 2)
            total += loss.item() * len(indices)
        predictor.eval()
        with torch.no_grad():
            mask = (torch.sigmoid(log_alpha) * 1.2 - 0.1).clamp(0, 1)
            x = torch.cat((vp, vn)) * mask
            labels = torch.cat((torch.ones(len(vp), device=device),
                                torch.zeros(len(vn), device=device)))
            logits = torch.cat([predictor(chunk).squeeze(-1) for chunk in x.split(a.batch_size)])
            val_loss = nn.functional.binary_cross_entropy_with_logits(logits, labels).item()
            accuracy = ((logits > 0) == (labels > 0.5)).float().mean().item()
            mask_np = mask.cpu().numpy()
            item = dict(epoch=epoch + 1, train_loss=total / len(pos), validation_bce=val_loss,
                        validation_accuracy=accuracy, mask=mask_spread(mask_np))
            history.append(item)
            print(json.dumps(item), flush=True)
            if val_loss < best and (epoch + 1 >= a.min_epochs):
                best, best_epoch = val_loss, epoch + 1
                np.save(output / "mask.npy", mask_np)
                torch.save({"predictor": predictor.state_dict(), "log_alpha": log_alpha.cpu(),
                            "log_beta": log_beta.cpu()}, output / "training_state.pt")
    np.save(output / "mask_final_epoch.npy", mask_np)
    final_mask = np.load(output / "mask.npy")
    summary = mask_spread(final_mask)
    informative = is_informative(final_mask)
    dump(output / "training.json", dict(
        encoder=MODEL, dimension=d, regularizer=a.regularizer, epochs=a.epochs, seed=a.seed,
        lr=a.lr, batch_size=a.batch_size, lam=a.lam, min_epochs=a.min_epochs,
        total_steps=total_steps, initial_log_alpha=0, initial_beta=2 / 3,
        beta_log_bounds=[-5, 2], best_epoch=best_epoch,
        selection="minimum deterministic validation BCE after min_epochs",
        selected_mask=summary, informative=informative,
        final_epoch_mask=mask_spread(mask_np), history=history,
        prepare_sha256=digest(a.workdir / "prepare.json")))
    print("Saved", output / "mask.npy")
    print("Selected mask:", json.dumps(summary))
    if not informative:
        print("WARNING: the mask is nearly uniform, so SPARSE would reduce to the Laplace "
              "mechanism. Consider an adapted setting chosen on validation data.")


def load_source(a):
    import chromadb

    client = chromadb.PersistentClient(path=str(a.project / "chroma_db"))
    source = client.get_collection(a.source)
    if source.metadata and source.metadata.get("hnsw:space", "l2") != "l2":
        raise ValueError("Expected L2 source collection")
    ids, vectors, metadata = [], [], []
    for offset in range(0, source.count(), 256):
        batch = source.get(limit=256, offset=offset, include=["embeddings", "metadatas"])
        ids.extend(batch["ids"])
        vectors.extend(batch["embeddings"])
        metadata.extend(batch["metadatas"] if batch["metadatas"] is not None
                        else [{} for _ in batch["ids"]])
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Invalid source vectors")
    return client, ids, vectors, metadata


def resolve_diagonal(a, dim):
    if a.mechanism == "laplace":
        return identity_diagonal(dim), None
    if a.mask is None:
        raise ValueError("--mask is required for the SPARSE mechanism")
    mask = np.load(a.mask)
    if len(mask) != dim:
        raise ValueError("Mask dimension does not match source vectors")
    if not is_informative(mask) and not a.allow_uniform_mask:
        raise ValueError("Mask is nearly uniform: " + json.dumps(mask_spread(mask)) +
                         ". Retrain the mask or pass --allow-uniform-mask to proceed anyway.")
    return shape_diagonal(mask), digest(a.mask)


def calibrate(a):
    _, ids, vectors, _ = load_source(a)
    index = {eid: i for i, eid in enumerate(ids)}
    db_rows = [index[r["encounter_id"]] for r in records(a.project, "train")]
    query_rows = [index[r["encounter_id"]] for r in records(a.project, "validation")]
    database, queries = vectors[db_rows], vectors[query_rows]
    norms = np.linalg.norm(vectors, axis=1)
    sparse_diagonal = None
    if a.mask is not None:
        a.mechanism = "sparse"
        sparse_diagonal, _ = resolve_diagonal(a, vectors.shape[1])
    rng = np.random.default_rng(a.seed)
    rows = []
    for eps in a.epsilons:
        row = {"epsilon": eps, "expected_noise_norm": vectors.shape[1] / eps,
               "mean_embedding_norm": float(norms.mean())}
        noisy = database + mahalanobis_noise(len(database), identity_diagonal(vectors.shape[1]), eps, rng)
        row["laplace_overlap_at_k"] = neighbor_overlap(database, noisy, queries, a.k)
        if sparse_diagonal is not None:
            noisy = database + mahalanobis_noise(len(database), sparse_diagonal, eps, rng)
            row["sparse_overlap_at_k"] = neighbor_overlap(database, noisy, queries, a.k)
        rows.append(row)
        print(json.dumps(row), flush=True)
    a.workdir.mkdir(parents=True, exist_ok=True)
    out = a.workdir / f"calibration_k{a.k}_seed{a.seed}.json"
    dump(out, dict(source=a.source, k=a.k, seed=a.seed,
                   mask=str(a.mask) if a.mask else None,
                   proxy="overlap between clean and noised top-k neighbors of validation queries over training vectors",
                   rows=rows))
    print("Saved", out)


def build(a):
    meta = json.loads((a.workdir / "prepare.json").read_text())
    for split, sha in meta["source_hashes"].items():
        if digest(a.project / "data/geia" / f"{split}.jsonl") != sha:
            raise ValueError("Splits changed since preparation")
    if digest(a.project / "data/ground_truth.json") != meta["ground_truth_sha256"]:
        raise ValueError("Ground truth changed since preparation")
    client, ids, vectors, metadata = load_source(a)
    diagonal, mask_sha = resolve_diagonal(a, vectors.shape[1])
    needed = {r["encounter_id"] for s in SPLITS for r in records(a.project, s)}
    if set(ids) != needed:
        raise ValueError("Source IDs differ from complete saved splits")
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(MODEL, device=a.device)
    texts = [(a.project / "data/notes" / f"{eid}.txt").read_text() for eid in ids[:5]]
    fresh = encoder.encode(texts, normalize_embeddings=False, convert_to_numpy=True)
    if not np.allclose(fresh, vectors[:5], rtol=1e-3, atol=1e-5):
        raise ValueError("Stored vectors differ from fresh MiniLM encodings")
    if a.debug_seed is not None:
        print("WARNING: --debug-seed makes the noise reproducible from the seed. "
              "Do not use this collection for privacy results.")
        rng = np.random.default_rng(a.debug_seed)
    else:
        rng = np.random.default_rng()
    raw = vectors + mahalanobis_noise(len(ids), diagonal, a.epsilon, rng)
    protected = raw
    if a.unit_normalize:
        protected = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    target = client.create_collection(a.target, metadata={"hnsw:space": "l2"})
    rows = []
    for old in metadata:
        row = dict(old or {})
        row.update(noise_mechanism=a.mechanism, noise_epsilon=float(a.epsilon),
                   noise_draw=int(a.draw), noise_debug_seeded=a.debug_seed is not None,
                   noise_privacy_definition="metric_ldp",
                   noise_mask_sha256=mask_sha or "", noise_normalized=bool(a.unit_normalize),
                   dp_applied=False)
        rows.append(row)
    for i in range(0, len(ids), 256):
        target.add(ids=ids[i:i + 256], embeddings=protected[i:i + 256].tolist(),
                   metadatas=rows[i:i + 256])
    if target.count() != len(ids):
        raise RuntimeError("Incomplete target collection")
    out = a.workdir / (a.target + "_build.json")
    dump(out, dict(
        target=a.target, source=a.source, count=len(ids), mechanism=a.mechanism,
        epsilon=a.epsilon, draw=a.draw,
        randomness="debug seed; not valid for privacy results" if a.debug_seed is not None
        else "fresh operating-system entropy; state not stored",
        debug_seed=a.debug_seed,
        privacy="Metric LDP for the stored vectors conditional on a fixed shape matrix; no DP claim for mask training data",
        mask=str(a.mask) if a.mask else None, mask_sha256=mask_sha,
        shape_diagonal=diagonal.tolist(), normalized_after_noise=a.unit_normalize,
        noise_radius_mean=float(np.linalg.norm((raw - vectors) / np.sqrt(diagonal), axis=1).mean()),
        expected_radius=len(diagonal) / a.epsilon))
    print("Created", a.target, "with", target.count(), "records; manifest:", out)


def main():
    p = argparse.ArgumentParser(description="SPARSE and Laplace embedding defenses for the encounter-level MiniLM project.")
    p.add_argument("--project", type=Path, default=Path.cwd())
    p.add_argument("--workdir", type=Path, default=Path("data/sparse"))
    p.add_argument("--device", default=None)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--fields", default="conditions,medications,procedures,allergies")
    prep.add_argument("--limit-pairs", type=int, default=None)
    tr = sub.add_parser("train")
    tr.add_argument("--regularizer", choices=["paper_literal", "positive_l0"], required=True)
    tr.add_argument("--epochs", type=int, default=100)
    tr.add_argument("--min-epochs", type=int, default=1)
    tr.add_argument("--batch-size", type=int, default=64)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--lam", type=float, default=0.001)
    tr.add_argument("--seed", type=int, default=42)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--source", default="encounter_embeddings")
    cal.add_argument("--mask", type=Path, default=None)
    cal.add_argument("--allow-uniform-mask", action="store_true")
    cal.add_argument("--epsilons", type=float, nargs="+",
                     default=[50, 100, 200, 500, 1000, 2000, 5000])
    cal.add_argument("--k", type=int, default=5)
    cal.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("build")
    b.add_argument("--mechanism", choices=["sparse", "laplace"], default="sparse")
    b.add_argument("--mask", type=Path, default=None)
    b.add_argument("--allow-uniform-mask", action="store_true")
    b.add_argument("--source", default="encounter_embeddings")
    b.add_argument("--target", required=True)
    b.add_argument("--epsilon", type=float, required=True)
    b.add_argument("--draw", type=int, required=True)
    b.add_argument("--debug-seed", type=int, default=None)
    b.add_argument("--unit-normalize", action="store_true")
    a = p.parse_args()
    a.project = a.project.resolve()
    a.workdir = a.workdir.resolve()
    if a.command == "prepare" and a.limit_pairs is not None and a.limit_pairs < 2:
        p.error("Need at least 2 pairs")
    if a.command == "train" and (a.epochs < 1 or a.batch_size < 1 or a.lr <= 0 or a.lam < 0
                                 or a.min_epochs < 1 or a.min_epochs > a.epochs):
        p.error("Invalid training arguments")
    if a.command == "build" and (a.epsilon <= 0 or not np.isfinite(a.epsilon)):
        p.error("Positive finite epsilon required")
    if a.command == "calibrate" and (a.k < 1 or any(e <= 0 for e in a.epsilons)):
        p.error("Invalid calibration arguments")
    {"prepare": prepare, "train": train, "calibrate": calibrate, "build": build}[a.command](a)


if __name__ == "__main__":
    main()
