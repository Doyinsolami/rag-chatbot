import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "chroma_db"
MANIFEST_DIR = PROJECT_ROOT / "data" / "results" / "dp_manifests"
EMBEDDING_DIM = 384


def phi(t):
    return 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))


def analytic_gaussian_sigma(epsilon, delta, sensitivity, tol=1e-12):
    def case_a(eps, s):
        return phi(math.sqrt(eps * s)) - math.exp(eps) * phi(-math.sqrt(eps * (s + 2.0)))

    def case_b(eps, s):
        return phi(-math.sqrt(eps * s)) - math.exp(eps) * phi(-math.sqrt(eps * (s + 2.0)))

    def doubling(predicate_stop, s_inf, s_sup):
        while not predicate_stop(s_sup):
            s_inf = s_sup
            s_sup = 2.0 * s_inf
        return s_inf, s_sup

    def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup - s_inf) / 2.0
        while not predicate_stop(s_mid):
            if predicate_left(s_mid):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup - s_inf) / 2.0
        return s_mid

    delta_thr = case_a(epsilon, 0.0)

    if delta == delta_thr:
        alpha = 1.0
    else:
        if delta > delta_thr:
            predicate_stop_dt = lambda s: case_a(epsilon, s) >= delta
            s_to_delta = lambda s: case_a(epsilon, s)
            predicate_left_bs = lambda s: s_to_delta(s) > delta
            s_to_alpha = lambda s: math.sqrt(1.0 + s / 2.0) - math.sqrt(s / 2.0)
        else:
            predicate_stop_dt = lambda s: case_b(epsilon, s) <= delta
            s_to_delta = lambda s: case_b(epsilon, s)
            predicate_left_bs = lambda s: s_to_delta(s) < delta
            s_to_alpha = lambda s: math.sqrt(1.0 + s / 2.0) + math.sqrt(s / 2.0)

        predicate_stop_bs = lambda s: abs(s_to_delta(s) - delta) <= tol
        s_inf, s_sup = doubling(predicate_stop_dt, 0.0, 1.0)
        s_final = binary_search(predicate_stop_bs, predicate_left_bs, s_inf, s_sup)
        alpha = s_to_alpha(s_final)

    return alpha * sensitivity / math.sqrt(2.0 * epsilon)


def classical_gaussian_sigma(epsilon, delta, sensitivity):
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


def clip_embeddings(embeddings, clip_norm):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    scale = np.minimum(1.0, clip_norm / np.maximum(norms, 1e-12))
    return embeddings * scale


def add_gaussian_noise(embeddings, sigma, rng):
    noise = rng.normal(0.0, sigma, size=embeddings.shape)
    return embeddings + noise


def l2_normalize(embeddings, eps=1e-12):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, eps)


def norm_stats(embeddings, clip_norm=None):
    norms = np.linalg.norm(embeddings, axis=1)
    stats = {
        "min": float(np.min(norms)),
        "mean": float(np.mean(norms)),
        "median": float(np.median(norms)),
        "p95": float(np.percentile(norms, 95)),
        "max": float(np.max(norms)),
    }
    if clip_norm is not None:
        exceeding = int(np.sum(norms > clip_norm))
        stats["num_exceeding_c"] = exceeding
        stats["pct_exceeding_c"] = float(100.0 * exceeding / len(norms))
    return stats


def load_all(collection, batch_size):
    ids = []
    embeddings = []
    metadatas = []
    documents = []
    offset = 0
    while True:
        batch = collection.get(
            limit=batch_size,
            offset=offset,
            include=["embeddings", "metadatas", "documents"],
        )
        batch_ids = batch["ids"]
        if not batch_ids:
            break
        ids.extend(batch_ids)
        embeddings.extend(batch["embeddings"])
        batch_meta = batch["metadatas"] if batch["metadatas"] is not None else [{}] * len(batch_ids)
        metadatas.extend(batch_meta)
        batch_docs = batch["documents"] if batch["documents"] is not None else [None] * len(batch_ids)
        documents.extend(batch_docs)
        offset += len(batch_ids)
    return ids, np.asarray(embeddings, dtype=np.float64), metadatas, documents


def validate(ids, embeddings):
    if embeddings.ndim != 2 or embeddings.shape[0] != len(ids):
        raise ValueError("Embedding count does not match id count")
    if embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"Expected {EMBEDDING_DIM} dimensions, found {embeddings.shape[1]}")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("Non-finite values found in source embeddings")


def build_dp_collection(args):
    import chromadb

    chroma_path = Path(args.chroma_path).resolve()
    client = chromadb.PersistentClient(path=str(chroma_path))

    source = client.get_collection(args.source_collection)
    source_count = source.count()

    ids, embeddings, metadatas, documents = load_all(source, args.batch_size)
    validate(ids, embeddings)
    print(f"Loaded {len(ids)} embeddings from {args.source_collection}")

    before_stats = norm_stats(embeddings, args.clip_norm)
    print("Source embedding norm statistics:")
    for key, value in before_stats.items():
        print(f"  {key}: {value}")

    sensitivity = 2.0 * args.clip_norm
    sigma = analytic_gaussian_sigma(args.epsilon, args.delta, sensitivity)
    classical = classical_gaussian_sigma(args.epsilon, args.delta, sensitivity)
    classical_valid = args.epsilon <= 1.0
    print(f"Sensitivity (2C): {sensitivity}")
    print(f"Analytic Gaussian sigma (Balle-Wang 2018): {sigma}")
    print(f"Classical formula sigma: {classical} (valid only for epsilon<=1: {classical_valid})")

    sample_source = embeddings[0].copy()

    clipped = clip_embeddings(embeddings, args.clip_norm)

    rng_main = np.random.default_rng()
    noised_raw = add_gaussian_noise(clipped, sigma, rng_main)
    noised = l2_normalize(noised_raw)

    noise = noised_raw - clipped
    observed_sigma = float(np.std(noise))
    if not 0.98 * sigma <= observed_sigma <= 1.02 * sigma:
        raise RuntimeError(f"Observed noise std {observed_sigma} does not match sigma {sigma}")
    if np.allclose(noised_raw, clipped):
        raise RuntimeError("Noise was not applied")
    print(f"Observed noise std: {observed_sigma} (target {sigma})")

    after_raw_stats = norm_stats(noised_raw)
    after_stats = norm_stats(noised)
    print("Noised embedding norm statistics (after clip + noise, before normalization):")
    for key, value in after_raw_stats.items():
        print(f"  {key}: {value}")
    print("Final embedding norm statistics (after L2 normalization, DP-safe post-processing):")
    for key, value in after_stats.items():
        print(f"  {key}: {value}")

    existing = [getattr(c, "name", c) for c in client.list_collections()]
    if args.target_collection in existing:
        if not args.overwrite:
            raise SystemExit(
                f"Target collection {args.target_collection} already exists. "
                f"Pass --overwrite to replace it."
            )
        client.delete_collection(args.target_collection)
        print(f"Overwrote existing target collection {args.target_collection}")

    target = client.create_collection(args.target_collection)

    dp_fields = {
        "dp_applied": True,
        "dp_epsilon": float(args.epsilon),
        "dp_delta": float(args.delta),
        "dp_clip_norm": float(args.clip_norm),
        "dp_sensitivity": float(sensitivity),
        "dp_noise_sigma": float(sigma),
        "dp_randomness": "os_entropy_no_seed_stored",
        "dp_draw": int(args.draw),
        "dp_mechanism": "gaussian",
        "dp_unit": "encounter",
        "dp_adjacency": "replace_one",
    }
    new_metadatas = []
    for meta in metadatas:
        merged = dict(meta) if meta else {}
        merged.update(dp_fields)
        new_metadatas.append(merged)

    noised_list = noised.tolist()
    has_docs = any(doc is not None for doc in documents)
    for start in range(0, len(ids), args.batch_size):
        end = start + args.batch_size
        add_kwargs = {
            "ids": ids[start:end],
            "embeddings": noised_list[start:end],
            "metadatas": new_metadatas[start:end],
        }
        if has_docs:
            add_kwargs["documents"] = documents[start:end]
        target.add(**add_kwargs)
        print(f"Wrote {min(end, len(ids))} / {len(ids)} to {args.target_collection}")

    source_after = client.get_collection(args.source_collection)
    if source_after.count() != source_count:
        raise RuntimeError("Source collection count changed during the DP run")
    refetch = source_after.get(ids=[ids[0]], include=["embeddings"])
    if not np.allclose(np.asarray(refetch["embeddings"][0], dtype=np.float64), sample_source):
        raise RuntimeError("A source embedding changed during the DP run")
    if target.count() != source_count:
        raise RuntimeError(
            f"Target count {target.count()} does not match source count {source_count}"
        )
    print("Verification passed: source unchanged, target count matches source")

    manifest = {
        "source_collection": args.source_collection,
        "target_collection": args.target_collection,
        "num_embeddings": len(ids),
        "embedding_dim": EMBEDDING_DIM,
        "epsilon": args.epsilon,
        "delta": args.delta,
        "clip_norm": args.clip_norm,
        "sensitivity": sensitivity,
        "sigma": sigma,
        "sigma_classical": classical,
        "classical_formula_valid": classical_valid,
        "randomness": "fresh OS entropy; no seed stored",
        "draw": args.draw,
        "observed_noise_std": observed_sigma,
        "mechanism": "gaussian",
        "calibration": "analytic_gaussian_balle_wang_2018",
        "adjacency": "replace_one",
        "unit": "encounter",
        "distance_metric": "l2",
        "source_embeddings_are_unit_norm": True,
        "l2_on_unit_vectors_equivalent_to_cosine_ranking": True,
        "normalized_after_noise": True,
        "normalization_note": "L2 normalization applied after noise addition as DP-safe "
                               "post-processing, to keep DP retrieval in the same effective "
                               "geometry as the clean collection (source embeddings are unit-norm, "
                               "so L2 ranking there is already equivalent to cosine ranking).",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_clipped": before_stats["num_exceeding_c"],
        "norm_stats_before_clipping": before_stats,
        "norm_stats_after_noise_before_normalization": after_raw_stats,
        "norm_stats_after_normalization": after_stats,
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / f"{args.target_collection}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest to {manifest_path}")
    print(f"Done. DP collection {args.target_collection} holds {target.count()} embeddings")


def run_self_tests():
    v = np.array([[3.0, 4.0]])
    clipped = clip_embeddings(v, 1.0)
    assert np.isclose(np.linalg.norm(clipped[0]), 1.0), "clipping to norm C failed"

    small = np.array([[0.1, 0.0]])
    assert np.allclose(clip_embeddings(small, 1.0), small), "clipping altered a sub-C vector"

    sensitivity = 2.0 * 1.0
    for eps in [0.5, 1.0, 2.0, 5.0]:
        sigma = analytic_gaussian_sigma(eps, 1e-5, sensitivity)
        delta_check = (
            phi(sensitivity / (2 * sigma) - eps * sigma / sensitivity)
            - math.exp(eps) * phi(-sensitivity / (2 * sigma) - eps * sigma / sensitivity)
        )
        assert abs(delta_check - 1e-5) < 1e-6, f"sigma at eps={eps} does not satisfy DP equation: {delta_check}"

    assert analytic_gaussian_sigma(2.0, 1e-5, sensitivity) < analytic_gaussian_sigma(1.0, 1e-5, sensitivity), \
        "larger epsilon should give smaller sigma"

    for eps in [1.0, 5.0, 10.0, 20.0, 50.0]:
        sigma = analytic_gaussian_sigma(eps, 1e-5, sensitivity)
        delta_check = (
            phi(sensitivity / (2 * sigma) - eps * sigma / sensitivity)
            - math.exp(eps) * phi(-sensitivity / (2 * sigma) - eps * sigma / sensitivity)
        )
        assert abs(delta_check - 1e-5) < 1e-7, f"sigma at eps={eps} does not satisfy DP equation: {delta_check}"

    base = np.zeros((2000, EMBEDDING_DIM))
    a = add_gaussian_noise(base, 1.0, np.random.default_rng())
    b = add_gaussian_noise(base, 1.0, np.random.default_rng())
    assert not np.allclose(a, b), "two fresh draws produced identical noise"
    assert abs(np.corrcoef(a.ravel(), b.ravel())[0, 1]) < 0.01, "two fresh draws are correlated"
    assert abs(np.std(a) - 1.0) < 0.01, "noise std does not match sigma"

    normed = l2_normalize(a)
    assert np.allclose(np.linalg.norm(normed, axis=1), 1.0), "normalization did not produce unit norm"

    print("All self-tests passed")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--clip-norm", type=float, default=None)
    parser.add_argument("--draw", type=int, default=1)
    parser.add_argument("--source-collection", default="encounter_embeddings")
    parser.add_argument("--target-collection", default=None)
    parser.add_argument("--chroma-path", default=str(DEFAULT_CHROMA_DIR))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.epsilon <= 0:
        raise SystemExit(f"--epsilon must be > 0, got {args.epsilon}")
    if not (0.0 < args.delta < 1.0):
        raise SystemExit(f"--delta must be in (0, 1), got {args.delta}")
    if args.clip_norm <= 0:
        raise SystemExit(f"--clip-norm must be > 0, got {args.clip_norm}")
    if args.batch_size <= 0:
        raise SystemExit(f"--batch-size must be > 0, got {args.batch_size}")
    if args.draw < 1:
        raise SystemExit(f"--draw must be >= 1, got {args.draw}")


def main():
    args = parse_args()
    if args.self_test:
        run_self_tests()
        return
    if args.epsilon is None or args.clip_norm is None:
        raise SystemExit("--epsilon and --clip-norm are required")
    validate_args(args)
    if args.target_collection is None:
        eps_str = f"{args.epsilon:g}".replace(".", "p")
        args.target_collection = f"encounters_dp_v2_eps{eps_str}_draw{args.draw}"
    build_dp_collection(args)


if __name__ == "__main__":
    main()