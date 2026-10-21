import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import secrets
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
NOTES_DIR = PROJECT_ROOT / "data" / "notes"
GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "ground_truth.json"
MANIFEST_DIR = PROJECT_ROOT / "data" / "results" / "lprag_manifests"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MECHANISM = "lprag_v2"

PRIVACY_LABELS = {
    "PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART",
    "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY",
    "ORDINAL", "CARDINAL",
}
SKIP_WORDS = {"year", "years", "old", "decade", "decades", "month", "months", "day", "days"}
NUMBER_RE = re.compile(r"[-+]?\d*\.\d+|\d+")


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    kind: str


def classify_entity(text):
    tokens = text.split()
    if len(tokens) == 1:
        return "NUMBER" if NUMBER_RE.fullmatch(tokens[0]) else "WORD"
    return "PHRASE"


def identify_entities(doc):
    return [
        Entity(ent.text, ent.label_, ent.start_char, ent.end_char, classify_entity(ent.text))
        for ent in doc.ents
        if ent.label_ in PRIVACY_LABELS and ent.text.strip()
    ]


class BertImportance:
    def __init__(self, device):
        import torch
        from transformers import BertModel, BertTokenizerFast
        self.torch = torch
        self.device = device
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased").to(device).eval()

    def piece_vectors(self, text):
        enc = self.tokenizer(text, return_offsets_mapping=True, return_overflowing_tokens=True,
                             truncation=True, max_length=512, stride=128, padding=True,
                             return_tensors="pt")
        offsets = enc.pop("offset_mapping").tolist()
        enc.pop("overflow_to_sample_mapping", None)
        with self.torch.no_grad():
            hidden = self.model(**{k: v.to(self.device) for k, v in enc.items()}).last_hidden_state
        hidden = hidden.float().cpu().numpy()
        spans = {}
        for window, window_offsets in enumerate(offsets):
            for position, (start, end) in enumerate(window_offsets):
                if end > start and (start, end) not in spans:
                    spans[(start, end)] = hidden[window, position]
        return spans

    def importances(self, text, entities):
        spans = self.piece_vectors(text)
        scores = []
        for ent in entities:
            word_vectors = []
            for match in re.finditer(r"\S+", ent.text):
                lo, hi = ent.start + match.start(), ent.start + match.end()
                pieces = [vec for (s, e), vec in spans.items() if s >= lo and e <= hi]
                if pieces:
                    word_vectors.append(np.mean(pieces, axis=0))
            scores.append(float(np.linalg.norm(np.mean(word_vectors, axis=0))) if word_vectors else None)
        known = [s for s in scores if s is not None]
        default = float(np.mean(known)) if known else 1.0
        return [default if s is None else s for s in scores]


def allocate_budgets(importances, total_epsilon):
    total = sum(importances)
    if total <= 0:
        return [total_epsilon / len(importances)] * len(importances)
    return [total_epsilon * score / total for score in importances]


class CandidateSource:
    def __init__(self):
        import gensim.downloader as api
        self.w2v = api.load("word2vec-google-news-300")
        self.cache = {}

    def lookup_key(self, word):
        stripped = re.sub(r"\d+", "", word)
        for key in (word, word.lower(), stripped, stripped.lower()):
            if key and key in self.w2v.key_to_index:
                return key
        return None

    def get(self, word, top_c):
        if word not in self.cache:
            key = self.lookup_key(word)
            if key is not None:
                candidates = [w.replace("_", " ") for w, _ in self.w2v.similar_by_word(key, topn=top_c)]
                source = "word2vec"
            else:
                candidates = []
                source = "oov_unchanged"
            self.cache[word] = ([c for c in candidates if c != word], source)
        return self.cache[word]


def random_response(word, candidates, epsilon, rng):
    pool = [word] + candidates
    if len(pool) <= 1:
        return word
    p_keep = math.exp(epsilon) / (math.exp(epsilon) + len(pool) - 1)
    if rng.random() < p_keep:
        return word
    return pool[1 + int(rng.integers(0, len(pool) - 1))]


def piecewise_mechanism(value, epsilon, rng):
    exp_half = math.exp(epsilon / 2.0)
    q = (exp_half + 1.0) / (exp_half - 1.0)
    left = (q + 1.0) / 2.0 * value - (q - 1.0) / 2.0
    right = left + q - 1.0
    if rng.random() < exp_half / (exp_half + 1.0):
        return rng.uniform(left, right)
    left_width, right_width = left + q, q - right
    if rng.random() < left_width / (left_width + right_width):
        return rng.uniform(-q, left)
    return rng.uniform(right, q)


def local_hashing(text, epsilon, rng, domain):
    salt = secrets.randbits(64)
    true_bucket = int(hashlib.sha256(f"{text}|{salt}".encode("utf-8")).hexdigest(), 16) % domain
    p_keep = math.exp(epsilon) / (math.exp(epsilon) + domain - 1)
    if rng.random() < p_keep:
        return true_bucket
    other = int(rng.integers(0, domain - 1))
    return other if other < true_bucket else other + 1


def is_discrete_number(text):
    digits = re.sub(r"\D", "", text)
    return len(digits) >= 6 or (text.startswith("0") and len(text) > 1 and "." not in text)


def perturb_number(text, epsilon, rng, args):
    if is_discrete_number(text):
        bucket = local_hashing(text, epsilon, rng, args.lh_domain)
        length = max(len(re.sub(r"\D", "", text)), 1)
        return str(bucket).zfill(length)[-length:]
    value = min(max(float(text), 0.0), args.number_max)
    unit = 2.0 * value / args.number_max - 1.0
    noisy = piecewise_mechanism(unit, epsilon, rng)
    restored = (noisy + 1.0) / 2.0 * args.number_max
    if text.isdigit():
        return str(round(restored))
    return f"{restored:.1f}"


def perturb_entity(ent, epsilon, rng, candidates, args, stats):
    if ent.kind == "NUMBER":
        stats["number_tokens"] += 1
        return perturb_number(ent.text, epsilon, rng, args)
    tokens = ent.text.split()
    active = [t for t in tokens if t.lower() not in SKIP_WORDS]
    if not active:
        return ent.text
    per_token = epsilon / len(active)
    out = []
    for token in tokens:
        if token.lower() in SKIP_WORDS:
            out.append(token)
            stats["skip_tokens"] += 1
        elif NUMBER_RE.fullmatch(token):
            out.append(perturb_number(token, per_token, rng, args))
            stats["number_tokens"] += 1
        else:
            pool, source = candidates.get(token, args.top_c)
            stats[f"word_tokens_{source}"] += 1
            out.append(random_response(token, pool, per_token, rng))
    return " ".join(out)


def perturb_note(text, doc, importance, candidates, rng, args, stats, label_counts):
    entities = identify_entities(doc)
    if not entities:
        return text, []
    budgets = allocate_budgets(importance.importances(text, entities), args.epsilon)
    label_counts.update(ent.label for ent in entities)
    records = []
    out = text
    for ent, eps in sorted(zip(entities, budgets), key=lambda pair: pair[0].start, reverse=True):
        replacement = perturb_entity(ent, eps, rng, candidates, args, stats)
        out = out[:ent.start] + replacement + out[ent.end:]
        records.append({"text": ent.text, "label": ent.label, "kind": ent.kind,
                        "epsilon": eps, "replacement": replacement})
    return out, records[::-1]


def package_versions(names):
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--target-collection", default=None)
    parser.add_argument("--chroma-path", type=Path, default=PROJECT_ROOT / "chroma_db")
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-c", type=int, default=5)
    parser.add_argument("--lh-domain", type=int, default=100)
    parser.add_argument("--number-max", type=float, default=300.0)
    parser.add_argument("--spacy-batch", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--preview", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.epsilon <= 0:
        parser.error("--epsilon must be > 0")
    if args.top_c < 1 or args.lh_domain < 2 or args.number_max <= 0 or args.preview < 0:
        parser.error("Invalid top-c, lh-domain, number-max, or preview")
    if args.target_collection is None:
        args.target_collection = f"encounters_lprag_v2_eps{args.epsilon:g}".replace(".", "p")
    return args


def main():
    args = parse_args()
    import spacy
    import torch
    from sentence_transformers import SentenceTransformer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    ids = sorted(ground_truth)
    if args.preview:
        ids = ids[:: max(1, len(ids) // args.preview)][: args.preview]
    texts = [(NOTES_DIR / f"{eid}.txt").read_text(encoding="utf-8") for eid in ids]

    client = None
    if not args.preview:
        import chromadb
        client = chromadb.PersistentClient(path=str(args.chroma_path.resolve()))
        names = {getattr(c, "name", c) for c in client.list_collections()}
        if args.target_collection in names and not args.overwrite:
            raise SystemExit("Collection already exists. Choose a new name or use --overwrite.")

    if device.startswith("cuda"):
        spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf")
    importance = BertImportance(device)
    victim = SentenceTransformer(EMBEDDING_MODEL, device=device)
    candidates = CandidateSource()
    rng = np.random.default_rng()

    perturbed, metadatas, examples = [], [], []
    stats, label_counts = Counter(), Counter()
    notes_without_entities = 0
    for index, (eid, text, doc) in enumerate(zip(ids, texts, nlp.pipe(texts, batch_size=args.spacy_batch)), 1):
        new_text, records = perturb_note(text, doc, importance, candidates, rng, args, stats, label_counts)
        notes_without_entities += int(not records)
        perturbed.append(new_text)
        metadatas.append({
            "patient_id": ground_truth[eid]["patient_id"],
            "lprag_applied": True,
            "lprag_mechanism": MECHANISM,
            "lprag_epsilon": float(args.epsilon),
            "lprag_num_entities": len(records),
        })
        if args.preview:
            examples.append({"encounter_id": eid, "original": text, "perturbed": new_text, "entities": records})
        if index % 100 == 0 or index == len(ids):
            print(f"Perturbed {index}/{len(ids)} notes", flush=True)

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    if args.preview:
        path = MANIFEST_DIR / f"{args.target_collection}_preview.json"
        path.write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Preview saved to {path}. No collection was created.")
        print(f"Entities by label: {dict(label_counts)}")
        print(f"Token stats: {dict(stats)}")
        return

    vectors = victim.encode(perturbed, convert_to_numpy=True, normalize_embeddings=False,
                            batch_size=args.batch_size, show_progress_bar=True)
    if len(vectors) != len(ids) or not np.isfinite(vectors).all():
        raise RuntimeError("Invalid embeddings; collection was not changed")

    names = {getattr(c, "name", c) for c in client.list_collections()}
    if args.target_collection in names:
        client.delete_collection(args.target_collection)
    target = client.create_collection(args.target_collection, metadata={"lprag_applied": True})
    batch_size = min(args.batch_size, client.get_max_batch_size()) if hasattr(client, "get_max_batch_size") else args.batch_size
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        target.add(ids=ids[start:end], embeddings=vectors[start:end].tolist(),
                   metadatas=metadatas[start:end], documents=perturbed[start:end])
    if target.count() != len(ids):
        raise RuntimeError("Collection count does not match source count")

    manifest = {
        "target_collection": args.target_collection,
        "mechanism": MECHANISM,
        "reference": "He et al. 2025, Information Processing and Management 62, 104150",
        "epsilon_per_note": args.epsilon,
        "num_notes": len(ids),
        "ner_model": "en_core_web_trf",
        "privacy_labels": sorted(PRIVACY_LABELS),
        "budget_allocation": "APB: L2 norm of mean contextual bert-base-uncased word vectors, aligned by character offsets",
        "word_mechanism": "random response over the word and its top-c word2vec-google-news-300 neighbours",
        "top_c": args.top_c,
        "oov_candidates": "word2vec lookup on word, lowercase, and digit-stripped forms; tokens with no match left unchanged, as in the reference code",
        "number_continuous": "piecewise mechanism on inputs clamped to the public range [0, number_max], output unclamped as in the paper",
        "number_max": args.number_max,
        "number_discrete": "local hashing, SHA-256 with fresh random salt, generalized random response over lh_domain buckets",
        "lh_domain": args.lh_domain,
        "discrete_rule": "6+ digits or leading zero integer",
        "phrase_budget": "epsilon split evenly over non-skip tokens",
        "skip_words": sorted(SKIP_WORDS),
        "replacement": "character offsets",
        "randomness": "fresh OS entropy; no seed stored",
        "embedding_model": EMBEDDING_MODEL,
        "normalize_embeddings": False,
        "entities_by_label": dict(label_counts),
        "token_stats": dict(stats),
        "notes_without_entities": notes_without_entities,
        "avg_entities_per_note": sum(label_counts.values()) / len(ids),
        "packages": package_versions(("spacy", "spacy-transformers", "transformers", "gensim",
                                      "torch", "sentence-transformers", "chromadb")),
        "device": device,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = MANIFEST_DIR / f"{args.target_collection}.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done: {target.count()} notes in {args.target_collection}. Manifest: {path}")


if __name__ == "__main__":
    main()