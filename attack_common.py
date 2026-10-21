import json
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.json"
GEIA_DIR = DATA_DIR / "geia"

FACT_FIELDS = ["conditions", "medications", "procedures", "allergies"]


def load_ground_truth():
    return json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))


def load_split(name):
    path = GEIA_DIR / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run create_geia_splits.py first"
        )
    encounter_ids = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            encounter_ids.append(json.loads(line)["encounter_id"])
    return encounter_ids


def fetch_embeddings(encounter_ids, use_collection=None):
    if use_collection is not None:
        source = use_collection
    else:
        from ingest import collection as source

    result = source.get(ids=list(encounter_ids), include=["embeddings"])
    id_to_embedding = dict(zip(result["ids"], result["embeddings"]))
    missing = [eid for eid in encounter_ids if eid not in id_to_embedding]
    if missing:
        raise KeyError(
            f"{len(missing)} encounter ids are missing from the vector store "
            f"(run ingest_notes.py first), e.g. {missing[:5]}"
        )
    return np.array([id_to_embedding[eid] for eid in encounter_ids], dtype=np.float32)


def note_terms(entry):
    terms = set()
    for field in FACT_FIELDS:
        for term in entry.get(field) or []:
            terms.add(term)
    return terms


def build_vocab(ground_truth, encounter_ids, min_df=5, max_df_ratio=0.6, top_k=200):
    counts = Counter()
    for eid in encounter_ids:
        for term in note_terms(ground_truth[eid]):
            counts[term] += 1

    max_df = int(len(encounter_ids) * max_df_ratio)
    vocab = [t for t, c in counts.items() if min_df <= c <= max_df]
    vocab.sort(key=lambda t: -counts[t])
    return vocab[:top_k]


def label_matrix(ground_truth, encounter_ids, vocab):
    vocab_index = {term: i for i, term in enumerate(vocab)}
    y = np.zeros((len(encounter_ids), len(vocab)), dtype=int)
    for row, eid in enumerate(encounter_ids):
        for term in note_terms(ground_truth[eid]):
            col = vocab_index.get(term)
            if col is not None:
                y[row, col] = 1
    return y