import argparse
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from attack_common import fetch_embeddings, load_split

PROJECT_ROOT = Path(__file__).resolve().parent
NOTES_DIR = PROJECT_ROOT / "data" / "notes"
VICTIM_MODEL = "all-MiniLM-L6-v2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    encounter_ids = load_split(args.split)[: args.n]
    print(f"Checking {len(encounter_ids)} encounters from the {args.split} split")

    stored = fetch_embeddings(encounter_ids)

    model = SentenceTransformer(VICTIM_MODEL)
    texts = [(NOTES_DIR / f"{eid}.txt").read_text(encoding="utf-8") for eid in encounter_ids]
    recomputed = np.array(model.encode(texts), dtype=np.float32)

    abs_diff = np.abs(stored - recomputed)
    max_abs_diff = float(abs_diff.max())
    mean_abs_diff = float(abs_diff.mean())

    cos_sims = []
    for i in range(len(encounter_ids)):
        a, b = stored[i], recomputed[i]
        cos_sims.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    min_cos_sim = min(cos_sims)

    print(f"Max absolute difference per element: {max_abs_diff:.8f}")
    print(f"Mean absolute difference per element: {mean_abs_diff:.8f}")
    print(f"Minimum cosine similarity across the {len(encounter_ids)} pairs: {min_cos_sim:.8f}")

    if max_abs_diff < 1e-5:
        print("\nResult: stored and recomputed embeddings match to floating-point precision.")
    elif min_cos_sim > 0.999:
        print("\nResult: stored and recomputed embeddings are extremely close but not "
              "bit-identical (likely CPU/GPU or batching floating-point differences). "
              "Practically equivalent for this attack, but worth noting as a caveat.")
    else:
        print("\nResult: stored and recomputed embeddings diverge meaningfully. "
              "Do not trust downstream attack numbers until this is explained -- check "
              "sentence-transformers version, pooling config, and whether ingest_notes.py "
              "used the same model settings as this script.")


if __name__ == "__main__":
    main()
