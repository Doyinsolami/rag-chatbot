import argparse
import json
import re
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
GEIA_DIR = PROJECT_ROOT / "data" / "geia"
SPLITS = ("train", "validation", "test")
WORD_RE = re.compile(r"[a-z0-9]+")


def load_split_records(name):
    path = GEIA_DIR / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run create_geia_splits.py first")
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def tokenize(text):
    return frozenset(WORD_RE.findall(text.lower()))


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compare_splits(records_a, records_b):
    tokens_b = [(r["encounter_id"], r["patient_id"], tokenize(r["text"])) for r in records_b]
    texts_b = {r["encounter_id"]: r["text"] for r in records_b}

    exact_matches = []
    near_matches = []

    for record in records_a:
        text_a = record["text"]
        tokens_a = tokenize(text_a)

        best_score = -1.0
        best_match = None
        for eid_b, pid_b, tset_b in tokens_b:
            if text_a == texts_b[eid_b]:
                exact_matches.append({
                    "encounter_id_a": record["encounter_id"],
                    "patient_id_a": record["patient_id"],
                    "encounter_id_b": eid_b,
                    "patient_id_b": pid_b,
                })
            score = jaccard(tokens_a, tset_b)
            if score > best_score:
                best_score = score
                best_match = (eid_b, pid_b)

        near_matches.append({
            "encounter_id": record["encounter_id"],
            "patient_id": record["patient_id"],
            "best_match_encounter_id": best_match[0] if best_match else None,
            "best_match_patient_id": best_match[1] if best_match else None,
            "best_match_jaccard": best_score,
        })

    return exact_matches, near_matches


def summarize(near_matches, threshold):
    scores = sorted((m["best_match_jaccard"] for m in near_matches), reverse=True)
    above_threshold = [m for m in near_matches if m["best_match_jaccard"] >= threshold]
    return {
        "n_compared": len(near_matches),
        "n_above_threshold": len(above_threshold),
        "threshold": threshold,
        "max_jaccard": scores[0] if scores else None,
        "mean_jaccard": sum(scores) / len(scores) if scores else None,
        "top_matches": sorted(above_threshold, key=lambda m: -m["best_match_jaccard"])[:20],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    records = {name: load_split_records(name) for name in SPLITS}

    report = {"threshold": args.threshold, "pairs": {}}

    for name_a, name_b in combinations(SPLITS, 2):
        print(f"Comparing {name_a} ({len(records[name_a])} notes) against "
              f"{name_b} ({len(records[name_b])} notes)...")
        exact_matches, near_matches = compare_splits(records[name_a], records[name_b])
        summary = summarize(near_matches, args.threshold)

        pair_key = f"{name_a}_vs_{name_b}"
        report["pairs"][pair_key] = {
            "n_exact_matches": len(exact_matches),
            "exact_matches": exact_matches,
            **summary,
        }

        print(f"  Exact text matches: {len(exact_matches)}")
        print(f"  Notes with a near-duplicate (Jaccard >= {args.threshold}) in the other split: "
              f"{summary['n_above_threshold']} / {summary['n_compared']}")
        print(f"  Max Jaccard similarity found: {summary['max_jaccard']:.3f}")
        print(f"  Mean of each note's best-match Jaccard similarity: {summary['mean_jaccard']:.3f}")

    output_path = GEIA_DIR / "duplicate_check_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved full report to {output_path}")


if __name__ == "__main__":
    main()
