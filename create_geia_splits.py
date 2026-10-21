import json
import random
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"
NOTES_DIR = DATA_DIR / "notes"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.json"
OUTPUT_DIR = DATA_DIR / "geia"

SEED = 42
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15


def assign_patient_splits(patient_ids):
    patient_ids = sorted(patient_ids)
    random.Random(SEED).shuffle(patient_ids)

    total = len(patient_ids)
    train_end = int(total * TRAIN_RATIO)
    validation_end = train_end + int(total * VALIDATION_RATIO)

    return {
        "train": set(patient_ids[:train_end]),
        "validation": set(patient_ids[train_end:validation_end]),
        "test": set(patient_ids[validation_end:]),
    }


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    patient_ids = {entry["patient_id"] for entry in ground_truth.values()}
    patient_splits = assign_patient_splits(patient_ids)

    patient_to_split = {
        patient_id: split
        for split, ids in patient_splits.items()
        for patient_id in ids
    }
    split_records = {"train": [], "validation": [], "test": []}

    for encounter_id, entry in ground_truth.items():
        note_path = NOTES_DIR / f"{encounter_id}.txt"
        if not note_path.exists():
            raise FileNotFoundError(f"Missing note for encounter {encounter_id}: {note_path}")

        text = note_path.read_text(encoding="utf-8").strip()
        split = patient_to_split[entry["patient_id"]]
        split_records[split].append(
            {
                "encounter_id": encounter_id,
                "patient_id": entry["patient_id"],
                "text": text,
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, records in split_records.items():
        records.sort(key=lambda record: record["encounter_id"])
        write_jsonl(OUTPUT_DIR / f"{split}.jsonl", records)

    overlap = (
        (patient_splits["train"] & patient_splits["validation"])
        | (patient_splits["train"] & patient_splits["test"])
        | (patient_splits["validation"] & patient_splits["test"])
    )
    if overlap:
        raise RuntimeError("Patient leakage detected across GEIA splits")

    all_word_counts = [
        len(record["text"].split())
        for records in split_records.values()
        for record in records
    ]
    manifest = {
        "seed": SEED,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VALIDATION_RATIO,
            "test": 1 - TRAIN_RATIO - VALIDATION_RATIO,
        },
        "patient_counts": {
            split: len(ids) for split, ids in patient_splits.items()
        },
        "encounter_counts": {
            split: len(records) for split, records in split_records.items()
        },
        "patient_overlap_count": len(overlap),
        "note_word_counts": {
            "minimum": min(all_word_counts),
            "maximum": max(all_word_counts),
            "mean": sum(all_word_counts) / len(all_word_counts),
        },
    }
    (OUTPUT_DIR / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("GEIA data preparation complete")
    for split in ("train", "validation", "test"):
        print(
            f"{split}: {len(patient_splits[split])} patients, "
            f"{len(split_records[split])} encounters"
        )
    print(f"Patient overlap across splits: {len(overlap)}")
    print(f"Files written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()