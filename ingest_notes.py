import json
from pathlib import Path

from ingest import model, collection

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
NOTES_DIR = DATA_DIR / "notes"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.json"
BATCH_SIZE = 256


def load_ground_truth():
    return json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))


def load_notes(ground_truth):
    ids = []
    texts = []
    metadatas = []
    for note_path in sorted(NOTES_DIR.glob("*.txt")):
        encounter_id = note_path.stem
        if encounter_id not in ground_truth:
            raise KeyError(
                f"Encounter {encounter_id} has a note file but no entry in ground_truth.json"
            )
        entry = ground_truth[encounter_id]
        ids.append(encounter_id)
        texts.append(note_path.read_text(encoding="utf-8"))
        metadatas.append({"patient_id": entry["patient_id"]})
    return ids, texts, metadatas

def main():
    ground_truth = load_ground_truth()
    ids, texts, metadatas = load_notes(ground_truth)
    print(f"Loaded {len(ids)} notes")

    for start in range(0, len(ids), BATCH_SIZE):
        batch_ids = ids[start:start + BATCH_SIZE]
        batch_texts = texts[start:start + BATCH_SIZE]
        batch_metadatas = metadatas[start:start + BATCH_SIZE]
        embeddings = model.encode(batch_texts).tolist()
        collection.upsert(
            ids=batch_ids,
            embeddings=embeddings,
            metadatas=batch_metadatas,
        )
        print(f"Embedded {start + len(batch_ids)} / {len(ids)}")

    print(f"Done. Collection now holds {collection.count()} documents")


if __name__ == "__main__":
    main()