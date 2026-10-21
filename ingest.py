from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
NOTES_DIR = DATA_DIR / "notes"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(
    name="encounter_embeddings"
)


def embed_text(text):
    return model.encode(text).tolist()


def get_collection(name):
    return chroma_client.get_collection(name)


def search(query_text, n_results=2, patient_id=None, use_collection=None):
    target = use_collection if use_collection is not None else collection
    where = {"patient_id": patient_id} if patient_id else None
    results = target.query(
        query_embeddings=[embed_text(query_text)],
        n_results=n_results,
        where=where,
        include=["metadatas", "distances"],
    )
    documents = []
    for encounter_id in results["ids"][0]:
        note_path = NOTES_DIR / f"{encounter_id}.txt"
        documents.append(note_path.read_text(encoding="utf-8"))
    results["documents"] = [documents]
    return results