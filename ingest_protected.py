from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
NOTES_DIR = DATA_DIR / "notes"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(name="encounter_embeddings")


def embed_text(text):
    return model.encode(text).tolist()


def get_collection(name):
    return chroma_client.get_collection(name)


def uses_protected_text(target, record_metadata):
    settings = getattr(target, "metadata", None) or {}
    metadata = record_metadata or {}
    name = getattr(target, "name", "")
    return (
        any(bool(metadata.get(key) or settings.get(key))
            for key in ("redaction_applied", "lprag_applied"))
        or name.startswith(("encounters_redacted", "encounters_lprag"))
    )


def search(query_text, n_results=2, patient_id=None, use_collection=None):
    target = use_collection if use_collection is not None else collection
    where = {"patient_id": patient_id} if patient_id else None
    results = target.query(
        query_embeddings=[embed_text(query_text)],
        n_results=n_results,
        where=where,
        include=["metadatas", "distances", "documents"],
    )
    ids = results["ids"][0]
    stored = results.get("documents")
    stored_docs = stored[0] if stored and stored[0] is not None else []
    metadata_batches = results.get("metadatas")
    metadatas = metadata_batches[0] if metadata_batches and metadata_batches[0] is not None else []
    documents, sources = [], []
    for index, encounter_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) else {}
        if uses_protected_text(target, metadata):
            document = stored_docs[index] if index < len(stored_docs) else None
            if not isinstance(document, str):
                raise RuntimeError(
                    f"Protected document missing for {encounter_id} in {target.name}. "
                    "Rebuild the text-defense collection with its protected documents. "
                    "Original notes will not be substituted."
                )
            documents.append(document)
            sources.append("protected_collection_document")
        else:
            note_path = NOTES_DIR / f"{encounter_id}.txt"
            documents.append(note_path.read_text(encoding="utf-8"))
            sources.append("original_note")
    results["documents"] = [documents]
    results["document_sources"] = [sources]
    return results
