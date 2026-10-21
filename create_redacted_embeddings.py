import argparse
from collections import Counter, defaultdict
import random
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parent
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MECHANISM = "template_aware_entity_generalization_v2"

REPLACEMENTS = {
    "person full name": "somebody",
    "date of birth": "a specific date",
    "person age": "a certain age",
    "gender": "a gender",
    "eye color": "an eye color",
    "person height": "a height",
    "social security number": "a social security number",
    "passport number": "a passport number",
    "job title": "a job title",
    "job area": "a field of work",
    "job type": "a type of employment",
    "street address": "a street address",
    "building number": "a building number",
    "secondary address unit": "an apartment or unit number",
    "city name": "a city",
    "county": "a county",
    "state or province": "a state or province",
    "zip postal code": "a postal code",
    "nearby GPS coordinate": "GPS coordinates",
    "email address": "an email address",
    "phone number": "a phone number",
    "symptom": "a symptom",
    "disease": "a disease",
    "diagnosis": "a medical diagnosis",
    "medication": "a medication",
    "dosage": "a medication dosage",
    "frequency": "a frequency of medication",
    "duration": "a duration of time",
    "medical test": "a medical test",
    "test result": "a test result",
    "procedure": "a medical procedure",
    "medical department": "a hospital department",
    "allergy": "an allergy",
    "encounter date": "a specific date",
    "patient identifier": "a patient identifier",
}


def text_windows(text, size=120, overlap=30):
    tokens = list(re.finditer(r"\w+(?:[-_]\w+)*|[^\w\s]", text))
    for start in range(0, len(tokens), size - overlap):
        end = min(start + size, len(tokens))
        lo, hi = tokens[start].start(), tokens[end - 1].end()
        yield lo, text[lo:hi]
        if end == len(tokens):
            break


def resolve_spans(spans):
    accepted = []
    for span in sorted(spans, key=lambda e: (
        -(e["end"] - e["start"]), -e["score"], e["start"], e["label"]
    )):
        if not any(span["start"] < other["end"] and other["start"] < span["end"]
                   for other in accepted):
            accepted.append(span)
    return sorted(accepted, key=lambda e: e["start"])



RULE_FIELDS = {
    "patient": ("person full name", "somebody"),
    "patient name": ("person full name", "somebody"),
    "encounter date": ("encounter date", "a specific date"),
    "date of birth": ("date of birth", "a specific date"),
    "patient id": ("patient identifier", "a patient identifier"),
    "patient identifier": ("patient identifier", "a patient identifier"),
}
EMPTY_VALUES = {"", "none", "unknown", "n/a", "not recorded", "not available"}
CATEGORY_TAG = re.compile(r"\((?:procedure|regime/therapy|finding|disorder|situation|observable entity|qualifier value)\)", re.I)
GENERIC_TERMS = {"procedure", "medication", "oral tablet", "oral gel", "disease",
                 "diagnosis", "medical test", "symptom", "dosage", "frequency"}


def trimmed_interval(text, start, end):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def subtract_ranges(start, end, exclusions):
    cursor = start
    for lo, hi in sorted(exclusions):
        if hi <= cursor or lo >= end:
            continue
        if lo > cursor:
            yield cursor, min(lo, end)
        cursor = max(cursor, hi)
    if cursor < end:
        yield cursor, end


def template_regions(text, vitals):
    """Return rule replacements and eligible clinical ranges in original offsets."""
    rules, eligible = [], []
    for line in re.finditer(r"[^\r\n]+", text):
        content = line.group()
        heading = re.match(r"[ \t]*([^:\r\n]+):[ \t]*", content)
        if not heading:
            eligible.append((line.start(), line.end()))
            continue
        field = heading.group(1).strip().casefold()
        start, end = trimmed_interval(text, line.start() + heading.end(), line.end())
        if field == "vitals":
            
            for item in re.finditer(r"[^;]+", text[start:end]):
                lo, hi = start + item.start(), start + item.end()
                colon = text.find(":", lo, hi)
                value_start = colon + 1 if colon >= 0 else lo
                value_start, value_end = trimmed_interval(text, value_start, hi)
                value = text[value_start:value_end]
                if vitals == "generalize" and value.casefold().rstrip(".") not in EMPTY_VALUES:
                    
                    stop = value_end - 1 if value.endswith(".") else value_end
                    rules.append(dict(start=value_start, end=stop, label="vital value",
                                      replacement="a measurement", score=1.0, source="template_rule"))
            
            continue
        if field in RULE_FIELDS:
            if text[start:end].casefold().rstrip(".") not in EMPTY_VALUES:
                label, replacement = RULE_FIELDS[field]
                rules.append(dict(start=start, end=end, label=label, replacement=replacement,
                                  score=1.0, source="template_rule"))
            continue
        if start < end:
            eligible.append((start, end))
    tags = [(m.start(), m.end()) for m in CATEGORY_TAG.finditer(text)]
    regions = []
    for start, end in eligible:
        regions.extend(subtract_ranges(start, end, tags))
    return rules, regions


def apply_spans(text, spans):
    """Copy every untouched character verbatim; never normalize whitespace."""
    parts, cursor = [], 0
    for entity in sorted(spans, key=lambda e: e["start"]):
        start, end = entity["start"], entity["end"]
        if not cursor <= start < end <= len(text):
            raise ValueError("Overlapping or invalid replacement spans")
        parts.extend([text[cursor:start], entity["replacement"]])
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def generalize_note(text, detector, threshold, window_tokens, overlap_tokens, vitals="keep"):
    rules, regions = template_regions(text, vitals)
    candidates = []
    
    
    for offset, chunk in text_windows(text, window_tokens, overlap_tokens):
        for entity in detector.predict_entities(chunk, list(REPLACEMENTS), threshold=threshold):
            lo, hi = int(entity["start"]), int(entity["end"])
            if entity["label"] not in REPLACEMENTS or not 0 <= lo < hi <= len(chunk):
                raise ValueError(f"Invalid detector span: {entity!r}")
            for region_start, region_end in regions:
                start, end = max(offset + lo, region_start), min(offset + hi, region_end)
                start, end = trimmed_interval(text, start, end)
                if start >= end:
                    continue
                
                
                if (start and text[start-1].isalnum() and text[start].isalnum()) or (
                    end < len(text) and text[end-1].isalnum() and text[end].isalnum()
                ):
                    continue
                original = text[start:end]
                if original.casefold().strip(" .,:;()") in GENERIC_TERMS:
                    continue
                candidates.append(dict(start=start, end=end, label=entity["label"],
                                       replacement=REPLACEMENTS[entity["label"]],
                                       score=float(entity["score"]), source="gliner"))
    spans = sorted(rules + resolve_spans(candidates), key=lambda e: e["start"])
    for entity in spans:
        entity["original_text"] = text[entity["start"]:entity["end"]]
    return apply_spans(text, spans), spans


def preview_ids(ground_truth, count, seed):
    
    groups = defaultdict(list)
    for eid in sorted(ground_truth):
        groups[str(ground_truth[eid]["patient_id"])].append(eid)
    rng = random.Random(seed)
    patients = sorted(groups)
    rng.shuffle(patients)
    selected = []
    for patient in patients:
        rng.shuffle(groups[patient])
        selected.append(groups[patient].pop())
        if len(selected) == count:
            return selected
    remaining = [eid for patient in patients for eid in groups[patient]]
    rng.shuffle(remaining)
    return (selected + remaining)[:count]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an entity-generalization defense collection: detected "
                    "personal and clinical entities are replaced with "
                    "natural-language category descriptors before embedding."
    )
    parser.add_argument("--target-collection", default=None)
    parser.add_argument("--vitals", choices=["keep", "generalize"], required=True,
                        help="Explicit scope choice for complete vital measurements, including units")
    parser.add_argument("--preview-seed", type=int, default=42)
    parser.add_argument("--chroma-path", type=Path, default=PROJECT_ROOT / "chroma_db")
    parser.add_argument("--notes-dir", type=Path, default=PROJECT_ROOT / "data/notes")
    parser.add_argument("--ground-truth", type=Path,
                        default=PROJECT_ROOT / "data/ground_truth.json")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "data/results/generalization_manifests")
    parser.add_argument("--ner-model", default="urchade/gliner_medium-v2.1")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--window-tokens", type=int, default=120)
    parser.add_argument("--overlap-tokens", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--preview", type=int, default=0,
                        help="Generalize N notes and save before/after JSON; no Chroma write")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 < args.threshold <= 1:
        parser.error("--threshold must be in (0, 1]")
    if not 0 <= args.overlap_tokens < args.window_tokens <= 120:
        parser.error("Require 0 <= overlap-tokens < window-tokens <= 120")
    if args.batch_size < 1 or args.preview < 0:
        parser.error("batch-size must be positive and preview nonnegative")
    if args.target_collection is None:
        args.target_collection = f"encounters_redacted_template_{args.vitals}_vitals"
    if not re.fullmatch(r"encounters_redacted(?:_[A-Za-z0-9_-]+)?", args.target_collection):
        parser.error("Use encounters_redacted or encounters_redacted_<suffix>")
    return args


def main():
    args = parse_args()
    ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    if not isinstance(ground_truth, dict) or not ground_truth:
        raise ValueError("Expected a nonempty encounter-ID -> record mapping")
    ids = sorted(ground_truth)
    for eid in ids:
        if Path(eid).name != eid or not (args.notes_dir / f"{eid}.txt").is_file():
            raise ValueError(f"Missing or invalid note ID: {eid}")
        if "patient_id" not in ground_truth[eid]:
            raise ValueError(f"Missing patient_id for {eid}")

    if args.preview:
        ids = preview_ids(ground_truth, args.preview, args.preview_seed)

    client = None
    if not args.preview:
        import chromadb
        client = chromadb.PersistentClient(path=str(args.chroma_path.resolve()))
        names = {getattr(c, "name", c) for c in client.list_collections()}
        if args.target_collection in names and not args.overwrite:
            raise SystemExit("Collection already exists. Choose a new name or use --overwrite.")

    from gliner import GLiNER
    import torch
    torch.manual_seed(42)
    detector = GLiNER.from_pretrained(args.ner_model).to(args.device)
    detector.eval()
    texts, metadatas, examples = [], [], []
    counts = Counter()
    unchanged = 0
    fingerprint = hashlib.sha256()
    with torch.inference_mode():
        for index, eid in enumerate(ids, 1):
            original = (args.notes_dir / f"{eid}.txt").read_text(encoding="utf-8")
            fingerprint.update(json.dumps([eid, original], ensure_ascii=False).encode("utf-8"))
            generalized, spans = generalize_note(original, detector, args.threshold,
                                                 args.window_tokens, args.overlap_tokens, args.vitals)
            counts.update(entity["label"] for entity in spans)
            unchanged += int(not spans)
            texts.append(generalized)
            metadatas.append({
                "patient_id": ground_truth[eid]["patient_id"],
                "redaction_applied": True,
                "redaction_mechanism": MECHANISM,
                "redaction_ner_model": args.ner_model,
                "redaction_threshold": float(args.threshold),
                "redaction_num_entities": len(spans),
                "redaction_vitals_policy": args.vitals,
            })
            if args.preview:
                examples.append({"encounter_id": eid, "patient_id": ground_truth[eid]["patient_id"],
                                 "vitals_policy": args.vitals, "original": original,
                                 "generalized": generalized, "spans": spans})
            if index % 100 == 0 or index == len(ids):
                print(f"Generalized {index}/{len(ids)} notes", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.preview:
        path = args.output_dir / f"{args.target_collection}_preview.json"
        path.write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Preview saved to {path}. No collection was created.")
        print(f"Detected entities by category: {dict(counts)}")
        return
    if not any(counts[label] for label in counts if label not in {"person full name", "encounter date", "date of birth", "patient identifier", "vital value"}):
        raise RuntimeError("No entities detected in the corpus. Inspect a preview before building.")
    del detector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import numpy as np
    from sentence_transformers import SentenceTransformer
    victim = SentenceTransformer(EMBEDDING_MODEL, device=args.device)
    vectors = victim.encode(texts, convert_to_numpy=True, normalize_embeddings=False,
                            batch_size=args.batch_size, show_progress_bar=True)
    if len(vectors) != len(ids) or not np.isfinite(vectors).all():
        raise RuntimeError("Invalid embeddings; collection was not changed")
    names = {getattr(c, "name", c) for c in client.list_collections()}
    if args.target_collection in names:
        if not args.overwrite:
            raise RuntimeError("Collection appeared during the run; refusing to overwrite")
        client.delete_collection(args.target_collection)
    target = client.create_collection(args.target_collection,
                                      metadata={"redaction_applied": True})
    batch_size = args.batch_size
    if hasattr(client, "get_max_batch_size"):
        batch_size = min(batch_size, client.get_max_batch_size())
    try:
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            target.add(ids=ids[start:end], embeddings=vectors[start:end].tolist(),
                       metadatas=metadatas[start:end], documents=texts[start:end])
        if target.count() != len(ids):
            raise RuntimeError("Collection count does not match source count")
    except Exception:
        print("Build failed; the target may be partial. Rebuild with --overwrite.")
        raise
    manifest = {
        "target_collection": args.target_collection,
        "method": "Template-aware entity generalization: structured-value rules plus clinical NER",
        "mechanism": MECHANISM,
        "vitals_policy": args.vitals,
        "template_rules": RULE_FIELDS,
        "preview_seed": args.preview_seed,
        "num_notes": len(ids), "ner_model": args.ner_model,
        "threshold": args.threshold, "label_mapping": REPLACEMENTS,
        "window_tokens": args.window_tokens, "overlap_tokens": args.overlap_tokens,
        "overlap_resolution": "longest span, then highest confidence",
        "embedding_model": EMBEDDING_MODEL, "normalize_embeddings": False,
        "chroma_distance": "default_l2", "device": args.device,
        "entities_by_label": dict(counts), "notes_without_detections": unchanged,
        "avg_entities_per_note": sum(counts.values()) / len(ids),
        "input_sha256": fingerprint.hexdigest(),
        "source_notes": str(args.notes_dir.resolve()),
        "ground_truth_used_for_detection": False,
        "ground_truth_use": "encounter IDs, patient IDs, and patient-diverse preview selection only",
        "scope": "Rule-matched structured values and NER-detected clinical entities; clinical misses remain possible",
        "design_choices": ["headings and category tags preserved",
                           "complete name/date fields from note text",
                           "explicit keep/generalize vital-value policy",
                           "all untouched whitespace copied verbatim",
                           "patient-diverse preview",
                           "healthcare-oriented label subset",
                           "encounter date and patient identifier labels",
                           "medium-v2.1 checkpoint and 0.5 default threshold chosen here",
                           "overlapping text windows",
                           "descriptive category phrases rather than opaque placeholders",
                           "existing project embedding and Chroma setup"],
        "packages": {name: importlib.metadata.version(name) for name in
                     ("gliner", "torch", "sentence-transformers", "chromadb")},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = args.output_dir / f"{args.target_collection}.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done: {target.count()} generalized notes in {args.target_collection}. "
          f"Manifest: {path}")


if __name__ == "__main__":
    main()
