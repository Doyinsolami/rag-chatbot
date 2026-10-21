import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

from attack_common import build_vocab, fetch_embeddings, load_ground_truth, load_split, note_terms
from results import save_result
from train_attacker import Projection, build_prefix, PREFIX_LENGTH, VICTIM_MODEL

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
NOTES_DIR = DATA_DIR / "notes"
GEIA_DIR = DATA_DIR / "geia"
MODEL_DIR = PROJECT_ROOT / "models" / "attacker"

MAX_NEW_TOKENS = 200
NUM_EXAMPLES_TO_SAVE = 3
NUM_CONSISTENCY_CHECKS = 5
SEED = 42

WORD_RE = re.compile(r"[a-z0-9]+")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_decoder():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    attacker = AutoModelForCausalLM.from_pretrained(MODEL_DIR / "gpt2").to(DEVICE).eval()
    projection = Projection().to(DEVICE)
    projection.load_state_dict(torch.load(MODEL_DIR / "projection.pt", map_location=DEVICE))
    projection.eval()
    return tokenizer, attacker, projection


def load_split_records(split):
    records = {}
    with open(GEIA_DIR / f"{split}.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            records[record["encounter_id"]] = record
    return records


@torch.no_grad()
def reconstruct(embedding, tokenizer, attacker, projection):
    embed_t = torch.tensor(embedding, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    prefix_embeds = build_prefix(embed_t, projection, attacker, tokenizer, DEVICE)
    output_ids = attacker.generate(
        inputs_embeds=prefix_embeds,
        attention_mask=torch.ones(
            (1, prefix_embeds.size(1)), dtype=torch.long, device=DEVICE
        ),
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=4,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


@torch.no_grad()
def conditioning_sanity_check(embedding_a, embedding_b, tokenizer, attacker, projection):
    vectors = torch.tensor(
        np.stack([embedding_a, embedding_b]), dtype=torch.float32, device=DEVICE
    )
    prefix_embeds = build_prefix(vectors, projection, attacker, tokenizer, DEVICE)
    logits = attacker(inputs_embeds=prefix_embeds).logits[:, -1, :]
    max_difference = torch.max(torch.abs(logits[0] - logits[1])).item()
    print(f"Embedding-conditioning sanity check, max first-token logit difference: "
          f"{max_difference:.8f}")
    if max_difference <= 1e-7:
        raise RuntimeError(
            "Different embeddings produced indistinguishable first-token logits. "
            "Do not trust reconstruction results until conditioning is fixed."
        )


def cross_patient_pairing(target_ids, patient_of, seed):
    if len({patient_of[eid] for eid in target_ids}) < 2:
        raise ValueError(
            "The shuffled control needs targets from at least 2 patients, use a larger --limit"
        )
    rng = np.random.default_rng(seed)
    n = len(target_ids)
    pairing = np.empty(n, dtype=int)
    for i in range(n):
        j = int(rng.integers(n))
        while patient_of[target_ids[j]] == patient_of[target_ids[i]]:
            j = int(rng.integers(n))
        pairing[i] = j
    return pairing


def found_in(term, haystack):
    return term.split(" (")[0].lower() in haystack


def token_f1(true_text, reconstructed_text):
    true_tokens = WORD_RE.findall(true_text.lower())
    recon_tokens = WORD_RE.findall(reconstructed_text.lower())
    if not true_tokens or not recon_tokens:
        return 0.0
    overlap = sum((Counter(true_tokens) & Counter(recon_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(recon_tokens)
    recall = overlap / len(true_tokens)
    return 2 * precision * recall / (precision + recall)


def name_recovered(patient_name, reconstructed_text):
    if not patient_name or patient_name == "Unknown":
        return None
    haystack = reconstructed_text.lower()
    parts = [p for p in WORD_RE.findall(patient_name.lower()) if p]
    if not parts:
        return None
    return all(part in haystack for part in parts)


def date_recovered(encounter_date, reconstructed_text):
    if not encounter_date:
        return None
    date_part = encounter_date.split("T")[0]
    return date_part in reconstructed_text


def mean_or_none(values):
    return sum(values) / len(values) if values else None


def score_against_targets(reconstructions, target_ids, ground_truth, scorer, vocab):
    rouge_scores = []
    token_f1s = []
    name_hits = []
    date_hits = []

    total_true = 0
    total_predicted = 0
    total_recovered_true = 0
    total_correct_predicted = 0

    for eid, reconstructed in zip(target_ids, reconstructions):
        entry = ground_truth[eid]
        true_text = (NOTES_DIR / f"{eid}.txt").read_text(encoding="utf-8")

        rouge_scores.append(scorer.score(true_text, reconstructed)["rougeL"].fmeasure)
        token_f1s.append(token_f1(true_text, reconstructed))

        haystack = reconstructed.lower()
        true_terms = set(note_terms(entry))
        predicted = {v for v in vocab if found_in(v, haystack)}
        recovered_true = {t for t in true_terms if found_in(t, haystack)}
        correct_predicted = predicted & true_terms

        total_true += len(true_terms)
        total_predicted += len(predicted)
        total_recovered_true += len(recovered_true)
        total_correct_predicted += len(correct_predicted)

        name_hit = name_recovered(entry.get("patient_name"), reconstructed)
        if name_hit is not None:
            name_hits.append(1.0 if name_hit else 0.0)

        date_hit = date_recovered(entry.get("encounter_date"), reconstructed)
        if date_hit is not None:
            date_hits.append(1.0 if date_hit else 0.0)

    precision = total_correct_predicted / total_predicted if total_predicted else None
    recall = total_recovered_true / total_true if total_true else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "mean_rouge_l_f1": mean_or_none(rouge_scores),
        "mean_token_f1": mean_or_none(token_f1s),
        "clinical_precision_micro": precision,
        "clinical_recall_micro": recall,
        "clinical_f1_micro": f1,
        "name_recovery_rate": mean_or_none(name_hits),
        "date_recovery_rate": mean_or_none(date_hits),
    }


def deltas(real, other):
    out = {}
    for key, value in real.items():
        baseline_value = other.get(key)
        if value is None or baseline_value is None:
            out[key] = None
        else:
            out[key] = value - baseline_value
    return out


def print_scores(label, scores):
    order = [
        ("ROUGE-L", "mean_rouge_l_f1"),
        ("token F1", "mean_token_f1"),
        ("clinical P", "clinical_precision_micro"),
        ("clinical R", "clinical_recall_micro"),
        ("clinical F1", "clinical_f1_micro"),
        ("name", "name_recovery_rate"),
        ("date", "date_recovery_rate"),
    ]
    parts = []
    for name, key in order:
        value = scores.get(key)
        if value is not None:
            parts.append(f"{name} {value:.3f}")
    print(f"{label}: " + ", ".join(parts))


def collection_noise_config(collection):
    sample = collection.get(limit=1, include=["metadatas"])
    metadata = (sample.get("metadatas") or [{}])[0] or {}
    return {
        "dp_applied": False,
        "noise_mechanism": metadata.get("noise_mechanism"),
        "noise_epsilon": metadata.get("noise_epsilon"),
        "noise_seed": metadata.get("noise_seed"),
        "privacy_definition": metadata.get("noise_privacy_definition"),
        "mask_sha256": metadata.get("noise_mask_sha256") or None,
        "normalized_after_noise": metadata.get("noise_normalized"),
    }


def attacker_provenance(model_dir, collection_name):
    config_path = model_dir / "training_config.json"
    if not config_path.exists():
        return "transfer_clean_trained", None
    trained_on = json.loads(config_path.read_text(encoding="utf-8")).get("collection")
    if trained_on in (None, "", "encounter_embeddings"):
        return "transfer_clean_trained", trained_on
    if trained_on != collection_name:
        raise ValueError(
            f"Attacker in {model_dir} was trained on {trained_on!r}, which does not match "
            f"the evaluated collection {collection_name!r}."
        )
    return "noise_aware", trained_on


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N test encounters.",
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="Chroma collection built by sparse_baseline.py build.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Result filename tag. Defaults to the collection name.",
    )
    parser.add_argument("--attacker-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    globals()["MODEL_DIR"] = args.attacker_dir.resolve()
    attacker_mode, attacker_training_collection = attacker_provenance(MODEL_DIR, args.collection)
    print(f"Attacker mode: {attacker_mode}")

    from ingest import get_collection

    target_collection = get_collection(args.collection)
    tag = args.tag if args.tag else args.collection
    run_id = f"geia_{tag}"
    noise_config = collection_noise_config(target_collection)
    if not noise_config.get("noise_mechanism"):
        raise ValueError(
            f"Collection {args.collection!r} has no noise_mechanism metadata. "
            "Use the original evaluator for clean or Gaussian DP collections."
        )

    selected_name = args.collection
    print(f"Selected Chroma collection: {selected_name} ({target_collection.count()} records)")

    ground_truth = load_ground_truth()
    target_ids = load_split("test")
    if args.limit is not None:
        if args.limit < 2:
            raise ValueError("--limit must be at least 2 so controls can be evaluated")
        target_ids = target_ids[:args.limit]
    train_ids = load_split("train")

    vocab = build_vocab(
        ground_truth,
        list(ground_truth.keys()),
        min_df=1,
        max_df_ratio=1.0,
        top_k=None,
    )
    print(f"Clinical detection vocabulary for precision: {len(vocab)} terms "
          f"(every clinical term in the corpus)")

    test_records = load_split_records("test")
    patient_of = {eid: test_records[eid]["patient_id"] for eid in target_ids}

    print(f"Loading GEIA attacker from {MODEL_DIR}")
    tokenizer, attacker, projection = load_decoder()

    embeddings = fetch_embeddings(target_ids, use_collection=target_collection)
    train_embeddings = fetch_embeddings(train_ids, use_collection=target_collection)
    baseline_embedding = train_embeddings.mean(axis=0)


    n = len(target_ids)
    rolled = np.roll(np.arange(n), 1)
    same_patient_in_old_roll = int(sum(
        patient_of[target_ids[i]] == patient_of[target_ids[rolled[i]]] for i in range(n)
    ))
    print(f"Old roll-by-one control would have paired a target with the same patient's "
          f"embedding in {same_patient_in_old_roll} of {n} targets")

    pairing = cross_patient_pairing(target_ids, patient_of, SEED)
    shuffled_embeddings = embeddings[pairing]

    conditioning_sanity_check(
        embeddings[0], embeddings[pairing[0]], tokenizer, attacker, projection
    )

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    print(f"Reconstructing {n} held-out test notes from embeddings only "
          f"(none of these patients were in the attacker's train split)...")
    reconstructions = [
        reconstruct(embeddings[i], tokenizer, attacker, projection)
        for i in range(n)
    ]
    real_scores = score_against_targets(
        reconstructions, target_ids, ground_truth, scorer, vocab
    )

    print("Reconstructing the shuffled-embedding control: every target is paired "
          "with a random embedding from a different test patient...")
    shuffled_reconstructions = [
        reconstruct(shuffled_embeddings[i], tokenizer, attacker, projection)
        for i in range(n)
    ]
    shuffled_scores = score_against_targets(
        shuffled_reconstructions, target_ids, ground_truth, scorer, vocab
    )

    print("Reconstructing the control condition: one decoding from the mean "
          "of the attacker's own train-split embeddings, carrying no "
          "information about which target patient is being attacked...")
    baseline_reconstruction = reconstruct(baseline_embedding, tokenizer, attacker, projection)
    baseline_scores = score_against_targets(
        [baseline_reconstruction] * n, target_ids, ground_truth, scorer, vocab
    )

    real_above_shuffled = deltas(real_scores, shuffled_scores)
    real_above_baseline = deltas(real_scores, baseline_scores)

    print()
    print_scores("Real attack", real_scores)
    print_scores("Shuffled control (different patient)", shuffled_scores)
    print_scores("Baseline (no per-target info)", baseline_scores)
    print()
    print_scores("Real above shuffled", real_above_shuffled)
    print_scores("Real above baseline", real_above_baseline)

    examples = [
        {
            "encounter_id": eid,
            "true_note": (NOTES_DIR / f"{eid}.txt").read_text(encoding="utf-8"),
            "reconstructed": reconstructions[i],
            "shuffled_reconstruction": shuffled_reconstructions[i],
        }
        for i, eid in enumerate(target_ids[:NUM_EXAMPLES_TO_SAVE])
    ]

    print("\nExample reconstruction:")
    example = examples[0]
    print(f"--- True note ({example['encounter_id']}) ---\n{example['true_note']}\n")
    print(f"--- Reconstructed from embedding only ---\n{example['reconstructed']}\n")
    print(f"--- Shuffled control (a different patient's embedding) ---\n"
          f"{example['shuffled_reconstruction']}\n")
    print(f"--- Baseline reconstruction (mean train embedding, same for every target) ---\n"
          f"{baseline_reconstruction}\n")

    save_result(
        run_id=run_id,
        config={
            "attack": "generative_inversion_geia_style",
            "threat_model": "white_box_stolen_vector_db",
            "collection": selected_name,
            **noise_config,
            "attacker_mode": attacker_mode,
            "attacker_training_collection": attacker_training_collection,
            "attacker_dir": str(MODEL_DIR),
            "embedding_model": VICTIM_MODEL,
            "decoder_backbone": "gpt2",
            "prefix_length": PREFIX_LENGTH,
            "target_encounters": n,
            "clinical_vocab_size": len(vocab),
            "clinical_prf_averaging": "micro",
            "baseline": "mean_of_train_split_embeddings",
            "shuffled_control": "each_target_paired_with_random_embedding_from_different_test_patient",
            "shuffled_control_seed": SEED,
            "old_roll_same_patient_pairs": same_patient_in_old_roll,
        },
        attack={
            "real": real_scores,
            "shuffled_control": shuffled_scores,
            "baseline": baseline_scores,
            "real_above_shuffled": real_above_shuffled,
            "real_above_baseline": real_above_baseline,
            "baseline_reconstruction": baseline_reconstruction,
            "examples": examples,
        },
    )


if __name__ == "__main__":
    main()
