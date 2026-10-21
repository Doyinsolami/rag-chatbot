import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from ingest import get_collection
from attack_common import build_vocab, fetch_embeddings, label_matrix, load_ground_truth, load_split
from results import save_result

TOP_K_VOCAB = 200
MIN_DF = 5
MAX_DF_RATIO = 0.6
TOP_N_GUESSES = 5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--adaptive", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.collection:
        target_collection = get_collection(args.collection)
        tag = args.tag if args.tag else args.collection
        run_id = f"attribute_{tag}"
        dp_applied = True
    else:
        target_collection = None
        run_id = "attack_attribute_inference"
        dp_applied = False

    shadow_collection = target_collection if args.adaptive else None

    ground_truth = load_ground_truth()
    shadow_ids = load_split("train")
    target_ids = load_split("test")

    vocab = build_vocab(
        ground_truth, shadow_ids, min_df=MIN_DF, max_df_ratio=MAX_DF_RATIO, top_k=TOP_K_VOCAB
    )
    print(f"Attacker vocabulary: {len(vocab)} candidate PHI terms "
          f"(built from {len(shadow_ids)} shadow notes)")

    x_shadow = fetch_embeddings(shadow_ids, use_collection=shadow_collection)
    y_shadow = label_matrix(ground_truth, shadow_ids, vocab)
    x_target = fetch_embeddings(target_ids, use_collection=target_collection)
    y_target = label_matrix(ground_truth, target_ids, vocab)

    keep = y_shadow.sum(axis=0) > 0
    vocab = [t for t, k in zip(vocab, keep) if k]
    y_shadow = y_shadow[:, keep]
    y_target = y_target[:, keep]

    print(f"Training {len(vocab)} per-term classifiers on "
          f"{x_shadow.shape[0]} shadow embeddings, attacking "
          f"{x_target.shape[0]} target embeddings...")

    predictions = np.zeros_like(y_target)
    scores = np.zeros(y_target.shape, dtype=float)
    aucs = []

    for j, term in enumerate(vocab):
        y = y_shadow[:, j]
        if y.sum() == len(y):
            predictions[:, j] = 1
            scores[:, j] = 1.0
            continue
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(x_shadow, y)
        proba = clf.predict_proba(x_target)[:, 1]
        scores[:, j] = proba
        predictions[:, j] = (proba >= 0.5).astype(int)
        if 0 < y_target[:, j].sum() < len(y_target):
            aucs.append(roc_auc_score(y_target[:, j], proba))

    precision = precision_score(y_target, predictions, average="micro", zero_division=0)
    recall = recall_score(y_target, predictions, average="micro", zero_division=0)
    f1 = f1_score(y_target, predictions, average="micro", zero_division=0)
    mean_auc = float(np.mean(aucs)) if aucs else None

    hit_at_n = []
    for i in range(len(target_ids)):
        true_terms = set(np.where(y_target[i] == 1)[0])
        if not true_terms:
            continue
        top_guess = set(np.argsort(-scores[i])[:TOP_N_GUESSES])
        hit_at_n.append(len(true_terms & top_guess) / len(true_terms))
    mean_hit_at_n = float(np.mean(hit_at_n)) if hit_at_n else None

    print(f"\nMicro precision: {precision:.3f}")
    print(f"Micro recall:    {recall:.3f}")
    print(f"Micro F1:        {f1:.3f}")
    if mean_auc is not None:
        print(f"Mean per-term AUC: {mean_auc:.3f}")
    if mean_hit_at_n is not None:
        print(f"Mean fraction of a victim's true PHI terms recovered in "
              f"attacker's top-{TOP_N_GUESSES} guesses: {mean_hit_at_n:.3f}")

    save_result(
        run_id=run_id,
        config={
            "attack": "attribute_inference",
            "threat_model": "white_box_stolen_vector_db",
            "embedding_model": "all-MiniLM-L6-v2",
            "dp_applied": dp_applied,
            "collection": args.collection,
            "attacker_mode": "adaptive" if args.adaptive else "transfer",
            "vocab_size": len(vocab),
            "shadow_encounters": len(shadow_ids),
            "target_encounters": len(target_ids),
            "top_n_guesses": TOP_N_GUESSES,
        },
        attack={
            "precision_micro": precision,
            "recall_micro": recall,
            "f1_micro": f1,
            "mean_auc": mean_auc,
            "mean_hit_at_top_n": mean_hit_at_n,
        },
    )


if __name__ == "__main__":
    main()