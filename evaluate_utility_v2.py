"""Evaluate retrieval and end-to-end QA on the validated utility_v2 benchmark.

Clear cases use deterministic rules. Ambiguous cases use a structured Ollama
judge by default. Manual review is optional and never required during a normal
run. Cached answers and grades survive interruption.
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import ollama

from generate import MODEL_NAME as GENERATOR_MODEL, generate_answer
from ingest import collection as clean_collection, get_collection, search
from results import RESULTS_DIR

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_EVAL_FILE = DATA_DIR / "eval_questions_v2.json"
K = 5


def normalize(text):
    text = str(text).lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9.]+", " ", text)
    return " ".join(text.split())


def token_f1(expected, actual):
    expected_tokens = normalize(expected).split()
    actual_tokens = normalize(actual).split()
    if not expected_tokens or not actual_tokens:
        return 0.0
    expected_counts, actual_counts = Counter(expected_tokens), Counter(actual_tokens)
    overlap = sum((expected_counts & actual_counts).values())
    if not overlap:
        return 0.0
    precision = overlap / len(actual_tokens)
    recall = overlap / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)


NEGATIVE_PATTERNS = (
    r"\bno\b.{0,80}\b(documented|recorded|listed|prescribed|performed|found)\b",
    r"\b(does not|do not|did not|doesn't|don't|didn't)\b.{0,80}"
    r"\b(contain|show|document|list|mention|indicate)\b",
    r"\brecords? (do|does) not contain\b",
    r"\bnot (explicitly )?(stated|documented|recorded|listed|mentioned|provided)\b",
    r"\b(no|without) (record|evidence|mention|documentation) of\b",
    r"\bnone (?:was|were|is|are) documented\b",
)
POSITIVE_ASSERTION_PATTERNS = (
    r"\byes\b", r"\bwas prescribed\b", r"\bwere prescribed\b",
    r"\bwas performed\b", r"\bwere performed\b",
    r"\bwas documented\b", r"\bwere documented\b",
)
REFUSAL_PATTERNS = (
    r"\brecords? (do|does) not contain\b",
    r"\bno information\b",
    r"\bnot (?:explicitly )?(?:stated|documented|recorded|listed|mentioned|provided)\b",
    r"\bcannot provide\b",
    r"\bcan't provide\b",
    r"\bunable to (?:answer|provide|determine)\b",
    r"\bi do not have enough information\b",
)


def deterministic_grade(question, expected_answers, answer):
    answer_norm = normalize(answer)
    if question.get("expected_answer_type") == "negative":
        if any(re.search(pattern, answer_norm) for pattern in NEGATIVE_PATTERNS):
            return True, "negative_answer_rule", 1.0
        if any(re.search(pattern, answer_norm) for pattern in POSITIVE_ASSERTION_PATTERNS):
            return False, "negative_contradicted", 0.0
        return None, "judge_required", 0.0

    contained = [
        bool(normalize(expected)) and normalize(expected) in answer_norm
        for expected in expected_answers
    ]
    scores = [token_f1(expected, answer) for expected in expected_answers]
    if question.get("expected_answer_match", "any") == "all":
        if contained and all(contained):
            return True, "all_expected_contained", min(scores)
    elif any(contained):
        return True, "expected_contained", max(scores)
    if any(re.search(pattern, answer_norm) for pattern in REFUSAL_PATTERNS):
        return False, "positive_question_refused", 0.0
    best = max(scores, default=0.0)
    if question.get("expected_answer_match", "any") == "any" and best >= 0.85:
        return True, "high_token_f1", best
    return None, "judge_required", best


def llm_grade(question, expected_answers, answer, model):
    match_rule = question.get("expected_answer_match", "any")
    expected_label = "ALL expected facts" if match_rule == "all" else "ANY expected fact"
    prompt = f"""You are grading a clinical chart question-answering benchmark.

Question: {question['question']}
Expected facts: {json.dumps(expected_answers)}
Matching rule: The answer must correctly express {expected_label}.
Candidate answer: {answer}

Decide whether the candidate correctly answers the question. Accept harmless
paraphrases, formatting differences, equivalent medication expressions, and
answers that correctly say none were documented for a negative question.
Reject missing required facts, a different medication/procedure/value, refusal
to answer a positive question, extra contradictory facts, or facts from the
wrong encounter. Judge against the expected facts, not outside medical knowledge.

Return JSON only:
{{"correct": true or false, "confidence": number from 0 to 1, "reason": "brief reason"}}
"""
    response = ollama.generate(
        model=model,
        prompt=prompt,
        format="json",
        options={"temperature": 0, "seed": 42},
    )
    payload = json.loads(response["response"])
    correct = payload.get("correct")
    if not isinstance(correct, bool):
        raise ValueError("Judge did not return a Boolean 'correct' value")
    confidence = float(payload.get("confidence", 0.0))
    reason = str(payload.get("reason", "")).strip()
    return correct, confidence, reason


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--questions", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--judge-model", default=GENERATOR_MODEL)
    parser.add_argument("--no-llm-judge", action="store_true",
                        help="Leave ambiguous answers pending instead of using Ollama judging.")
    parser.add_argument("--manual-review", action="store_true",
                        help="Ask y/n only for answers still pending after automatic grading.")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    target = get_collection(args.collection) if args.collection else clean_collection
    tag = args.tag or (args.collection if args.collection else "clean")
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[:args.limit]
    if any(q.get("benchmark_version") != "utility_v2" for q in questions):
        raise ValueError("This evaluator accepts only utility_v2 questions")

    cache_path = DATA_DIR / "results" / f"utility_v2_answers_{tag}.json"
    cache = {}
    if cache_path.exists():
        cache = {
            row["question_id"]: row
            for row in json.loads(cache_path.read_text(encoding="utf-8"))
        }
    question_order = {q["question_id"]: index for index, q in enumerate(questions)}

    def checkpoint():
        current_ids = set(question_order)
        rows = [cache[qid] for qid in current_ids if qid in cache]
        rows.sort(key=lambda row: question_order[row["question_id"]])
        save_json(cache_path, rows)

    for number, q in enumerate(questions, 1):
        previous = cache.get(q["question_id"], {})
        results = search(
            q["question"], n_results=K, patient_id=q["patient_id"],
            use_collection=target,
        )
        retrieved_ids = results["ids"][0]
        documents = results["documents"][0]
        relevant = set(q["relevant_encounter_ids"])
        rank = next(
            (index for index, encounter_id in enumerate(retrieved_ids, 1)
             if encounter_id in relevant),
            None,
        )

        answer = previous.get("generated_answer")
        if answer is None:
            print(f"[{number}/{len(questions)}] Generating: {q['question']}")
            answer = generate_answer(q["question"], documents)

        correct, method, similarity = deterministic_grade(
            q, q["expected_answers"], answer
        )
        judge_confidence, judge_reason = None, None
        if correct is None and previous.get("manual_correct") is not None:
            correct = bool(previous["manual_correct"])
            method = "cached_manual_review"
        elif correct is None and previous.get("judge_correct") is not None:
            correct = bool(previous["judge_correct"])
            method = "cached_llm_judge"
            judge_confidence = previous.get("judge_confidence")
            judge_reason = previous.get("judge_reason")
        elif correct is None and not args.no_llm_judge:
            try:
                correct, judge_confidence, judge_reason = llm_grade(
                    q, q["expected_answers"], answer, args.judge_model
                )
                method = "llm_judge"
            except Exception as error:
                method = "judge_error"
                judge_reason = str(error)
        if correct is None and args.manual_review:
            separator = " AND " if q.get("expected_answer_match") == "all" else " OR "
            print("\nQuestion:", q["question"])
            print("Expected:", separator.join(q["expected_answers"]))
            print("Answer:", answer)
            while True:
                decision = input("Factually correct? [y/n]: ").strip().lower()
                if decision in {"y", "n"}:
                    correct = decision == "y"
                    method = "manual_review"
                    break

        row = {
            **q,
            "retrieved_ids": retrieved_ids,
            "relevant_rank": rank,
            "retrieval_hit_at_1": bool(rank and rank <= 1),
            "retrieval_hit_at_3": bool(rank and rank <= 3),
            "retrieval_hit_at_5": bool(rank and rank <= 5),
            "generated_answer": answer,
            "correct": correct,
            "manual_correct": (
                correct if method in {"manual_review", "cached_manual_review"}
                else previous.get("manual_correct")
            ),
            "judge_correct": (
                correct if method in {"llm_judge", "cached_llm_judge"}
                else previous.get("judge_correct")
            ),
            "judge_confidence": judge_confidence,
            "judge_reason": judge_reason,
            "grading_method": method,
            "best_token_f1": similarity,
        }
        cache[q["question_id"]] = row
        checkpoint()

    rows = [cache[q["question_id"]] for q in questions]
    total = len(rows)
    judged = [row for row in rows if row["correct"] is not None]
    correct_rows = [row for row in judged if row["correct"]]
    hit1 = sum(row["retrieval_hit_at_1"] for row in rows)
    hit3 = sum(row["retrieval_hit_at_3"] for row in rows)
    hit5 = sum(row["retrieval_hit_at_5"] for row in rows)
    reciprocal_rank = sum(
        1 / row["relevant_rank"] for row in rows if row["relevant_rank"]
    )

    category = defaultdict(lambda: {"n": 0, "hit5": 0, "judged": 0, "correct": 0})
    for row in rows:
        item = category[row["field"]]
        item["n"] += 1
        item["hit5"] += int(row["retrieval_hit_at_5"])
        if row["correct"] is not None:
            item["judged"] += 1
            item["correct"] += int(row["correct"])

    pending = total - len(judged)
    report = {
        "benchmark": "utility_v2",
        "collection": target.name,
        "tag": tag,
        "questions": total,
        "generator_model": GENERATOR_MODEL,
        "judge_model": None if args.no_llm_judge else args.judge_model,
        "retrieval": {
            "recall_at_1": hit1 / total if total else 0,
            "recall_at_3": hit3 / total if total else 0,
            "recall_at_5": hit5 / total if total else 0,
            "mrr_at_5": reciprocal_rank / total if total else 0,
        },
        "answers": {
            "judged": len(judged),
            "pending_review": pending,
            "end_to_end_accuracy": (
                len(correct_rows) / total if total and pending == 0 else None
            ),
            "accuracy_among_judged": (
                len(correct_rows) / len(judged) if judged else None
            ),
            "conservative_lower_bound": len(correct_rows) / total if total else 0,
            "grading_methods": dict(Counter(row["grading_method"] for row in rows)),
        },
        "by_category": {},
    }
    for name, item in sorted(category.items()):
        report["by_category"][name] = {
            **item,
            "recall_at_5": item["hit5"] / item["n"] if item["n"] else 0,
            "answer_accuracy": (
                item["correct"] / item["judged"]
                if item["judged"] == item["n"] and item["n"] else None
            ),
        }

    result_path = RESULTS_DIR / f"utility_v2_{tag}.json"
    save_json(result_path, report)
    print(json.dumps(report, indent=2))
    print(f"Saved answers: {cache_path}")
    print(f"Saved summary: {result_path}")


if __name__ == "__main__":
    main()
