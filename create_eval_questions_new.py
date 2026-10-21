import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

SEED = 42
TARGET_TOTAL = 120
MAX_PER_PATIENT = 2
DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_FILE = DATA_DIR / "ground_truth.json"
OUTPUT_FILE = DATA_DIR / "eval_questions_new.json"

QUOTAS = {
    "allergies": 10,
    "medications": 25,
    "conditions": 25,
    "procedures": 20,
    "observations": 20,
    "negative": 20,
}

LOW_VALUE = {
    "Medication review due (situation)",
    "Medication reconciliation (procedure)",
    "Allergic disposition (finding)",
}

MAX_SAME_POSITIVE_ANSWER = 2


def core(value):
    return re.sub(r"\s*\([^)]*\)\s*$", "", str(value)).strip()


def date_obj(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def date_text(value):
    return date_obj(value).strftime("%B %d, %Y").replace(" 0", " ")


def values(encounter, field):
    value = encounter.get(field)
    if isinstance(value, list):
        return [x for x in value if x and x not in LOW_VALUE]
    return [value] if value else []


def clinical_tokens(value):
    stop = {"of", "the", "and", "for", "with", "without", "history", "finding",
            "disorder", "situation", "procedure", "requested", "patient"}
    return {
        token for token in re.findall(r"[a-z0-9]+", core(value).lower())
        if token not in stop and len(token) > 2
    }


def clue_leaks_answer(clue, answer):
    clue_tokens = clinical_tokens(clue)
    answer_tokens = clinical_tokens(answer)
    if not clue_tokens or not answer_tokens:
        return False
    overlap = len(clue_tokens & answer_tokens)
    return overlap / min(len(clue_tokens), len(answer_tokens)) >= 0.5


def normalize_answer_key(answers):
    return " | ".join(sorted(core(answer).lower() for answer in answers))


def add_candidate(bucket, category, patient_id, question, answers, encounter_ids,
                  question_type, metadata=None, answer_type="positive",
                  answer_match="any"):
    answers = list(dict.fromkeys(str(x).strip() for x in answers if str(x).strip()))
    encounter_ids = list(dict.fromkeys(encounter_ids))
    if not answers or not encounter_ids:
        return
    bucket[category].append({
        "question": question,
        "expected_answers": answers,
        "field": category,
        "patient_id": patient_id,
        "relevant_encounter_ids": encounter_ids,
        "question_type": question_type,
        "expected_answer_type": answer_type,
        "expected_answer_match": answer_match,
        "metadata": metadata or {},
    })


def main():
    rng = random.Random(SEED)
    data = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    patients = defaultdict(list)
    for encounter_id, encounter in data.items():
        patients[encounter["patient_id"]].append((encounter_id, encounter))
    for encounters in patients.values():
        encounters.sort(key=lambda x: date_obj(x[1]["encounter_date"]))

    candidates = defaultdict(list)
    all_observations = [
        observation
        for encounter in data.values()
        for observation in values(encounter, "observations")
    ]
    observation_total = len(all_observations)
    observation_parseable = sum(
        1 for observation in all_observations
        if ":" in observation
        and observation.split(":", 1)[0].strip()
        and observation.split(":", 1)[1].strip()
    )
    observation_unparseable = [
        observation for observation in all_observations
        if not (
            ":" in observation
            and observation.split(":", 1)[0].strip()
            and observation.split(":", 1)[1].strip()
        )
    ]
    for patient_id, encounters in patients.items():
        reason_counts = Counter()
        type_counts = Counter()
        for _, encounter in encounters:
            reason = encounter.get("reason_for_visit")
            if reason:
                reason_counts[core(reason).lower()] += 1
            encounter_type = core(encounter.get("encounter_type") or "visit").lower()
            type_counts[encounter_type] += 1

        for encounter_id, encounter in encounters:
            visit = core(encounter.get("encounter_type") or "visit").lower()
            reason = encounter.get("reason_for_visit")
            reason_text = core(reason) if reason else None
            locator_options = []
            if reason_text and reason_counts[reason_text.lower()] == 1:
                locator_options.append(
                    (0, "problem_anchored_lookup", f"during the patient's visit for {reason_text}")
                )
            if type_counts[visit] == 1:
                locator_options.append(
                    (1, "unique_visit_lookup", f"during the patient's only {visit}")
                )
            if not locator_options:
                continue

            for field, noun in (
                ("medications", "medication"),
                ("conditions", "condition"),
                ("procedures", "procedure"),
            ):
                facts = values(encounter, field)
                if len(facts) == 1:
                    if field == "conditions" and reason_text and clue_leaks_answer(reason_text, facts[0]):
                        continue
                    for priority, lookup_type, locator in locator_options:
                        if field == "conditions" and lookup_type == "problem_anchored_lookup":
                            question = (
                                "What additional condition or relevant medical history was "
                                f"documented {locator}?"
                            )
                        else:
                            question = f"What {noun} was documented {locator}?"
                        add_candidate(
                            candidates, field, patient_id,
                            question,
                            facts, [encounter_id], lookup_type,
                            {"encounter_date": encounter["encounter_date"],
                             "encounter_type": encounter.get("encounter_type"),
                             "selection_priority": priority},
                        )

            observations = values(encounter, "observations")
            parsed = []
            for observation in observations:
                if ":" in observation:
                    label, result = observation.split(":", 1)
                    if label.strip() and result.strip():
                        parsed.append((label.strip(), result.strip(), observation))
            if parsed:
                label, result, full = rng.choice(parsed)
                for priority, lookup_type, locator in locator_options:
                    add_candidate(candidates, "observations", patient_id,
                                  f"What was the patient's {label.lower()} {locator}?",
                                  [result, full], [encounter_id], lookup_type,
                                  {"observation_name": label,
                                   "encounter_date": encounter["encounter_date"],
                                   "selection_priority": priority})

            
            
            
            empty_fields = []
            for field, plural in (
                ("medications", "medications"),
                ("conditions", "conditions"),
                ("procedures", "procedures"),
                ("allergies", "allergies"),
            ):
                if not values(encounter, field):
                    empty_fields.append((field, plural))
            if empty_fields:
                negative_field, plural = rng.choice(empty_fields)
                for priority, lookup_type, locator in locator_options:
                    add_candidate(
                        candidates, "negative", patient_id,
                        f"Were any {plural} documented {locator}?",
                        [f"No {plural} were documented."], [encounter_id],
                        f"negative_{lookup_type}",
                        {"negative_field": negative_field,
                         "encounter_date": encounter["encounter_date"],
                         "encounter_type": encounter.get("encounter_type"),
                         "selection_priority": priority},
                        answer_type="negative",
                    )

        allergy_sources = defaultdict(list)
        for encounter_id, encounter in encounters:
            for allergy in values(encounter, "allergies"):
                allergy_sources[allergy].append(encounter_id)
        if allergy_sources:
            all_allergies = sorted(allergy_sources)
            all_source_ids = list(dict.fromkeys(
                encounter_id
                for source_ids in allergy_sources.values()
                for encounter_id in source_ids
            ))
            add_candidate(
                candidates, "allergies", patient_id,
                "What allergies does this patient have?",
                all_allergies, all_source_ids, "patient_chart_lookup",
                {"selection_priority": 0}, answer_match="all",
            )

    
    
    for category in candidates:
        unique = {}
        for item in candidates[category]:
            unique[(item["patient_id"], item["question"])] = item
        candidates[category] = list(unique.values())
        rng.shuffle(candidates[category])

    selected = []
    per_patient = Counter()
    answer_counts = defaultdict(Counter)
    for category, quota in QUOTAS.items():
        taken = 0
        selected_keys = set()
        style_targets = {0: (quota * 2 + 2) // 3, 1: quota // 3}

        def eligible(item):
            if per_patient[item["patient_id"]] >= MAX_PER_PATIENT:
                return False
            key = (item["patient_id"], tuple(item["relevant_encounter_ids"]),
                   tuple(item["expected_answers"]), category)
            if key in selected_keys:
                return False
            if category not in {"negative", "allergies", "observations"}:
                answer_key = normalize_answer_key(item["expected_answers"])
                if answer_counts[category][answer_key] >= MAX_SAME_POSITIVE_ANSWER:
                    return False
            return True

        for priority in (0, 1):
            for item in candidates[category]:
                if taken >= quota or sum(
                    1 for chosen in selected
                    if chosen["field"] == category and
                    chosen.get("metadata", {}).get("selection_priority") == priority
                ) >= style_targets[priority]:
                    break
                if item.get("metadata", {}).get("selection_priority") != priority or not eligible(item):
                    continue
                selected.append(item)
                per_patient[item["patient_id"]] += 1
                selected_keys.add((item["patient_id"], tuple(item["relevant_encounter_ids"]),
                                   tuple(item["expected_answers"]), category))
                answer_counts[category][normalize_answer_key(item["expected_answers"])] += 1
                taken += 1

        if taken < quota:
            for item in candidates[category]:
                if taken >= quota:
                    break
                if not eligible(item):
                    continue
                selected.append(item)
                per_patient[item["patient_id"]] += 1
                selected_keys.add((item["patient_id"], tuple(item["relevant_encounter_ids"]),
                                   tuple(item["expected_answers"]), category))
                answer_counts[category][normalize_answer_key(item["expected_answers"])] += 1
                taken += 1
        if taken < quota:
            print(f"Warning: requested {quota} {category} questions; found {taken} under patient cap")

    rng.shuffle(selected)
    for index, item in enumerate(selected, 1):
        item["question_id"] = f"new_{index:03d}"
        item["benchmark_version"] = "new_v1"

    OUTPUT_FILE.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    counts = Counter(q["field"] for q in selected)
    types = Counter(q["question_type"] for q in selected)
    quota_errors = {
        category: {"expected": quota, "actual": counts.get(category, 0)}
        for category, quota in QUOTAS.items()
        if counts.get(category, 0) != quota
    }
    if len(selected) != TARGET_TOTAL or quota_errors:
        OUTPUT_FILE.unlink(missing_ok=True)
        raise RuntimeError(
            f"Benchmark validation failed: total={len(selected)}/{TARGET_TOTAL}, "
            f"quota_errors={quota_errors}. No benchmark file was kept."
        )
    print(f"Wrote {len(selected)} questions to {OUTPUT_FILE}")
    print(f"Patients represented: {len({q['patient_id'] for q in selected})}")
    print("Categories:", dict(sorted(counts.items())))
    print("Question types:", dict(sorted(types.items())))
    repeated = {
        category: dict(counts)
        for category, counts in answer_counts.items()
        if category not in {"negative", "allergies", "observations"}
        if any(value > MAX_SAME_POSITIVE_ANSWER for value in counts.values())
    }
    if repeated:
        raise RuntimeError(f"Answer repetition validation failed: {repeated}")
    print(f"Observation strings parseable as 'label: value': {observation_parseable}/{observation_total}")
    if observation_unparseable:
        print(f"Warning: {len(observation_unparseable)} observation strings were not parseable")
        for value in observation_unparseable[:5]:
            print(f"  Unparseable example: {value!r}")


if __name__ == "__main__":
    main()