import argparse
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

SEED = 42
TARGET_TOTAL = 120
MAX_PER_PATIENT = 2
MAX_SAME_POSITIVE_ANSWER = 2
QUOTAS = {
    "allergies": 10,
    "medications": 25,
    "clinical_facts": 25,
    "procedures": 20,
    "observations": 20,
    "negative": 20,
}
LOW_VALUE = {
    "Medication review due (situation)",
    "Medication reconciliation (procedure)",
    "Allergic disposition (finding)",
}
INVALID_PROCEDURE_TAGS = {
    "physical object", "finding", "situation", "observable entity"
}


def core(value):
    return re.sub(r"\s*\([^)]*\)\s*$", "", str(value)).strip()


def terminal_tag(value):
    tags = re.findall(r"\(([^()]*)\)", str(value))
    return tags[-1].lower() if tags else None


def valid_procedure(value):
    return terminal_tag(value) not in INVALID_PROCEDURE_TAGS


def date_obj(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def values(encounter, field):
    value = encounter.get(field)
    if isinstance(value, list):
        return [x for x in value if x and x not in LOW_VALUE]
    return [value] if value else []


def clinical_tokens(value):
    stop = {
        "of", "the", "and", "for", "with", "without", "history",
        "finding", "disorder", "situation", "procedure", "requested",
        "patient",
    }
    return {
        token for token in re.findall(r"[a-z0-9]+", core(value).lower())
        if token not in stop and len(token) > 2
    }


def clue_leaks_answer(clue, answer):
    clue_tokens, answer_tokens = clinical_tokens(clue), clinical_tokens(answer)
    if not clue_tokens or not answer_tokens:
        return False
    return len(clue_tokens & answer_tokens) / min(
        len(clue_tokens), len(answer_tokens)
    ) >= 0.5


def answer_key(answers):
    return " | ".join(sorted(core(x).lower() for x in answers))


def add_candidate(bucket, category, patient_id, question, answers,
                  encounter_ids, question_type, metadata,
                  answer_type="positive", answer_match="any"):
    answers = list(dict.fromkeys(str(x).strip() for x in answers if str(x).strip()))
    encounter_ids = list(dict.fromkeys(encounter_ids))
    if answers and encounter_ids:
        bucket[category].append({
            "question": question,
            "expected_answers": answers,
            "field": category,
            "patient_id": patient_id,
            "relevant_encounter_ids": encounter_ids,
            "question_type": question_type,
            "expected_answer_type": answer_type,
            "expected_answer_match": answer_match,
            "metadata": metadata,
        })


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=root / "data" / "ground_truth.json")
    parser.add_argument("--output", type=Path,
                        default=root / "data" / "eval_questions_v2.json")
    parser.add_argument("--base", type=Path,
                        default=root / "data" / "eval_questions_new.json",
                        help="Existing v1 benchmark to repair without resampling valid targets.")
    return parser.parse_args()


def locator_options(encounter, reason_counts, type_counts):
    visit = core(encounter.get("encounter_type") or "visit").lower()
    reason = encounter.get("reason_for_visit")
    reason_text = core(reason) if reason else None
    options = []
    if reason_text and reason_counts[reason_text.lower()] == 1:
        options.append((
            0, "problem_anchored_lookup",
            f"during the patient's visit for {reason_text}",
            "reason_for_visit", reason_text,
        ))
    if type_counts[visit] == 1:
        options.append((
            1, "unique_visit_lookup", f"during the patient's only {visit}",
            "encounter_type", visit,
        ))
    return options


def repair_v1(base_questions, data, patients, rng):
    """Preserve valid v1 targets and replace only rule-invalid questions."""
    kept = []
    rejected = []
    for original in base_questions:
        item = json.loads(json.dumps(original))
        invalid_procedure = (
            item.get("field") == "procedures" and any(
                terminal_tag(answer) in INVALID_PROCEDURE_TAGS
                for answer in item.get("expected_answers", [])
            )
        )
        invalid_negative = (
            item.get("field") == "negative" and
            item.get("metadata", {}).get("negative_field") not in
            {"medications", "procedures"}
        )
        if invalid_procedure or invalid_negative:
            rejected.append(item)
            continue

        old_field = item.get("field")
        source_map = {
            "conditions": "conditions", "clinical_facts": "conditions",
            "medications": "medications", "procedures": "procedures",
            "observations": "observations", "allergies": "allergies",
        }
        if old_field in source_map:
            item.setdefault("metadata", {})["source_field"] = source_map[old_field]
        if old_field == "conditions":
            item["question"] = item["question"].replace(
            "What additional condition or relevant medical history was",
            "What additional clinical finding or relevant medical history was",
        ).replace(
            "What condition was documented",
            "What clinical finding or relevant medical history was documented",
        )
        kept.append(item)

    needed = Counter({
        category: quota - sum(q["field"] == category for q in kept)
        for category, quota in QUOTAS.items()
    })
    per_patient = Counter(q["patient_id"] for q in kept)
    used = {
        (q["patient_id"], tuple(q["relevant_encounter_ids"]), q["field"])
        for q in kept
    }
    replacement_candidates = defaultdict(list)

    for patient_id, encounters in patients.items():
        reason_counts = Counter(
            core(e["reason_for_visit"]).lower()
            for _, e in encounters if e.get("reason_for_visit")
        )
        type_counts = Counter(
            core(e.get("encounter_type") or "visit").lower()
            for _, e in encounters
        )
        for encounter_id, encounter in encounters:
            locators = locator_options(encounter, reason_counts, type_counts)
            if not locators:
                continue
            raw_procedures = values(encounter, "procedures")
            if len(raw_procedures) == 1 and valid_procedure(raw_procedures[0]):
                for priority, lookup_type, locator, locator_field, locator_value in locators:
                    add_candidate(
                        replacement_candidates, "procedures", patient_id,
                        f"What procedure was documented {locator}?",
                        raw_procedures, [encounter_id], lookup_type,
                        {
                            "source_field": "procedures",
                            "encounter_date": encounter["encounter_date"],
                            "encounter_type": encounter.get("encounter_type"),
                            "selection_priority": priority,
                            "locator_field": locator_field,
                            "locator_value": locator_value,
                        },
                    )
            for negative_field, plural in (
                ("medications", "medications"),
                ("procedures", "procedures"),
            ):
                if values(encounter, negative_field):
                    continue
                for priority, lookup_type, locator, locator_field, locator_value in locators:
                    add_candidate(
                        replacement_candidates, "negative", patient_id,
                        f"Were any {plural} documented {locator}?",
                        [f"No {plural} were documented."], [encounter_id],
                        f"negative_{lookup_type}",
                        {
                            "source_field": negative_field,
                            "negative_field": negative_field,
                            "encounter_date": encounter["encounter_date"],
                            "encounter_type": encounter.get("encounter_type"),
                            "selection_priority": priority,
                            "locator_field": locator_field,
                            "locator_value": locator_value,
                        }, answer_type="negative",
                    )

    for category in ("procedures", "negative"):
        unique = {}
        for item in replacement_candidates[category]:
            key = (item["patient_id"], item["question"], tuple(item["expected_answers"]))
            unique[key] = item
        choices = list(unique.values())
        rng.shuffle(choices)
        choices.sort(key=lambda item: item["metadata"]["selection_priority"])
        added = 0
        for item in choices:
            key = (item["patient_id"], tuple(item["relevant_encounter_ids"]), category)
            if per_patient[item["patient_id"]] >= MAX_PER_PATIENT or key in used:
                continue
            kept.append(item)
            per_patient[item["patient_id"]] += 1
            used.add(key)
            added += 1
            if added == needed[category]:
                break
        if added != needed[category]:
            raise RuntimeError(
                f"Could replace only {added}/{needed[category]} invalid {category} questions"
            )

    rng.shuffle(kept)
    for index, item in enumerate(kept, 1):
        item["question_id"] = f"v2_{index:03d}"
        item["benchmark_version"] = "utility_v2"
    print(f"Preserved {len(base_questions) - len(rejected)} valid v1 targets")
    print(f"Replaced {len(rejected)} invalid v1 targets")
    return kept


def main():
    args = parse_args()
    rng = random.Random(SEED)
    data = json.loads(args.input.read_text(encoding="utf-8"))
    patients = defaultdict(list)
    for encounter_id, encounter in data.items():
        patients[encounter["patient_id"]].append((encounter_id, encounter))
    for encounters in patients.values():
        encounters.sort(key=lambda pair: date_obj(pair[1]["encounter_date"]))

    if args.base.exists():
        base_questions = json.loads(args.base.read_text(encoding="utf-8"))
        selected = repair_v1(base_questions, data, patients, rng)
        counts = Counter(item["field"] for item in selected)
        if len(selected) != TARGET_TOTAL or any(
            counts[category] != quota for category, quota in QUOTAS.items()
        ):
            raise RuntimeError(f"Repaired benchmark quota validation failed: {counts}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(selected, indent=2), encoding="utf-8")
        print(f"Wrote {len(selected)} questions to {args.output}")
        print(f"Patients represented: {len({q['patient_id'] for q in selected})}")
        print("Categories:", dict(sorted(counts.items())))
        print("Question types:", dict(sorted(Counter(q["question_type"] for q in selected).items())))
        return

    candidates = defaultdict(list)
    all_observations = [
        observation for encounter in data.values()
        for observation in values(encounter, "observations")
    ]
    parseable = sum(
        ":" in item and all(part.strip() for part in item.split(":", 1))
        for item in all_observations
    )

    for patient_id, encounters in patients.items():
        reason_counts = Counter(
            core(e["reason_for_visit"]).lower()
            for _, e in encounters if e.get("reason_for_visit")
        )
        type_counts = Counter(
            core(e.get("encounter_type") or "visit").lower()
            for _, e in encounters
        )

        for encounter_id, encounter in encounters:
            visit = core(encounter.get("encounter_type") or "visit").lower()
            reason = encounter.get("reason_for_visit")
            reason_text = core(reason) if reason else None
            locators = []
            if reason_text and reason_counts[reason_text.lower()] == 1:
                locators.append((
                    0, "problem_anchored_lookup",
                    f"during the patient's visit for {reason_text}",
                    "reason_for_visit", reason_text,
                ))
            if type_counts[visit] == 1:
                locators.append((
                    1, "unique_visit_lookup",
                    f"during the patient's only {visit}",
                    "encounter_type", visit,
                ))
            if not locators:
                continue

            positive_fields = (
                ("medications", "medications", "medication"),
                ("conditions", "clinical_facts", "clinical fact"),
                ("procedures", "procedures", "procedure"),
            )
            for source_field, category, noun in positive_fields:
                facts = values(encounter, source_field)
                if source_field == "procedures":
                    facts = [fact for fact in facts if valid_procedure(fact)]
                if len(facts) != 1:
                    continue
                if (source_field == "conditions" and reason_text and
                        clue_leaks_answer(reason_text, facts[0])):
                    continue
                for priority, lookup_type, locator, locator_field, locator_value in locators:
                    if source_field == "conditions":
                        prefix = "What additional clinical finding or relevant medical history was"
                    else:
                        prefix = f"What {noun} was documented"
                    add_candidate(
                        candidates, category, patient_id,
                        f"{prefix} {locator}?", facts, [encounter_id], lookup_type,
                        {
                            "source_field": source_field,
                            "encounter_date": encounter["encounter_date"],
                            "encounter_type": encounter.get("encounter_type"),
                            "selection_priority": priority,
                            "locator_field": locator_field,
                            "locator_value": locator_value,
                        },
                    )

            parsed = []
            for observation in values(encounter, "observations"):
                if ":" in observation:
                    label, result = observation.split(":", 1)
                    if label.strip() and result.strip():
                        parsed.append((label.strip(), result.strip(), observation))
            if parsed:
                label, result, full = rng.choice(parsed)
                for priority, lookup_type, locator, locator_field, locator_value in locators:
                    add_candidate(
                        candidates, "observations", patient_id,
                        f"What was the patient's {label.lower()} {locator}?",
                        [result, full], [encounter_id], lookup_type,
                        {
                            "source_field": "observations",
                            "observation_name": label,
                            "encounter_date": encounter["encounter_date"],
                            "selection_priority": priority,
                            "locator_field": locator_field,
                            "locator_value": locator_value,
                        },
                    )

            
            
            empty_fields = [
                (field, plural) for field, plural in (
                    ("medications", "medications"),
                    ("procedures", "procedures"),
                ) if not values(encounter, field)
            ]
            if empty_fields:
                negative_field, plural = rng.choice(empty_fields)
                for priority, lookup_type, locator, locator_field, locator_value in locators:
                    add_candidate(
                        candidates, "negative", patient_id,
                        f"Were any {plural} documented {locator}?",
                        [f"No {plural} were documented."], [encounter_id],
                        f"negative_{lookup_type}",
                        {
                            "source_field": negative_field,
                            "negative_field": negative_field,
                            "encounter_date": encounter["encounter_date"],
                            "encounter_type": encounter.get("encounter_type"),
                            "selection_priority": priority,
                            "locator_field": locator_field,
                            "locator_value": locator_value,
                        }, answer_type="negative",
                    )

        allergy_sources = defaultdict(list)
        for encounter_id, encounter in encounters:
            for allergy in values(encounter, "allergies"):
                allergy_sources[allergy].append(encounter_id)
        if allergy_sources:
            all_allergies = sorted(allergy_sources)
            source_ids = list(dict.fromkeys(
                eid for ids in allergy_sources.values() for eid in ids
            ))
            add_candidate(
                candidates, "allergies", patient_id,
                "What allergies does this patient have?", all_allergies,
                source_ids, "patient_chart_lookup",
                {"source_field": "allergies", "selection_priority": 0},
                answer_match="all",
            )

    for category in candidates:
        unique = {}
        for item in candidates[category]:
            unique[(item["patient_id"], item["question"])] = item
        candidates[category] = list(unique.values())
        rng.shuffle(candidates[category])

    selected, per_patient = [], Counter()
    repeated_answers = defaultdict(Counter)
    for category, quota in QUOTAS.items():
        chosen_keys = set()
        style_targets = {0: (quota * 2 + 2) // 3, 1: quota // 3}

        def eligible(item):
            key = (
                item["patient_id"], tuple(item["relevant_encounter_ids"]),
                tuple(item["expected_answers"]), category,
            )
            if per_patient[item["patient_id"]] >= MAX_PER_PATIENT or key in chosen_keys:
                return False
            if category not in {"negative", "allergies", "observations"}:
                if repeated_answers[category][answer_key(item["expected_answers"])] >= MAX_SAME_POSITIVE_ANSWER:
                    return False
            return True

        chosen = []
        for priority in (0, 1):
            for item in candidates[category]:
                if sum(x["metadata"]["selection_priority"] == priority for x in chosen) >= style_targets[priority]:
                    break
                if item["metadata"]["selection_priority"] == priority and eligible(item):
                    chosen.append(item)
                    key = (item["patient_id"], tuple(item["relevant_encounter_ids"]),
                           tuple(item["expected_answers"]), category)
                    chosen_keys.add(key)
                    per_patient[item["patient_id"]] += 1
                    repeated_answers[category][answer_key(item["expected_answers"])] += 1
        for item in candidates[category]:
            if len(chosen) >= quota:
                break
            if eligible(item):
                chosen.append(item)
                key = (item["patient_id"], tuple(item["relevant_encounter_ids"]),
                       tuple(item["expected_answers"]), category)
                chosen_keys.add(key)
                per_patient[item["patient_id"]] += 1
                repeated_answers[category][answer_key(item["expected_answers"])] += 1
        if len(chosen) != quota:
            raise RuntimeError(f"Could create only {len(chosen)}/{quota} {category} questions")
        selected.extend(chosen)

    rng.shuffle(selected)
    for index, item in enumerate(selected, 1):
        item["question_id"] = f"v2_{index:03d}"
        item["benchmark_version"] = "utility_v2"

    counts = Counter(item["field"] for item in selected)
    if len(selected) != TARGET_TOTAL or any(counts[k] != v for k, v in QUOTAS.items()):
        raise RuntimeError(f"Final quota validation failed: {counts}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(f"Wrote {len(selected)} questions to {args.output}")
    print(f"Patients represented: {len({q['patient_id'] for q in selected})}")
    print("Categories:", dict(sorted(counts.items())))
    print("Question types:", dict(sorted(Counter(q["question_type"] for q in selected).items())))
    print(f"Observation strings parseable as 'label: value': {parseable}/{len(all_observations)}")


if __name__ == "__main__":
    main()
