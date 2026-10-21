"""Fail-fast validation for the utility_v2 benchmark."""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_QUOTAS = {
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


def values(encounter, field):
    value = encounter.get(field)
    if isinstance(value, list):
        return [x for x in value if x and x not in LOW_VALUE]
    return [value] if value else []


def terminal_tag(value):
    tags = re.findall(r"\(([^()]*)\)", str(value))
    return tags[-1].lower() if tags else None


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path,
                        default=root / "data" / "ground_truth.json")
    parser.add_argument("--questions", type=Path,
                        default=root / "data" / "eval_questions_v2.json")
    return parser.parse_args()


def main():
    args = parse_args()
    ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    patients = defaultdict(list)
    for encounter_id, encounter in ground_truth.items():
        patients[encounter["patient_id"]].append((encounter_id, encounter))

    errors, warnings = [], []
    ids = [q.get("question_id") for q in questions]
    if len(ids) != len(set(ids)):
        errors.append("Question IDs are not unique")
    if len(questions) != sum(EXPECTED_QUOTAS.values()):
        errors.append(f"Expected 120 questions, found {len(questions)}")
    counts = Counter(q.get("field") for q in questions)
    if counts != Counter(EXPECTED_QUOTAS):
        errors.append(f"Category quotas differ: {dict(counts)}")

    for q in questions:
        qid = q.get("question_id", "<missing>")
        if q.get("benchmark_version") != "utility_v2":
            errors.append(f"{qid}: wrong benchmark version")
        patient_id = q.get("patient_id")
        relevant = q.get("relevant_encounter_ids") or []
        if not relevant:
            errors.append(f"{qid}: no relevant encounter IDs")
            continue
        for encounter_id in relevant:
            encounter = ground_truth.get(encounter_id)
            if encounter is None:
                errors.append(f"{qid}: missing encounter {encounter_id}")
            elif encounter["patient_id"] != patient_id:
                errors.append(f"{qid}: encounter belongs to a different patient")

        metadata = q.get("metadata", {})
        source_field = metadata.get("source_field")
        expected = set(q.get("expected_answers") or [])
        if q.get("field") == "allergies":
            chart_allergies = {
                allergy for _, encounter in patients[patient_id]
                for allergy in values(encounter, "allergies")
            }
            if expected != chart_allergies:
                errors.append(
                    f"{qid}: allergy answer mismatch; expected={sorted(expected)}, "
                    f"chart={sorted(chart_allergies)}"
                )
            if q.get("expected_answer_match") != "all":
                errors.append(f"{qid}: allergy question must require all answers")

        if q.get("field") == "procedures":
            for answer in expected:
                if terminal_tag(answer) in INVALID_PROCEDURE_TAGS:
                    errors.append(f"{qid}: invalid procedure target {answer!r}")

        if q.get("field") == "negative":
            negative_field = metadata.get("negative_field")
            if negative_field not in {"medications", "procedures"}:
                errors.append(f"{qid}: invalid negative field {negative_field!r}")
            for encounter_id in relevant:
                if encounter_id in ground_truth and values(
                    ground_truth[encounter_id], negative_field
                ):
                    errors.append(f"{qid}: negative field is not empty")
        elif source_field and q.get("field") != "allergies":
            source_values = {
                value for encounter_id in relevant if encounter_id in ground_truth
                for value in values(ground_truth[encounter_id], source_field)
            }
            
            
            
            
            answer_found = (
                bool(expected & source_values)
                if q.get("field") == "observations"
                else expected.issubset(source_values)
            )
            if not answer_found:
                errors.append(f"{qid}: expected answer absent from source field")

        locator_field = metadata.get("locator_field")
        locator_value = str(metadata.get("locator_value", "")).lower()
        if locator_field and locator_value:
            matches = 0
            for _, encounter in patients[patient_id]:
                raw = encounter.get(locator_field)
                candidate = re.sub(r"\s*\([^)]*\)\s*$", "", str(raw)).strip().lower()
                if candidate == locator_value:
                    matches += 1
            if matches != 1:
                errors.append(f"{qid}: locator is not unique ({matches} matches)")

    per_patient = Counter(q["patient_id"] for q in questions)
    if per_patient and max(per_patient.values()) > 2:
        errors.append(f"Patient cap exceeded: maximum {max(per_patient.values())}")

    print(f"Questions: {len(questions)}")
    print("Categories:", dict(sorted(counts.items())))
    print(f"Patients represented: {len(per_patient)}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    for item in errors:
        print("ERROR:", item)
    for item in warnings:
        print("WARNING:", item)
    if errors:
        raise SystemExit(1)
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
