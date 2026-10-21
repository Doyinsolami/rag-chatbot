import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/synthea_csv")
NOTES_DIR = Path("data/notes")
GROUND_TRUTH_PATH = Path("data/ground_truth.json")

patients = pd.read_csv(DATA_DIR / "patients.csv")
encounters = pd.read_csv(DATA_DIR / "encounters.csv")
conditions = pd.read_csv(DATA_DIR / "conditions.csv")
medications = pd.read_csv(DATA_DIR / "medications.csv")
procedures = pd.read_csv(DATA_DIR / "procedures.csv")
allergies = pd.read_csv(DATA_DIR / "allergies.csv")
observations = pd.read_csv(DATA_DIR / "observations.csv")

CORE_VITALS = {
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Heart rate",
    "Respiratory rate",
    "Body temperature",
    "Oxygen saturation in Arterial blood",
    "Body Weight",
    "Body Height",
}

NON_CLINICAL_CONDITIONS = {
    "Educated to high school level (finding)",
    "Full-time employment (finding)",
    "Has a criminal record (finding)",
    "Homeless(finding)",
    "Housing unsatisfactory (finding)",
    "Lack of access to transportation (finding)",
    "Only received primary school education (finding)",
    "Part-time employment (finding)",
    "Received higher education (finding)",
    "Refugee (person)",
    "Risk activity involvement (finding)",
    "Social isolation (finding)",
    "Unemployed (finding)",
}

conditions_by_encounter = {k: v for k, v in conditions.groupby("ENCOUNTER")}
medications_by_encounter = {k: v for k, v in medications.groupby("ENCOUNTER")}
procedures_by_encounter = {k: v for k, v in procedures.groupby("ENCOUNTER")}
allergies_by_encounter = {k: v for k, v in allergies.groupby("ENCOUNTER")}
observations_by_encounter = {k: v for k, v in observations.groupby("ENCOUNTER")}


def format_name(row):
    parts = [str(row[field]) for field in ("FIRST", "LAST") if pd.notna(row[field])]
    return " ".join(parts) if parts else "Unknown"


patient_names = {row["Id"]: format_name(row) for _, row in patients.iterrows()}


def entities_for_encounter(grouped, encounter_id, exclude=None):
    rows = grouped.get(encounter_id)
    if rows is None:
        return []
    values = rows["DESCRIPTION"].dropna().astype(str).drop_duplicates().tolist()
    if exclude:
        values = [v for v in values if v not in exclude]
    return values


def allergies_for_encounter(encounter_id):
    rows = allergies_by_encounter.get(encounter_id)
    if rows is None:
        return []
    entries = []
    for _, row in rows.iterrows():
        allergen = row["DESCRIPTION"]
        reaction = row["DESCRIPTION1"] if pd.notna(row.get("DESCRIPTION1")) else None
        if reaction:
            entries.append(f"{allergen} (reaction: {reaction})")
        else:
            entries.append(str(allergen))
    return list(dict.fromkeys(entries))


def observations_for_encounter(encounter_id):
    rows = observations_by_encounter.get(encounter_id)
    if rows is None:
        return []
    rows = rows[rows["DESCRIPTION"].isin(CORE_VITALS)]
    entries = []
    for _, row in rows.iterrows():
        value = row["VALUE"]
        if pd.isna(value):
            continue
        units = f" {row['UNITS']}" if pd.notna(row.get("UNITS")) else ""
        entries.append(f"{row['DESCRIPTION']}: {value}{units}")
    return entries


def build_note(patient_name, encounter, conditions_list, medications_list,
                procedures_list, allergies_list, observations_list):
    lines = [
        f"Patient: {patient_name}",
        f"Encounter date: {encounter['START']}",
        f"Encounter type: {encounter['DESCRIPTION']}",
    ]
    if pd.notna(encounter.get("REASONDESCRIPTION")):
        lines.append(f"Reason for visit: {encounter['REASONDESCRIPTION']}.")
    if conditions_list:
        lines.append("Assessment: " + "; ".join(conditions_list) + ".")
    if procedures_list:
        lines.append("Procedures performed: " + "; ".join(procedures_list) + ".")
    if medications_list:
        lines.append("Medications prescribed: " + "; ".join(medications_list) + ".")
    if allergies_list:
        lines.append("Allergies: " + "; ".join(allergies_list) + ".")
    if observations_list:
        lines.append("Vitals: " + "; ".join(observations_list) + ".")
    return "\n".join(lines)


def main():
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in NOTES_DIR.glob("*.txt"):
        old_file.unlink()
    ground_truth = {}

    for _, encounter in encounters.iterrows():
        encounter_id = encounter["Id"]
        patient_id = encounter["PATIENT"]

        conditions_list = entities_for_encounter(
            conditions_by_encounter, encounter_id, exclude=NON_CLINICAL_CONDITIONS
        )
        medications_list = entities_for_encounter(medications_by_encounter, encounter_id)
        procedures_list = entities_for_encounter(procedures_by_encounter, encounter_id)
        allergies_list = allergies_for_encounter(encounter_id)
        observations_list = observations_for_encounter(encounter_id)

        reason = encounter.get("REASONDESCRIPTION")
        has_reason = pd.notna(reason)

        if not (conditions_list or medications_list or procedures_list
                or allergies_list or observations_list or has_reason):
            continue

        patient_name = patient_names.get(patient_id, "Unknown")
        note = build_note(
            patient_name,
            encounter,
            conditions_list,
            medications_list,
            procedures_list,
            allergies_list,
            observations_list,
        )

        note_path = NOTES_DIR / f"{encounter_id}.txt"
        note_path.write_text(note, encoding="utf-8")

        ground_truth[encounter_id] = {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "encounter_date": encounter["START"],
        "encounter_type": encounter["DESCRIPTION"],
        "reason_for_visit": reason if has_reason else None,
        "conditions": conditions_list,
        "medications": medications_list,
        "procedures": procedures_list,
        "allergies": allergies_list,
        "observations": observations_list,
}

    GROUND_TRUTH_PATH.write_text(
        json.dumps(ground_truth, indent=2), encoding="utf-8"
    )

    print(f"Wrote {len(ground_truth)} notes to {NOTES_DIR}")
    print(f"Wrote ground truth to {GROUND_TRUTH_PATH}")


if __name__ == "__main__":
    main()