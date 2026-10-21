import re
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PATIENTS_CSV = DATA_DIR / "synthea_csv" / "patients.csv"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.json"


def load_encounter_counts():
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    counts = {}
    for entry in ground_truth.values():
        pid = entry["patient_id"]
        counts[pid] = counts.get(pid, 0) + 1
    return counts


def load_patient_directory():
    patients = pd.read_csv(PATIENTS_CSV)
    counts = load_encounter_counts()
    directory = []
    for _, row in patients.iterrows():
        pid = row["Id"]
        directory.append({
            "patient_id": pid,
            "first": row["FIRST"],
            "last": row["LAST"],
            "name": re.sub(r"\d+\b", "", f"{row['FIRST']} {row['LAST']}"),
            "dob": row["BIRTHDATE"],
            "gender": row["GENDER"],
            "encounter_count": counts.get(pid, 0),
        })
    directory.sort(key=lambda p: (p["last"], p["first"]))
    return directory


def format_patient_label(patient):
    return f"{patient['name']} (DOB {patient['dob']}, {patient['gender']})"