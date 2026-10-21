import json
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "data" / "results"


def save_result(run_id, config, utility=None, attack=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    result = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "utility": utility,
        "attack": attack,
    }

    out_path = RESULTS_DIR / f"{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Saved result to {out_path}")