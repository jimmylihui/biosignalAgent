from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.eeg_tools import EEG_compute_bandpower, EEG_screen_seizure_like_activity
from scripts.benchmark_metrics import binary_metrics


def evaluate(manifest_path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text())
    rows = []
    for record in manifest.get("records", []):
        seizure = EEG_screen_seizure_like_activity(record["path"], float(record["sampling_rate"]), column=None)
        bandpower = EEG_compute_bandpower(record["path"], float(record["sampling_rate"]), column=None)
        prediction = "seizure" if seizure.get("seizure_like_risk") == "possible_seizure_like_activity_proxy" else "non_seizure"
        rows.append({
            "record": record["record"],
            "truth": record["label"],
            "prediction": prediction,
            "spike_count": seizure.get("spike_count"),
            "spike_rate_per_min": seizure.get("spike_rate_per_min"),
            "fast_power_ratio": seizure.get("fast_power_ratio"),
            "seizure_like_risk": seizure.get("seizure_like_risk"),
            "total_power": bandpower.get("total_power"),
            "tool_error": seizure.get("error") or bandpower.get("error"),
        })
    return {
        "manifest": str(manifest_path),
        "num_windows": len(rows),
        "truth_counts": dict(Counter(row["truth"] for row in rows)),
        "prediction_counts": dict(Counter(row["prediction"] for row in rows)),
        "metrics": binary_metrics(rows, positive="seizure"),
        "rows": rows,
    }


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CHB-MIT seizure-window EEG baseline.")
    parser.add_argument("--manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/chbmit_seizure_manifest.json")
    parser.add_argument("--out-json", default="/data1/jiahui/biosignal-agent/outputs/chbmit_seizure_eval.json")
    parser.add_argument("--out-csv", default="/data1/jiahui/biosignal-agent/outputs/chbmit_seizure_eval.csv")
    args = parser.parse_args()
    report = evaluate(args.manifest)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    write_csv(report["rows"], args.out_csv)
    print(json.dumps({key: report[key] for key in ["num_windows", "truth_counts", "prediction_counts", "metrics"]}, indent=2))


if __name__ == "__main__":
    main()
