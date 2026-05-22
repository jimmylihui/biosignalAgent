from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.ppg_tools import PPG_assess_perfusion_variability, PPG_screen_pulse_irregularity
from scripts.benchmark_metrics import binary_metrics


def evaluate(manifest_path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text())
    rows = []
    for record in manifest.get("records", []):
        irregular = PPG_screen_pulse_irregularity(record["path"], float(record["sampling_rate"]), column=None)
        perfusion = PPG_assess_perfusion_variability(record["path"], float(record["sampling_rate"]), column=None)
        prediction = "af" if irregular.get("irregular_pulse_risk") == "elevated_irregular_pulse_proxy" else "non_af"
        rows.append({
            "record": record["record"],
            "truth": record["label"],
            "prediction": prediction,
            "heart_rate_bpm": irregular.get("heart_rate_bpm"),
            "pulse_interval_cv": irregular.get("pulse_interval_cv"),
            "normalized_rmssd": irregular.get("normalized_rmssd"),
            "successive_change_fraction": irregular.get("successive_change_fraction"),
            "irregular_pulse_score": irregular.get("irregular_pulse_score"),
            "irregular_pulse_risk": irregular.get("irregular_pulse_risk"),
            "pulse_amplitude_proxy": perfusion.get("pulse_amplitude_proxy"),
            "tool_error": irregular.get("error") or perfusion.get("error"),
        })
    return {
        "manifest": str(manifest_path),
        "num_windows": len(rows),
        "truth_counts": dict(Counter(row["truth"] for row in rows)),
        "prediction_counts": dict(Counter(row["prediction"] for row in rows)),
        "metrics": binary_metrics(rows, positive="af"),
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
    parser = argparse.ArgumentParser(description="Evaluate PPG AF/non-AF pulse-irregularity baseline.")
    parser.add_argument("--manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/ppg_af_manifest.json")
    parser.add_argument("--out-json", default="/data1/jiahui/biosignal-agent/outputs/ppg_af_eval.json")
    parser.add_argument("--out-csv", default="/data1/jiahui/biosignal-agent/outputs/ppg_af_eval.csv")
    args = parser.parse_args()
    report = evaluate(args.manifest)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    write_csv(report["rows"], args.out_csv)
    print(json.dumps({key: report[key] for key in ["num_windows", "truth_counts", "prediction_counts", "metrics"]}, indent=2))


if __name__ == "__main__":
    main()
