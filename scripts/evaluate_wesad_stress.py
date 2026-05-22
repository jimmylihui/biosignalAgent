from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.eda_tools import EDA_detect_arousal_events, EDA_screen_stress_proxy, EDA_summarize
from scripts.benchmark_metrics import binary_metrics


def evaluate(manifest_path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text())
    rows = []
    for record in manifest.get("records", []):
        stress = EDA_screen_stress_proxy(record["path"], float(record["sampling_rate"]), column=None)
        summary = EDA_summarize(record["path"], float(record["sampling_rate"]), column=None)
        events = EDA_detect_arousal_events(record["path"], float(record["sampling_rate"]), column=None)
        prediction = "stress" if stress.get("stress_arousal_level") == "elevated_stress_arousal_proxy" else "non_stress"
        rows.append({
            "record": record["record"],
            "truth": record["label"],
            "prediction": prediction,
            "stress_arousal_score": stress.get("stress_arousal_score"),
            "stress_arousal_level": stress.get("stress_arousal_level"),
            "arousal_rate_per_min": stress.get("arousal_rate_per_min"),
            "normalized_phasic_std": stress.get("normalized_phasic_std"),
            "mean_level": summary.get("mean_level"),
            "phasic_std": summary.get("phasic_std"),
            "arousal_event_count": events.get("arousal_event_count"),
            "tool_error": stress.get("error") or summary.get("error") or events.get("error"),
        })
    return {
        "manifest": str(manifest_path),
        "num_windows": len(rows),
        "truth_counts": dict(Counter(row["truth"] for row in rows)),
        "prediction_counts": dict(Counter(row["prediction"] for row in rows)),
        "metrics": binary_metrics(rows, positive="stress"),
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
    parser = argparse.ArgumentParser(description="Evaluate WESAD EDA stress baseline.")
    parser.add_argument("--manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/wesad_stress_manifest.json")
    parser.add_argument("--out-json", default="/data1/jiahui/biosignal-agent/outputs/wesad_stress_eval.json")
    parser.add_argument("--out-csv", default="/data1/jiahui/biosignal-agent/outputs/wesad_stress_eval.csv")
    args = parser.parse_args()
    report = evaluate(args.manifest)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    write_csv(report["rows"], args.out_csv)
    print(json.dumps({key: report[key] for key in ["num_windows", "truth_counts", "prediction_counts", "metrics"]}, indent=2))


if __name__ == "__main__":
    main()
