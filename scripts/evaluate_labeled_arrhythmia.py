from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.ecg_tools import ECG_screen_arrhythmia


def confusion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(1 for row in rows if row["truth_binary"] == "abnormal" and row["predicted_binary"] == "abnormal")
    tn = sum(1 for row in rows if row["truth_binary"] == "normal" and row["predicted_binary"] == "normal")
    fp = sum(1 for row in rows if row["truth_binary"] == "normal" and row["predicted_binary"] == "abnormal")
    fn = sum(1 for row in rows if row["truth_binary"] == "abnormal" and row["predicted_binary"] == "normal")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(rows) if rows else 0.0
    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
    }


def evaluate(manifest_path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text())
    rows = []
    for record in manifest.get("records", []):
        result = ECG_screen_arrhythmia(record["path"], float(record["sampling_rate"]), column=None)
        predicted = "abnormal" if result.get("arrhythmia_risk") == "elevated" else "normal"
        rows.append({
            "record": record["record"],
            "window_start_s": record["window_start_s"],
            "truth_binary": record["binary_label"],
            "truth_label": record["label"],
            "predicted_binary": predicted,
            "arrhythmia_risk": result.get("arrhythmia_risk"),
            "arrhythmia_flags": result.get("arrhythmia_flags", []),
            "heart_rate_bpm": result.get("heart_rate_bpm"),
            "rr_cv": result.get("rr_cv"),
            "pause_count": result.get("pause_count"),
            "ectopy_proxy_fraction": result.get("ectopy_proxy_fraction"),
            "confidence": result.get("confidence"),
            "error": result.get("error"),
            "num_beats": record.get("num_beats"),
            "num_abnormal_beats": record.get("num_abnormal_beats"),
            "abnormal_fraction": record.get("abnormal_fraction"),
        })
    return {
        "manifest": str(manifest_path),
        "num_windows": len(rows),
        "truth_counts": dict(Counter(row["truth_binary"] for row in rows)),
        "prediction_counts": dict(Counter(row["predicted_binary"] for row in rows)),
        "label_counts": dict(Counter(row["truth_label"] for row in rows)),
        "metrics": confusion(rows),
        "rows": rows,
    }


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record", "window_start_s", "truth_binary", "truth_label", "predicted_binary", "arrhythmia_risk",
        "arrhythmia_flags", "heart_rate_bpm", "rr_cv", "pause_count", "ectopy_proxy_fraction", "confidence",
        "error", "num_beats", "num_abnormal_beats", "abnormal_fraction"
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key)) if isinstance(row.get(key), list) else row.get(key) for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ECG arrhythmia screening on labeled MIT-BIH windows.")
    parser.add_argument("--manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/labeled_arrhythmia_manifest.json")
    parser.add_argument("--out-json", default="/data1/jiahui/biosignal-agent/outputs/labeled_arrhythmia_eval.json")
    parser.add_argument("--out-csv", default="/data1/jiahui/biosignal-agent/outputs/labeled_arrhythmia_eval.csv")
    args = parser.parse_args()
    report = evaluate(args.manifest)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    write_csv(report["rows"], args.out_csv)
    print(json.dumps({key: report[key] for key in ["num_windows", "truth_counts", "prediction_counts", "label_counts", "metrics"]}, indent=2))


if __name__ == "__main__":
    main()
