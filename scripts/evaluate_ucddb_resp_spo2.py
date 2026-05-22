from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.resp_tools import RESP_detect_apnea, RESP_detect_hypopnea, RESP_estimate_rate
from biosignal_agent.tools.spo2_tools import SpO2_detect_desaturation, SpO2_summarize


def confusion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(1 for row in rows if row["truth"] == "respiratory_event" and row["prediction"] == "respiratory_event")
    tn = sum(1 for row in rows if row["truth"] == "normal" and row["prediction"] == "normal")
    fp = sum(1 for row in rows if row["truth"] == "normal" and row["prediction"] == "respiratory_event")
    fn = sum(1 for row in rows if row["truth"] == "respiratory_event" and row["prediction"] == "normal")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(rows) if rows else 0.0
    return {"true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn, "precision": precision, "recall_sensitivity": recall, "specificity": specificity, "f1": f1, "accuracy": accuracy}


def predict(resp_result: dict[str, Any], spo2_result: dict[str, Any], hypopnea_result: dict[str, Any] | None = None) -> str:
    if resp_result.get("apnea_event_count", 0) > 0:
        return "respiratory_event"
    if hypopnea_result and hypopnea_result.get("hypopnea_event_count", 0) > 0:
        return "respiratory_event"
    if spo2_result.get("desaturation_event_count", 0) > 0:
        return "respiratory_event"
    if spo2_result.get("time_below_90_fraction", 0.0) >= 0.2:
        return "respiratory_event"
    return "normal"


def evaluate(manifest_path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text())
    rows = []
    for record in manifest.get("records", []):
        resp_apnea = RESP_detect_apnea(record["resp_path"], float(record["resp_sampling_rate"]), column=None)
        resp_rate = RESP_estimate_rate(record["resp_path"], float(record["resp_sampling_rate"]), column=None)
        resp_hypopnea = RESP_detect_hypopnea(record["resp_path"], float(record["resp_sampling_rate"]), column=None)
        spo2_desat = SpO2_detect_desaturation(record["spo2_path"], float(record["spo2_sampling_rate"]), column=None)
        spo2_summary = SpO2_summarize(record["spo2_path"], float(record["spo2_sampling_rate"]), column=None)
        prediction = predict(resp_apnea, spo2_desat, resp_hypopnea)
        rows.append({
            "record": record["record"],
            "window_start_s": record["window_start_s"],
            "truth": record["label"],
            "prediction": prediction,
            "event_count": record.get("event_count"),
            "event_types": record.get("event_types", []),
            "resp_apnea_event_count": resp_apnea.get("apnea_event_count"),
            "resp_apnea_index_per_hour": resp_apnea.get("apnea_index_per_hour"),
            "resp_hypopnea_event_count": resp_hypopnea.get("hypopnea_event_count"),
            "resp_hypopnea_index_per_hour": resp_hypopnea.get("hypopnea_index_per_hour"),
            "respiratory_rate_bpm": resp_rate.get("respiratory_rate_bpm"),
            "desaturation_event_count": spo2_desat.get("desaturation_event_count"),
            "oxygen_desaturation_index_per_hour": spo2_desat.get("oxygen_desaturation_index_per_hour"),
            "time_below_90_fraction": spo2_desat.get("time_below_90_fraction"),
            "min_spo2_percent": spo2_desat.get("min_spo2_percent"),
            "mean_spo2_percent": spo2_summary.get("mean_spo2_percent"),
            "resp_error": resp_apnea.get("error"),
            "spo2_error": spo2_desat.get("error"),
        })
    return {
        "manifest": str(manifest_path),
        "num_windows": len(rows),
        "truth_counts": dict(Counter(row["truth"] for row in rows)),
        "prediction_counts": dict(Counter(row["prediction"] for row in rows)),
        "metrics": confusion(rows),
        "rows": rows,
    }


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["record", "window_start_s", "truth", "prediction", "event_count", "event_types", "resp_apnea_event_count", "resp_apnea_index_per_hour", "resp_hypopnea_event_count", "resp_hypopnea_index_per_hour", "respiratory_rate_bpm", "desaturation_event_count", "oxygen_desaturation_index_per_hour", "time_below_90_fraction", "min_spo2_percent", "mean_spo2_percent", "resp_error", "spo2_error"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key)) if isinstance(row.get(key), list) else row.get(key) for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RESP/SpO2 respiratory-event screening on UCDDB labels.")
    parser.add_argument("--manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/ucddb_resp_spo2_manifest.json")
    parser.add_argument("--out-json", default="/data1/jiahui/biosignal-agent/outputs/ucddb_resp_spo2_eval.json")
    parser.add_argument("--out-csv", default="/data1/jiahui/biosignal-agent/outputs/ucddb_resp_spo2_eval.csv")
    args = parser.parse_args()
    report = evaluate(args.manifest)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    write_csv(report["rows"], args.out_csv)
    print(json.dumps({key: report[key] for key in ["num_windows", "truth_counts", "prediction_counts", "metrics"]}, indent=2))


if __name__ == "__main__":
    main()
