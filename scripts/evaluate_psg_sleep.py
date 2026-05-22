from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.eeg_tools import EEG_estimate_sleep_stage_features
from biosignal_agent.tools.resp_tools import RESP_detect_apnea, RESP_detect_hypopnea
from biosignal_agent.tools.spo2_tools import SpO2_detect_desaturation


def binary_metrics(rows: list[dict[str, Any]], truth_key: str, pred_key: str, positive: str) -> dict[str, Any]:
    tp = sum(1 for row in rows if row[truth_key] == positive and row[pred_key] == positive)
    tn = sum(1 for row in rows if row[truth_key] != positive and row[pred_key] != positive)
    fp = sum(1 for row in rows if row[truth_key] != positive and row[pred_key] == positive)
    fn = sum(1 for row in rows if row[truth_key] == positive and row[pred_key] != positive)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(rows) if rows else 0.0
    return {"true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn, "precision": precision, "recall_sensitivity": recall, "specificity": specificity, "f1": f1, "accuracy": accuracy}


def multiclass_metrics(rows: list[dict[str, Any]], truth_key: str, pred_key: str) -> dict[str, Any]:
    labels = sorted({row[truth_key] for row in rows} | {row[pred_key] for row in rows})
    per_label = {}
    f1s = []
    for label in labels:
        one_vs_all = binary_metrics(rows, truth_key, pred_key, label)
        support = sum(1 for row in rows if row[truth_key] == label)
        per_label[label] = {"support": support, **one_vs_all}
        if support:
            f1s.append(one_vs_all["f1"])
    accuracy = sum(1 for row in rows if row[truth_key] == row[pred_key]) / len(rows) if rows else 0.0
    return {"accuracy": accuracy, "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0, "per_label": per_label}


def predict_stage(result: dict[str, Any]) -> str:
    hint = result.get("sleep_stage_hint")
    if hint == "n3_like_slow_wave":
        return "n3"
    if hint == "n1_n2_like":
        return "n1_n2"
    if hint == "wake_rem_like":
        return "wake_rem"
    return "unknown"


def predict_event(resp_apnea: dict[str, Any], resp_hypopnea: dict[str, Any], spo2_desat: dict[str, Any]) -> str:
    if resp_apnea.get("apnea_event_count", 0) > 0:
        return "respiratory_event"
    if resp_hypopnea.get("hypopnea_event_count", 0) > 0:
        return "respiratory_event"
    if spo2_desat.get("desaturation_event_count", 0) > 0:
        return "respiratory_event"
    return "normal"


def evaluate(manifest_path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text())
    rows = []
    for record in manifest.get("records", []):
        sleep = EEG_estimate_sleep_stage_features(record["eeg_path"], float(record["eeg_sampling_rate"]), column=None)
        apnea = RESP_detect_apnea(record["resp_path"], float(record["resp_sampling_rate"]), column=None)
        hypopnea = RESP_detect_hypopnea(record["resp_path"], float(record["resp_sampling_rate"]), column=None)
        desat = SpO2_detect_desaturation(record["spo2_path"], float(record["spo2_sampling_rate"]), column=None)
        rows.append({
            "record": record["record"],
            "window_start_s": record["window_start_s"],
            "truth_sleep_stage": record["coarse_sleep_stage"],
            "truth_sleep_stage_detail": record["sleep_stage"],
            "predicted_sleep_stage": predict_stage(sleep),
            "truth_resp_event": record["respiratory_event_label"],
            "predicted_resp_event": predict_event(apnea, hypopnea, desat),
            "sleep_stage_hint": sleep.get("sleep_stage_hint"),
            "delta_ratio": sleep.get("delta_ratio"),
            "theta_ratio": sleep.get("theta_ratio"),
            "alpha_ratio": sleep.get("alpha_ratio"),
            "beta_ratio": sleep.get("beta_ratio"),
            "apnea_event_count": apnea.get("apnea_event_count"),
            "hypopnea_event_count": hypopnea.get("hypopnea_event_count"),
            "desaturation_event_count": desat.get("desaturation_event_count"),
            "time_below_90_fraction": desat.get("time_below_90_fraction"),
            "event_types": record.get("event_types", []),
            "sleep_error": sleep.get("error"),
            "resp_error": apnea.get("error") or hypopnea.get("error"),
            "spo2_error": desat.get("error"),
        })
    return {
        "manifest": str(manifest_path),
        "num_windows": len(rows),
        "sleep_truth_counts": dict(Counter(row["truth_sleep_stage"] for row in rows)),
        "sleep_prediction_counts": dict(Counter(row["predicted_sleep_stage"] for row in rows)),
        "resp_truth_counts": dict(Counter(row["truth_resp_event"] for row in rows)),
        "resp_prediction_counts": dict(Counter(row["predicted_resp_event"] for row in rows)),
        "sleep_stage_metrics": multiclass_metrics(rows, "truth_sleep_stage", "predicted_sleep_stage"),
        "respiratory_event_metrics": binary_metrics(rows, "truth_resp_event", "predicted_resp_event", "respiratory_event"),
        "rows": rows,
    }


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key)) if isinstance(row.get(key), list) else row.get(key) for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PSG sleep-stage and respiratory-event baselines on UCDDB labels.")
    parser.add_argument("--manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/psg_sleep_manifest.json")
    parser.add_argument("--out-json", default="/data1/jiahui/biosignal-agent/outputs/psg_sleep_eval.json")
    parser.add_argument("--out-csv", default="/data1/jiahui/biosignal-agent/outputs/psg_sleep_eval.csv")
    args = parser.parse_args()
    report = evaluate(args.manifest)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    write_csv(report["rows"], args.out_csv)
    print(json.dumps({key: report[key] for key in ["num_windows", "sleep_truth_counts", "sleep_prediction_counts", "resp_truth_counts", "resp_prediction_counts", "sleep_stage_metrics", "respiratory_event_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
