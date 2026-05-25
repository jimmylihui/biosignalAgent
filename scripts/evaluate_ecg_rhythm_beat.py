from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.ecg_tools import ECG_detect_r_peaks, ECG_screen_arrhythmia


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


def multiclass_accuracy(rows: list[dict[str, Any]], truth_key: str, pred_key: str) -> dict[str, Any]:
    labels = sorted({row[truth_key] for row in rows} | {row[pred_key] for row in rows})
    per_label = {}
    for label in labels:
        support = sum(1 for row in rows if row[truth_key] == label)
        correct = sum(1 for row in rows if row[truth_key] == label and row[pred_key] == label)
        per_label[label] = {"support": support, "accuracy": correct / support if support else 0.0}
    return {"accuracy": sum(1 for row in rows if row[truth_key] == row[pred_key]) / len(rows) if rows else 0.0, "per_label": per_label}


def predict_rhythm(arrhythmia: dict[str, Any]) -> str:
    model_rhythm = arrhythmia.get("predicted_rhythm")
    if model_rhythm in {"normal", "af", "other_rhythm"}:
        return model_rhythm
    flags = set(arrhythmia.get("arrhythmia_flags", []))
    rr_cv = arrhythmia.get("rr_cv") or 0.0
    if "irregular_rr_pattern" in flags and rr_cv >= 0.18:
        return "af"
    if arrhythmia.get("arrhythmia_risk") == "elevated":
        return "other_rhythm"
    return "normal"


def beat_predictions(record: dict[str, Any], peak_result: dict[str, Any], tolerance_ms: float) -> list[dict[str, Any]]:
    fs = float(record["sampling_rate"])
    peaks = np.asarray(peak_result.get("r_peak_indices", []), dtype=int)
    annotations = record.get("beat_annotations", [])
    tol = int(tolerance_ms / 1000.0 * fs)
    rr = np.diff(peaks) / fs if len(peaks) > 1 else np.array([])
    rr_median = float(np.median(rr)) if len(rr) else None
    rows = []
    for ann in annotations:
        sample = int(ann["sample"])
        if len(peaks):
            idx = int(np.argmin(np.abs(peaks - sample)))
            distance = int(abs(peaks[idx] - sample))
            matched = distance <= tol
        else:
            idx = -1
            distance = None
            matched = False
        pred_label = "missed"
        if matched:
            local_rr = []
            if idx > 0:
                local_rr.append((peaks[idx] - peaks[idx - 1]) / fs)
            if idx + 1 < len(peaks):
                local_rr.append((peaks[idx + 1] - peaks[idx]) / fs)
            outlier = bool(rr_median and any(abs(x - rr_median) / rr_median > 0.2 for x in local_rr))
            pred_label = "abnormal" if outlier else "normal"
        truth_binary = "normal" if ann["label"] == "normal" else "abnormal"
        rows.append({
            "record": record["record"],
            "window_start_s": record["window_start_s"],
            "sample": sample,
            "truth_symbol": ann["symbol"],
            "truth_label": ann["label"],
            "truth_binary": truth_binary,
            "predicted_binary": pred_label,
            "matched_peak": matched,
            "match_distance_samples": distance,
        })
    return rows


def evaluate(manifest_path: str | Path, tolerance_ms: float) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text())
    rhythm_rows = []
    beat_rows = []
    for record in manifest.get("records", []):
        arrhythmia = ECG_screen_arrhythmia(record["path"], float(record["sampling_rate"]), column=None)
        peaks = ECG_detect_r_peaks(record["path"], float(record["sampling_rate"]), column=None)
        rhythm_rows.append({
            "record": record["record"],
            "window_start_s": record["window_start_s"],
            "truth_rhythm": record["coarse_rhythm_label"],
            "truth_rhythm_detail": record["rhythm_label"],
            "predicted_rhythm": predict_rhythm(arrhythmia),
            "arrhythmia_risk": arrhythmia.get("arrhythmia_risk"),
            "arrhythmia_flags": arrhythmia.get("arrhythmia_flags", []),
            "rr_cv": arrhythmia.get("rr_cv"),
            "heart_rate_bpm": arrhythmia.get("heart_rate_bpm"),
            "predicted_rhythm_model": arrhythmia.get("predicted_rhythm"),
            "rhythm_probabilities": arrhythmia.get("rhythm_probabilities"),
            "error": arrhythmia.get("error"),
        })
        beat_rows.extend(beat_predictions(record, peaks, tolerance_ms))
    return {
        "manifest": str(manifest_path),
        "num_windows": len(rhythm_rows),
        "num_beats": len(beat_rows),
        "rhythm_truth_counts": dict(Counter(row["truth_rhythm"] for row in rhythm_rows)),
        "rhythm_prediction_counts": dict(Counter(row["predicted_rhythm"] for row in rhythm_rows)),
        "beat_truth_counts": dict(Counter(row["truth_binary"] for row in beat_rows)),
        "beat_prediction_counts": dict(Counter(row["predicted_binary"] for row in beat_rows)),
        "af_metrics": binary_metrics(rhythm_rows, "truth_rhythm", "predicted_rhythm", "af"),
        "rhythm_multiclass": multiclass_accuracy(rhythm_rows, "truth_rhythm", "predicted_rhythm"),
        "beat_abnormal_metrics": binary_metrics(beat_rows, "truth_binary", "predicted_binary", "abnormal"),
        "beat_detection_recall": sum(1 for row in beat_rows if row["matched_peak"]) / len(beat_rows) if beat_rows else 0.0,
        "rhythm_rows": rhythm_rows,
        "beat_rows": beat_rows,
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
    parser = argparse.ArgumentParser(description="Evaluate ECG rhythm/AF and beat-label proxies on MIT-BIH labels.")
    parser.add_argument("--manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/ecg_rhythm_beat_manifest.json")
    parser.add_argument("--out-json", default="/data1/jiahui/biosignal-agent/outputs/ecg_rhythm_beat_eval.json")
    parser.add_argument("--rhythm-csv", default="/data1/jiahui/biosignal-agent/outputs/ecg_rhythm_eval.csv")
    parser.add_argument("--beat-csv", default="/data1/jiahui/biosignal-agent/outputs/ecg_beat_eval.csv")
    parser.add_argument("--tolerance-ms", type=float, default=100.0)
    args = parser.parse_args()
    report = evaluate(args.manifest, args.tolerance_ms)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    write_csv(report["rhythm_rows"], args.rhythm_csv)
    write_csv(report["beat_rows"], args.beat_csv)
    print(json.dumps({key: report[key] for key in ["num_windows", "num_beats", "rhythm_truth_counts", "rhythm_prediction_counts", "beat_truth_counts", "beat_prediction_counts", "af_metrics", "beat_abnormal_metrics", "beat_detection_recall"]}, indent=2))


if __name__ == "__main__":
    main()
