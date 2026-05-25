from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.digitize_tools import ML_MODEL_PATH, Signal_digitize_waveform_image, Signal_digitize_waveform_image_ml
from biosignal_agent.tools.digitize_unet_tools import UNET_MODEL_PATH, Signal_digitize_waveform_image_unet
from biosignal_agent.tools.modality_tools import Signal_classify_modality


def load_signal(path: str) -> np.ndarray:
    frame = pd.read_csv(path)
    col = "signal" if "signal" in frame.columns else frame.select_dtypes("number").columns[-1]
    values = frame[col].to_numpy(dtype=float)
    return values[np.isfinite(values)]


def corrcoef(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or len(b) < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def nrmse(a: np.ndarray, b: np.ndarray) -> float | None:
    denom = float(np.percentile(a, 99) - np.percentile(a, 1))
    if denom < 1e-8:
        return None
    return float(np.sqrt(np.mean((a - b) ** 2)) / denom)


def detect_peaks(values: np.ndarray, sampling_rate: float) -> np.ndarray:
    centered = values - np.median(values)
    prominence = max(float(np.std(centered)) * 0.5, 1e-8)
    distance = max(1, int(float(sampling_rate) * 0.25))
    peaks, _ = scipy_signal.find_peaks(centered, distance=distance, prominence=prominence)
    return peaks.astype(int)


def peak_f1(reference: np.ndarray, predicted: np.ndarray, tolerance: int = 5) -> dict[str, Any]:
    used = set()
    tp = 0
    for peak in predicted:
        candidates = [idx for idx, ref in enumerate(reference) if idx not in used and abs(int(ref) - int(peak)) <= tolerance]
        if candidates:
            best = min(candidates, key=lambda idx: abs(int(reference[idx]) - int(peak)))
            used.add(best)
            tp += 1
    fp = len(predicted) - tp
    fn = len(reference) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn, "precision": precision, "recall": recall, "f1": f1}


def heart_rate(peaks: np.ndarray, sampling_rate: float) -> float | None:
    if len(peaks) < 2:
        return None
    intervals = np.diff(peaks) / float(sampling_rate)
    intervals = intervals[intervals > 0]
    if len(intervals) == 0:
        return None
    return float(60.0 / np.median(intervals))


def evaluate_record(record: dict[str, Any], out_dir: Path, method: str, model_path: str | None, probability_threshold: float, trace_method: str = "median") -> dict[str, Any]:
    out_csv = out_dir / f"{record['record']}_digitized.csv"
    common_args = dict(
        image_path=record["image_path"],
        sampling_rate=float(record["sampling_rate"]),
        out_csv=str(out_csv),
        value_min=float(record["value_min"]),
        value_max=float(record["value_max"]),
        crop_left=int(record["crop_left"]),
        crop_right=int(record["crop_right"]),
        crop_top=int(record["crop_top"]),
        crop_bottom=int(record["crop_bottom"]),
        smooth_window=1,
        trace_method=trace_method,
    )
    if method == "ml":
        digitized = Signal_digitize_waveform_image_ml(**common_args, model_path=model_path, probability_threshold=probability_threshold)
    elif method == "unet":
        digitized = Signal_digitize_waveform_image_unet(**common_args, model_path=model_path, probability_threshold=probability_threshold)
    else:
        digitized = Signal_digitize_waveform_image(**common_args, threshold=80)
    row = {"record": record["record"], "modality": record["modality"], "variant": record.get("variant"), "method": method, "trace_method": trace_method, "digitizer_error": digitized.get("error")}
    if digitized.get("error"):
        return row
    ref = load_signal(record["reference_path"])
    pred = load_signal(digitized["out_csv"])
    n = min(len(ref), len(pred))
    ref = ref[:n]
    pred = pred[:n]
    ref_peaks = detect_peaks(ref, float(record["sampling_rate"]))
    pred_peaks = detect_peaks(pred, float(record["sampling_rate"]))
    peak_metrics = peak_f1(ref_peaks, pred_peaks)
    ref_hr = heart_rate(ref_peaks, float(record["sampling_rate"]))
    pred_hr = heart_rate(pred_peaks, float(record["sampling_rate"]))
    modality_out = Signal_classify_modality(digitized["out_csv"], float(record["sampling_rate"]))
    row.update({
        "num_points": int(n),
        "pixel_coverage": digitized.get("pixel_coverage"),
        "waveform_correlation": corrcoef(ref, pred),
        "nrmse": nrmse(ref, pred),
        "low_dynamic_range_reference": bool((np.percentile(ref, 99) - np.percentile(ref, 1)) < 1e-8),
        "reference_peaks": int(len(ref_peaks)),
        "digitized_peaks": int(len(pred_peaks)),
        "peak_precision": peak_metrics["precision"],
        "peak_recall": peak_metrics["recall"],
        "peak_f1": peak_metrics["f1"],
        "reference_hr_bpm": ref_hr,
        "digitized_hr_bpm": pred_hr,
        "hr_abs_error_bpm": abs(ref_hr - pred_hr) if ref_hr is not None and pred_hr is not None else None,
        "modality_prediction": modality_out.get("predicted_modality"),
        "modality_confidence": modality_out.get("confidence"),
        "modality_correct": modality_out.get("predicted_modality") == record["modality"],
        "digitized_csv": digitized["out_csv"],
    })
    return row


def mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate image-to-signal digitization on rendered waveform benchmark.")
    parser.add_argument("--manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark_manifest.json")
    parser.add_argument("--out-dir", default="/data1/jiahui/biosignal-agent/outputs/digitized")
    parser.add_argument("--out-json", default="/data1/jiahui/biosignal-agent/outputs/waveform_digitization_eval.json")
    parser.add_argument("--out-csv", default="/data1/jiahui/biosignal-agent/outputs/waveform_digitization_eval.csv")
    parser.add_argument("--method", choices=["rule", "ml", "unet"], default="rule")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--probability-threshold", type=float, default=0.5)
    parser.add_argument("--trace-method", choices=["median", "path", "momentum", "full", "lazy", "fragmented"], default="median")
    args = parser.parse_args()
    if args.model_path is None:
        args.model_path = str(UNET_MODEL_PATH if args.method == "unet" else ML_MODEL_PATH)

    manifest = json.loads(Path(args.manifest).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [evaluate_record(record, out_dir, args.method, args.model_path, args.probability_threshold, args.trace_method) for record in manifest.get("records", [])]
    ok_rows = [row for row in rows if not row.get("digitizer_error")]
    metrics = {
        "num_records": len(rows),
        "num_ok": len(ok_rows),
        "mean_waveform_correlation": mean_metric(ok_rows, "waveform_correlation"),
        "mean_nrmse": mean_metric(ok_rows, "nrmse"),
        "mean_peak_f1": mean_metric(ok_rows, "peak_f1"),
        "mean_hr_abs_error_bpm": mean_metric(ok_rows, "hr_abs_error_bpm"),
        "modality_retention_accuracy": sum(1 for row in ok_rows if row.get("modality_correct")) / len(ok_rows) if ok_rows else 0.0,
    }
    by_variant = {}
    for variant in sorted({row.get("variant") for row in ok_rows}):
        subset = [row for row in ok_rows if row.get("variant") == variant]
        by_variant[variant] = {
            "num_records": len(subset),
            "mean_waveform_correlation": mean_metric(subset, "waveform_correlation"),
            "mean_nrmse": mean_metric(subset, "nrmse"),
            "mean_peak_f1": mean_metric(subset, "peak_f1"),
            "modality_retention_accuracy": sum(1 for row in subset if row.get("modality_correct")) / len(subset) if subset else 0.0,
        }
    report = {
        "manifest": args.manifest,
        "truth_counts": dict(Counter(row.get("modality") for row in rows)),
        "prediction_counts": dict(Counter(row.get("modality_prediction") for row in ok_rows)),
        "method": args.method,
        "trace_method": args.trace_method,
        "model_path": args.model_path if args.method in {"ml", "unet"} else None,
        "metrics": metrics,
        "by_variant": by_variant,
        "rows": rows,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    fieldnames = sorted({key for row in rows for key in row})
    with Path(args.out_csv).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"metrics": metrics, "by_variant": by_variant, "out_json": args.out_json}, indent=2))


if __name__ == "__main__":
    main()
