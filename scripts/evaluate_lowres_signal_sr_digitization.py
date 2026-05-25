from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import interpolate, signal as scipy_signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.digitize_tools import Signal_digitize_waveform_image_ml
from scripts.evaluate_lowres_recovery_digitization import load_records
from scripts.evaluate_waveform_digitization import (
    corrcoef,
    detect_peaks,
    heart_rate,
    load_signal,
    mean_metric,
    nrmse,
    peak_f1,
)


SR_METHODS = ["linear", "cubic", "pchip", "fft", "polyphase", "savgol_cubic"]


def resample_signal(values: np.ndarray, target_len: int, method: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("empty signal")
    if len(values) == target_len:
        return values.copy()
    x_old = np.linspace(0.0, 1.0, len(values))
    x_new = np.linspace(0.0, 1.0, target_len)
    if method == "linear":
        return np.interp(x_new, x_old, values)
    if method == "cubic":
        if len(values) < 4:
            return np.interp(x_new, x_old, values)
        return interpolate.CubicSpline(x_old, values, bc_type="natural")(x_new)
    if method == "pchip":
        if len(values) < 3:
            return np.interp(x_new, x_old, values)
        return interpolate.PchipInterpolator(x_old, values)(x_new)
    if method == "fft":
        return scipy_signal.resample(values, target_len)
    if method == "polyphase":
        up = target_len
        down = len(values)
        gcd = int(np.gcd(up, down))
        out = scipy_signal.resample_poly(values, up // gcd, down // gcd)
        if len(out) != target_len:
            out = np.interp(x_new, np.linspace(0.0, 1.0, len(out)), out)
        return out[:target_len]
    if method == "savgol_cubic":
        if len(values) >= 9:
            window = min(len(values) // 8 * 2 + 1, 31)
            window = max(5, window if window % 2 else window + 1)
            if window < len(values):
                values = scipy_signal.savgol_filter(values, window_length=window, polyorder=3)
        return resample_signal(values, target_len, "cubic")
    raise ValueError(f"unknown method: {method}")


def write_signal(path: Path, values: np.ndarray, sampling_rate: float) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time_s": np.arange(len(values), dtype=float) / float(sampling_rate), "signal": values}).to_csv(path, index=False)


def evaluate_record(low_record: dict[str, Any], high_record: dict[str, Any], out_dir: Path, method: str, model_path: str, threshold: float, trace_method: str, use_lowres_reference: bool) -> dict[str, Any]:
    low_csv = out_dir / "lowres_csv" / f"{low_record['record']}_lowres_digitized.csv"
    if use_lowres_reference:
        low_values = load_signal(low_record["reference_path"])
        low_digitized = {"out_csv": str(low_record["reference_path"]), "error": None, "pixel_coverage": None, "mask_pixel_fraction": None}
    else:
        low_digitized = Signal_digitize_waveform_image_ml(
            image_path=low_record["image_path"],
            sampling_rate=float(low_record["sampling_rate"]),
            out_csv=str(low_csv),
            value_min=float(low_record["value_min"]),
            value_max=float(low_record["value_max"]),
            crop_left=int(low_record["crop_left"]),
            crop_right=int(low_record["crop_right"]),
            crop_top=int(low_record["crop_top"]),
            crop_bottom=int(low_record["crop_bottom"]),
            model_path=model_path,
            probability_threshold=threshold,
            smooth_window=1,
            trace_method=trace_method,
        )
        if low_digitized.get("error"):
            return {
                "record": low_record["record"],
                "modality": low_record["modality"],
                "variant": low_record.get("variant"),
                "method": method,
                "failure_stage": "lowres_digitize",
                "error": low_digitized.get("error"),
            }
        low_values = load_signal(low_digitized["out_csv"])

    ref = load_signal(high_record["reference_path"])
    try:
        pred = resample_signal(low_values, len(ref), method)
    except Exception as exc:
        return {
            "record": low_record["record"],
            "modality": low_record["modality"],
            "variant": low_record.get("variant"),
            "method": method,
            "failure_stage": "signal_sr",
            "error": str(exc),
        }
    out_csv = out_dir / "sr_csv" / f"{low_record['record']}_{method}_sr.csv"
    write_signal(out_csv, pred, float(high_record["sampling_rate"]))

    n = min(len(ref), len(pred))
    ref = ref[:n]
    pred = pred[:n]
    fs = float(high_record["sampling_rate"])
    ref_peaks = detect_peaks(ref, fs)
    pred_peaks = detect_peaks(pred, fs)
    peak_metrics = peak_f1(ref_peaks, pred_peaks, tolerance=max(5, int(round(fs * 0.02))))
    ref_hr = heart_rate(ref_peaks, fs)
    pred_hr = heart_rate(pred_peaks, fs)
    return {
        "record": low_record["record"],
        "modality": low_record["modality"],
        "variant": low_record.get("variant"),
        "method": method,
        "source": "lowres_reference_oracle" if use_lowres_reference else "lowres_digitized_image",
        "failure_stage": "",
        "error": "",
        "num_lowres_points": int(len(low_values)),
        "num_points": int(n),
        "lowres_pixel_coverage": low_digitized.get("pixel_coverage"),
        "lowres_mask_pixel_fraction": low_digitized.get("mask_pixel_fraction"),
        "waveform_correlation": corrcoef(ref, pred),
        "nrmse": nrmse(ref, pred),
        "reference_peaks": int(len(ref_peaks)),
        "digitized_peaks": int(len(pred_peaks)),
        "peak_precision": peak_metrics["precision"],
        "peak_recall": peak_metrics["recall"],
        "peak_f1": peak_metrics["f1"],
        "reference_hr_bpm": ref_hr,
        "digitized_hr_bpm": pred_hr,
        "hr_abs_error_bpm": abs(ref_hr - pred_hr) if ref_hr is not None and pred_hr is not None else None,
        "sr_csv": str(out_csv),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if not row.get("error")]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ok_rows:
        grouped[str(row.get("modality"))].append(row)
    by_modality = {
        modality: {
            "num_records": len(subset),
            "mean_waveform_correlation": mean_metric(subset, "waveform_correlation"),
            "mean_nrmse": mean_metric(subset, "nrmse"),
            "mean_peak_f1": mean_metric(subset, "peak_f1"),
            "mean_hr_abs_error_bpm": mean_metric(subset, "hr_abs_error_bpm"),
        }
        for modality, subset in sorted(grouped.items())
    }
    return {
        "num_records": len(rows),
        "num_ok": len(ok_rows),
        "mean_waveform_correlation": mean_metric(ok_rows, "waveform_correlation"),
        "mean_nrmse": mean_metric(ok_rows, "nrmse"),
        "mean_peak_f1": mean_metric(ok_rows, "peak_f1"),
        "mean_hr_abs_error_bpm": mean_metric(ok_rows, "hr_abs_error_bpm"),
        "failure_counts": dict(Counter(row.get("error") for row in rows if row.get("error"))),
        "by_modality": by_modality,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover high-resolution signals from low-resolution digitized waveform images in 1D signal space.")
    parser.add_argument("--lowres-manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark_manifest.json")
    parser.add_argument("--highres-manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark_highres_aligned_manifest.json")
    parser.add_argument("--method", choices=SR_METHODS, default="pchip")
    parser.add_argument("--out-dir", default="/data1/jiahui/biosignal-agent/outputs/lowres_signal_sr_digitization")
    parser.add_argument("--out-json", default="/data1/jiahui/biosignal-agent/outputs/lowres_signal_sr_digitization_eval.json")
    parser.add_argument("--out-csv", default="/data1/jiahui/biosignal-agent/outputs/lowres_signal_sr_digitization_eval.csv")
    parser.add_argument("--model-path", default="/data1/jiahui/biosignal-agent/outputs/waveform_digitization_pixel_model.joblib")
    parser.add_argument("--probability-threshold", type=float, default=0.5)
    parser.add_argument("--trace-method", choices=["median", "path", "momentum", "full", "lazy", "fragmented"], default="median")
    parser.add_argument("--include-modality", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--use-lowres-reference", action="store_true", help="Use the low-res reference CSV as an oracle coarse signal instead of digitizing the image.")
    args = parser.parse_args()

    low_records = load_records(args.lowres_manifest)
    high_records = load_records(args.highres_manifest)
    wanted = {item.lower() for item in args.include_modality} if args.include_modality else None
    ids = [record_id for record_id in low_records if record_id in high_records]
    if wanted:
        ids = [record_id for record_id in ids if str(low_records[record_id].get("modality", "")).lower() in wanted]
    if args.limit is not None:
        ids = ids[: args.limit]
    out_dir = Path(args.out_dir) / args.method
    rows = [
        evaluate_record(low_records[record_id], high_records[record_id], out_dir, args.method, args.model_path, args.probability_threshold, args.trace_method, args.use_lowres_reference)
        for record_id in ids
    ]
    metrics = summarize(rows)
    report = {
        "lowres_manifest": args.lowres_manifest,
        "highres_manifest": args.highres_manifest,
        "method": args.method,
        "source": "lowres_reference_oracle" if args.use_lowres_reference else "lowres_digitized_image",
        "model_path": args.model_path,
        "probability_threshold": args.probability_threshold,
        "trace_method": args.trace_method,
        "metrics": metrics,
        "rows": rows,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    fieldnames = sorted({key for row in rows for key in row})
    with Path(args.out_csv).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"metrics": metrics, "out_json": args.out_json, "out_csv": args.out_csv}, indent=2))


if __name__ == "__main__":
    main()
