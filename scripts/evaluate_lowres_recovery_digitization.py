from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.digitize_tools import Signal_digitize_waveform_image_ml
from biosignal_agent.tools.digitize_unet_tools import Signal_digitize_waveform_image_unet
from scripts.evaluate_waveform_digitization import (
    corrcoef,
    detect_peaks,
    heart_rate,
    load_signal,
    mean_metric,
    nrmse,
    peak_f1,
)


RESAMPLE_METHODS = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def load_records(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    return {str(record["record"]): record for record in payload.get("records", [])}


def upscale_image(src: str, dst: Path, scale: int, method: str, blur_radius: float = 0.0) -> None:
    image = Image.open(src).convert("RGB")
    if blur_radius > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    width, height = image.size
    upscaled = image.resize((int(width * scale), int(height * scale)), resample=RESAMPLE_METHODS[method])
    dst.parent.mkdir(parents=True, exist_ok=True)
    upscaled.save(dst)


def recovered_record(low_record: dict[str, Any], high_record: dict[str, Any], image_path: Path, scale: int) -> dict[str, Any]:
    return {
        **high_record,
        "source_lowres_image_path": low_record["image_path"],
        "image_path": str(image_path),
        "crop_left": int(low_record.get("crop_left") or 0) * scale,
        "crop_right": int(low_record.get("crop_right") or 0) * scale,
        "crop_top": int(low_record.get("crop_top") or 0) * scale,
        "crop_bottom": int(low_record.get("crop_bottom") or 0) * scale,
        "sampling_rate": float(high_record["sampling_rate"]),
        "num_points": int(high_record.get("num_points") or 0),
        "value_min": float(high_record["value_min"]),
        "value_max": float(high_record["value_max"]),
        "reference_path": high_record["reference_path"],
    }


def evaluate_record(record: dict[str, Any], out_dir: Path, method: str, model_path: str, threshold: float, trace_method: str) -> dict[str, Any]:
    out_csv = out_dir / "csv" / f"{record['record']}_recovered_digitized.csv"
    digitizer = Signal_digitize_waveform_image_unet if method == "unet" else Signal_digitize_waveform_image_ml
    digitized = digitizer(
        image_path=record["image_path"],
        sampling_rate=float(record["sampling_rate"]),
        out_csv=str(out_csv),
        value_min=float(record["value_min"]),
        value_max=float(record["value_max"]),
        crop_left=int(record["crop_left"]),
        crop_right=int(record["crop_right"]),
        crop_top=int(record["crop_top"]),
        crop_bottom=int(record["crop_bottom"]),
        model_path=model_path,
        probability_threshold=threshold,
        smooth_window=1,
        trace_method=trace_method,
    )
    row = {
        "record": record["record"],
        "modality": record["modality"],
        "variant": record.get("variant"),
        "method": method,
        "digitizer_error": digitized.get("error"),
        "image_path": record["image_path"],
        "source_lowres_image_path": record["source_lowres_image_path"],
    }
    if digitized.get("error"):
        return row
    ref = load_signal(record["reference_path"])
    pred = load_signal(digitized["out_csv"])
    n = min(len(ref), len(pred))
    ref = ref[:n]
    pred = pred[:n]
    fs = float(record["sampling_rate"])
    ref_peaks = detect_peaks(ref, fs)
    pred_peaks = detect_peaks(pred, fs)
    peak_metrics = peak_f1(ref_peaks, pred_peaks, tolerance=max(5, int(round(fs * 0.02))))
    ref_hr = heart_rate(ref_peaks, fs)
    pred_hr = heart_rate(pred_peaks, fs)
    row.update({
        "num_points": int(n),
        "pixel_coverage": digitized.get("pixel_coverage"),
        "mask_pixel_fraction": digitized.get("mask_pixel_fraction"),
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
        "digitized_csv": digitized["out_csv"],
    })
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if not row.get("digitizer_error")]
    by_modality = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ok_rows:
        grouped[str(row.get("modality"))].append(row)
    for modality, subset in sorted(grouped.items()):
        by_modality[modality] = {
            "num_records": len(subset),
            "mean_waveform_correlation": mean_metric(subset, "waveform_correlation"),
            "mean_nrmse": mean_metric(subset, "nrmse"),
            "mean_peak_f1": mean_metric(subset, "peak_f1"),
            "mean_hr_abs_error_bpm": mean_metric(subset, "hr_abs_error_bpm"),
        }
    return {
        "num_records": len(rows),
        "num_ok": len(ok_rows),
        "mean_waveform_correlation": mean_metric(ok_rows, "waveform_correlation"),
        "mean_nrmse": mean_metric(ok_rows, "nrmse"),
        "mean_peak_f1": mean_metric(ok_rows, "peak_f1"),
        "mean_hr_abs_error_bpm": mean_metric(ok_rows, "hr_abs_error_bpm"),
        "failure_counts": dict(Counter(row.get("digitizer_error") for row in rows if row.get("digitizer_error"))),
        "by_modality": by_modality,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate low-resolution waveform image recovery before digitization.")
    parser.add_argument("--lowres-manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark_manifest.json")
    parser.add_argument("--highres-manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark_highres_aligned_manifest.json")
    parser.add_argument("--out-dir", default="/data1/jiahui/biosignal-agent/outputs/lowres_recovery_digitization")
    parser.add_argument("--out-json", default="/data1/jiahui/biosignal-agent/outputs/lowres_recovery_digitization_eval.json")
    parser.add_argument("--out-csv", default="/data1/jiahui/biosignal-agent/outputs/lowres_recovery_digitization_eval.csv")
    parser.add_argument("--method", choices=["ml", "unet"], default="ml")
    parser.add_argument("--model-path", default="/data1/jiahui/biosignal-agent/outputs/waveform_digitization_pixel_model_highres.joblib")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--upscale-method", choices=sorted(RESAMPLE_METHODS), default="lanczos")
    parser.add_argument("--pre-blur-radius", type=float, default=0.0)
    parser.add_argument("--probability-threshold", type=float, default=0.5)
    parser.add_argument("--trace-method", choices=["median", "path", "momentum", "full", "lazy", "fragmented"], default="median")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-modality", action="append", default=None)
    args = parser.parse_args()

    low_records = load_records(args.lowres_manifest)
    high_records = load_records(args.highres_manifest)
    wanted = {item.lower() for item in args.include_modality} if args.include_modality else None
    shared_ids = [record_id for record_id in low_records if record_id in high_records]
    if wanted:
        shared_ids = [record_id for record_id in shared_ids if str(low_records[record_id].get("modality", "")).lower() in wanted]
    if args.limit is not None:
        shared_ids = shared_ids[: args.limit]

    out_dir = Path(args.out_dir)
    image_dir = out_dir / f"{args.upscale_method}_x{args.scale}"
    rows = []
    for record_id in shared_ids:
        low = low_records[record_id]
        high = high_records[record_id]
        suffix = Path(low["image_path"]).suffix.lower()
        out_image = image_dir / f"{record_id}_{args.upscale_method}_x{args.scale}{suffix if suffix else '.png'}"
        upscale_image(low["image_path"], out_image, args.scale, args.upscale_method, args.pre_blur_radius)
        record = recovered_record(low, high, out_image, args.scale)
        rows.append(evaluate_record(record, out_dir, args.method, args.model_path, args.probability_threshold, args.trace_method))

    metrics = summarize(rows)
    report = {
        "lowres_manifest": args.lowres_manifest,
        "highres_manifest": args.highres_manifest,
        "method": args.method,
        "model_path": args.model_path,
        "scale": args.scale,
        "upscale_method": args.upscale_method,
        "pre_blur_radius": args.pre_blur_radius,
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
