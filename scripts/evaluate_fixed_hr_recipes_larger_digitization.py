from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy import signal as scipy_signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.digitize_tools import ML_MODEL_PATH, Signal_digitize_waveform_image_ml


RECIPES: dict[str, dict[str, Any]] = {
    "pcg": {"candidate": "x4_momentum", "distance_s": 0.2, "prominence_scale": 0.5, "band": (20, 150)},
    "eeg": {"candidate": "lowres_direct", "distance_s": 0.2, "prominence_scale": 0.15, "band": (0.3, 8)},
    "eda": {"candidate": "lowres_direct", "distance_s": 0.18, "prominence_scale": 0.5, "band": None},
    "bcg": {"candidate": "lowres_direct", "distance_s": 0.25, "prominence_scale": 0.35, "band": (20, 150)},
    "emg": {"candidate": "x4_path", "distance_s": 0.18, "prominence_scale": 0.35, "band": None},
    "scg": {"candidate": "x4_median", "distance_s": 0.25, "prominence_scale": 0.35, "band": (1, 35)},
}

TRACE_BY_CANDIDATE = {
    "lowres_direct": "median",
    "x4_median": "median",
    "x4_path": "path",
    "x4_momentum": "momentum",
}


def load_signal(path: str | Path) -> np.ndarray:
    frame = pd.read_csv(path)
    col = "signal" if "signal" in frame.columns else frame.select_dtypes("number").columns[-1]
    values = frame[col].to_numpy(dtype=float)
    return values[np.isfinite(values)]


def peaks(values: np.ndarray, fs: float, distance_s: float, prominence_scale: float, band: tuple[float, float] | None) -> np.ndarray:
    z = values - np.nanmedian(values)
    if band is not None:
        low, high = band
        high = min(float(high), float(fs) * 0.45)
        if low < high and len(z) > 12:
            sos = scipy_signal.butter(2, [float(low), high], btype="bandpass", fs=float(fs), output="sos")
            try:
                z = scipy_signal.sosfiltfilt(sos, z)
            except ValueError:
                z = scipy_signal.sosfilt(sos, z)
    prom = max(float(np.nanstd(z)) * float(prominence_scale), 1e-8)
    p, _ = scipy_signal.find_peaks(z, distance=max(1, int(float(fs) * float(distance_s))), prominence=prom)
    return p.astype(int)


def hr_from_peaks(p: np.ndarray, fs: float) -> float | None:
    if len(p) < 2:
        return None
    intervals = np.diff(p) / float(fs)
    intervals = intervals[intervals > 0]
    if len(intervals) == 0:
        return None
    return float(60.0 / np.median(intervals))


def reference_hr(record: dict[str, Any]) -> float | None:
    fs = float(record["sampling_rate"])
    ref = load_signal(record["reference_path"])
    return hr_from_peaks(peaks(ref, fs, 0.25, 0.5, None), fs)


def upscale_image(src: str | Path, dst: Path, scale: int) -> None:
    image = Image.open(src).convert("RGB")
    width, height = image.size
    dst.parent.mkdir(parents=True, exist_ok=True)
    image.resize((width * scale, height * scale), Image.Resampling.LANCZOS).save(dst)


def digitize(record: dict[str, Any], out_dir: Path, candidate: str, model_path: str, threshold: float, scale: int) -> tuple[str | None, float, str | None]:
    trace_method = TRACE_BY_CANDIDATE[candidate]
    img = Path(record["image_path"])
    image_path = img
    crop_scale = 1
    fs_scale = 1
    if candidate.startswith("x4_"):
        crop_scale = scale
        fs_scale = scale
        image_path = out_dir / "upscaled" / f"{record['record']}_lanczos_x{scale}{img.suffix or '.png'}"
        if not image_path.exists():
            upscale_image(img, image_path, scale)
    out_csv = out_dir / "csv" / candidate / f"{record['record']}_digitized.csv"
    res = Signal_digitize_waveform_image_ml(
        image_path=str(image_path),
        sampling_rate=float(record["sampling_rate"]) * fs_scale,
        out_csv=str(out_csv),
        value_min=float(record["value_min"]),
        value_max=float(record["value_max"]),
        crop_left=int(record["crop_left"]) * crop_scale,
        crop_right=int(record["crop_right"]) * crop_scale,
        crop_top=int(record["crop_top"]) * crop_scale,
        crop_bottom=int(record["crop_bottom"]) * crop_scale,
        model_path=model_path,
        probability_threshold=threshold,
        smooth_window=1,
        trace_method=trace_method,
    )
    if res.get("error"):
        return None, float(record["sampling_rate"]) * fs_scale, str(res.get("error"))
    return str(res["out_csv"]), float(record["sampling_rate"]) * fs_scale, None


def mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if not r.get("digitizer_error")]
    by_modality: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ok:
        grouped[str(row["modality"])].append(row)
    for modality, subset in sorted(grouped.items()):
        by_modality[modality] = {
            "num_records": len(subset),
            "baseline_hr_mae_bpm": mean_metric(subset, "baseline_hr_abs_error_bpm"),
            "fixed_recipe_hr_mae_bpm": mean_metric(subset, "fixed_recipe_hr_abs_error_bpm"),
            "improvement_bpm": None if mean_metric(subset, "baseline_hr_abs_error_bpm") is None or mean_metric(subset, "fixed_recipe_hr_abs_error_bpm") is None else mean_metric(subset, "baseline_hr_abs_error_bpm") - mean_metric(subset, "fixed_recipe_hr_abs_error_bpm"),
            "recipe_counts": dict(Counter(r.get("recipe_candidate") for r in subset)),
        }
    return {
        "num_records": len(rows),
        "num_ok": len(ok),
        "baseline_hr_mae_bpm": mean_metric(ok, "baseline_hr_abs_error_bpm"),
        "fixed_recipe_hr_mae_bpm": mean_metric(ok, "fixed_recipe_hr_abs_error_bpm"),
        "failure_counts": dict(Counter(r.get("digitizer_error") for r in rows if r.get("digitizer_error"))),
        "by_modality": by_modality,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply fixed modality-specific HR detector recipes to a larger digitization benchmark.")
    ap.add_argument("--manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark_more_10s_manifest.json")
    ap.add_argument("--out-dir", default="/data1/jiahui/biosignal-agent/outputs/fixed_hr_recipes_more_10s")
    ap.add_argument("--out-json", default="/data1/jiahui/biosignal-agent/outputs/fixed_hr_recipes_more_10s_eval.json")
    ap.add_argument("--out-csv", default="/data1/jiahui/biosignal-agent/outputs/fixed_hr_recipes_more_10s_eval.csv")
    ap.add_argument("--model-path", default=str(ML_MODEL_PATH))
    ap.add_argument("--probability-threshold", type=float, default=0.5)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--include-modality", action="append", default=None)
    args = ap.parse_args()

    records = json.loads(Path(args.manifest).read_text()).get("records", [])
    wanted = {m.lower() for m in args.include_modality} if args.include_modality else set(RECIPES)
    records = [r for r in records if str(r.get("modality", "")).lower() in wanted]
    out_dir = Path(args.out_dir)
    rows = []
    cache: dict[tuple[str, str], tuple[str | None, float, str | None]] = {}
    for record in records:
        modality = str(record["modality"]).lower()
        recipe = RECIPES[modality]
        ref_hr = reference_hr(record)
        baseline_key = (record["record"], "lowres_direct")
        if baseline_key not in cache:
            cache[baseline_key] = digitize(record, out_dir, "lowres_direct", args.model_path, args.probability_threshold, args.scale)
        baseline_csv, baseline_fs, baseline_err = cache[baseline_key]
        candidate = str(recipe["candidate"])
        recipe_key = (record["record"], candidate)
        if recipe_key not in cache:
            cache[recipe_key] = digitize(record, out_dir, candidate, args.model_path, args.probability_threshold, args.scale)
        recipe_csv, recipe_fs, recipe_err = cache[recipe_key]
        row: dict[str, Any] = {
            "record": record["record"],
            "modality": modality,
            "variant": record.get("variant"),
            "reference_hr_bpm": ref_hr,
            "recipe_candidate": candidate,
            "recipe_distance_s": recipe["distance_s"],
            "recipe_prominence_scale": recipe["prominence_scale"],
            "recipe_band": recipe["band"],
            "baseline_csv": baseline_csv,
            "recipe_csv": recipe_csv,
            "digitizer_error": baseline_err or recipe_err,
        }
        if row["digitizer_error"]:
            rows.append(row)
            continue
        baseline_hr = hr_from_peaks(peaks(load_signal(baseline_csv), baseline_fs, 0.25, 0.5, None), baseline_fs) if baseline_csv else None
        recipe_hr = hr_from_peaks(peaks(load_signal(recipe_csv), recipe_fs, float(recipe["distance_s"]), float(recipe["prominence_scale"]), recipe["band"]), recipe_fs) if recipe_csv else None
        row.update({
            "baseline_hr_bpm": baseline_hr,
            "fixed_recipe_hr_bpm": recipe_hr,
            "baseline_hr_abs_error_bpm": abs(ref_hr - baseline_hr) if ref_hr is not None and baseline_hr is not None else None,
            "fixed_recipe_hr_abs_error_bpm": abs(ref_hr - recipe_hr) if ref_hr is not None and recipe_hr is not None else None,
        })
        rows.append(row)

    report = {"manifest": args.manifest, "recipes": RECIPES, "metrics": summarize(rows), "rows": rows}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    with Path(args.out_csv).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"out_json": args.out_json, "out_csv": args.out_csv, "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
