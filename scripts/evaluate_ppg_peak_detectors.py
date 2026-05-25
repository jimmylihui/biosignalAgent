from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import neurokit2 as nk
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.peak_detectors import neurokit_nabian2018_peaks, ppg_multiscale_systolic_peaks  # noqa: E402


def match_ppg_to_ecg(
    ecg_peaks: np.ndarray,
    ppg_peaks: np.ndarray,
    fs: float,
    min_delay_s: float = 0.08,
    max_delay_s: float = 0.60,
    lag_s: float = 0.0,
) -> dict[str, Any]:
    shifted_ppg = np.asarray(ppg_peaks, dtype=int) + int(round(lag_s * fs))
    if len(ecg_peaks) == 0 or len(shifted_ppg) == 0:
        return {
            "matched": 0,
            "reference": int(len(ecg_peaks)),
            "detected": int(len(ppg_peaks)),
            "sensitivity": 0.0,
            "ppv": 0.0,
            "f1": 0.0,
            "median_delay_s": None,
            "applied_lag_s": float(lag_s),
        }
    used = np.zeros(len(shifted_ppg), dtype=bool)
    delays = []
    for r_peak in ecg_peaks:
        lo = r_peak + int(min_delay_s * fs)
        hi = r_peak + int(max_delay_s * fs)
        candidates = np.where((shifted_ppg >= lo) & (shifted_ppg <= hi) & (~used))[0]
        if len(candidates) == 0:
            continue
        idx = candidates[np.argmin(shifted_ppg[candidates] - r_peak)]
        used[idx] = True
        delays.append((shifted_ppg[idx] - r_peak) / fs)
    matched = len(delays)
    sensitivity = matched / len(ecg_peaks) if len(ecg_peaks) else 0.0
    ppv = matched / len(ppg_peaks) if len(ppg_peaks) else 0.0
    f1 = 2 * sensitivity * ppv / (sensitivity + ppv) if sensitivity + ppv else 0.0
    return {
        "matched": int(matched),
        "reference": int(len(ecg_peaks)),
        "detected": int(len(ppg_peaks)),
        "sensitivity": float(sensitivity),
        "ppv": float(ppv),
        "f1": float(f1),
        "median_delay_s": float(np.median(delays)) if delays else None,
        "applied_lag_s": float(lag_s),
    }


def estimate_best_lag_match(
    ecg_peaks: np.ndarray,
    ppg_peaks: np.ndarray,
    fs: float,
    search_min_s: float = -0.75,
    search_max_s: float = 0.75,
    step_s: float | None = None,
) -> dict[str, Any]:
    step_s = step_s or (1.0 / fs)
    best = None
    for lag_s in np.arange(search_min_s, search_max_s + step_s / 2.0, step_s):
        current = match_ppg_to_ecg(ecg_peaks, ppg_peaks, fs, lag_s=float(lag_s))
        if best is None or current["f1"] > best["f1"]:
            best = current
    return best or match_ppg_to_ecg(ecg_peaks, ppg_peaks, fs)


def record_paths(raw_dir: Path) -> list[Path]:
    return sorted(raw_dir.rglob("*_data.csv"))


def evaluate_record(path: Path, seconds: float) -> dict[str, Any] | None:
    frame = pd.read_csv(path)
    if "PPG" not in frame.columns or "ECG" not in frame.columns:
        return None
    fs = 1.0 / float(np.median(np.diff(frame["Time"].to_numpy(float)))) if "Time" in frame.columns else 125.0
    n = int(seconds * fs)
    ppg = frame["PPG"].to_numpy(float)[:n]
    ecg = frame["ECG"].to_numpy(float)[:n]
    if len(ppg) < fs * 20 or len(ecg) < fs * 20:
        return None
    try:
        _, info = nk.ecg_peaks(ecg, sampling_rate=fs, method="nabian2018", correct_artifacts=True)
        ecg_peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
    except Exception:
        return None
    rows = {"record": path.stem.replace("_data", ""), "sampling_rate": fs, "ecg_peaks": int(len(ecg_peaks))}
    detectors = {
        "nabian_on_ppg": lambda x: neurokit_nabian2018_peaks(x, fs, low_hz=0.4, high_hz=min(8.0, fs * 0.45), fallback_threshold_scale=0.35)[0],
        "ppg_multiscale": lambda x: ppg_multiscale_systolic_peaks(x, fs)[0],
    }
    for name, fn in detectors.items():
        peaks = fn(ppg)
        fixed = match_ppg_to_ecg(ecg_peaks, peaks, fs)
        lag_corrected = estimate_best_lag_match(ecg_peaks, peaks, fs)
        rows[name] = {
            "fixed_ptt_window": fixed,
            "lag_corrected": lag_corrected,
            "estimated_channel_lag_s": lag_corrected["applied_lag_s"],
        }
    return rows


def _summarize_metric(vals: list[dict[str, Any]]) -> dict[str, Any]:
    delays = [v["median_delay_s"] for v in vals if v.get("median_delay_s") is not None]
    return {
        "mean_sensitivity": float(np.mean([v["sensitivity"] for v in vals])),
        "mean_ppv": float(np.mean([v["ppv"] for v in vals])),
        "mean_f1": float(np.mean([v["f1"] for v in vals])),
        "median_delay_s": float(np.nanmedian(delays)) if delays else None,
        "total_matched": int(sum(v["matched"] for v in vals)),
        "total_reference": int(sum(v["reference"] for v in vals)),
        "total_detected": int(sum(v["detected"] for v in vals)),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    detector_names = [key for key in rows[0] if isinstance(rows[0].get(key), dict)] if rows else []
    summary = {"num_records": len(rows), "reference": "ECG-derived R peaks; lag-corrected metric is preferred when no manual PPG beat labels are available.", "detectors": {}}
    for name in detector_names:
        fixed_vals = [row[name]["fixed_ptt_window"] for row in rows]
        lag_vals = [row[name]["lag_corrected"] for row in rows]
        lags = [row[name]["estimated_channel_lag_s"] for row in rows]
        summary["detectors"][name] = {
            "fixed_ptt_window": _summarize_metric(fixed_vals),
            "lag_corrected": _summarize_metric(lag_vals),
            "estimated_channel_lag_s_median": float(np.median(lags)) if lags else None,
            "estimated_channel_lag_s_iqr": [float(np.percentile(lags, 25)), float(np.percentile(lags, 75))] if lags else None,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPG peak detectors against ECG-derived references with both fixed and lag-corrected timing windows.")
    parser.add_argument("--raw-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/raw/mimic_perform_af"))
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--out-json", type=Path, default=Path("/data1/jiahui/biosignal-agent/outputs/ppg_peak_detector_eval.json"))
    args = parser.parse_args()
    rows = [row for path in record_paths(args.raw_dir) if (row := evaluate_record(path, args.seconds)) is not None]
    report = summarize(rows)
    report["rows"] = rows
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))
    print(json.dumps({"num_records": report["num_records"], "detectors": report["detectors"]}, indent=2))


if __name__ == "__main__":
    main()
