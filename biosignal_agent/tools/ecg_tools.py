from __future__ import annotations

import numpy as np

from .common import bpm_from_peaks, load_csv_signal, signal_quality_summary
from .peak_detectors import neurokit_nabian2018_peaks

try:
    import neurokit2 as nk
except ImportError:  # pragma: no cover
    nk = None


def ECG_assess_quality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    return {"tool": "ECG_assess_quality", "source": data.source, **signal_quality_summary(data.values)}


def ECG_detect_r_peaks(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    try:
        if nk is None:
            raise RuntimeError("neurokit2 is not installed")
        cleaned = nk.ecg_clean(values, sampling_rate=data.sampling_rate, method="pantompkins1985")
        _, info = nk.ecg_peaks(cleaned, sampling_rate=data.sampling_rate, method="pantompkins1985", correct_artifacts=True)
        peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
        details = {"method": "pantompkins1985", "median_prominence": None}
    except Exception as exc:
        peaks, details = neurokit_nabian2018_peaks(values, data.sampling_rate, low_hz=None, high_hz=None, fallback_threshold_scale=0.6)
        details["fallback_reason"] = str(exc)
    heart_rate = bpm_from_peaks(peaks, data.sampling_rate)
    confidence = 0.82 if details["method"] == "pantompkins1985" else 0.65
    if heart_rate is None or not 35 <= heart_rate <= 220:
        confidence = 0.3
    return {"tool": "ECG_detect_r_peaks", "r_peak_indices": peaks.tolist(), "num_peaks": int(len(peaks)), "heart_rate_bpm": heart_rate, "confidence": confidence, **details}


def ECG_compute_hrv(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    peak_result = ECG_detect_r_peaks(signal_path, sampling_rate, column)
    peaks = np.asarray(peak_result["r_peak_indices"], dtype=float)
    if len(peaks) < 3:
        return {"tool": "ECG_compute_hrv", "error": "not enough R peaks", "confidence": 0.1}
    rr_ms = np.diff(peaks) / float(sampling_rate) * 1000.0
    rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))) if len(rr_ms) > 1 else None
    return {"tool": "ECG_compute_hrv", "mean_rr_ms": float(np.mean(rr_ms)), "sdnn_ms": float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else 0.0, "rmssd_ms": rmssd, "confidence": peak_result["confidence"]}



def ECG_screen_arrhythmia(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    peak_result = ECG_detect_r_peaks(signal_path, sampling_rate, column)
    peaks = np.asarray(peak_result.get("r_peak_indices", []), dtype=float)
    if len(peaks) < 4:
        return {"tool": "ECG_screen_arrhythmia", "error": "not enough R peaks", "confidence": 0.1}
    rr_s = np.diff(peaks) / float(sampling_rate)
    rr_s = rr_s[(rr_s >= 0.25) & (rr_s <= 3.0)]
    if len(rr_s) < 3:
        return {"tool": "ECG_screen_arrhythmia", "error": "not enough valid RR intervals", "confidence": 0.1}
    heart_rate = float(60.0 / np.median(rr_s))
    rr_cv = float(np.std(rr_s) / np.mean(rr_s)) if np.mean(rr_s) > 0 else None
    pause_count = int(np.sum(rr_s > 2.0))
    short_long = np.abs(np.diff(rr_s)) / rr_s[:-1] if len(rr_s) > 1 else np.array([])
    ectopy_proxy_fraction = float(np.mean(short_long > 0.2)) if len(short_long) else 0.0
    flags = []
    if heart_rate < 50:
        flags.append("bradycardia_pattern")
    if heart_rate > 110:
        flags.append("tachycardia_pattern")
    if rr_cv is not None and rr_cv > 0.18:
        flags.append("irregular_rr_pattern")
    if pause_count:
        flags.append("long_pause_pattern")
    if ectopy_proxy_fraction > 0.15:
        flags.append("ectopy_proxy_pattern")
    risk = "elevated" if flags else "low"
    confidence = min(float(peak_result.get("confidence", 0.5)), 0.7)
    return {
        "tool": "ECG_screen_arrhythmia",
        "heart_rate_bpm": heart_rate,
        "rr_cv": rr_cv,
        "pause_count": pause_count,
        "ectopy_proxy_fraction": ectopy_proxy_fraction,
        "arrhythmia_flags": flags,
        "arrhythmia_risk": risk,
        "confidence": confidence,
        "method": "rr_interval_screening",
        "disclaimer": "Screening heuristic only; not a diagnostic rhythm classifier.",
    }



def ECG_screen_sleep_apnea(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    hrv = ECG_compute_hrv(signal_path, sampling_rate, column)
    arrhythmia = ECG_screen_arrhythmia(signal_path, sampling_rate, column)
    if hrv.get("error"):
        return {"tool": "ECG_screen_sleep_apnea", "error": hrv["error"], "confidence": 0.1}
    mean_rr = float(hrv.get("mean_rr_ms") or 0.0)
    rmssd = float(hrv.get("rmssd_ms") or 0.0)
    sdnn = float(hrv.get("sdnn_ms") or 0.0)
    rr_cv = arrhythmia.get("rr_cv")
    heart_rate = arrhythmia.get("heart_rate_bpm")
    score = 0
    flags = []
    if heart_rate is not None and (heart_rate < 55 or heart_rate > 95):
        score += 1
        flags.append("sleep_epoch_heart_rate_extreme")
    if rr_cv is not None and rr_cv > 0.10:
        score += 1
        flags.append("elevated_rr_variability")
    if rmssd > 80 or sdnn > 90:
        score += 1
        flags.append("high_short_term_hrv")
    if mean_rr > 1200:
        score += 1
        flags.append("bradycardic_rr_pattern")
    apnea_risk = "elevated" if score >= 2 else "low"
    return {
        "tool": "ECG_screen_sleep_apnea",
        "apnea_risk": apnea_risk,
        "apnea_proxy_score": score,
        "apnea_proxy_flags": flags,
        "heart_rate_bpm": heart_rate,
        "mean_rr_ms": mean_rr,
        "sdnn_ms": sdnn,
        "rmssd_ms": rmssd,
        "rr_cv": rr_cv,
        "confidence": 0.5,
        "method": "ecg_hrv_sleep_apnea_proxy",
        "disclaimer": "ECG-only apnea proxy for benchmarking; respiratory effort and SpO2 labels are preferred for clinical apnea detection.",
    }
