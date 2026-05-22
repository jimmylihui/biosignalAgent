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
