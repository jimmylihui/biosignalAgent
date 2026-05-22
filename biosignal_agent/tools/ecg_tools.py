from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from .common import bandpass_filter, bpm_from_peaks, load_csv_signal, signal_quality_summary


def ECG_assess_quality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    return {"tool": "ECG_assess_quality", "source": data.source, **signal_quality_summary(data.values)}


def ECG_detect_r_peaks(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    filtered = bandpass_filter(data.values, data.sampling_rate, low_hz=0.5, high_hz=40.0)
    distance = max(1, int(0.25 * data.sampling_rate))
    prominence = max(float(np.std(filtered)) * 0.6, 1e-8)
    peaks, properties = scipy_signal.find_peaks(filtered, distance=distance, prominence=prominence)
    heart_rate = bpm_from_peaks(peaks, data.sampling_rate)
    return {"tool": "ECG_detect_r_peaks", "r_peak_indices": peaks.tolist(), "num_peaks": int(len(peaks)), "heart_rate_bpm": heart_rate, "median_prominence": float(np.median(properties.get("prominences", [0]))), "confidence": 0.75 if heart_rate is not None and 35 <= heart_rate <= 220 else 0.35}


def ECG_compute_hrv(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    peak_result = ECG_detect_r_peaks(signal_path, sampling_rate, column)
    peaks = np.asarray(peak_result["r_peak_indices"], dtype=float)
    if len(peaks) < 3:
        return {"tool": "ECG_compute_hrv", "error": "not enough R peaks", "confidence": 0.1}
    rr_ms = np.diff(peaks) / float(sampling_rate) * 1000.0
    rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))) if len(rr_ms) > 1 else None
    return {"tool": "ECG_compute_hrv", "mean_rr_ms": float(np.mean(rr_ms)), "sdnn_ms": float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else 0.0, "rmssd_ms": rmssd, "confidence": peak_result["confidence"]}
