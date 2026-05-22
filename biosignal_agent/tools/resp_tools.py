from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from .common import bandpass_filter, load_csv_signal, signal_quality_summary


def RESP_assess_quality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    return {"tool": "RESP_assess_quality", "source": data.source, **signal_quality_summary(data.values)}


def RESP_estimate_rate(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    if len(values) < data.sampling_rate * 10:
        return {"tool": "RESP_estimate_rate", "error": "signal too short", "confidence": 0.0}
    filtered = bandpass_filter(values, data.sampling_rate, low_hz=0.05, high_hz=0.7, order=3)
    min_distance = max(1, int(1.5 * data.sampling_rate))
    prominence = max(float(np.nanstd(filtered)) * 0.2, 1e-8)
    peaks, _ = scipy_signal.find_peaks(filtered, distance=min_distance, prominence=prominence)
    if len(peaks) < 2:
        peaks, _ = scipy_signal.find_peaks(-filtered, distance=min_distance, prominence=prominence)
    intervals = np.diff(peaks) / data.sampling_rate if len(peaks) >= 2 else np.array([])
    intervals = intervals[(intervals >= 1.5) & (intervals <= 12.0)]
    respiratory_rate = float(60.0 / np.median(intervals)) if len(intervals) else None
    confidence = 0.75 if respiratory_rate is not None and 5 <= respiratory_rate <= 40 else 0.3
    return {
        "tool": "RESP_estimate_rate",
        "breath_indices": peaks.tolist(),
        "num_breaths": int(len(peaks)),
        "respiratory_rate_bpm": respiratory_rate,
        "confidence": confidence,
        "method": "bandpass_find_peaks",
    }
