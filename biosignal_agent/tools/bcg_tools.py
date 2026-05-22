from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from .common import bandpass_filter, bpm_from_peaks, load_csv_signal, signal_quality_summary


def BCG_assess_quality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    return {"tool": "BCG_assess_quality", "source": data.source, **signal_quality_summary(data.values)}


def BCG_detect_j_peaks(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    filtered = bandpass_filter(data.values, data.sampling_rate, low_hz=0.8, high_hz=12.0)
    distance = max(1, int(0.3 * data.sampling_rate))
    prominence = max(float(np.std(filtered)) * 0.35, 1e-8)
    peaks, properties = scipy_signal.find_peaks(filtered, distance=distance, prominence=prominence)
    heart_rate = bpm_from_peaks(peaks, data.sampling_rate)
    return {"tool": "BCG_detect_j_peaks", "j_peak_indices": peaks.tolist(), "num_peaks": int(len(peaks)), "heart_rate_bpm": heart_rate, "median_prominence": float(np.median(properties.get("prominences", [0]))), "confidence": 0.55 if heart_rate is not None and 35 <= heart_rate <= 220 else 0.25}
