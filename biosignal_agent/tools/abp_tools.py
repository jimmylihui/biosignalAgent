from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from .common import bpm_from_peaks, interval_regularity, load_csv_signal, signal_quality_summary


def ABP_assess_quality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    return {"tool": "ABP_assess_quality", "source": data.source, **signal_quality_summary(data.values)}


def ABP_detect_pulses(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    min_distance = max(1, int(0.3 * data.sampling_rate))
    prominence = max(float(np.nanstd(values)) * 0.25, 1e-8)
    peaks, _ = scipy_signal.find_peaks(values, distance=min_distance, prominence=prominence)
    heart_rate = bpm_from_peaks(peaks, data.sampling_rate)
    regularity = interval_regularity(peaks, data.sampling_rate)
    confidence = min(0.75, regularity["regularity_confidence"]) if heart_rate is not None and 35 <= heart_rate <= 220 else 0.3
    systolic = float(np.nanmedian(values[peaks])) if len(peaks) else None
    diastolic = float(np.nanpercentile(values, 10)) if len(values) else None
    return {
        "tool": "ABP_detect_pulses",
        "pulse_indices": peaks.tolist(),
        "num_pulses": int(len(peaks)),
        "heart_rate_bpm": heart_rate,
        "median_systolic_value": systolic,
        "approx_diastolic_value": diastolic,
        "confidence": confidence,
        **regularity,
        "method": "find_peaks",
    }
