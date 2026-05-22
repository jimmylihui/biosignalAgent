from __future__ import annotations

from .common import bpm_from_peaks, load_csv_signal, signal_quality_summary
from .peak_detectors import neurokit_nabian2018_peaks


def SCG_assess_quality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    return {"tool": "SCG_assess_quality", "source": data.source, **signal_quality_summary(data.values)}


def SCG_detect_j_peaks(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    peaks, details = neurokit_nabian2018_peaks(data.values, data.sampling_rate, low_hz=0.8, high_hz=20.0, fallback_threshold_scale=0.30)
    heart_rate = bpm_from_peaks(peaks, data.sampling_rate)
    confidence = 0.58 if details["method"] == "nabian2018" else 0.45
    if heart_rate is None or not 35 <= heart_rate <= 220:
        confidence = 0.25
    return {"tool": "SCG_detect_j_peaks", "j_peak_indices": peaks.tolist(), "num_peaks": int(len(peaks)), "heart_rate_bpm": heart_rate, "confidence": confidence, **details}
