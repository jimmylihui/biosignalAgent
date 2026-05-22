from __future__ import annotations

import numpy as np

from .common import bpm_from_peaks, interval_regularity, load_csv_signal, signal_quality_summary
from .peak_detectors import neurokit_nabian2018_peaks


def PPG_assess_quality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    return {"tool": "PPG_assess_quality", "source": data.source, **signal_quality_summary(data.values)}


def PPG_detect_peaks(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    peaks, details = neurokit_nabian2018_peaks(data.values, data.sampling_rate, low_hz=0.4, high_hz=8.0, fallback_threshold_scale=0.35)
    heart_rate = bpm_from_peaks(peaks, data.sampling_rate)
    regularity = interval_regularity(peaks, data.sampling_rate)
    base_confidence = 0.72 if details["method"] == "nabian2018" else 0.6
    confidence = min(base_confidence, regularity["regularity_confidence"])
    if heart_rate is None or not 35 <= heart_rate <= 220:
        confidence = 0.3
    return {"tool": "PPG_detect_peaks", "peak_indices": peaks.tolist(), "num_peaks": int(len(peaks)), "heart_rate_bpm": heart_rate, "confidence": confidence, **regularity, **details}



def PPG_assess_perfusion_variability(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    if len(values) == 0:
        return {"tool": "PPG_assess_perfusion_variability", "error": "empty signal", "confidence": 0.0}
    peaks_result = PPG_detect_peaks(signal_path, sampling_rate, column)
    peaks = np.asarray(peaks_result.get("peak_indices", []), dtype=int)
    finite = values[np.isfinite(values)]
    dynamic_range = float(np.nanpercentile(finite, 95) - np.nanpercentile(finite, 5)) if len(finite) else 0.0
    median_abs = float(np.nanmedian(np.abs(finite))) if len(finite) else 0.0
    amplitude_proxy = float(dynamic_range / (median_abs + 1e-8))
    if len(peaks) >= 3:
        intervals = np.diff(peaks) / float(data.sampling_rate)
        pulse_interval_cv = float(np.std(intervals) / np.mean(intervals)) if np.mean(intervals) > 0 else None
    else:
        pulse_interval_cv = None
    perfusion_level = "low_perfusion_proxy" if amplitude_proxy < 0.05 or dynamic_range < 1e-6 else "adequate_perfusion_proxy"
    variability_risk = "high_pulse_variability_proxy" if pulse_interval_cv is not None and pulse_interval_cv > 0.2 else "no_high_variability_proxy"
    return {
        "tool": "PPG_assess_perfusion_variability",
        "pulse_amplitude_proxy": amplitude_proxy,
        "dynamic_range": dynamic_range,
        "pulse_interval_cv": pulse_interval_cv,
        "heart_rate_bpm": peaks_result.get("heart_rate_bpm"),
        "perfusion_level": perfusion_level,
        "pulse_variability_risk": variability_risk,
        "confidence": max(0.5, min(0.65, float(peaks_result.get("confidence", 0.5)))),
        "method": "ppg_peak_amplitude_interval_variability_screening",
        "disclaimer": "Screening heuristic only; low perfusion and vascular interpretations require calibrated PPG and context.",
    }
