from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

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



def PPG_screen_pulse_irregularity(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    peaks_result = PPG_detect_peaks(signal_path, sampling_rate, column)
    peaks = np.asarray(peaks_result.get("peak_indices", []), dtype=float)
    if len(peaks) < 6:
        return {"tool": "PPG_screen_pulse_irregularity", "error": "not enough PPG peaks", "confidence": 0.1}
    intervals_s = np.diff(peaks) / float(data.sampling_rate)
    intervals_s = intervals_s[(intervals_s >= 0.25) & (intervals_s <= 3.0)]
    if len(intervals_s) < 5:
        return {"tool": "PPG_screen_pulse_irregularity", "error": "not enough valid pulse intervals", "confidence": 0.1}
    mean_interval = float(np.mean(intervals_s))
    pulse_interval_cv = float(np.std(intervals_s) / mean_interval) if mean_interval > 0 else None
    rmssd_s = float(np.sqrt(np.mean(np.diff(intervals_s) ** 2))) if len(intervals_s) > 1 else None
    normalized_rmssd = float(rmssd_s / mean_interval) if rmssd_s is not None and mean_interval > 0 else None
    successive_change_fraction = float(np.mean(np.abs(np.diff(intervals_s)) > 0.12)) if len(intervals_s) > 1 else 0.0
    score = 0
    flags = []
    if pulse_interval_cv is not None and pulse_interval_cv > 0.16:
        score += 1
        flags.append("high_pulse_interval_cv")
    if normalized_rmssd is not None and normalized_rmssd > 0.18:
        score += 1
        flags.append("high_pulse_interval_rmssd")
    if successive_change_fraction > 0.25:
        score += 1
        flags.append("frequent_successive_pulse_changes")
    risk = "elevated_irregular_pulse_proxy" if score >= 2 else "low_irregular_pulse_proxy"
    return {
        "tool": "PPG_screen_pulse_irregularity",
        "heart_rate_bpm": peaks_result.get("heart_rate_bpm"),
        "pulse_interval_cv": pulse_interval_cv,
        "normalized_rmssd": normalized_rmssd,
        "successive_change_fraction": successive_change_fraction,
        "irregular_pulse_score": score,
        "irregular_pulse_flags": flags,
        "irregular_pulse_risk": risk,
        "confidence": min(float(peaks_result.get("confidence", 0.5)), 0.62),
        "method": "ppg_pulse_interval_irregularity_screening",
        "disclaimer": "PPG irregular-pulse proxy only; AF screening requires ECG reference labels and artifact-aware validation.",
    }


def PPG_estimate_respiration_modulation(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    if len(values) < data.sampling_rate * 20:
        return {"tool": "PPG_estimate_respiration_modulation", "error": "signal too short", "confidence": 0.0}
    peaks_result = PPG_detect_peaks(signal_path, sampling_rate, column)
    peaks = np.asarray(peaks_result.get("peak_indices", []), dtype=int)
    if len(peaks) < 5:
        return {"tool": "PPG_estimate_respiration_modulation", "error": "not enough PPG peaks", "confidence": 0.1}
    envelope = np.abs(scipy_signal.hilbert(values - np.nanmedian(values)))
    high = min(0.7, data.sampling_rate * 0.45)
    if high <= 0.08:
        return {"tool": "PPG_estimate_respiration_modulation", "error": "sampling rate too low", "confidence": 0.1}
    resp_band = scipy_signal.sosfiltfilt(scipy_signal.butter(3, [0.08 / (0.5 * data.sampling_rate), high / (0.5 * data.sampling_rate)], btype="bandpass", output="sos"), envelope)
    freqs, psd = scipy_signal.welch(resp_band, fs=data.sampling_rate, nperseg=min(len(resp_band), int(data.sampling_rate * 16)))
    mask = (freqs >= 0.08) & (freqs <= high)
    resp_rate = None
    if np.any(mask):
        resp_rate = float(freqs[mask][np.argmax(psd[mask])] * 60.0)
    modulation_index = float(np.nanstd(resp_band) / (np.nanstd(values) + 1e-8))
    return {
        "tool": "PPG_estimate_respiration_modulation",
        "respiratory_rate_bpm": resp_rate,
        "respiratory_modulation_index": modulation_index,
        "heart_rate_bpm": peaks_result.get("heart_rate_bpm"),
        "confidence": 0.55 if resp_rate is not None and 5 <= resp_rate <= 40 else 0.5,
        "method": "ppg_envelope_respiration_bandpower_proxy",
        "disclaimer": "PPG-derived respiration proxy only; validate against respiratory reference signals.",
    }
