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



def ABP_screen_pressure_events(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    pulses = ABP_detect_pulses(signal_path, sampling_rate, column)
    if pulses.get("error"):
        return {"tool": "ABP_screen_pressure_events", "error": pulses["error"], "confidence": 0.0}
    systolic = pulses.get("median_systolic_value")
    diastolic = pulses.get("approx_diastolic_value")
    flags = []
    if systolic is not None and systolic < 90:
        flags.append("low_systolic_proxy")
    if diastolic is not None and diastolic < 60:
        flags.append("low_diastolic_proxy")
    if systolic is not None and systolic >= 140:
        flags.append("high_systolic_proxy")
    if diastolic is not None and diastolic >= 90:
        flags.append("high_diastolic_proxy")
    if any(flag.startswith("low") for flag in flags):
        pressure_risk = "hypotension_proxy"
    elif any(flag.startswith("high") for flag in flags):
        pressure_risk = "hypertension_proxy"
    else:
        pressure_risk = "no_pressure_event_proxy"
    return {
        "tool": "ABP_screen_pressure_events",
        "median_systolic_value": systolic,
        "approx_diastolic_value": diastolic,
        "heart_rate_bpm": pulses.get("heart_rate_bpm"),
        "pressure_flags": flags,
        "pressure_risk": pressure_risk,
        "confidence": max(0.5, min(0.7, float(pulses.get("confidence", 0.5)))),
        "method": "abp_peak_percentile_threshold_screening",
        "disclaimer": "Screening heuristic only; ABP calibration and clinical context are required for blood-pressure interpretation.",
    }



def ABP_compute_hemodynamics(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    pulses = ABP_detect_pulses(signal_path, sampling_rate, column)
    if len(data.values) == 0:
        return {"tool": "ABP_compute_hemodynamics", "error": "empty signal", "confidence": 0.0}
    systolic = pulses.get("median_systolic_value")
    diastolic = pulses.get("approx_diastolic_value")
    mean_pressure = float(np.nanmean(data.values))
    pulse_pressure = float(systolic - diastolic) if systolic is not None and diastolic is not None else None
    map_formula = float(diastolic + (pulse_pressure / 3.0)) if pulse_pressure is not None and diastolic is not None else mean_pressure
    flags = []
    if map_formula < 65:
        flags.append("low_map_proxy")
    if pulse_pressure is not None and pulse_pressure < 25:
        flags.append("narrow_pulse_pressure_proxy")
    if pulse_pressure is not None and pulse_pressure > 80:
        flags.append("wide_pulse_pressure_proxy")
    hemodynamic_risk = "elevated" if flags else "low"
    return {
        "tool": "ABP_compute_hemodynamics",
        "mean_arterial_pressure_proxy": map_formula,
        "mean_pressure_value": mean_pressure,
        "pulse_pressure_proxy": pulse_pressure,
        "median_systolic_value": systolic,
        "approx_diastolic_value": diastolic,
        "heart_rate_bpm": pulses.get("heart_rate_bpm"),
        "hemodynamic_flags": flags,
        "hemodynamic_risk": hemodynamic_risk,
        "confidence": max(0.5, min(0.7, float(pulses.get("confidence", 0.5)))),
        "method": "abp_map_pulse_pressure_proxy",
        "disclaimer": "Screening heuristic only; ABP units and calibration are required for clinical hemodynamic interpretation.",
    }
