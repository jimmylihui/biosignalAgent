from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from .common import load_csv_signal, signal_quality_summary


def EDA_assess_quality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    return {"tool": "EDA_assess_quality", "source": data.source, **signal_quality_summary(data.values)}


def EDA_summarize(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    if len(values) == 0:
        return {"tool": "EDA_summarize", "error": "empty signal", "confidence": 0.0}
    smoothed = scipy_signal.medfilt(values, kernel_size=max(3, int(data.sampling_rate) // 2 * 2 + 1)) if len(values) > 5 else values
    phasic = values - smoothed
    return {"tool": "EDA_summarize", "mean_level": float(np.nanmean(values)), "tonic_median": float(np.nanmedian(smoothed)), "phasic_std": float(np.nanstd(phasic)), "num_samples": int(len(values)), "confidence": 0.65, "method": "median_filter_tonic_phasic_summary"}



def EDA_detect_arousal_events(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    if len(values) < max(5, int(data.sampling_rate * 5)):
        return {"tool": "EDA_detect_arousal_events", "error": "signal too short", "confidence": 0.0}
    kernel_size = max(3, int(data.sampling_rate) // 2 * 2 + 1)
    smoothed = scipy_signal.medfilt(values, kernel_size=kernel_size) if len(values) > kernel_size else values
    phasic = values - smoothed
    distance = max(1, int(1.0 * data.sampling_rate))
    prominence = max(float(np.nanstd(phasic)) * 0.75, 1e-8)
    peaks, _ = scipy_signal.find_peaks(phasic, distance=distance, prominence=prominence)
    duration_min = len(values) / float(data.sampling_rate) / 60.0 if data.sampling_rate else 0.0
    rate = float(len(peaks) / duration_min) if duration_min > 0 else None
    arousal_level = "high_arousal_proxy" if rate is not None and rate > 6 else "low_arousal_proxy"
    return {
        "tool": "EDA_detect_arousal_events",
        "arousal_event_indices": peaks.tolist(),
        "arousal_event_count": int(len(peaks)),
        "arousal_rate_per_min": rate,
        "arousal_level": arousal_level,
        "confidence": 0.6,
        "method": "eda_phasic_peak_screening",
        "disclaimer": "Screening heuristic only; EDA arousal events are not equivalent to emotion or stress labels.",
    }


def EDA_screen_stress_proxy(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    summary = EDA_summarize(signal_path, sampling_rate, column)
    events = EDA_detect_arousal_events(signal_path, sampling_rate, column)
    if summary.get("error"):
        return {"tool": "EDA_screen_stress_proxy", "error": summary["error"], "confidence": 0.1}
    phasic_std = float(summary.get("phasic_std") or 0.0)
    mean_level = abs(float(summary.get("mean_level") or 0.0))
    normalized_phasic = float(phasic_std / (mean_level + 1e-8))
    arousal_rate = events.get("arousal_rate_per_min") if not events.get("error") else None
    score = 0
    flags = []
    if arousal_rate is not None and arousal_rate > 6.0:
        score += 1
        flags.append("high_scr_rate")
    if normalized_phasic > 0.08:
        score += 1
        flags.append("high_phasic_variability")
    if float(summary.get("tonic_median") or 0.0) > float(summary.get("mean_level") or 0.0) * 1.2:
        score += 1
        flags.append("elevated_tonic_level_proxy")
    stress_level = "elevated_stress_arousal_proxy" if score >= 2 else "low_stress_arousal_proxy"
    return {
        "tool": "EDA_screen_stress_proxy",
        "stress_arousal_score": score,
        "stress_arousal_level": stress_level,
        "stress_arousal_flags": flags,
        "arousal_rate_per_min": arousal_rate,
        "normalized_phasic_std": normalized_phasic,
        "mean_level": summary.get("mean_level"),
        "tonic_median": summary.get("tonic_median"),
        "confidence": 0.55,
        "method": "eda_tonic_phasic_arousal_score_baseline",
        "disclaimer": "Stress/arousal proxy only; validated stress classification requires labeled protocol data and multimodal context.",
    }
