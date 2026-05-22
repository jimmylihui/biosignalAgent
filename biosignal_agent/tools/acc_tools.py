from __future__ import annotations

import numpy as np

from .common import load_csv_signal, signal_quality_summary


def ACC_assess_quality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    return {"tool": "ACC_assess_quality", "source": data.source, **signal_quality_summary(data.values)}


def ACC_summarize_activity(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    activity = float(np.nanstd(values)) if len(values) else 0.0
    dynamic_range = float(np.nanpercentile(values, 95) - np.nanpercentile(values, 5)) if len(values) else 0.0
    label = "high" if activity > 0.5 else "moderate" if activity > 0.05 else "low"
    return {"tool": "ACC_summarize_activity", "activity_std": activity, "dynamic_range": dynamic_range, "activity_level": label, "confidence": 0.65, "method": "std_threshold_summary"}



def ACC_estimate_sleep_wake(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    summary = ACC_summarize_activity(signal_path, sampling_rate, column)
    activity = float(summary.get("activity_std", 0.0))
    sleep_wake_hint = "wake_or_active" if activity > 0.08 else "sleep_or_rest"
    return {
        "tool": "ACC_estimate_sleep_wake",
        "activity_std": activity,
        "activity_level": summary.get("activity_level"),
        "sleep_wake_hint": sleep_wake_hint,
        "confidence": 0.5,
        "method": "activity_threshold_sleep_wake_proxy",
        "disclaimer": "Actigraphy proxy only; sleep staging requires labeled sleep data and multimodal context.",
    }



def ACC_detect_activity_bouts(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    if len(values) < max(4, int(data.sampling_rate * 5)):
        return {"tool": "ACC_detect_activity_bouts", "error": "signal too short", "confidence": 0.0}
    window = max(1, int(data.sampling_rate * 2.0))
    kernel = np.ones(window) / window
    centered = values - np.nanmedian(values)
    activity = np.sqrt(np.convolve(centered ** 2, kernel, mode="same"))
    threshold = max(float(np.nanmedian(activity) + 2.0 * np.nanstd(activity)), 1e-8)
    active = activity > threshold
    events = []
    start = None
    for idx, flag in enumerate(active):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            if idx - start >= max(1, int(data.sampling_rate)):
                events.append((start, idx))
            start = None
    if start is not None and len(active) - start >= max(1, int(data.sampling_rate)):
        events.append((start, len(active)))
    duration_min = len(values) / data.sampling_rate / 60.0
    return {
        "tool": "ACC_detect_activity_bouts",
        "activity_bout_count": int(len(events)),
        "activity_bout_rate_per_min": float(len(events) / duration_min) if duration_min > 0 else None,
        "activity_threshold": threshold,
        "event_intervals_s": [[float(a / data.sampling_rate), float(b / data.sampling_rate)] for a, b in events[:20]],
        "confidence": 0.6,
        "method": "accelerometer_rms_bout_detection",
        "disclaimer": "Activity bout proxy only; validated activity classification needs labeled posture/activity data.",
    }


def ACC_detect_fall_proxy(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    if len(values) < max(4, int(data.sampling_rate * 3)):
        return {"tool": "ACC_detect_fall_proxy", "error": "signal too short", "confidence": 0.0}
    centered = values - np.nanmedian(values)
    impact_threshold = max(float(np.nanpercentile(np.abs(centered), 99) * 0.9), float(np.nanstd(centered) * 4.0), 1e-8)
    impact_indices = np.flatnonzero(np.abs(centered) >= impact_threshold)
    refractory = max(1, int(data.sampling_rate))
    events = []
    last = -refractory
    for idx in impact_indices:
        if idx - last >= refractory:
            events.append(idx)
            last = idx
    risk = "possible_fall_or_impact_proxy" if events else "no_fall_proxy"
    return {
        "tool": "ACC_detect_fall_proxy",
        "impact_event_count": int(len(events)),
        "impact_indices": [int(x) for x in events[:20]],
        "impact_threshold": impact_threshold,
        "fall_risk": risk,
        "confidence": 0.5,
        "method": "accelerometer_impact_threshold_proxy",
        "disclaimer": "Fall proxy only; fall detection requires tri-axial acceleration, posture transition, and labeled events.",
    }
