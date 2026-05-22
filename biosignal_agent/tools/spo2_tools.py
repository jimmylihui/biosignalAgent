from __future__ import annotations

import numpy as np

from .common import load_csv_signal


def _spo2_values(signal_path: str, sampling_rate: float, column: str | None = None):
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values[np.isfinite(data.values)]
    return data, values


def SpO2_assess_quality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data, values = _spo2_values(signal_path, sampling_rate, column)
    if len(values) == 0:
        return {"tool": "SpO2_assess_quality", "source": data.source, "quality": "bad", "reason": "empty signal", "confidence": 0.0}
    plausible_ratio = float(((values >= 50) & (values <= 100)).mean())
    finite_ratio = 1.0
    dynamic_range = float(np.nanpercentile(values, 95) - np.nanpercentile(values, 5))
    step_changes = np.abs(np.diff(values)) if len(values) > 1 else np.array([])
    jump_fraction = float((step_changes > 5).mean()) if len(step_changes) else 0.0
    if plausible_ratio < 0.95 or jump_fraction > 0.1:
        quality = "bad"
        confidence = 0.25
    elif dynamic_range > 8:
        quality = "moderate"
        confidence = 0.65
    else:
        quality = "good"
        confidence = 0.9
    return {
        "tool": "SpO2_assess_quality",
        "source": data.source,
        "quality": quality,
        "finite_ratio": finite_ratio,
        "plausible_ratio": plausible_ratio,
        "dynamic_range": dynamic_range,
        "jump_fraction": jump_fraction,
        "confidence": confidence,
    }


def SpO2_summarize(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data, values = _spo2_values(signal_path, sampling_rate, column)
    if len(values) == 0:
        return {"tool": "SpO2_summarize", "error": "empty signal", "confidence": 0.0}
    plausible = values[(values >= 50) & (values <= 100)]
    confidence = 0.85 if len(plausible) / len(values) > 0.95 else 0.4
    target = plausible if len(plausible) else values
    return {
        "tool": "SpO2_summarize",
        "mean_spo2_percent": float(np.mean(target)),
        "min_spo2_percent": float(np.min(target)),
        "time_below_90_fraction": float(np.mean(target < 90)),
        "num_samples": int(len(target)),
        "confidence": confidence,
    }



def SpO2_detect_desaturation(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data, values = _spo2_values(signal_path, sampling_rate, column)
    if len(values) == 0:
        return {"tool": "SpO2_detect_desaturation", "error": "empty signal", "confidence": 0.0}
    target = values[(values >= 50) & (values <= 100)]
    if len(target) < max(3, int(sampling_rate * 10)):
        target = values
    baseline = float(np.nanpercentile(target, 90))
    below90 = target < 90
    desat = target <= baseline - 3.0
    min_len = max(1, int(10.0 * sampling_rate))
    events = []
    start = None
    for idx, flag in enumerate(desat):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            if idx - start >= min_len:
                events.append((start, idx))
            start = None
    if start is not None and len(desat) - start >= min_len:
        events.append((start, len(desat)))
    duration_hours = len(target) / float(sampling_rate) / 3600.0 if sampling_rate else 0.0
    odi = float(len(events) / duration_hours) if duration_hours > 0 else None
    return {
        "tool": "SpO2_detect_desaturation",
        "baseline_spo2_percent": baseline,
        "min_spo2_percent": float(np.nanmin(target)),
        "time_below_90_fraction": float(np.mean(below90)),
        "desaturation_event_count": int(len(events)),
        "oxygen_desaturation_index_per_hour": odi,
        "event_intervals_s": [[float(start / sampling_rate), float(end / sampling_rate)] for start, end in events[:20]],
        "confidence": 0.65,
        "method": "three_percent_drop_screening",
        "disclaimer": "Screening heuristic only; ODI should be validated against labeled sleep-study events.",
    }



def SpO2_assess_hypoxemia_burden(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data, values = _spo2_values(signal_path, sampling_rate, column)
    if len(values) == 0:
        return {"tool": "SpO2_assess_hypoxemia_burden", "error": "empty signal", "confidence": 0.0}
    target = values[(values >= 50) & (values <= 100)]
    if len(target) == 0:
        target = values
    time_below_90_fraction = float(np.mean(target < 90))
    time_below_88_fraction = float(np.mean(target < 88))
    nadir = float(np.nanmin(target))
    mean_spo2 = float(np.nanmean(target))
    if time_below_88_fraction > 0.05 or nadir < 85:
        burden = "high_hypoxemia_burden_proxy"
    elif time_below_90_fraction > 0.05:
        burden = "moderate_hypoxemia_burden_proxy"
    else:
        burden = "low_hypoxemia_burden_proxy"
    return {
        "tool": "SpO2_assess_hypoxemia_burden",
        "mean_spo2_percent": mean_spo2,
        "min_spo2_percent": nadir,
        "time_below_90_fraction": time_below_90_fraction,
        "time_below_88_fraction": time_below_88_fraction,
        "hypoxemia_burden": burden,
        "num_samples": int(len(target)),
        "confidence": 0.75 if len(target) / len(values) > 0.95 else 0.45,
        "method": "spo2_threshold_burden_screening",
        "disclaimer": "Screening heuristic only; clinical hypoxemia interpretation requires validated oximetry and clinical context.",
    }
