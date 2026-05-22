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
