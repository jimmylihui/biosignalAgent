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
