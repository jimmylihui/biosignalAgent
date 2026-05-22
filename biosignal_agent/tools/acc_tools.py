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
