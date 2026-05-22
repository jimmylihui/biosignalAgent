from __future__ import annotations

import numpy as np

from .common import load_csv_signal, signal_quality_summary


def SpO2_assess_quality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    return {"tool": "SpO2_assess_quality", "source": data.source, **signal_quality_summary(data.values)}


def SpO2_summarize(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values[np.isfinite(data.values)]
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
