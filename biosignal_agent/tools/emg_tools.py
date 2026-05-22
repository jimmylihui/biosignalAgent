from __future__ import annotations

import numpy as np

from .common import bandpass_filter, load_csv_signal, signal_quality_summary


def EMG_assess_quality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    return {"tool": "EMG_assess_quality", "source": data.source, **signal_quality_summary(data.values)}


def EMG_summarize_activation(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    if len(data.values) == 0:
        return {"tool": "EMG_summarize_activation", "error": "empty signal", "confidence": 0.0}
    high = min(150.0, data.sampling_rate * 0.45)
    filtered = bandpass_filter(data.values, data.sampling_rate, low_hz=20.0, high_hz=high, order=3) if high > 25 else data.values
    rms = float(np.sqrt(np.nanmean(filtered ** 2)))
    mav = float(np.nanmean(np.abs(filtered)))
    return {"tool": "EMG_summarize_activation", "rms": rms, "mean_absolute_value": mav, "num_samples": int(len(filtered)), "confidence": 0.65, "method": "bandpass_rms_summary"}
