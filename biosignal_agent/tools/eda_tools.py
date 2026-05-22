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
