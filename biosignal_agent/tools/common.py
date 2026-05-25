from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal


@dataclass
class SignalData:
    values: np.ndarray
    sampling_rate: float
    source: str


def load_csv_signal(signal_path: str, sampling_rate: float, column: str | None = None) -> SignalData:
    path = Path(signal_path)
    frame = pd.read_csv(path)
    if column is None:
        column = "signal" if "signal" in frame.columns else frame.select_dtypes("number").columns[0]
    values = frame[column].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return SignalData(values=values, sampling_rate=float(sampling_rate), source=str(path))


def bandpass_filter(values: np.ndarray, sampling_rate: float, low_hz: float, high_hz: float, order: int = 3) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    centered = values - np.nanmedian(values)
    if sampling_rate <= 0:
        return centered
    nyquist = 0.5 * float(sampling_rate)
    high_hz = min(float(high_hz), nyquist * 0.90)
    low_hz = max(float(low_hz), 1e-6)
    if nyquist <= 0 or high_hz <= low_hz or len(values) < max(12, order * 6):
        return centered
    low = max(low_hz / nyquist, 1e-5)
    high = min(high_hz / nyquist, 0.999)
    if not (0 < low < high < 1):
        return centered
    sos = scipy_signal.butter(order, [low, high], btype="bandpass", output="sos")
    try:
        return scipy_signal.sosfiltfilt(sos, centered)
    except ValueError:
        return centered


def signal_quality_summary(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"quality": "bad", "reason": "empty signal", "confidence": 0.0}
    finite_ratio = float(np.isfinite(values).mean())
    dynamic_range = float(np.nanpercentile(values, 95) - np.nanpercentile(values, 5))
    max_abs = float(np.nanmax(np.abs(values)))
    clipped_ratio = float((np.abs(values) >= max_abs * 0.999).mean()) if max_abs > 0 else 1.0
    if finite_ratio < 0.99 or dynamic_range == 0 or clipped_ratio > 0.2:
        return {"quality": "bad", "finite_ratio": finite_ratio, "dynamic_range": dynamic_range, "clipped_ratio": clipped_ratio, "confidence": 0.2}
    quality = "moderate" if clipped_ratio > 0.05 else "good"
    confidence = 0.6 if quality == "moderate" else 0.85
    return {"quality": quality, "finite_ratio": finite_ratio, "dynamic_range": dynamic_range, "clipped_ratio": clipped_ratio, "confidence": confidence}


def bpm_from_peaks(peaks: np.ndarray, sampling_rate: float) -> float | None:
    if len(peaks) < 2:
        return None
    intervals = np.diff(peaks) / sampling_rate
    intervals = intervals[intervals > 0]
    if len(intervals) == 0:
        return None
    return float(60.0 / np.median(intervals))


def interval_regularity(peaks: np.ndarray, sampling_rate: float) -> dict:
    if len(peaks) < 3:
        return {"interval_cv": None, "regularity_confidence": 0.3}
    intervals = np.diff(peaks) / sampling_rate
    intervals = intervals[intervals > 0]
    if len(intervals) < 2:
        return {"interval_cv": None, "regularity_confidence": 0.3}
    cv = float(np.std(intervals) / np.mean(intervals)) if np.mean(intervals) > 0 else None
    if cv is None:
        confidence = 0.3
    elif cv <= 0.12:
        confidence = 0.9
    elif cv <= 0.25:
        confidence = 0.75
    elif cv <= 0.45:
        confidence = 0.55
    else:
        confidence = 0.35
    return {"interval_cv": cv, "regularity_confidence": confidence}
