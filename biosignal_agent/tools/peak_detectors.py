from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from .common import bandpass_filter

try:
    import neurokit2 as nk
except ImportError:  # pragma: no cover
    nk = None


def robust_std(values: np.ndarray) -> float:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return float(1.4826 * mad) if mad > 0 else float(np.std(values))


def scipy_adaptive_peaks(
    values: np.ndarray,
    sampling_rate: float,
    low_hz: float,
    high_hz: float,
    min_bpm: float = 35.0,
    max_bpm: float = 220.0,
    threshold_scale: float = 0.35,
) -> tuple[np.ndarray, dict]:
    filtered = bandpass_filter(values, sampling_rate, low_hz=low_hz, high_hz=high_hz)
    window = max(1, int(0.08 * sampling_rate))
    smoothed = np.convolve(filtered, np.ones(window) / window, mode="same") if window > 1 else filtered
    min_distance = max(1, int((60.0 / max_bpm) * sampling_rate))
    max_distance = max(min_distance + 1, int((60.0 / min_bpm) * sampling_rate))
    prominence = max(robust_std(smoothed) * threshold_scale, 1e-8)
    peaks, properties = scipy_signal.find_peaks(smoothed, distance=min_distance, prominence=prominence)
    long_gap_count = int(np.sum(np.diff(peaks) > max_distance)) if len(peaks) > 2 else 0
    return peaks.astype(int), {
        "method": "scipy_adaptive_fallback",
        "median_prominence": float(np.median(properties.get("prominences", [0]))),
        "threshold_prominence": float(prominence),
        "long_gap_count": long_gap_count,
    }


def neurokit_nabian2018_peaks(
    values: np.ndarray,
    sampling_rate: float,
    low_hz: float | None = None,
    high_hz: float | None = None,
    fallback_threshold_scale: float = 0.35,
) -> tuple[np.ndarray, dict]:
    """Detect peaks with NeuroKit2's Nabian 2018 ECG detector.

    For ECG, pass the raw/cleaned ECG through directly. For PPG and BCG, callers
    pass modality-specific bandpass limits first, then reuse the same Nabian 2018
    detector core requested for ECG.
    """
    if nk is None:
        raise RuntimeError("neurokit2 is not installed")
    detector_input = values
    if low_hz is not None and high_hz is not None:
        detector_input = bandpass_filter(values, sampling_rate, low_hz=low_hz, high_hz=high_hz)
    try:
        _, info = nk.ecg_peaks(detector_input, sampling_rate=sampling_rate, method="nabian2018", correct_artifacts=True)
        peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
        return peaks, {"method": "nabian2018", "median_prominence": None}
    except Exception as exc:
        if low_hz is None or high_hz is None:
            low_hz, high_hz = 0.5, min(40.0, sampling_rate / 2.0 * 0.9)
        peaks, details = scipy_adaptive_peaks(values, sampling_rate, low_hz, high_hz, threshold_scale=fallback_threshold_scale)
        details["fallback_reason"] = str(exc)
        return peaks, details
