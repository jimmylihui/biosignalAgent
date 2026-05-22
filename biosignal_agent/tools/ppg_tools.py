from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from .common import bandpass_filter, bpm_from_peaks, load_csv_signal, signal_quality_summary

try:
    import neurokit2 as nk
except ImportError:  # pragma: no cover
    nk = None


def PPG_assess_quality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    return {"tool": "PPG_assess_quality", "source": data.source, **signal_quality_summary(data.values)}


def _detect_ppg_peaks_scipy(values: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, dict]:
    filtered = bandpass_filter(values, sampling_rate, low_hz=0.4, high_hz=8.0)
    distance = max(1, int(0.3 * sampling_rate))
    prominence = max(float(np.std(filtered)) * 0.4, 1e-8)
    peaks, properties = scipy_signal.find_peaks(filtered, distance=distance, prominence=prominence)
    return peaks, {"method": "scipy_find_peaks", "median_prominence": float(np.median(properties.get("prominences", [0])))}


def _detect_ppg_peaks_neurokit(values: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, dict]:
    if nk is None:
        raise RuntimeError("neurokit2 is not installed")
    cleaned = nk.ppg_clean(values, sampling_rate=sampling_rate, method="elgendi")
    _, info = nk.ppg_peaks(cleaned, sampling_rate=sampling_rate, method="elgendi")
    peaks = np.asarray(info.get("PPG_Peaks", []), dtype=int)
    return peaks, {"method": "neurokit2_elgendi", "median_prominence": None}


def PPG_detect_peaks(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    try:
        peaks, details = _detect_ppg_peaks_neurokit(data.values, data.sampling_rate)
    except Exception as exc:
        peaks, details = _detect_ppg_peaks_scipy(data.values, data.sampling_rate)
        details["fallback_reason"] = str(exc)
    heart_rate = bpm_from_peaks(peaks, data.sampling_rate)
    confidence = 0.8 if details["method"] == "neurokit2_elgendi" else 0.6
    if heart_rate is None or not 35 <= heart_rate <= 220:
        confidence = 0.3
    return {"tool": "PPG_detect_peaks", "peak_indices": peaks.tolist(), "num_peaks": int(len(peaks)), "heart_rate_bpm": heart_rate, "confidence": confidence, **details}
