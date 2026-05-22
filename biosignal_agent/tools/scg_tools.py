from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from .common import bpm_from_peaks, interval_regularity, load_csv_signal, signal_quality_summary
from .peak_detectors import neurokit_nabian2018_peaks


def SCG_assess_quality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    return {"tool": "SCG_assess_quality", "source": data.source, **signal_quality_summary(data.values)}


def SCG_detect_j_peaks(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    peaks, details = neurokit_nabian2018_peaks(data.values, data.sampling_rate, low_hz=0.8, high_hz=20.0, fallback_threshold_scale=0.30)
    heart_rate = bpm_from_peaks(peaks, data.sampling_rate)
    regularity = interval_regularity(peaks, data.sampling_rate)
    base_confidence = 0.62 if details["method"] == "nabian2018" else 0.45
    confidence = min(base_confidence, regularity["regularity_confidence"])
    if heart_rate is None or not 35 <= heart_rate <= 220:
        confidence = 0.25
    return {"tool": "SCG_detect_j_peaks", "j_peak_indices": peaks.tolist(), "num_peaks": int(len(peaks)), "heart_rate_bpm": heart_rate, "confidence": confidence, **regularity, **details}



def SCG_estimate_respiration(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    if len(values) < data.sampling_rate * 20:
        return {"tool": "SCG_estimate_respiration", "error": "signal too short", "confidence": 0.0}
    high = min(0.7, data.sampling_rate * 0.45)
    if high <= 0.08:
        return {"tool": "SCG_estimate_respiration", "error": "sampling rate too low", "confidence": 0.1}
    filtered = scipy_signal.sosfiltfilt(scipy_signal.butter(3, [0.08 / (0.5 * data.sampling_rate), high / (0.5 * data.sampling_rate)], btype="bandpass", output="sos"), values - np.nanmedian(values))
    freqs, psd = scipy_signal.welch(filtered, fs=data.sampling_rate, nperseg=min(len(filtered), int(data.sampling_rate * 16)))
    mask = (freqs >= 0.08) & (freqs <= high)
    respiratory_rate = float(freqs[mask][np.argmax(psd[mask])] * 60.0) if np.any(mask) else None
    respiration_power_ratio = float(np.trapz(psd[mask], freqs[mask]) / (np.trapz(psd, freqs) + 1e-12)) if len(freqs) and np.any(mask) else 0.0
    return {
        "tool": "SCG_estimate_respiration",
        "respiratory_rate_bpm": respiratory_rate,
        "respiration_power_ratio": respiration_power_ratio,
        "confidence": 0.55 if respiratory_rate is not None and 5 <= respiratory_rate <= 40 else 0.5,
        "method": "mechanical_signal_respiration_bandpower_proxy",
        "disclaimer": "Mechanical-signal respiration proxy only; validate against respiratory reference signals.",
    }
