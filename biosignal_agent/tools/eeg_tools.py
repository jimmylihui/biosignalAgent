from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from .common import load_csv_signal, signal_quality_summary


def EEG_assess_quality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    return {"tool": "EEG_assess_quality", "source": data.source, **signal_quality_summary(data.values)}


def EEG_compute_bandpower(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    if len(data.values) < data.sampling_rate:
        return {"tool": "EEG_compute_bandpower", "error": "signal too short", "confidence": 0.0}
    freqs, psd = scipy_signal.welch(data.values, fs=data.sampling_rate, nperseg=min(len(data.values), int(data.sampling_rate * 4)))
    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, min(45, data.sampling_rate * 0.45))}
    powers = {}
    total = float(np.trapz(psd[(freqs >= 0.5) & (freqs <= min(45, data.sampling_rate * 0.45))], freqs[(freqs >= 0.5) & (freqs <= min(45, data.sampling_rate * 0.45))]))
    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        powers[f"{name}_power"] = float(np.trapz(psd[mask], freqs[mask])) if np.any(mask) else 0.0
    powers["total_power"] = total
    powers["confidence"] = 0.65
    powers["method"] = "welch_bandpower"
    powers["tool"] = "EEG_compute_bandpower"
    return powers
