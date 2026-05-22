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



def EEG_estimate_sleep_stage_features(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    bandpower = EEG_compute_bandpower(signal_path, sampling_rate, column)
    if bandpower.get("error"):
        return {"tool": "EEG_estimate_sleep_stage_features", "error": bandpower["error"], "confidence": 0.0}
    total = float(bandpower.get("total_power") or 0.0)
    if total <= 0:
        return {"tool": "EEG_estimate_sleep_stage_features", "error": "zero EEG power", "confidence": 0.1}
    delta_ratio = float(bandpower.get("delta_power", 0.0) / total)
    theta_ratio = float(bandpower.get("theta_power", 0.0) / total)
    alpha_ratio = float(bandpower.get("alpha_power", 0.0) / total)
    beta_ratio = float(bandpower.get("beta_power", 0.0) / total)
    if delta_ratio > 0.45:
        stage_hint = "n3_like_slow_wave"
    elif theta_ratio > 0.30 and alpha_ratio < 0.20:
        stage_hint = "n1_n2_like"
    elif alpha_ratio > 0.25 or beta_ratio > 0.25:
        stage_hint = "wake_rem_like"
    else:
        stage_hint = "uncertain"
    return {
        "tool": "EEG_estimate_sleep_stage_features",
        "delta_ratio": delta_ratio,
        "theta_ratio": theta_ratio,
        "alpha_ratio": alpha_ratio,
        "beta_ratio": beta_ratio,
        "sleep_stage_hint": stage_hint,
        "confidence": 0.5,
        "method": "single_channel_bandpower_rules",
        "disclaimer": "Feature heuristic only; sleep staging requires labeled epochs and usually EEG/EOG/EMG context.",
    }
