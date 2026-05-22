from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

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



def EMG_estimate_fatigue(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    if len(data.values) < max(8, int(data.sampling_rate)):
        return {"tool": "EMG_estimate_fatigue", "error": "signal too short", "confidence": 0.0}
    high = min(250.0, data.sampling_rate * 0.45)
    filtered = bandpass_filter(data.values, data.sampling_rate, low_hz=20.0, high_hz=high, order=3) if high > 25 else data.values
    freqs, psd = scipy_signal.welch(filtered, fs=data.sampling_rate, nperseg=min(len(filtered), int(data.sampling_rate * 2)))
    mask = (freqs >= 20) & (freqs <= high)
    if not np.any(mask):
        return {"tool": "EMG_estimate_fatigue", "error": "insufficient EMG bandwidth", "confidence": 0.1}
    f = freqs[mask]
    pxx = psd[mask]
    cumulative = np.cumsum(pxx)
    median_frequency = float(f[np.searchsorted(cumulative, cumulative[-1] / 2.0)]) if cumulative[-1] > 0 else None
    rms = float(np.sqrt(np.nanmean(filtered ** 2)))
    fatigue_proxy = "possible_fatigue_proxy" if median_frequency is not None and median_frequency < 60 else "no_fatigue_proxy"
    return {
        "tool": "EMG_estimate_fatigue",
        "median_frequency_hz": median_frequency,
        "rms": rms,
        "fatigue_proxy": fatigue_proxy,
        "confidence": 0.55,
        "method": "emg_median_frequency_screening",
        "disclaimer": "Screening heuristic only; muscle fatigue needs task protocol, normalization, and repeated contractions.",
    }
