from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from .common import bandpass_filter, bpm_from_peaks, interval_regularity, load_csv_signal, signal_quality_summary


def PCG_assess_quality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    return {"tool": "PCG_assess_quality", "source": data.source, **signal_quality_summary(data.values)}


def PCG_detect_heart_sounds(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    filtered = bandpass_filter(data.values, data.sampling_rate, low_hz=20.0, high_hz=min(150.0, data.sampling_rate * 0.45), order=3)
    envelope = np.abs(scipy_signal.hilbert(filtered))
    min_distance = max(1, int(0.25 * data.sampling_rate))
    peaks, _ = scipy_signal.find_peaks(envelope, distance=min_distance, prominence=max(float(np.std(envelope)) * 0.3, 1e-8))
    beat_peaks = peaks[::2] if len(peaks) >= 4 else peaks
    heart_rate = bpm_from_peaks(beat_peaks, data.sampling_rate)
    regularity = interval_regularity(beat_peaks, data.sampling_rate)
    confidence = min(0.6, regularity["regularity_confidence"]) if heart_rate is not None and 35 <= heart_rate <= 220 else 0.25
    return {"tool": "PCG_detect_heart_sounds", "sound_indices": peaks.tolist(), "num_sounds": int(len(peaks)), "heart_rate_bpm": heart_rate, "confidence": confidence, **regularity, "method": "hilbert_envelope_find_peaks"}



def PCG_screen_murmur_proxy(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    if len(values) < max(8, int(data.sampling_rate)):
        return {"tool": "PCG_screen_murmur_proxy", "error": "signal too short", "confidence": 0.0}
    high = min(400.0, data.sampling_rate * 0.45)
    freqs, psd = scipy_signal.welch(values, fs=data.sampling_rate, nperseg=min(len(values), int(data.sampling_rate * 2)))
    low_mask = (freqs >= 20) & (freqs < min(150, high))
    high_mask = (freqs >= 150) & (freqs <= high)
    low_power = float(np.trapz(psd[low_mask], freqs[low_mask])) if np.any(low_mask) else 0.0
    high_power = float(np.trapz(psd[high_mask], freqs[high_mask])) if np.any(high_mask) else 0.0
    high_frequency_ratio = float(high_power / (low_power + high_power + 1e-12))
    envelope = np.abs(scipy_signal.hilbert(bandpass_filter(values, data.sampling_rate, 20.0, min(200.0, high), order=3))) if high > 30 else np.abs(values)
    continuous_fraction = float(np.mean(envelope > np.nanpercentile(envelope, 60))) if len(envelope) else 0.0
    score = min(1.0, high_frequency_ratio * 1.5 + max(0.0, continuous_fraction - 0.4))
    murmur_risk = "possible_murmur_proxy" if score >= 0.35 else "no_murmur_proxy"
    return {
        "tool": "PCG_screen_murmur_proxy",
        "high_frequency_ratio": high_frequency_ratio,
        "continuous_sound_fraction": continuous_fraction,
        "murmur_proxy_score": float(score),
        "murmur_risk": murmur_risk,
        "confidence": 0.5,
        "method": "pcg_high_frequency_continuity_screening",
        "disclaimer": "Screening heuristic only; murmur detection requires validated PCG segmentation and labeled clinical data.",
    }
