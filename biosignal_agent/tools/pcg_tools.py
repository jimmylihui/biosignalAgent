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



def PCG_segment_s1_s2_proxy(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    sounds = PCG_detect_heart_sounds(signal_path, sampling_rate, column)
    peaks = np.asarray(sounds.get("sound_indices", []), dtype=int)
    if len(peaks) < 4:
        return {"tool": "PCG_segment_s1_s2_proxy", "error": "not enough heart sounds", "confidence": 0.1}
    intervals = np.diff(peaks) / float(sampling_rate)
    short_intervals = intervals[::2]
    long_intervals = intervals[1::2]
    systole_duration_s = float(np.nanmedian(short_intervals)) if len(short_intervals) else None
    diastole_duration_s = float(np.nanmedian(long_intervals)) if len(long_intervals) else None
    s1_indices = peaks[::2]
    s2_indices = peaks[1::2]
    return {
        "tool": "PCG_segment_s1_s2_proxy",
        "s1_indices": s1_indices[:20].tolist(),
        "s2_indices": s2_indices[:20].tolist(),
        "num_s1": int(len(s1_indices)),
        "num_s2": int(len(s2_indices)),
        "systole_duration_s": systole_duration_s,
        "diastole_duration_s": diastole_duration_s,
        "heart_rate_bpm": sounds.get("heart_rate_bpm"),
        "confidence": min(0.5, float(sounds.get("confidence", 0.5))),
        "method": "alternating_pcg_peak_s1_s2_proxy",
        "disclaimer": "S1/S2 segmentation proxy only; validated PCG segmentation requires dedicated algorithms and labels.",
    }


def PCG_extract_murmur_features(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = np.asarray(data.values, dtype=float)
    if len(values) < max(8, int(data.sampling_rate)):
        return {"tool": "PCG_extract_murmur_features", "error": "signal too short", "confidence": 0.0}
    centered = values - np.nanmedian(values)
    high = min(500.0, data.sampling_rate * 0.45)
    low_cut = 20.0 if high > 40 else max(1.0, data.sampling_rate * 0.05)
    filtered = bandpass_filter(centered, data.sampling_rate, low_cut, high, order=3) if high > low_cut else centered
    envelope = np.abs(scipy_signal.hilbert(filtered))
    freqs, psd = scipy_signal.welch(filtered, fs=data.sampling_rate, nperseg=min(len(filtered), int(data.sampling_rate * 2)))

    def band_power(low: float, high_band: float) -> float:
        mask = (freqs >= low) & (freqs < min(high_band, high))
        return float(np.trapz(psd[mask], freqs[mask])) if np.any(mask) else 0.0

    low_power = band_power(20.0, 60.0)
    mid_power = band_power(60.0, 150.0)
    high_power = band_power(150.0, 400.0)
    very_high_power = band_power(400.0, high)
    total_power = float(low_power + mid_power + high_power + very_high_power + 1e-12)
    spectral_centroid_hz = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-12)) if len(freqs) else None
    psd_norm = psd / (np.sum(psd) + 1e-12) if len(psd) else np.array([])
    spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)) / np.log2(len(psd_norm))) if len(psd_norm) > 1 else 0.0
    zcr = float(np.mean(np.diff(np.signbit(filtered)) != 0)) if len(filtered) > 1 else 0.0
    env_median = float(np.nanmedian(envelope)) if len(envelope) else 0.0
    env_p90 = float(np.nanpercentile(envelope, 90)) if len(envelope) else 0.0
    env_p95 = float(np.nanpercentile(envelope, 95)) if len(envelope) else 0.0
    env_p99 = float(np.nanpercentile(envelope, 99)) if len(envelope) else 0.0
    env_std = float(np.nanstd(envelope)) if len(envelope) else 0.0
    continuous_fraction_60 = float(np.mean(envelope > np.nanpercentile(envelope, 60))) if len(envelope) else 0.0
    continuous_fraction_75 = float(np.mean(envelope > np.nanpercentile(envelope, 75))) if len(envelope) else 0.0
    sounds = PCG_detect_heart_sounds(signal_path, sampling_rate, column)
    peaks = np.asarray(sounds.get("sound_indices", []), dtype=int)
    intervals = np.diff(peaks) / float(data.sampling_rate) if len(peaks) > 1 else np.array([])
    interval_cv = float(np.nanstd(intervals) / (np.nanmean(intervals) + 1e-12)) if len(intervals) else None
    return {
        "tool": "PCG_extract_murmur_features",
        "low_band_power": low_power,
        "mid_band_power": mid_power,
        "high_band_power": high_power,
        "very_high_band_power": very_high_power,
        "mid_band_ratio": float(mid_power / total_power),
        "high_band_ratio": float((high_power + very_high_power) / total_power),
        "spectral_centroid_hz": spectral_centroid_hz,
        "spectral_entropy": spectral_entropy,
        "zero_crossing_rate": zcr,
        "envelope_std": env_std,
        "envelope_p90_median_ratio": float(env_p90 / (env_median + 1e-12)),
        "envelope_p95_median_ratio": float(env_p95 / (env_median + 1e-12)),
        "envelope_p99_median_ratio": float(env_p99 / (env_median + 1e-12)),
        "continuous_fraction_60": continuous_fraction_60,
        "continuous_fraction_75": continuous_fraction_75,
        "num_sounds": sounds.get("num_sounds"),
        "heart_rate_bpm": sounds.get("heart_rate_bpm"),
        "sound_interval_cv": interval_cv,
        "confidence": 0.55,
        "method": "pcg_spectral_envelope_timing_features",
        "disclaimer": "Feature extraction only; classification requires labeled training/evaluation.",
    }
