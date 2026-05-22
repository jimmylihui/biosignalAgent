from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from .common import bandpass_filter, bpm_from_peaks, load_csv_signal, signal_quality_summary


def PCG_assess_quality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    return {"tool": "PCG_assess_quality", "source": data.source, **signal_quality_summary(data.values)}


def PCG_detect_heart_sounds(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    filtered = bandpass_filter(data.values, data.sampling_rate, low_hz=20.0, high_hz=min(150.0, data.sampling_rate * 0.45), order=3)
    envelope = np.abs(scipy_signal.hilbert(filtered))
    min_distance = max(1, int(0.25 * data.sampling_rate))
    peaks, _ = scipy_signal.find_peaks(envelope, distance=min_distance, prominence=max(float(np.std(envelope)) * 0.3, 1e-8))
    heart_rate = bpm_from_peaks(peaks[::2] if len(peaks) >= 4 else peaks, data.sampling_rate)
    confidence = 0.55 if heart_rate is not None and 35 <= heart_rate <= 220 else 0.25
    return {"tool": "PCG_detect_heart_sounds", "sound_indices": peaks.tolist(), "num_sounds": int(len(peaks)), "heart_rate_bpm": heart_rate, "confidence": confidence, "method": "hilbert_envelope_find_peaks"}
