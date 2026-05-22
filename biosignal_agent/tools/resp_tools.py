from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from .common import bandpass_filter, load_csv_signal, signal_quality_summary


def RESP_assess_quality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    return {"tool": "RESP_assess_quality", "source": data.source, **signal_quality_summary(data.values)}


def RESP_estimate_rate(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    if len(values) < data.sampling_rate * 10:
        return {"tool": "RESP_estimate_rate", "error": "signal too short", "confidence": 0.0}
    filtered = bandpass_filter(values, data.sampling_rate, low_hz=0.05, high_hz=0.7, order=3)
    min_distance = max(1, int(1.5 * data.sampling_rate))
    prominence = max(float(np.nanstd(filtered)) * 0.2, 1e-8)
    peaks, _ = scipy_signal.find_peaks(filtered, distance=min_distance, prominence=prominence)
    if len(peaks) < 2:
        peaks, _ = scipy_signal.find_peaks(-filtered, distance=min_distance, prominence=prominence)
    intervals = np.diff(peaks) / data.sampling_rate if len(peaks) >= 2 else np.array([])
    intervals = intervals[(intervals >= 1.5) & (intervals <= 12.0)]
    respiratory_rate = float(60.0 / np.median(intervals)) if len(intervals) else None
    confidence = 0.75 if respiratory_rate is not None and 5 <= respiratory_rate <= 40 else 0.3
    return {
        "tool": "RESP_estimate_rate",
        "breath_indices": peaks.tolist(),
        "num_breaths": int(len(peaks)),
        "respiratory_rate_bpm": respiratory_rate,
        "confidence": confidence,
        "method": "bandpass_find_peaks",
    }



def RESP_detect_apnea(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    if len(values) < data.sampling_rate * 20:
        return {"tool": "RESP_detect_apnea", "error": "signal too short for apnea screening", "confidence": 0.0}
    filtered = bandpass_filter(values, data.sampling_rate, low_hz=0.05, high_hz=0.7, order=3)
    window = max(1, int(2.0 * data.sampling_rate))
    kernel = np.ones(window) / window
    envelope = np.sqrt(np.convolve(filtered ** 2, kernel, mode="same"))
    baseline = float(np.nanmedian(envelope))
    if baseline <= 0:
        return {"tool": "RESP_detect_apnea", "error": "flat respiration envelope", "confidence": 0.1}
    low = envelope < baseline * 0.25
    min_len = int(10.0 * data.sampling_rate)
    events = []
    start = None
    for idx, is_low in enumerate(low):
        if is_low and start is None:
            start = idx
        elif not is_low and start is not None:
            if idx - start >= min_len:
                events.append((start, idx))
            start = None
    if start is not None and len(low) - start >= min_len:
        events.append((start, len(low)))
    duration_hours = len(values) / data.sampling_rate / 3600.0
    apnea_index = float(len(events) / duration_hours) if duration_hours > 0 else None
    longest_event_s = float(max(((end - start) / data.sampling_rate for start, end in events), default=0.0))
    return {
        "tool": "RESP_detect_apnea",
        "apnea_event_count": int(len(events)),
        "apnea_index_per_hour": apnea_index,
        "longest_event_s": longest_event_s,
        "low_respiration_fraction": float(np.mean(low)),
        "event_intervals_s": [[float(start / data.sampling_rate), float(end / data.sampling_rate)] for start, end in events[:20]],
        "confidence": 0.55,
        "method": "respiration_envelope_drop_screening",
        "disclaimer": "Screening heuristic only; validated apnea labels are required for diagnosis.",
    }



def RESP_detect_hypopnea(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    if len(values) < data.sampling_rate * 20:
        return {"tool": "RESP_detect_hypopnea", "error": "signal too short for hypopnea screening", "confidence": 0.0}
    filtered = bandpass_filter(values, data.sampling_rate, low_hz=0.05, high_hz=0.7, order=3)
    window = max(1, int(2.0 * data.sampling_rate))
    kernel = np.ones(window) / window
    envelope = np.sqrt(np.convolve(filtered ** 2, kernel, mode="same"))
    baseline = float(np.nanpercentile(envelope, 75))
    if baseline <= 0:
        return {"tool": "RESP_detect_hypopnea", "error": "flat respiration envelope", "confidence": 0.1}
    reduced = (envelope < baseline * 0.7) & (envelope >= baseline * 0.25)
    min_len = int(10.0 * data.sampling_rate)
    events = []
    start = None
    for idx, is_low in enumerate(reduced):
        if is_low and start is None:
            start = idx
        elif not is_low and start is not None:
            if idx - start >= min_len:
                events.append((start, idx))
            start = None
    if start is not None and len(reduced) - start >= min_len:
        events.append((start, len(reduced)))
    duration_hours = len(values) / data.sampling_rate / 3600.0
    hypopnea_index = float(len(events) / duration_hours) if duration_hours > 0 else None
    return {
        "tool": "RESP_detect_hypopnea",
        "hypopnea_event_count": int(len(events)),
        "hypopnea_index_per_hour": hypopnea_index,
        "reduced_respiration_fraction": float(np.mean(reduced)),
        "event_intervals_s": [[float(start / data.sampling_rate), float(end / data.sampling_rate)] for start, end in events[:20]],
        "confidence": 0.55,
        "method": "respiration_envelope_reduction_screening",
        "disclaimer": "Screening heuristic only; hypopnea scoring also needs desaturation/arousal context and validated PSG labels.",
    }
