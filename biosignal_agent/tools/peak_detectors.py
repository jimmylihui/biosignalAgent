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



def ppg_multiscale_systolic_peaks(
    values: np.ndarray,
    sampling_rate: float,
    min_bpm: float = 35.0,
    max_bpm: float = 220.0,
) -> tuple[np.ndarray, dict]:
    """Detect PPG systolic peaks with a lightweight PPG-native multi-scale vote.

    The detector is inspired by multi-scale PPG peak detectors: candidate peaks are
    generated from several smoothed versions of a 0.4-8 Hz PPG band, then retained
    when they have consistent local support and a plausible neighboring trough.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < max(8, int(sampling_rate * 2)):
        return np.asarray([], dtype=int), {"method": "ppg_multiscale_systolic", "error": "signal too short"}
    high_hz = min(8.0, sampling_rate * 0.45)
    filtered = bandpass_filter(values, sampling_rate, low_hz=0.4, high_hz=high_hz)
    filtered = filtered - np.nanmedian(filtered)
    min_distance = max(1, int((60.0 / max_bpm) * sampling_rate))
    max_distance = max(min_distance + 1, int((60.0 / min_bpm) * sampling_rate))
    base_scale = max(robust_std(filtered), 1e-8)
    vote = np.zeros(len(filtered), dtype=float)
    scale_windows = [0.04, 0.08, 0.12, 0.18]
    candidate_sets = []
    for window_s in scale_windows:
        window = max(1, int(window_s * sampling_rate))
        if window > 1:
            kernel = np.ones(window, dtype=float) / float(window)
            smooth = np.convolve(filtered, kernel, mode="same")
        else:
            smooth = filtered
        prominence = max(robust_std(smooth) * 0.35, base_scale * 0.12, 1e-8)
        peaks, properties = scipy_signal.find_peaks(smooth, distance=min_distance, prominence=prominence)
        candidate_sets.append(len(peaks))
        for peak in peaks:
            left = max(0, peak - max(1, window // 2))
            right = min(len(vote), peak + max(1, window // 2) + 1)
            vote[left:right] += 1.0
    min_votes = max(2.0, 0.55 * len(scale_windows))
    vote_peaks, _ = scipy_signal.find_peaks(vote, distance=min_distance, height=min_votes)
    if len(vote_peaks) == 0:
        peaks, details = scipy_adaptive_peaks(values, sampling_rate, 0.4, high_hz, min_bpm=min_bpm, max_bpm=max_bpm, threshold_scale=0.28)
        details["method"] = "ppg_multiscale_systolic_fallback"
        details["scale_candidate_counts"] = candidate_sets
        return peaks, details

    refined = []
    prominences = []
    trough_depths = []
    local_radius = max(1, int(0.08 * sampling_rate))
    trough_radius = max(1, int(0.32 * sampling_rate))
    for peak in vote_peaks:
        left = max(0, int(peak) - local_radius)
        right = min(len(filtered), int(peak) + local_radius + 1)
        local_peak = left + int(np.argmax(filtered[left:right]))
        trough_left = max(0, local_peak - trough_radius)
        trough_right = min(len(filtered), local_peak + trough_radius + 1)
        local_trough = float(np.nanmin(filtered[trough_left:trough_right])) if trough_right > trough_left else 0.0
        amplitude = float(filtered[local_peak] - local_trough)
        if amplitude >= max(base_scale * 0.30, 1e-8):
            refined.append(local_peak)
            trough_depths.append(amplitude)
    if not refined:
        refined = vote_peaks.tolist()
    refined = np.asarray(sorted(set(int(x) for x in refined)), dtype=int)
    if len(refined) > 1:
        keep = [int(refined[0])]
        for peak in refined[1:]:
            if int(peak) - keep[-1] < min_distance:
                if filtered[int(peak)] > filtered[keep[-1]]:
                    keep[-1] = int(peak)
            else:
                keep.append(int(peak))
        refined = np.asarray(keep, dtype=int)
    if len(refined) > 1:
        intervals = np.diff(refined)
        long_gap_count = int(np.sum(intervals > max_distance))
    else:
        long_gap_count = 0
    if len(refined):
        local_span = max(1, int(0.20 * sampling_rate))
        approx_prominences = []
        for peak in refined:
            left = max(0, int(peak) - local_span)
            right = min(len(filtered), int(peak) + local_span + 1)
            local_floor = np.nanpercentile(filtered[left:right], 10) if right > left else 0.0
            approx_prominences.append(float(filtered[int(peak)] - local_floor))
        prominences = np.asarray(approx_prominences, dtype=float)
    return refined.astype(int), {
        "method": "ppg_multiscale_systolic",
        "median_prominence": float(np.median(prominences)) if len(prominences) else None,
        "median_trough_to_peak_amplitude": float(np.median(trough_depths)) if trough_depths else None,
        "scale_candidate_counts": candidate_sets,
        "long_gap_count": long_gap_count,
    }
