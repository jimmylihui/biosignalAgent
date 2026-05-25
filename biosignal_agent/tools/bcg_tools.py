from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from .common import bandpass_filter, bpm_from_peaks, interval_regularity, load_csv_signal, signal_quality_summary
from .peak_detectors import neurokit_nabian2018_peaks, robust_std, scipy_adaptive_peaks


def _safe_trapz(y: np.ndarray, x: np.ndarray) -> float:
    integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(integrate(y, x)) if len(y) and len(x) else 0.0


def _clean_intervals(peaks: np.ndarray, sampling_rate: float) -> np.ndarray:
    intervals = np.diff(np.asarray(peaks, dtype=float)) / float(sampling_rate)
    intervals = intervals[np.isfinite(intervals) & (intervals >= 0.28) & (intervals <= 2.2)]
    if len(intervals) < 4:
        return intervals
    median = float(np.median(intervals))
    mad = float(np.median(np.abs(intervals - median))) + 1e-8
    return intervals[np.abs(intervals - median) <= max(0.22, 4.5 * mad)]


def _bcg_hilbert_peaks(values: np.ndarray, sampling_rate: float, low_hz: float = 2.5, high_hz: float = 12.0, prominence_scale: float = 0.7) -> tuple[np.ndarray, dict]:
    filtered = bandpass_filter(values - np.nanmedian(values), sampling_rate, low_hz, min(high_hz, sampling_rate * 0.45), order=3)
    envelope = np.abs(scipy_signal.hilbert(filtered))
    window = max(5, int(0.16 * sampling_rate) | 1)
    envelope = scipy_signal.savgol_filter(envelope, window, 2, mode="interp")
    min_distance = max(1, int((60.0 / 180.0) * sampling_rate))
    prominence = max(robust_std(envelope) * prominence_scale, 1e-8)
    peaks, properties = scipy_signal.find_peaks(envelope, distance=min_distance, prominence=prominence)
    refined: list[int] = []
    radius = max(1, int(0.12 * sampling_rate))
    for peak in peaks:
        left = max(0, int(peak) - radius)
        right = min(len(filtered), int(peak) + radius + 1)
        if right > left:
            refined.append(left + int(np.argmax(filtered[left:right])))
    return np.asarray(sorted(set(refined)), dtype=int), {
        "method": "hilbert_envelope_peak_train",
        "candidate": f"hilbert_{low_hz:g}_{high_hz:g}",
        "threshold_prominence": float(prominence),
        "median_prominence": float(np.median(properties.get("prominences", [0]))),
    }


def _bcg_spectral_hr(values: np.ndarray, sampling_rate: float) -> dict:
    filtered = bandpass_filter(values - np.nanmedian(values), sampling_rate, 0.8, min(12.0, sampling_rate * 0.45), order=3)
    envelope = np.abs(scipy_signal.hilbert(filtered))
    window = max(5, int(0.24 * sampling_rate) | 1)
    envelope = scipy_signal.savgol_filter(envelope, window, 2, mode="interp")
    freqs, psd = scipy_signal.welch(envelope - np.nanmedian(envelope), fs=sampling_rate, nperseg=min(len(envelope), int(sampling_rate * 16)))
    mask = (freqs >= 0.7) & (freqs <= 3.0)
    if not np.any(mask):
        return {"spectral_hr_bpm": None, "spectral_hr_candidates_bpm": []}
    band_freqs = freqs[mask]
    band_psd = psd[mask]
    peak_idx, _ = scipy_signal.find_peaks(band_psd)
    if len(peak_idx):
        top = peak_idx[np.argsort(band_psd[peak_idx])[-3:]][::-1]
    else:
        top = np.asarray([int(np.argmax(band_psd))])
    candidates = [float(band_freqs[i] * 60.0) for i in top]
    return {"spectral_hr_bpm": candidates[0], "spectral_hr_candidates_bpm": candidates}


def _score_peak_candidate(peaks: np.ndarray, sampling_rate: float, spectral_candidates: list[float]) -> float:
    hr = bpm_from_peaks(peaks, sampling_rate)
    if hr is None or not 35 <= hr <= 180:
        return -1.0
    regularity = interval_regularity(peaks, sampling_rate)
    cv = regularity.get("interval_cv")
    regularity_score = max(0.0, 1.0 - float(cv)) if cv is not None else 0.2
    if spectral_candidates:
        nearest = min(abs(hr - candidate) for candidate in spectral_candidates)
        spectral_score = max(0.0, 1.0 - nearest / 35.0)
    else:
        spectral_score = 0.4
    return 0.85 * spectral_score + 0.15 * regularity_score


def _detect_bcg_peaks(values: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, dict]:
    spectral = _bcg_spectral_hr(values, sampling_rate)
    spectral_candidates = spectral.get("spectral_hr_candidates_bpm") or []
    candidates: list[tuple[float, np.ndarray, dict]] = []
    candidate_fns = [
        lambda: (*neurokit_nabian2018_peaks(values, sampling_rate, low_hz=3.0, high_hz=12.0, fallback_threshold_scale=0.30), "nabian_3_12"),
        lambda: (*neurokit_nabian2018_peaks(values, sampling_rate, low_hz=1.0, high_hz=8.0, fallback_threshold_scale=0.30), "nabian_1_8"),
        lambda: (*scipy_adaptive_peaks(values, sampling_rate, low_hz=1.0, high_hz=min(12.0, sampling_rate * 0.45), min_bpm=35, max_bpm=180, threshold_scale=0.7), "adaptive_1_12"),
        lambda: (*_bcg_hilbert_peaks(values, sampling_rate, 2.5, 12.0, 0.7), "hilbert_2p5_12"),
        lambda: (*_bcg_hilbert_peaks(values, sampling_rate, 3.0, 10.0, 1.0), "hilbert_3_10"),
    ]
    for fn in candidate_fns:
        try:
            peaks, details, name = fn()
            details = dict(details)
            details["candidate"] = name
            details["heart_rate_bpm"] = bpm_from_peaks(peaks, sampling_rate)
            details["num_peaks"] = int(len(peaks))
            score = _score_peak_candidate(peaks, sampling_rate, spectral_candidates)
            candidates.append((score, peaks, details))
        except Exception as exc:
            candidates.append((-1.0, np.asarray([], dtype=int), {"candidate": getattr(fn, "__name__", "candidate"), "error": str(exc)}))
    candidates.sort(key=lambda item: item[0], reverse=True)
    score, peaks, details = candidates[0]
    candidate_hrs = [d.get("heart_rate_bpm") for _, _, d in candidates if d.get("heart_rate_bpm") is not None]
    candidate_hrs = [float(hr) for hr in candidate_hrs if 35 <= float(hr) <= 180]
    harmonic_ambiguity = False
    if candidate_hrs:
        low_hr = min(candidate_hrs)
        high_hr = max(candidate_hrs)
        selected_hr = details.get("heart_rate_bpm")
        harmonic_ambiguity = bool(selected_hr is not None and high_hr / max(low_hr, 1e-8) > 1.45 and float(selected_hr) > low_hr * 1.25)
    details.update(spectral)
    details["peak_detector_selected"] = details.get("candidate")
    details["peak_detector_score"] = float(score)
    details["candidate_hr_min_bpm"] = float(min(candidate_hrs)) if candidate_hrs else None
    details["candidate_hr_max_bpm"] = float(max(candidate_hrs)) if candidate_hrs else None
    details["harmonic_ambiguity"] = harmonic_ambiguity
    details["all_peak_candidates"] = [candidate_details | {"score": float(candidate_score)} for candidate_score, _, candidate_details in candidates]
    return peaks.astype(int), details


def _bcg_feature_summary(values: np.ndarray, sampling_rate: float, peaks: np.ndarray | None = None) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    duration_s = float(len(values) / sampling_rate) if sampling_rate else 0.0
    centered = values - float(np.nanmedian(values)) if len(values) else np.zeros(1)
    quality = signal_quality_summary(values)
    if peaks is None:
        peaks, _ = _detect_bcg_peaks(values, sampling_rate)
    intervals = _clean_intervals(peaks, sampling_rate)
    diff = np.diff(intervals) if len(intervals) > 1 else np.asarray([])
    features = {
        "duration_s": duration_s,
        "num_samples": int(len(values)),
        "dynamic_range": float(np.nanpercentile(values, 95) - np.nanpercentile(values, 5)) if len(values) else 0.0,
        "motion_energy": float(np.nanmean(np.abs(np.diff(centered)))) if len(centered) > 1 else 0.0,
        "peak_count": int(len(peaks)),
        "clean_interval_count": int(len(intervals)),
        "heart_rate_bpm": bpm_from_peaks(peaks, sampling_rate),
        "interval_cv": float(np.std(intervals) / np.mean(intervals)) if len(intervals) > 1 and np.mean(intervals) > 0 else None,
        "rmssd_ms": float(np.sqrt(np.mean((diff * 1000.0) ** 2))) if len(diff) else None,
        "sdnn_ms": float(np.std(intervals * 1000.0, ddof=1)) if len(intervals) > 1 else None,
        "quality": quality.get("quality"),
        "quality_confidence": quality.get("confidence"),
    }
    return features


def BCG_assess_quality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    spectral = _bcg_spectral_hr(data.values, data.sampling_rate) if len(data.values) >= int(data.sampling_rate * 8) else {}
    return {"tool": "BCG_assess_quality", "source": data.source, **signal_quality_summary(data.values), **spectral}


def BCG_detect_j_peaks(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    peaks, details = _detect_bcg_peaks(data.values, data.sampling_rate)
    heart_rate = bpm_from_peaks(peaks, data.sampling_rate)
    regularity = interval_regularity(peaks, data.sampling_rate)
    confidence = 0.78 * max(0.0, float(details.get("peak_detector_score", 0.0))) + 0.22 * float(regularity.get("regularity_confidence", 0.3))
    if details.get("harmonic_ambiguity"):
        confidence *= 0.65
    if heart_rate is None or not 35 <= heart_rate <= 180:
        confidence = 0.2
    return {"tool": "BCG_detect_j_peaks", "j_peak_indices": peaks.tolist(), "num_peaks": int(len(peaks)), "heart_rate_bpm": heart_rate, "confidence": float(confidence), **regularity, **details}


def BCG_compute_hrv(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    peak_result = BCG_detect_j_peaks(signal_path, sampling_rate, column)
    peaks = np.asarray(peak_result.get("j_peak_indices", []), dtype=float)
    intervals = _clean_intervals(peaks, data.sampling_rate)
    if len(intervals) < 3:
        return {"tool": "BCG_compute_hrv", "error": "not enough clean BCG beat intervals", "confidence": 0.1, "num_peaks": int(len(peaks))}
    rr_ms = intervals * 1000.0
    diff_ms = np.diff(rr_ms)
    mean_rr = float(np.mean(rr_ms))
    median_rr = float(np.median(rr_ms))
    duration_s = float((peaks[-1] - peaks[0]) / data.sampling_rate) if len(peaks) > 1 else 0.0
    metrics = {
        "tool": "BCG_compute_hrv",
        "num_clean_intervals": int(len(intervals)),
        "mean_hr_bpm": float(60000.0 / mean_rr) if mean_rr > 0 else None,
        "median_hr_bpm": float(60000.0 / median_rr) if median_rr > 0 else None,
        "mean_nn_ms": mean_rr,
        "sdnn_ms": float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else 0.0,
        "rmssd_ms": float(np.sqrt(np.mean(diff_ms ** 2))) if len(diff_ms) else None,
        "pnn50": float(np.mean(np.abs(diff_ms) > 50.0)) if len(diff_ms) else None,
        "interval_cv": float(np.std(intervals) / np.mean(intervals)) if np.mean(intervals) > 0 else None,
        "duration_s": duration_s,
        "confidence": float(peak_result.get("confidence", 0.5)) * (0.75 if duration_s < 120 else 1.0),
        "method": "BCG beat-to-beat interval HRV proxy",
        "disclaimer": "BCG HRV depends on J-peak reliability and is not interchangeable with ECG HRV without validation.",
    }
    if len(intervals) >= 8 and duration_s >= 30:
        t = np.cumsum(intervals) - intervals[0]
        grid = np.arange(0, t[-1], 0.25)
        if len(grid) >= 16:
            tach = np.interp(grid, t, rr_ms)
            tach = scipy_signal.detrend(tach, type="constant")
            freqs, psd = scipy_signal.welch(tach, fs=4.0, nperseg=min(256, len(tach)))
            def band(lo: float, hi: float) -> float:
                mask = (freqs >= lo) & (freqs < hi)
                return _safe_trapz(psd[mask], freqs[mask]) if np.any(mask) else 0.0
            lf = band(0.04, 0.15)
            hf = band(0.15, 0.40)
            metrics.update({"lf_power_ms2": float(lf), "hf_power_ms2": float(hf), "lf_hf_ratio": float(lf / (hf + 1e-12)), "frequency_method": "welch_interpolated_bcg_interval_tachogram_4hz"})
    return metrics


def BCG_estimate_respiration(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    if len(values) < data.sampling_rate * 20:
        return {"tool": "BCG_estimate_respiration", "error": "signal too short", "confidence": 0.0}
    high = min(0.7, data.sampling_rate * 0.45)
    if high <= 0.08:
        return {"tool": "BCG_estimate_respiration", "error": "sampling rate too low", "confidence": 0.1}
    raw_resp = scipy_signal.sosfiltfilt(scipy_signal.butter(3, [0.08 / (0.5 * data.sampling_rate), high / (0.5 * data.sampling_rate)], btype="bandpass", output="sos"), values - np.nanmedian(values))
    cardiac = bandpass_filter(values - np.nanmedian(values), data.sampling_rate, 1.0, min(12.0, data.sampling_rate * 0.45))
    envelope = np.abs(scipy_signal.hilbert(cardiac))
    envelope_resp = scipy_signal.sosfiltfilt(scipy_signal.butter(3, [0.08 / (0.5 * data.sampling_rate), high / (0.5 * data.sampling_rate)], btype="bandpass", output="sos"), envelope - np.nanmedian(envelope))
    candidates = []
    for source, filtered in [("baseline", raw_resp), ("cardiac_amplitude_envelope", envelope_resp)]:
        freqs, psd = scipy_signal.welch(filtered, fs=data.sampling_rate, nperseg=min(len(filtered), int(data.sampling_rate * 32)))
        mask = (freqs >= 0.08) & (freqs <= high)
        if np.any(mask):
            rate = float(freqs[mask][np.argmax(psd[mask])] * 60.0)
            ratio = float(_safe_trapz(psd[mask], freqs[mask]) / (_safe_trapz(psd, freqs) + 1e-12))
            candidates.append((ratio, rate, source))
    if not candidates:
        respiratory_rate = None
        ratio = 0.0
        source = "none"
    else:
        ratio, respiratory_rate, source = max(candidates, key=lambda item: item[0])
    return {
        "tool": "BCG_estimate_respiration",
        "respiratory_rate_bpm": respiratory_rate,
        "respiration_power_ratio": float(ratio),
        "respiration_source": source,
        "confidence": 0.65 if respiratory_rate is not None and 5 <= respiratory_rate <= 40 and ratio > 0.2 else 0.45,
        "method": "respiratory-band baseline plus Hilbert cardiac-envelope modulation",
        "disclaimer": "Mechanical-signal respiration proxy only; validate against respiratory reference signals.",
    }


def BCG_screen_arrhythmia(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    hrv = BCG_compute_hrv(signal_path, sampling_rate, column)
    if "error" in hrv:
        return {"tool": "BCG_screen_arrhythmia", **hrv, "arrhythmia_risk": "unknown"}
    interval_cv = hrv.get("interval_cv") or 0.0
    rmssd = hrv.get("rmssd_ms") or 0.0
    pnn50 = hrv.get("pnn50") or 0.0
    score = float(min(1.0, 0.55 * min(interval_cv / 0.35, 1.0) + 0.25 * min(rmssd / 180.0, 1.0) + 0.20 * min(pnn50 / 0.45, 1.0)))
    if score >= 0.65:
        risk = "elevated_irregularity"
    elif score >= 0.38:
        risk = "borderline_irregularity"
    else:
        risk = "low_irregularity"
    return {
        "tool": "BCG_screen_arrhythmia",
        "arrhythmia_risk": risk,
        "irregularity_score": score,
        "interval_cv": hrv.get("interval_cv"),
        "rmssd_ms": hrv.get("rmssd_ms"),
        "pnn50": hrv.get("pnn50"),
        "confidence": float(hrv.get("confidence", 0.4)) * 0.75,
        "method": "BCG beat-interval irregularity proxy inspired by AF screening literature",
        "disclaimer": "BCG cannot confirm AF or rhythm subtype; route elevated cases to ECG-grade rhythm assessment.",
    }


def BCG_assess_bed_presence_motion(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    if len(values) < 2:
        return {"tool": "BCG_assess_bed_presence_motion", "presence_state": "unknown", "confidence": 0.0}
    dynamic_range = float(np.nanpercentile(values, 95) - np.nanpercentile(values, 5))
    motion = np.abs(np.diff(values - np.nanmedian(values)))
    motion_energy = float(np.nanmedian(motion))
    burst_threshold = float(np.nanmedian(motion) + 6.0 * (np.nanmedian(np.abs(motion - np.nanmedian(motion))) + 1e-8))
    burst_fraction = float(np.mean(motion > burst_threshold))
    peaks, details = _detect_bcg_peaks(values, data.sampling_rate)
    hr = bpm_from_peaks(peaks, data.sampling_rate)
    if dynamic_range <= 1e-8 or len(peaks) < 3:
        state = "likely_empty_or_no_cardiac_signal"
        confidence = 0.45
    elif burst_fraction > 0.08:
        state = "present_with_motion_artifact"
        confidence = 0.65
    else:
        state = "present_quiet"
        confidence = 0.75
    return {"tool": "BCG_assess_bed_presence_motion", "presence_state": state, "motion_burst_fraction": burst_fraction, "motion_energy": motion_energy, "dynamic_range": dynamic_range, "heart_rate_bpm": hr, "num_peaks": int(len(peaks)), "confidence": confidence, "peak_detector_selected": details.get("peak_detector_selected")}


def BCG_estimate_sleep_features(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    hrv = BCG_compute_hrv(signal_path, sampling_rate, column)
    resp = BCG_estimate_respiration(signal_path, sampling_rate, column)
    motion = BCG_assess_bed_presence_motion(signal_path, sampling_rate, column)
    sleep_proxy_score = 0.0
    if motion.get("presence_state") == "present_quiet":
        sleep_proxy_score += 0.35
    hr = hrv.get("median_hr_bpm") or hrv.get("mean_hr_bpm") or 0.0
    if 40 <= hr <= 85:
        sleep_proxy_score += 0.25
    rr = resp.get("respiratory_rate_bpm") or 0.0
    if 8 <= rr <= 24:
        sleep_proxy_score += 0.25
    if (hrv.get("interval_cv") or 1.0) < 0.25:
        sleep_proxy_score += 0.15
    state = "sleep_compatible_quiet_rest" if sleep_proxy_score >= 0.65 else "wake_or_uncertain"
    return {"tool": "BCG_estimate_sleep_features", "sleep_wake_proxy": state, "sleep_proxy_score": float(sleep_proxy_score), "heart_rate_bpm": hr or None, "respiratory_rate_bpm": resp.get("respiratory_rate_bpm"), "motion_burst_fraction": motion.get("motion_burst_fraction"), "rmssd_ms": hrv.get("rmssd_ms"), "sdnn_ms": hrv.get("sdnn_ms"), "confidence": min(float(hrv.get("confidence", 0.4)), float(resp.get("confidence", 0.4))), "method": "BCG HRV/RRV/motion feature proxy", "disclaimer": "Sleep staging requires trained sleep labels/PSG validation; this tool exposes features and a conservative sleep-wake proxy only."}


def BCG_estimate_bp_proxy(signal_path: str, sampling_rate: float, column: str | None = None, calibration_sbp: float | None = None, calibration_dbp: float | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    peaks, details = _detect_bcg_peaks(data.values, data.sampling_rate)
    features = _bcg_feature_summary(data.values, data.sampling_rate, peaks)
    centered = data.values - float(np.nanmedian(data.values))
    morphology_amplitude = float(np.nanmedian(np.abs(centered[np.asarray(peaks, dtype=int)]))) if len(peaks) else None
    output = {"tool": "BCG_estimate_bp_proxy", "heart_rate_bpm": features.get("heart_rate_bpm"), "morphology_amplitude_proxy": morphology_amplitude, "interval_cv": features.get("interval_cv"), "confidence": 0.25, "method": "BCG morphology/timing BP proxy", "disclaimer": "Cuffless BP from BCG needs subject/device calibration and external validation; without calibration this is not a BP estimate."}
    if calibration_sbp is not None and calibration_dbp is not None and morphology_amplitude is not None:
        # Conservative placeholder: report calibration baseline plus a relative variation index, not an unvalidated absolute predictor.
        variation_index = float(np.tanh((morphology_amplitude / (features.get("dynamic_range") or 1.0) - 0.25) * 2.0))
        output.update({"calibrated_baseline_sbp": float(calibration_sbp), "calibrated_baseline_dbp": float(calibration_dbp), "relative_bp_variation_index": variation_index, "confidence": 0.4})
    return output


def BCG_route_task_recommendation(task: str) -> dict:
    text = task.lower()
    mapping = []
    if any(k in text for k in ["heart rate", "hr", "j-peak", "j peak", "heartbeat", "心率", "峰"]):
        mapping.append("BCG_detect_j_peaks")
    if any(k in text for k in ["hrv", "recovery", "stress", "恢复", "压力"]):
        mapping.append("BCG_compute_hrv")
    if any(k in text for k in ["resp", "breath", "呼吸"]):
        mapping.append("BCG_estimate_respiration")
    if any(k in text for k in ["sleep", "睡眠"]):
        mapping.append("BCG_estimate_sleep_features")
    if any(k in text for k in ["af", "arrhythmia", "irregular", "房颤", "心律"]):
        mapping.append("BCG_screen_arrhythmia")
    if any(k in text for k in ["motion", "bed", "presence", "离床", "体动"]):
        mapping.append("BCG_assess_bed_presence_motion")
    if any(k in text for k in ["blood pressure", "bp", "hypertension", "血压"]):
        mapping.append("BCG_estimate_bp_proxy")
    if any(k in text for k in ["heart failure", "cardiac function", "心衰", "心功能"]):
        return {"tool": "BCG_route_task_recommendation", "recommended_tools": ["BCG_compute_hrv", "BCG_estimate_bp_proxy"], "standalone_supported": False, "reason": "BCG cardiac-function monitoring is investigational and should be longitudinal, calibrated, and clinically validated."}
    return {"tool": "BCG_route_task_recommendation", "recommended_tools": sorted(set(mapping)), "standalone_supported": bool(mapping), "reason": "Use BCG tools as unobtrusive monitoring/proxy features; route diagnostic decisions to ECG/clinical references."}
