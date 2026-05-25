from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew

from .common import bandpass_filter, load_csv_signal, signal_quality_summary


RESP_SPO2_EVENT_MODEL_PATH = Path("/data1/jiahui/biosignal-agent/outputs/resp_spo2/resp_spo2_ucddb_event_fusion_ensemble.joblib")
RESP_EVENT_FLOW_MODEL_PATH = Path("/data1/jiahui/biosignal-agent/outputs/resp_spo2/resp_ucddb_event_flow_ensemble.joblib")
RESP_SPO2_EVENT_METRICS = {
    "benchmark": "UCDDB ucddb002 Flow+SpO2 30 s windows, 5-fold stratified window CV",
    "accuracy": 0.6705,
    "balanced_accuracy": 0.6452613178863981,
    "macro_f1": 0.6472947264674266,
    "weighted_f1": 0.6634886600437644,
    "auroc": 0.7240962685402996,
    "caveat": "single UCDDB record window CV; respiratory-event labels combine apnea/hypopnea annotations and are not full AASM scoring",
}
RESP_EVENT_FLOW_METRICS = {
    "benchmark": "UCDDB ucddb002 Flow-only 30 s windows, 5-fold stratified window CV",
    "accuracy": 0.6165,
    "balanced_accuracy": 0.5874396539522853,
    "macro_f1": 0.586943204690645,
    "weighted_f1": 0.6067213966920452,
    "auroc": 0.6387879032066441,
}


def _read_numeric_series(path: str, column: str | None = None) -> np.ndarray:
    df = pd.read_csv(path)
    if column and column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    numeric = df.apply(pd.to_numeric, errors="coerce").select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        return np.array([], dtype=float)
    return numeric.iloc[:, 0].to_numpy(dtype=float)


def _clean_resp_values(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float).ravel()
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=float)
    x = np.where(finite, x, float(np.nanmedian(x[finite])))
    lo, hi = np.percentile(x, [0.5, 99.5])
    if hi > lo:
        x = np.clip(x, lo, hi)
    return x


def _resp_ml_features(values: np.ndarray, sampling_rate: float) -> list[float]:
    x = _clean_resp_values(values)
    n = len(x)
    if n == 0:
        return [0.0] * 35
    fs = float(sampling_rate)
    try:
        filtered = scipy_signal.sosfiltfilt(scipy_signal.butter(3, [0.05, 0.7], btype="bandpass", fs=fs, output="sos"), x)
    except Exception:
        filtered = x - np.nanmean(x)
    window = max(1, int(2.0 * fs))
    envelope = np.sqrt(np.convolve(filtered * filtered, np.ones(window) / window, mode="same"))
    baseline = float(np.percentile(envelope, 75) + 1e-12)
    low = envelope < baseline * 0.25
    reduced = (envelope < baseline * 0.7) & (envelope >= baseline * 0.25)
    peaks, _ = scipy_signal.find_peaks(filtered, distance=max(1, int(1.5 * fs)), prominence=max(np.std(filtered) * 0.2, 1e-8))
    if len(peaks) < 2:
        peaks, _ = scipy_signal.find_peaks(-filtered, distance=max(1, int(1.5 * fs)), prominence=max(np.std(filtered) * 0.2, 1e-8))
    intervals = np.diff(peaks) / fs if len(peaks) > 1 else np.array([])
    intervals = intervals[(intervals >= 1.0) & (intervals <= 15.0)]
    rate = float(60.0 / np.median(intervals)) if len(intervals) else 0.0
    cv = float(np.std(intervals) / (np.mean(intervals) + 1e-12)) if len(intervals) else 0.0
    def bp(lo: float, hi: float) -> float:
        if len(filtered) < 8:
            return 0.0
        freqs, psd = scipy_signal.welch(filtered - np.mean(filtered), fs=fs, nperseg=min(len(filtered), int(fs * 8)))
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.trapezoid(psd[mask], freqs[mask])) if np.any(mask) else 0.0
    bands = [bp(0.05, 0.15), bp(0.15, 0.35), bp(0.35, 0.7)]
    total = sum(bands) + 1e-12
    q = np.percentile(filtered, [1, 5, 25, 50, 75, 95, 99])
    eq = np.percentile(envelope, [1, 5, 25, 50, 75, 95, 99])
    feats = [float(np.mean(filtered)), float(np.std(filtered)), float(np.ptp(filtered)), *[float(v) for v in q], float(skew(filtered)), float(kurtosis(filtered)), float(np.mean(envelope)), float(np.std(envelope)), float(np.min(envelope)), float(np.max(envelope)), *[float(v) for v in eq], float(np.mean(low)), float(np.mean(reduced)), float(np.mean(envelope < baseline * 0.5)), float(np.mean(envelope < baseline * 0.8)), rate, cv, float(len(peaks)), float(len(peaks) / (n / fs / 60.0 + 1e-12)), *bands, *[b / total for b in bands]]
    return [0.0 if not np.isfinite(v) else float(v) for v in feats]


def _spo2_event_features(values: np.ndarray, sampling_rate: float) -> list[float]:
    x = _clean_resp_values(values)
    plausible = x[(x >= 50) & (x <= 100)]
    if len(plausible) == 0:
        plausible = x
    if len(plausible) == 0:
        return [0.0] * 23
    target = plausible
    fs = float(sampling_rate)
    baseline = float(np.percentile(target, 90))
    desat = target <= baseline - 3.0
    dx = np.diff(target, prepend=target[0]) * fs
    q = np.percentile(target, [1, 5, 25, 50, 75, 95, 99])
    min_len = max(1, int(10 * fs))
    events = 0
    start = None
    for idx, flag in enumerate(desat):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            if idx - start >= min_len:
                events += 1
            start = None
    if start is not None and len(desat) - start >= min_len:
        events += 1
    duration_h = len(target) / fs / 3600.0 if fs else 0.0
    odi = float(events / duration_h) if duration_h > 0 else 0.0
    feats = [float(np.mean(target)), float(np.std(target)), float(np.min(target)), float(np.max(target)), float(np.ptp(target)), *[float(v) for v in q], baseline, float(np.mean(target < 90)), float(np.mean(target < 88)), float(np.mean(desat)), float(events), odi, float(np.mean(dx)), float(np.std(dx)), float(np.percentile(np.abs(dx), 95)), float(np.mean(np.abs(dx) > 1)), float(len(target) / max(1, len(x)))]
    return [0.0 if not np.isfinite(v) else float(v) for v in feats]


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



def _merge_event_intervals(intervals: list[list[float]], gap_s: float = 2.0) -> list[list[float]]:
    if not intervals:
        return []
    ordered = sorted([[float(a), float(b)] for a, b in intervals if b > a], key=lambda x: x[0])
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + gap_s:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def RESP_summarize_event_burden(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    if len(data.values) < data.sampling_rate * 20:
        rate = RESP_estimate_rate(signal_path, sampling_rate, column)
        return {"tool": "RESP_summarize_event_burden", "error": "signal too short for respiratory event burden", "rate_fallback": rate, "confidence": 0.15}
    rate = RESP_estimate_rate(signal_path, sampling_rate, column)
    apnea = RESP_detect_apnea(signal_path, sampling_rate, column)
    hypopnea = RESP_detect_hypopnea(signal_path, sampling_rate, column)
    pattern = RESP_screen_rate_pattern(signal_path, sampling_rate, column)
    apnea_intervals = apnea.get("event_intervals_s", []) if not apnea.get("error") else []
    hypopnea_intervals = hypopnea.get("event_intervals_s", []) if not hypopnea.get("error") else []
    merged = _merge_event_intervals(apnea_intervals + hypopnea_intervals)
    duration_h = len(data.values) / data.sampling_rate / 3600.0
    rei = float(len(merged) / duration_h) if duration_h > 0 else None
    event_seconds = float(sum(max(0.0, end - start) for start, end in merged))
    burden_fraction = float(event_seconds / (len(data.values) / data.sampling_rate)) if len(data.values) else 0.0
    if rei is not None and rei >= 30:
        severity = "severe_event_burden_proxy"
    elif rei is not None and rei >= 15:
        severity = "moderate_event_burden_proxy"
    elif rei is not None and rei >= 5:
        severity = "mild_event_burden_proxy"
    elif len(merged):
        severity = "isolated_event_proxy"
    else:
        severity = "no_event_burden_proxy"
    flags = []
    if apnea.get("apnea_event_count", 0):
        flags.append("apnea_like_flow_drop")
    if hypopnea.get("hypopnea_event_count", 0):
        flags.append("hypopnea_like_flow_reduction")
    if pattern.get("respiratory_pattern_flags"):
        flags.extend(pattern.get("respiratory_pattern_flags", []))
    return {
        "tool": "RESP_summarize_event_burden",
        "respiratory_rate_bpm": rate.get("respiratory_rate_bpm"),
        "num_breaths": rate.get("num_breaths"),
        "respiratory_event_index_per_hour": rei,
        "respiratory_event_burden_fraction": burden_fraction,
        "event_burden_severity": severity,
        "event_flags": flags,
        "apnea_event_count": apnea.get("apnea_event_count", 0),
        "hypopnea_event_count": hypopnea.get("hypopnea_event_count", 0),
        "merged_event_count": int(len(merged)),
        "apnea_intervals_s": apnea_intervals[:20],
        "hypopnea_intervals_s": hypopnea_intervals[:20],
        "merged_event_intervals_s": merged[:20],
        "longest_event_s": float(max([end - start for start, end in merged], default=0.0)),
        "rate_result": rate,
        "pattern_result": pattern,
        "confidence": min(0.72, max(0.35, float(rate.get("confidence", 0.3)) - 0.05)),
        "method": "resp_flow_envelope_event_burden_summary",
        "disclaimer": "Standalone respiration event burden is a screen; AASM apnea/hypopnea scoring needs airflow/effort, SpO2 desaturation/arousal context, sleep time, and event-level labels.",
    }


def RESP_screen_rate_pattern(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    if len(values) < data.sampling_rate * 20:
        return {"tool": "RESP_screen_rate_pattern", "error": "signal too short for respiratory pattern screening", "confidence": 0.0}
    filtered = bandpass_filter(values, data.sampling_rate, low_hz=0.05, high_hz=0.7, order=3)
    min_distance = max(1, int(1.5 * data.sampling_rate))
    prominence = max(float(np.nanstd(filtered)) * 0.2, 1e-8)
    peaks, _ = scipy_signal.find_peaks(filtered, distance=min_distance, prominence=prominence)
    if len(peaks) < 3:
        return {"tool": "RESP_screen_rate_pattern", "error": "not enough breaths", "confidence": 0.1}
    intervals = np.diff(peaks) / data.sampling_rate
    intervals = intervals[(intervals >= 1.0) & (intervals <= 15.0)]
    if len(intervals) < 2:
        return {"tool": "RESP_screen_rate_pattern", "error": "not enough valid breath intervals", "confidence": 0.1}
    rate = float(60.0 / np.nanmedian(intervals))
    interval_cv = float(np.nanstd(intervals) / np.nanmean(intervals)) if np.nanmean(intervals) > 0 else None
    amps = filtered[peaks]
    amplitude_cv = float(np.nanstd(amps) / (np.nanmean(np.abs(amps)) + 1e-8)) if len(amps) else None
    flags = []
    if rate < 8:
        flags.append("bradypnea_proxy")
    if rate > 24:
        flags.append("tachypnea_proxy")
    if interval_cv is not None and interval_cv > 0.35:
        flags.append("irregular_breathing_proxy")
    if amplitude_cv is not None and amplitude_cv > 0.8:
        flags.append("periodic_or_variable_breathing_proxy")
    pattern_risk = "elevated" if flags else "low"
    return {
        "tool": "RESP_screen_rate_pattern",
        "respiratory_rate_bpm": rate,
        "breath_interval_cv": interval_cv,
        "breath_amplitude_cv": amplitude_cv,
        "respiratory_pattern_flags": flags,
        "respiratory_pattern_risk": pattern_risk,
        "confidence": 0.6,
        "method": "resp_rate_variability_pattern_screening",
        "disclaimer": "Screening heuristic only; respiratory pattern diagnosis requires validated airflow/effort signals and labels.",
    }



def RESP_screen_sleep_apnea_ml(
    signal_path: str,
    sampling_rate: float,
    column: str | None = None,
    spo2_path: str | None = None,
    spo2_sampling_rate: float | None = None,
    spo2_column: str | None = None,
) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    if len(data.values) < max(20, int(data.sampling_rate * 20)):
        fallback = RESP_detect_apnea(signal_path, sampling_rate, column)
        return {"tool": "RESP_screen_sleep_apnea_ml", "error": "signal too short for 30 s UCDDB respiratory-event classifier", "fallback_result": fallback, "confidence": 0.2}
    resp_feats = _resp_ml_features(data.values, data.sampling_rate)
    has_spo2 = spo2_path is not None and spo2_sampling_rate is not None
    if has_spo2 and RESP_SPO2_EVENT_MODEL_PATH.exists():
        spo2_values = _read_numeric_series(spo2_path, spo2_column)
        X = np.asarray([resp_feats + _spo2_event_features(spo2_values, float(spo2_sampling_rate))], dtype=float)
        model_path = RESP_SPO2_EVENT_MODEL_PATH
        metrics = RESP_SPO2_EVENT_METRICS
        method = "ucddb_resp_flow_spo2_event_fusion_ensemble"
    else:
        X = np.asarray([resp_feats], dtype=float)
        model_path = RESP_EVENT_FLOW_MODEL_PATH
        metrics = RESP_EVENT_FLOW_METRICS
        method = "ucddb_resp_flow_only_event_ensemble"
    if not model_path.exists():
        apnea = RESP_detect_apnea(signal_path, sampling_rate, column)
        hypopnea = RESP_detect_hypopnea(signal_path, sampling_rate, column)
        return {"tool": "RESP_screen_sleep_apnea_ml", "error": f"trained model not found: {model_path}", "apnea_fallback": apnea, "hypopnea_fallback": hypopnea, "confidence": 0.2}
    bundle = joblib.load(model_path)
    model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
    classes = list(getattr(model, "classes_", []))
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        probabilities = {str(cls): float(prob) for cls, prob in zip(classes, probs)}
        event_probability = probabilities.get("respiratory_event", float(np.max(probs)))
    else:
        pred = str(model.predict(X)[0])
        event_probability = 1.0 if pred == "respiratory_event" else 0.0
        probabilities = {pred: 1.0}
    risk = "respiratory_event_likely" if event_probability >= 0.5 else "normal_likely"
    return {
        "tool": "RESP_screen_sleep_apnea_ml",
        "respiratory_event_probability": float(event_probability),
        "respiratory_event_risk": risk,
        "class_probabilities": probabilities,
        "used_spo2": bool(has_spo2 and model_path == RESP_SPO2_EVENT_MODEL_PATH),
        "model_source": str(model_path),
        "model_metrics": metrics,
        "confidence": float(min(0.85, max(0.25, abs(event_probability - 0.5) * 1.2 + 0.35))),
        "method": method,
        "disclaimer": "Research respiratory-event screen trained on one UCDDB record. It is not full AASM apnea/hypopnea scoring, not subject-independent, and not diagnostic.",
    }
