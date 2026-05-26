from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from scipy import signal as scipy_signal

from .common import load_csv_signal

SPO2_APNEA_MODEL_PATH = Path('/data1/jiahui/biosignal-agent/outputs/spo2_ucddb_event_feature_model.joblib')


def _spo2_values(signal_path: str, sampling_rate: float, column: str | None = None):
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values[np.isfinite(data.values)]
    return data, values


def _clean_spo2(values: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, dict]:
    raw = np.asarray(values, dtype=float).ravel()
    finite = np.isfinite(raw)
    if not finite.any():
        return np.asarray([], dtype=float), {"valid_fraction": 0.0, "artifact_fraction": 1.0, "interpolated_fraction": 1.0}
    plausible = finite & (raw >= 50.0) & (raw <= 100.0)
    valid_fraction = float(plausible.mean())
    x = raw.copy()
    if plausible.sum() == 0:
        x = np.asarray([], dtype=float)
        return x, {"valid_fraction": valid_fraction, "artifact_fraction": 1.0, "interpolated_fraction": 1.0}
    idx = np.arange(len(raw))
    x[~plausible] = np.interp(idx[~plausible], idx[plausible], raw[plausible])
    if len(x) >= max(5, int(sampling_rate * 3)):
        kernel = max(3, int(round(sampling_rate * 1.5)) | 1)
        kernel = min(kernel, len(x) if len(x) % 2 else len(x) - 1)
        if kernel >= 3:
            x = scipy_signal.medfilt(x, kernel_size=kernel)
    jumps = np.abs(np.diff(x)) if len(x) > 1 else np.asarray([])
    jump_fraction = float(np.mean(jumps > 4.0)) if len(jumps) else 0.0
    return x.astype(float), {"valid_fraction": valid_fraction, "artifact_fraction": float(1.0 - valid_fraction), "interpolated_fraction": float(1.0 - valid_fraction), "jump_fraction": jump_fraction}


def _rolling_baseline(values: np.ndarray, sampling_rate: float, window_s: float = 120.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    window = max(1, int(round(window_s * sampling_rate)))
    baseline = np.empty(len(values), dtype=float)
    for idx in range(len(values)):
        left = max(0, idx - window)
        segment = values[left:idx + 1]
        baseline[idx] = np.nanpercentile(segment, 90)
    baseline = np.maximum.accumulate(np.minimum(baseline, 100.0)) if len(values) < window else baseline
    return baseline


def _desaturation_events(values: np.ndarray, sampling_rate: float, threshold_percent: float = 3.0, min_duration_s: float = 10.0, baseline_window_s: float = 120.0, recovery_percent: float = 1.0) -> tuple[list[dict], np.ndarray]:
    x, _ = _clean_spo2(values, sampling_rate)
    if len(x) == 0:
        return [], np.asarray([], dtype=float)
    baseline = _rolling_baseline(x, sampling_rate, baseline_window_s)
    min_len = max(1, int(round(min_duration_s * sampling_rate)))
    max_gap = max(0, int(round(3.0 * sampling_rate)))
    below = x <= baseline - float(threshold_percent)
    events = []
    start = None
    last_below = None
    for idx, flag in enumerate(below):
        if flag:
            if start is None:
                start = idx
            last_below = idx
        elif start is not None:
            recovered = x[idx] >= baseline[start] - float(recovery_percent)
            gap = idx - (last_below if last_below is not None else idx)
            if recovered or gap > max_gap:
                end = (last_below + 1) if last_below is not None else idx
                if end - start >= min_len:
                    segment = x[start:end]
                    base = float(baseline[start])
                    nadir = float(np.nanmin(segment))
                    area = float(np.trapezoid(np.maximum(0.0, base - segment), dx=1.0 / sampling_rate))
                    events.append({
                        "start_s": float(start / sampling_rate),
                        "end_s": float(end / sampling_rate),
                        "duration_s": float((end - start) / sampling_rate),
                        "baseline_spo2_percent": base,
                        "nadir_spo2_percent": nadir,
                        "depth_percent": float(base - nadir),
                        "desaturation_area_percent_seconds": area,
                    })
                start = None
                last_below = None
    if start is not None:
        end = (last_below + 1) if last_below is not None else len(x)
        if end - start >= min_len:
            segment = x[start:end]
            base = float(baseline[start])
            nadir = float(np.nanmin(segment))
            area = float(np.trapezoid(np.maximum(0.0, base - segment), dx=1.0 / sampling_rate))
            events.append({"start_s": float(start / sampling_rate), "end_s": float(end / sampling_rate), "duration_s": float((end - start) / sampling_rate), "baseline_spo2_percent": base, "nadir_spo2_percent": nadir, "depth_percent": float(base - nadir), "desaturation_area_percent_seconds": area})
    return events, baseline


def _odi(events: list[dict], values: np.ndarray, sampling_rate: float) -> float | None:
    duration_hours = len(values) / float(sampling_rate) / 3600.0 if sampling_rate else 0.0
    return float(len(events) / duration_hours) if duration_hours > 0 else None


def _severity_from_odi(odi: float | None) -> str:
    if odi is None:
        return "unknown"
    if odi >= 30:
        return "severe_oximetry_sdb_burden"
    if odi >= 15:
        return "moderate_oximetry_sdb_burden"
    if odi >= 5:
        return "mild_oximetry_sdb_burden"
    return "low_oximetry_sdb_burden"


def SpO2_assess_quality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data, values = _spo2_values(signal_path, sampling_rate, column)
    if len(values) == 0:
        return {"tool": "SpO2_assess_quality", "source": data.source, "quality": "bad", "reason": "empty signal", "confidence": 0.0}
    cleaned, clean_info = _clean_spo2(values, sampling_rate)
    dynamic_range = float(np.nanpercentile(cleaned, 95) - np.nanpercentile(cleaned, 5)) if len(cleaned) else 0.0
    if clean_info["valid_fraction"] < 0.90 or clean_info["jump_fraction"] > 0.1:
        quality = "bad"
        confidence = 0.25
    elif clean_info["valid_fraction"] < 0.98 or clean_info["jump_fraction"] > 0.03:
        quality = "moderate"
        confidence = 0.65
    else:
        quality = "good"
        confidence = 0.9
    return {"tool": "SpO2_assess_quality", "source": data.source, "quality": quality, "finite_ratio": 1.0, "plausible_ratio": clean_info["valid_fraction"], "dynamic_range": dynamic_range, "jump_fraction": clean_info["jump_fraction"], "artifact_fraction": clean_info["artifact_fraction"], "confidence": confidence}



def _spo2_event_point(sample_index: int | None, values: np.ndarray, sampling_rate: float) -> dict | None:
    if sample_index is None:
        return None
    idx = int(sample_index)
    if idx < 0 or idx >= len(values):
        return None
    return {
        "sample_index": idx,
        "time_s": float(idx / float(sampling_rate)),
        "amplitude": float(values[idx]),
        "spo2_percent": float(values[idx]),
    }


def SpO2_detect_peaks_troughs(signal_path: str, sampling_rate: float, column: str | None = None, min_distance_s: float = 5.0) -> dict:
    data, values = _spo2_values(signal_path, sampling_rate, column)
    if len(values) == 0:
        return {"tool": "SpO2_detect_peaks_troughs", "error": "empty signal", "confidence": 0.0}
    target, clean_info = _clean_spo2(values, sampling_rate)
    if len(target) == 0:
        return {"tool": "SpO2_detect_peaks_troughs", "error": "no plausible SpO2 samples", "confidence": 0.0}
    fs = float(sampling_rate)
    min_distance = max(1, int(round(float(min_distance_s) * fs)))
    dynamic_range = float(np.nanpercentile(target, 95) - np.nanpercentile(target, 5)) if len(target) else 0.0
    prominence = max(0.15, 0.15 * dynamic_range)
    peaks, _ = scipy_signal.find_peaks(target, distance=min_distance, prominence=prominence)
    troughs, _ = scipy_signal.find_peaks(-target, distance=min_distance, prominence=prominence)
    if len(peaks) == 0 and dynamic_range > 0:
        peaks, _ = scipy_signal.find_peaks(target, distance=min_distance, prominence=max(0.05, 0.05 * dynamic_range))
    if len(troughs) == 0 and dynamic_range > 0:
        troughs, _ = scipy_signal.find_peaks(-target, distance=min_distance, prominence=max(0.05, 0.05 * dynamic_range))
    nadir_points = [_spo2_event_point(int(x), target, fs) for x in troughs[:5000]]
    peak_points = [_spo2_event_point(int(x), target, fs) for x in peaks[:5000]]
    cycles = []
    for i, trough_idx in enumerate(troughs[:5000]):
        prior_peaks = peaks[peaks < trough_idx]
        next_peaks = peaks[peaks > trough_idx]
        cycles.append({
            "cycle_index": int(i),
            "pre_trough_peak": _spo2_event_point(int(prior_peaks[-1]), target, fs) if len(prior_peaks) else None,
            "trough": _spo2_event_point(int(trough_idx), target, fs),
            "post_trough_peak": _spo2_event_point(int(next_peaks[0]), target, fs) if len(next_peaks) else None,
        })
    confidence = 0.8 if clean_info["valid_fraction"] > 0.95 else 0.45
    if dynamic_range < 0.5:
        confidence = min(confidence, 0.45)
    return {
        "tool": "SpO2_detect_peaks_troughs",
        "source": data.source,
        "peak_indices": peaks.tolist(),
        "trough_indices": troughs.tolist(),
        "peak_points": peak_points,
        "trough_points": nadir_points,
        "nadir_points": nadir_points,
        "cycles": cycles,
        "num_peaks": int(len(peaks)),
        "num_troughs": int(len(troughs)),
        "min_spo2_percent": float(np.nanmin(target)),
        "max_spo2_percent": float(np.nanmax(target)),
        "dynamic_range_percent": dynamic_range,
        "artifact_fraction": clean_info["artifact_fraction"],
        "confidence": float(confidence),
        "method": "cleaned_spo2_local_peak_trough_detection",
        "limitation": "SpO2 extrema are slow oximetry maxima/nadirs, not cardiac pulse peaks; event interpretation needs desaturation duration and clinical context.",
    }

def SpO2_summarize(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data, values = _spo2_values(signal_path, sampling_rate, column)
    if len(values) == 0:
        return {"tool": "SpO2_summarize", "error": "empty signal", "confidence": 0.0}
    target, clean_info = _clean_spo2(values, sampling_rate)
    if len(target) == 0:
        return {"tool": "SpO2_summarize", "error": "no plausible SpO2 samples", "confidence": 0.0}
    duration_h = len(target) / float(sampling_rate) / 3600.0 if sampling_rate else 0.0
    return {
        "tool": "SpO2_summarize",
        "mean_spo2_percent": float(np.mean(target)),
        "median_spo2_percent": float(np.median(target)),
        "min_spo2_percent": float(np.min(target)),
        "nadir_spo2_percent": float(np.min(target)),
        "time_below_90_fraction": float(np.mean(target < 90)),
        "time_below_88_fraction": float(np.mean(target < 88)),
        "time_below_85_fraction": float(np.mean(target < 85)),
        "ct90_minutes": float(np.sum(target < 90) / sampling_rate / 60.0) if sampling_rate else None,
        "recording_duration_hours": duration_h,
        "num_samples": int(len(target)),
        "artifact_fraction": clean_info["artifact_fraction"],
        "confidence": 0.85 if clean_info["valid_fraction"] > 0.95 else 0.4,
    }


def SpO2_detect_desaturation(signal_path: str, sampling_rate: float, column: str | None = None, threshold_percent: float = 3.0, min_duration_s: float = 10.0, baseline_window_s: float = 120.0) -> dict:
    data, values = _spo2_values(signal_path, sampling_rate, column)
    if len(values) == 0:
        return {"tool": "SpO2_detect_desaturation", "error": "empty signal", "confidence": 0.0}
    target, clean_info = _clean_spo2(values, sampling_rate)
    if len(target) == 0:
        return {"tool": "SpO2_detect_desaturation", "error": "no plausible SpO2 samples", "confidence": 0.0}
    events3, baseline3 = _desaturation_events(target, sampling_rate, 3.0, min_duration_s, baseline_window_s)
    events4, _ = _desaturation_events(target, sampling_rate, 4.0, min_duration_s, baseline_window_s)
    selected = events3 if float(threshold_percent) <= 3.5 else events4
    odi3 = _odi(events3, target, sampling_rate)
    odi4 = _odi(events4, target, sampling_rate)
    depths = [event["depth_percent"] for event in selected]
    durations = [event["duration_s"] for event in selected]
    areas = [event["desaturation_area_percent_seconds"] for event in selected]
    return {
        "tool": "SpO2_detect_desaturation",
        "baseline_spo2_percent": float(np.nanmedian(baseline3)) if len(baseline3) else None,
        "min_spo2_percent": float(np.nanmin(target)),
        "time_below_90_fraction": float(np.mean(target < 90)),
        "desaturation_event_count": int(len(selected)),
        "desaturation_event_count_3pct": int(len(events3)),
        "desaturation_event_count_4pct": int(len(events4)),
        "oxygen_desaturation_index_per_hour": _odi(selected, target, sampling_rate),
        "odi3_per_hour": odi3,
        "odi4_per_hour": odi4,
        "mean_desaturation_depth_percent": float(np.mean(depths)) if depths else 0.0,
        "max_desaturation_depth_percent": float(np.max(depths)) if depths else 0.0,
        "median_desaturation_duration_s": float(np.median(durations)) if durations else 0.0,
        "desaturation_area_percent_minutes_per_hour": float(np.sum(areas) / 60.0 / (len(target) / sampling_rate / 3600.0)) if areas and sampling_rate else 0.0,
        "event_intervals_s": [[event["start_s"], event["end_s"]] for event in selected[:20]],
        "events": selected[:20],
        "artifact_fraction": clean_info["artifact_fraction"],
        "confidence": 0.75 if clean_info["valid_fraction"] > 0.95 else 0.45,
        "method": "rolling_baseline_odi3_odi4_desaturation_detection",
        "disclaimer": "ODI-style screening heuristic only; sleep-apnea diagnosis requires PSG/HSAT respiratory-event scoring and clinical context.",
    }


def SpO2_assess_hypoxemia_burden(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data, values = _spo2_values(signal_path, sampling_rate, column)
    if len(values) == 0:
        return {"tool": "SpO2_assess_hypoxemia_burden", "error": "empty signal", "confidence": 0.0}
    target, clean_info = _clean_spo2(values, sampling_rate)
    if len(target) == 0:
        return {"tool": "SpO2_assess_hypoxemia_burden", "error": "no plausible SpO2 samples", "confidence": 0.0}
    duration_h = len(target) / float(sampling_rate) / 3600.0 if sampling_rate else 0.0
    time_below_90_fraction = float(np.mean(target < 90))
    time_below_88_fraction = float(np.mean(target < 88))
    time_below_85_fraction = float(np.mean(target < 85))
    nadir = float(np.nanmin(target))
    mean_spo2 = float(np.nanmean(target))
    hypoxic_area_90 = float(np.trapezoid(np.maximum(0.0, 90.0 - target), dx=1.0 / sampling_rate) / 60.0 / max(duration_h, 1e-12)) if sampling_rate else 0.0
    if time_below_88_fraction > 0.05 or nadir < 85 or hypoxic_area_90 > 5:
        burden = "high_hypoxemia_burden_proxy"
    elif time_below_90_fraction > 0.05 or hypoxic_area_90 > 1:
        burden = "moderate_hypoxemia_burden_proxy"
    else:
        burden = "low_hypoxemia_burden_proxy"
    return {"tool": "SpO2_assess_hypoxemia_burden", "mean_spo2_percent": mean_spo2, "min_spo2_percent": nadir, "nadir_spo2_percent": nadir, "time_below_90_fraction": time_below_90_fraction, "time_below_88_fraction": time_below_88_fraction, "time_below_85_fraction": time_below_85_fraction, "ct90_minutes": float(np.sum(target < 90) / sampling_rate / 60.0) if sampling_rate else None, "hypoxic_burden_percent_minutes_per_hour_below90": hypoxic_area_90, "hypoxemia_burden": burden, "num_samples": int(len(target)), "artifact_fraction": clean_info["artifact_fraction"], "confidence": 0.8 if clean_info["valid_fraction"] > 0.95 else 0.45, "method": "threshold_time_and_area_hypoxemia_burden", "disclaimer": "Screening heuristic only; clinical hypoxemia interpretation requires validated oximetry and clinical context."}


def SpO2_extract_oximetry_features(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data, values = _spo2_values(signal_path, sampling_rate, column)
    if len(values) == 0:
        return {"tool": "SpO2_extract_oximetry_features", "error": "empty signal", "confidence": 0.0}
    x, clean_info = _clean_spo2(values, sampling_rate)
    if len(x) == 0:
        return {"tool": "SpO2_extract_oximetry_features", "error": "no plausible SpO2 samples", "confidence": 0.0}
    desat = SpO2_detect_desaturation(signal_path, sampling_rate, column)
    burden = SpO2_assess_hypoxemia_burden(signal_path, sampling_rate, column)
    dx = np.diff(x, prepend=x[0]) * sampling_rate
    q = np.percentile(x, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    features = {
        "tool": "SpO2_extract_oximetry_features",
        "mean_spo2": float(np.mean(x)), "std_spo2": float(np.std(x)), "min_spo2": float(np.min(x)), "max_spo2": float(np.max(x)), "range_spo2": float(np.ptp(x)),
        "spo2_p01": float(q[0]), "spo2_p05": float(q[1]), "spo2_p10": float(q[2]), "spo2_p25": float(q[3]), "spo2_p50": float(q[4]), "spo2_p75": float(q[5]), "spo2_p90": float(q[6]), "spo2_p95": float(q[7]), "spo2_p99": float(q[8]),
        "time_below_95_fraction": float(np.mean(x < 95)), "time_below_93_fraction": float(np.mean(x < 93)), "time_below_90_fraction": float(np.mean(x < 90)), "time_below_88_fraction": float(np.mean(x < 88)), "time_below_85_fraction": float(np.mean(x < 85)),
        "odi3_per_hour": desat.get("odi3_per_hour") or 0.0, "odi4_per_hour": desat.get("odi4_per_hour") or 0.0, "desaturation_event_count_3pct": float(desat.get("desaturation_event_count_3pct") or 0), "desaturation_event_count_4pct": float(desat.get("desaturation_event_count_4pct") or 0),
        "mean_desaturation_depth_percent": float(desat.get("mean_desaturation_depth_percent") or 0.0), "max_desaturation_depth_percent": float(desat.get("max_desaturation_depth_percent") or 0.0), "desaturation_area_percent_minutes_per_hour": float(desat.get("desaturation_area_percent_minutes_per_hour") or 0.0),
        "hypoxic_burden_percent_minutes_per_hour_below90": float(burden.get("hypoxic_burden_percent_minutes_per_hour_below90") or 0.0),
        "delta_spo2_std_per_s": float(np.std(dx)), "delta_spo2_p95_abs_per_s": float(np.percentile(np.abs(dx), 95)), "artifact_fraction": clean_info["artifact_fraction"],
        "confidence": 0.8 if clean_info["valid_fraction"] > 0.95 else 0.45,
    }
    return features


def SpO2_screen_sleep_apnea_oximetry(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    desat = SpO2_detect_desaturation(signal_path, sampling_rate, column)
    if "error" in desat:
        return {"tool": "SpO2_screen_sleep_apnea_oximetry", **desat, "sleep_apnea_oximetry_risk": "unknown"}
    burden = SpO2_assess_hypoxemia_burden(signal_path, sampling_rate, column)
    odi3 = desat.get("odi3_per_hour")
    odi4 = desat.get("odi4_per_hour")
    severity = _severity_from_odi(odi3)
    risk_score = 0.0
    if odi3 is not None:
        risk_score += min(float(odi3) / 30.0, 1.0) * 0.55
    if odi4 is not None:
        risk_score += min(float(odi4) / 20.0, 1.0) * 0.25
    risk_score += min(float(burden.get("time_below_90_fraction") or 0.0) / 0.15, 1.0) * 0.20
    return {"tool": "SpO2_screen_sleep_apnea_oximetry", "sleep_apnea_oximetry_risk": severity, "oximetry_risk_score": float(min(risk_score, 1.0)), "odi3_per_hour": odi3, "odi4_per_hour": odi4, "nadir_spo2_percent": burden.get("nadir_spo2_percent"), "time_below_90_fraction": burden.get("time_below_90_fraction"), "hypoxic_burden_percent_minutes_per_hour_below90": burden.get("hypoxic_burden_percent_minutes_per_hour_below90"), "confidence": min(float(desat.get("confidence", 0.5)), float(burden.get("confidence", 0.5))), "method": "oximetry_only_odi_and_hypoxic_burden_screen", "disclaimer": "Oximetry-only screening can miss arousal-only hypopneas and cannot replace PSG/HSAT AHI scoring."}


def SpO2_screen_sleep_apnea_ml(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    features = SpO2_extract_oximetry_features(signal_path, sampling_rate, column)
    if "error" in features:
        return {"tool": "SpO2_screen_sleep_apnea_ml", **features}
    if not SPO2_APNEA_MODEL_PATH.exists():
        proxy = SpO2_screen_sleep_apnea_oximetry(signal_path, sampling_rate, column)
        return {"tool": "SpO2_screen_sleep_apnea_ml", **proxy, "model_error": f"missing model {SPO2_APNEA_MODEL_PATH}", "model_used": False}
    payload = joblib.load(SPO2_APNEA_MODEL_PATH)
    cols = payload["feature_columns"]
    x = np.asarray([[float(features.get(col, 0.0) or 0.0) for col in cols]], dtype=float)
    model = payload["model"]
    labels = list(payload.get("labels", getattr(model, "classes_", [])))
    proba = model.predict_proba(x)[0]
    pred = labels[int(np.argmax(proba))]
    pos_prob = float(proba[labels.index("respiratory_event")]) if "respiratory_event" in labels else float(np.max(proba))
    return {"tool": "SpO2_screen_sleep_apnea_ml", "prediction": pred, "respiratory_event_probability": pos_prob, "class_probabilities": {str(label): float(prob) for label, prob in zip(labels, proba)}, "model_used": True, "model_source": str(SPO2_APNEA_MODEL_PATH), "cv_metrics": payload.get("metrics"), "confidence": float(max(proba)), "disclaimer": "SpO2-only model trained on UCDDB window labels; use as screening evidence, not standalone AHI diagnosis."}
