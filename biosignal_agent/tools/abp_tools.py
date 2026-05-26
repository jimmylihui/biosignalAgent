from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from .common import bpm_from_peaks, interval_regularity, load_csv_signal, signal_quality_summary



def _abp_pulse_summary(values: np.ndarray, sampling_rate: float) -> dict:
    x = np.asarray(values, dtype=float).ravel()
    finite = np.isfinite(x)
    if not finite.any():
        return {"error": "empty or non-finite ABP signal"}
    x = np.where(finite, x, float(np.nanmedian(x[finite])))
    lo, hi = np.nanpercentile(x, [0.5, 99.5])
    if hi > lo:
        x = np.clip(x, lo, hi)
    fs = float(sampling_rate)
    if len(x) >= max(9, int(0.2 * fs)):
        kernel = max(3, int(0.06 * fs) | 1)
        smooth = scipy_signal.medfilt(x, kernel_size=kernel) if kernel < len(x) else x
        try:
            smooth = scipy_signal.savgol_filter(smooth, window_length=max(5, int(0.08 * fs) | 1), polyorder=2, mode="interp")
        except Exception:
            pass
    else:
        smooth = x
    min_distance = max(1, int(0.3 * fs))
    prominence = max(float(np.nanstd(smooth)) * 0.12, 2.0)
    peaks, props = scipy_signal.find_peaks(smooth, distance=min_distance, prominence=prominence)
    if len(peaks) == 0:
        return {"error": "no ABP pulses detected"}
    systolic_vals = smooth[peaks]
    plausible_peak = (systolic_vals >= 40) & (systolic_vals <= 250)
    peaks = peaks[plausible_peak]
    if len(peaks) == 0:
        return {"error": "no physiologically plausible ABP pulses detected"}
    diastolic_vals = []
    valid_peaks = []
    for i, peak in enumerate(peaks):
        left = peaks[i - 1] if i > 0 else max(0, peak - int(0.8 * fs))
        seg = smooth[left:peak + 1]
        if len(seg) == 0:
            continue
        dia = float(np.nanmin(seg))
        sys = float(smooth[peak])
        pp = sys - dia
        if 20 <= dia <= 180 and 5 <= pp <= 140 and sys > dia:
            diastolic_vals.append(dia)
            valid_peaks.append(int(peak))
    peaks = np.asarray(valid_peaks, dtype=int)
    if len(peaks) == 0:
        return {"error": "no ABP pulses passed systolic/diastolic plausibility checks"}
    systolic_vals = smooth[peaks]
    diastolic_vals = np.asarray(diastolic_vals, dtype=float)
    intervals = np.diff(peaks) / fs if len(peaks) > 1 else np.array([])
    intervals = intervals[(intervals >= 0.3) & (intervals <= 2.0)]
    heart_rate = float(60.0 / np.nanmedian(intervals)) if len(intervals) else None
    interval_cv = float(np.nanstd(intervals) / (np.nanmean(intervals) + 1e-12)) if len(intervals) else None
    pulse_pressure_vals = systolic_vals - diastolic_vals
    map_vals = diastolic_vals + pulse_pressure_vals / 3.0
    return {
        "pulse_indices": peaks.tolist(),
        "num_pulses": int(len(peaks)),
        "heart_rate_bpm": heart_rate,
        "median_systolic_value": float(np.nanmedian(systolic_vals)),
        "approx_diastolic_value": float(np.nanmedian(diastolic_vals)),
        "median_map_value": float(np.nanmedian(map_vals)),
        "median_pulse_pressure": float(np.nanmedian(pulse_pressure_vals)),
        "beat_systolic_values": [float(v) for v in systolic_vals],
        "beat_diastolic_values": [float(v) for v in diastolic_vals],
        "beat_map_values": [float(v) for v in map_vals],
        "beat_pulse_pressure_values": [float(v) for v in pulse_pressure_vals],
        "pulse_interval_cv": interval_cv,
        "artifact_rejected_fraction": float(1.0 - len(peaks) / max(1, len(plausible_peak))),
        "smoothed_values": smooth,
    }


def ABP_assess_quality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    return {"tool": "ABP_assess_quality", "source": data.source, **signal_quality_summary(data.values)}



def _fiducial_point(sample_index: int | None, values: np.ndarray, sampling_rate: float) -> dict | None:
    if sample_index is None:
        return None
    idx = int(max(0, min(len(values) - 1, sample_index)))
    return {
        "sample_index": idx,
        "time_s": float(idx / float(sampling_rate)),
        "amplitude": float(values[idx]),
    }


def _abp_fiducial_summary(values: np.ndarray, sampling_rate: float) -> dict:
    summary = _abp_pulse_summary(values, sampling_rate)
    if summary.get("error"):
        return summary
    smooth = np.asarray(summary.get("smoothed_values"), dtype=float)
    peaks = np.asarray(summary.get("pulse_indices", []), dtype=int)
    fs = float(sampling_rate)
    fiducials = []
    missing = {"systolic_onset": 0, "systolic_peak": 0, "dicrotic_notch": 0, "diastolic_peak": 0, "diastolic_phase_endpoint": 0}
    for beat_index, peak in enumerate(peaks):
        prev_peak = int(peaks[beat_index - 1]) if beat_index > 0 else max(0, int(peak - 0.8 * fs))
        next_peak = int(peaks[beat_index + 1]) if beat_index + 1 < len(peaks) else min(len(smooth) - 1, int(peak + 0.8 * fs))
        onset_left = max(0, prev_peak)
        onset_right = max(onset_left + 1, int(peak) + 1)
        onset_seg = smooth[onset_left:onset_right]
        systolic_onset = int(onset_left + int(np.nanargmin(onset_seg))) if len(onset_seg) else None
        systolic_peak = int(peak)

        notch_start = min(len(smooth) - 1, int(peak + 0.08 * fs))
        notch_limit = min(int(next_peak), int(peak + 0.45 * fs))
        notch_end = min(len(smooth), max(notch_start + 1, notch_limit))
        notch_seg = smooth[notch_start:notch_end]
        dicrotic_notch = None
        if len(notch_seg):
            valleys, _ = scipy_signal.find_peaks(-notch_seg, distance=max(1, int(0.06 * fs)))
            if len(valleys):
                dicrotic_notch = int(notch_start + valleys[0])
            else:
                dicrotic_notch = int(notch_start + int(np.nanargmin(notch_seg)))

        diastolic_peak = None
        if dicrotic_notch is not None:
            dia_start = min(len(smooth) - 1, int(dicrotic_notch + 0.03 * fs))
            dia_limit = min(int(next_peak), int(dicrotic_notch + 0.35 * fs))
            dia_end = min(len(smooth), max(dia_start + 1, dia_limit))
            dia_seg = smooth[dia_start:dia_end]
            if len(dia_seg):
                dia_peaks, _ = scipy_signal.find_peaks(dia_seg, distance=max(1, int(0.05 * fs)))
                if len(dia_peaks):
                    diastolic_peak = int(dia_start + dia_peaks[0])
                else:
                    diastolic_peak = int(dia_start + int(np.nanargmax(dia_seg)))

        end_left = int(diastolic_peak if diastolic_peak is not None else (dicrotic_notch if dicrotic_notch is not None else peak))
        end_right = min(len(smooth), max(end_left + 1, next_peak + 1))
        end_seg = smooth[end_left:end_right]
        diastolic_phase_endpoint = int(end_left + int(np.nanargmin(end_seg))) if len(end_seg) else None
        beat = {
            "beat_index": int(beat_index),
            "systolic_onset": _fiducial_point(systolic_onset, smooth, fs),
            "systolic_peak": _fiducial_point(systolic_peak, smooth, fs),
            "dicrotic_notch": _fiducial_point(dicrotic_notch, smooth, fs),
            "diastolic_peak": _fiducial_point(diastolic_peak, smooth, fs),
            "diastolic_phase_endpoint": _fiducial_point(diastolic_phase_endpoint, smooth, fs),
        }
        for key in missing:
            if beat[key] is None:
                missing[key] += 1
        fiducials.append(beat)
    n = max(1, len(fiducials))
    out = dict(summary)
    out["fiducials"] = fiducials
    out["missing_fiducial_fraction"] = {key: float(val / n) for key, val in missing.items()}
    out["systolic_onset_indices"] = [b["systolic_onset"]["sample_index"] for b in fiducials if b["systolic_onset"]]
    out["systolic_peak_indices"] = [b["systolic_peak"]["sample_index"] for b in fiducials if b["systolic_peak"]]
    out["dicrotic_notch_indices"] = [b["dicrotic_notch"]["sample_index"] for b in fiducials if b["dicrotic_notch"]]
    out["diastolic_peak_indices"] = [b["diastolic_peak"]["sample_index"] for b in fiducials if b["diastolic_peak"]]
    out["diastolic_phase_endpoint_indices"] = [b["diastolic_phase_endpoint"]["sample_index"] for b in fiducials if b["diastolic_phase_endpoint"]]
    return out


def ABP_detect_fiducial_points(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    summary = _abp_fiducial_summary(data.values, data.sampling_rate)
    if summary.get("error"):
        return {"tool": "ABP_detect_fiducial_points", "error": summary["error"], "confidence": 0.0, "method": "abp_fiducial_detection_from_smoothed_pulses"}
    peaks = np.asarray(summary["systolic_peak_indices"], dtype=int)
    regularity = interval_regularity(peaks, data.sampling_rate)
    heart_rate = summary.get("heart_rate_bpm")
    missing = summary.get("missing_fiducial_fraction", {})
    hard_missing = max(float(missing.get("dicrotic_notch", 0.0)), float(missing.get("diastolic_peak", 0.0)))
    confidence = min(0.86, max(0.35, float(regularity.get("regularity_confidence", 0.4)) + 0.12 - 0.25 * hard_missing)) if heart_rate is not None else 0.35
    return {
        "tool": "ABP_detect_fiducial_points",
        "fiducials": summary["fiducials"],
        "num_beats": int(len(summary["fiducials"])),
        "systolic_onset_indices": summary["systolic_onset_indices"],
        "systolic_peak_indices": summary["systolic_peak_indices"],
        "dicrotic_notch_indices": summary["dicrotic_notch_indices"],
        "diastolic_peak_indices": summary["diastolic_peak_indices"],
        "diastolic_phase_endpoint_indices": summary["diastolic_phase_endpoint_indices"],
        "heart_rate_bpm": heart_rate,
        "median_systolic_value": summary["median_systolic_value"],
        "approx_diastolic_value": summary["approx_diastolic_value"],
        "median_map_value": summary["median_map_value"],
        "median_pulse_pressure": summary["median_pulse_pressure"],
        "missing_fiducial_fraction": missing,
        "artifact_rejected_fraction": summary["artifact_rejected_fraction"],
        "confidence": confidence,
        **regularity,
        "method": "abp_fiducial_detection_from_smoothed_pulses",
        "limitations": ["Dicrotic notch and diastolic peak positions are less reliable in damped, noisy, or low-pulse-pressure ABP waveforms."],
    }

def ABP_detect_pulses(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    fid = ABP_detect_fiducial_points(signal_path, sampling_rate, column)
    if fid.get("error"):
        return {"tool": "ABP_detect_pulses", "error": fid["error"], "confidence": 0.0, "method": "legacy_wrapper_over_abp_fiducial_detection"}
    return {
        "tool": "ABP_detect_pulses",
        "pulse_indices": fid.get("systolic_peak_indices", []),
        "num_pulses": fid.get("num_beats", 0),
        "heart_rate_bpm": fid.get("heart_rate_bpm"),
        "median_systolic_value": fid.get("median_systolic_value"),
        "approx_diastolic_value": fid.get("approx_diastolic_value"),
        "median_map_value": fid.get("median_map_value"),
        "median_pulse_pressure": fid.get("median_pulse_pressure"),
        "artifact_rejected_fraction": fid.get("artifact_rejected_fraction"),
        "confidence": fid.get("confidence", 0.0),
        "interval_cv": fid.get("interval_cv"),
        "regularity_confidence": fid.get("regularity_confidence"),
        "method": "legacy_wrapper_over_abp_fiducial_detection",
    }



def _intervals_from_beat_flags(peaks: list[int], flags: np.ndarray, sampling_rate: float) -> list[list[float]]:
    intervals = []
    start = None
    last_idx = None
    for peak, flag in zip(peaks, flags):
        if bool(flag) and start is None:
            start = int(peak)
        elif not bool(flag) and start is not None:
            intervals.append([float(start / sampling_rate), float(int(last_idx or peak) / sampling_rate)])
            start = None
        last_idx = int(peak)
    if start is not None and last_idx is not None:
        intervals.append([float(start / sampling_rate), float(last_idx / sampling_rate)])
    return intervals


def ABP_classify_pressure_events(signal_path: str, sampling_rate: float, column: str | None = None, hypotension_map_threshold: float = 65.0, severe_map_threshold: float = 55.0) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    summary = _abp_pulse_summary(data.values, data.sampling_rate)
    if summary.get("error"):
        return {"tool": "ABP_classify_pressure_events", "error": summary["error"], "confidence": 0.0}
    peaks = summary["pulse_indices"]
    maps = np.asarray(summary.get("beat_map_values", []), dtype=float)
    systolic = np.asarray(summary.get("beat_systolic_values", []), dtype=float)
    diastolic = np.asarray(summary.get("beat_diastolic_values", []), dtype=float)
    pp = np.asarray(summary.get("beat_pulse_pressure_values", []), dtype=float)
    if len(maps) == 0:
        return {"tool": "ABP_classify_pressure_events", "error": "no beat-level MAP values", "confidence": 0.0}
    hypo = (maps < float(hypotension_map_threshold)) | (systolic < 90)
    severe = maps < float(severe_map_threshold)
    hyper = (systolic >= 140) | (diastolic >= 90)
    narrow_pp = pp < 25
    wide_pp = pp > 80
    duration_s = float(len(data.values) / data.sampling_rate) if len(data.values) else 0.0
    hypo_intervals = _intervals_from_beat_flags(peaks, hypo, data.sampling_rate)
    severe_intervals = _intervals_from_beat_flags(peaks, severe, data.sampling_rate)
    hypotension_fraction = float(np.mean(hypo))
    severe_fraction = float(np.mean(severe))
    flags = []
    if hypotension_fraction >= 0.2:
        flags.append("sustained_hypotension_burden")
    elif hypotension_fraction > 0:
        flags.append("transient_hypotension_beats")
    if severe_fraction > 0:
        flags.append("severe_map_below_55")
    if float(np.mean(narrow_pp)) >= 0.2:
        flags.append("narrow_pulse_pressure_burden")
    if float(np.mean(wide_pp)) >= 0.2:
        flags.append("wide_pulse_pressure_burden")
    if float(np.mean(hyper)) >= 0.2:
        flags.append("hypertension_burden")
    if severe_fraction >= 0.1 or (hypotension_fraction >= 0.3 and float(np.mean(narrow_pp)) >= 0.2):
        risk = "shock_or_severe_hypotension_proxy"
    elif hypotension_fraction >= 0.2:
        risk = "hypotension_event_proxy"
    elif float(np.mean(hyper)) >= 0.2:
        risk = "hypertension_event_proxy"
    else:
        risk = "no_pressure_event_proxy"
    return {
        "tool": "ABP_classify_pressure_events",
        "pressure_event_risk": risk,
        "event_flags": flags,
        "num_beats": int(len(maps)),
        "duration_s": duration_s,
        "median_map_value": float(np.nanmedian(maps)),
        "map_p05": float(np.nanpercentile(maps, 5)),
        "map_p95": float(np.nanpercentile(maps, 95)),
        "median_systolic_value": float(np.nanmedian(systolic)),
        "approx_diastolic_value": float(np.nanmedian(diastolic)),
        "median_pulse_pressure": float(np.nanmedian(pp)),
        "hypotensive_beat_fraction": hypotension_fraction,
        "severe_hypotensive_beat_fraction": severe_fraction,
        "hypertensive_beat_fraction": float(np.mean(hyper)),
        "narrow_pulse_pressure_fraction": float(np.mean(narrow_pp)),
        "wide_pulse_pressure_fraction": float(np.mean(wide_pp)),
        "hypotension_intervals_s": hypo_intervals[:20],
        "severe_hypotension_intervals_s": severe_intervals[:20],
        "confidence": max(0.52, min(0.78, 1.0 - float(summary.get("artifact_rejected_fraction", 0.4)))),
        "method": "beat_level_map_sbp_pulse_pressure_event_burden",
        "benchmark_direction": "PhysioNet/CinC Challenge 2009 acute hypotensive episode prediction is the appropriate labeled benchmark; this wrapper is an event-burden screen until challenge labels are wired in.",
        "disclaimer": "Screening heuristic only; shock prediction requires clinical context, vasopressor/lactate/outcome labels, and calibrated invasive ABP.",
    }


def ABP_screen_pressure_events(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    fiducials = ABP_detect_fiducial_points(signal_path, sampling_rate, column)
    if fiducials.get("error"):
        return {"tool": "ABP_screen_pressure_events", "error": fiducials["error"], "confidence": 0.0}
    systolic = fiducials.get("median_systolic_value")
    diastolic = fiducials.get("approx_diastolic_value")
    flags = []
    if systolic is not None and systolic < 90:
        flags.append("low_systolic_proxy")
    if diastolic is not None and diastolic < 60:
        flags.append("low_diastolic_proxy")
    if systolic is not None and systolic >= 140:
        flags.append("high_systolic_proxy")
    if diastolic is not None and diastolic >= 90:
        flags.append("high_diastolic_proxy")
    event_result = ABP_classify_pressure_events(signal_path, sampling_rate, column)
    pressure_risk = event_result.get("pressure_event_risk") if not event_result.get("error") else ("hypotension_proxy" if any(flag.startswith("low") for flag in flags) else "hypertension_proxy" if any(flag.startswith("high") for flag in flags) else "no_pressure_event_proxy")
    return {
        "tool": "ABP_screen_pressure_events",
        "median_systolic_value": systolic,
        "approx_diastolic_value": diastolic,
        "heart_rate_bpm": fiducials.get("heart_rate_bpm"),
        "pressure_flags": flags,
        "pressure_risk": pressure_risk,
        "event_burden": None if event_result.get("error") else event_result,
        "confidence": max(0.5, min(0.74, float(fiducials.get("confidence", 0.5)))),
        "method": "abp_beat_level_pressure_event_screening",
        "disclaimer": "Screening heuristic only; ABP calibration and clinical context are required for blood-pressure interpretation.",
    }



def ABP_detect_acute_hypotensive_episode_proxy(signal_path: str, sampling_rate: float, column: str | None = None, map_threshold: float = 60.0, window_minutes: float = 30.0, required_fraction: float = 0.9) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    if len(data.values) < int(float(sampling_rate) * 60):
        return {"tool": "ABP_detect_acute_hypotensive_episode_proxy", "error": "signal too short for minute-level AHE screening", "confidence": 0.0}
    summary = _abp_pulse_summary(data.values, data.sampling_rate)
    if summary.get("error"):
        return {"tool": "ABP_detect_acute_hypotensive_episode_proxy", "error": summary["error"], "confidence": 0.0}
    peaks = np.asarray(summary.get("pulse_indices", []), dtype=int)
    maps = np.asarray(summary.get("beat_map_values", []), dtype=float)
    if len(peaks) == 0 or len(maps) == 0:
        return {"tool": "ABP_detect_acute_hypotensive_episode_proxy", "error": "no beat-level MAP values", "confidence": 0.0}
    minute_idx = np.floor(peaks / data.sampling_rate / 60.0).astype(int)
    n_minutes = int(np.floor(len(data.values) / data.sampling_rate / 60.0))
    minute_maps = []
    for minute in range(n_minutes):
        vals = maps[minute_idx == minute]
        minute_maps.append(float(np.nanmean(vals)) if len(vals) else np.nan)
    minute_maps_arr = np.asarray(minute_maps, dtype=float)
    valid = np.isfinite(minute_maps_arr) & (minute_maps_arr > 10.0)
    low = valid & (minute_maps_arr <= float(map_threshold))
    win = max(1, int(round(window_minutes)))
    required = int(np.ceil(float(required_fraction) * win))
    intervals = []
    low_counts = []
    for end in range(win, len(minute_maps_arr) + 1):
        segment_valid = valid[end - win:end]
        segment_low = low[end - win:end]
        if int(np.sum(segment_valid)) < max(1, int(0.5 * win)):
            low_counts.append(None)
            continue
        count = int(np.sum(segment_low))
        low_counts.append(count)
        if count >= required:
            intervals.append([float((end - win) * 60.0), float(end * 60.0)])
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1] + 60.0:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    valid_minutes = int(np.sum(valid))
    low_minutes = int(np.sum(low))
    low_fraction = float(low_minutes / valid_minutes) if valid_minutes else 0.0
    recent = minute_maps_arr[-win:] if len(minute_maps_arr) else np.array([])
    recent_valid = recent[np.isfinite(recent)]
    recent_low_fraction = float(np.mean(recent_valid <= float(map_threshold))) if len(recent_valid) else None
    risk = "acute_hypotensive_episode_proxy" if merged else "no_ahe_proxy"
    return {
        "tool": "ABP_detect_acute_hypotensive_episode_proxy",
        "ahe_risk": risk,
        "ahe_event_count": int(len(merged)),
        "ahe_intervals_s": merged[:20],
        "valid_minute_count": valid_minutes,
        "low_map_minute_count": low_minutes,
        "low_map_minute_fraction": low_fraction,
        "recent_window_low_map_fraction": recent_low_fraction,
        "minute_map_threshold": float(map_threshold),
        "window_minutes": float(window_minutes),
        "required_low_minutes": required,
        "median_minute_map": float(np.nanmedian(minute_maps_arr[valid])) if valid_minutes else None,
        "min_minute_map": float(np.nanmin(minute_maps_arr[valid])) if valid_minutes else None,
        "confidence": 0.72 if valid_minutes >= win else 0.45,
        "method": "physionet_cinc2009_minute_map_30min_ahe_rule_proxy",
        "source_note": "Implements the public PhysioNet/CinC 2009 ahe-detect rule: low MAP for at least 90% of a 30-minute window, using beat-derived MAP when ABP Mean is unavailable.",
        "disclaimer": "AHE proxy requires calibrated ABP and enough continuous data; challenge prediction still requires forecasting future AHE, not just detecting current low MAP.",
    }


def ABP_compute_hemodynamics(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    fiducials = ABP_detect_fiducial_points(signal_path, sampling_rate, column)
    if len(data.values) == 0:
        return {"tool": "ABP_compute_hemodynamics", "error": "empty signal", "confidence": 0.0}
    if fiducials.get("error"):
        return {"tool": "ABP_compute_hemodynamics", "error": fiducials["error"], "confidence": 0.0}
    systolic = fiducials.get("median_systolic_value")
    diastolic = fiducials.get("approx_diastolic_value")
    mean_pressure = float(np.nanmean(data.values))
    pulse_pressure = float(systolic - diastolic) if systolic is not None and diastolic is not None else None
    map_formula = float(diastolic + (pulse_pressure / 3.0)) if pulse_pressure is not None and diastolic is not None else mean_pressure
    flags = []
    if map_formula < 65:
        flags.append("low_map_proxy")
    if pulse_pressure is not None and pulse_pressure < 25:
        flags.append("narrow_pulse_pressure_proxy")
    if pulse_pressure is not None and pulse_pressure > 80:
        flags.append("wide_pulse_pressure_proxy")
    hemodynamic_risk = "elevated" if flags else "low"
    return {
        "tool": "ABP_compute_hemodynamics",
        "mean_arterial_pressure_proxy": map_formula,
        "mean_pressure_value": mean_pressure,
        "pulse_pressure_proxy": pulse_pressure,
        "median_systolic_value": systolic,
        "approx_diastolic_value": diastolic,
        "heart_rate_bpm": fiducials.get("heart_rate_bpm"),
        "fiducial_summary": None if fiducials.get("error") else {
            "num_beats": fiducials.get("num_beats"),
            "missing_fiducial_fraction": fiducials.get("missing_fiducial_fraction"),
            "systolic_peak_indices": fiducials.get("systolic_peak_indices"),
            "systolic_onset_indices": fiducials.get("systolic_onset_indices"),
            "dicrotic_notch_indices": fiducials.get("dicrotic_notch_indices"),
            "diastolic_peak_indices": fiducials.get("diastolic_peak_indices"),
            "diastolic_phase_endpoint_indices": fiducials.get("diastolic_phase_endpoint_indices"),
        },
        "hemodynamic_flags": flags,
        "hemodynamic_risk": hemodynamic_risk,
        "confidence": max(0.5, min(0.7, float(fiducials.get("confidence", 0.5)))),
        "method": "abp_map_pulse_pressure_proxy",
        "disclaimer": "Screening heuristic only; ABP units and calibration are required for clinical hemodynamic interpretation.",
    }
