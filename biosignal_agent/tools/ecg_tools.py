from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from .common import bpm_from_peaks, load_csv_signal, signal_quality_summary
from .peak_detectors import neurokit_nabian2018_peaks

try:
    import neurokit2 as nk
except ImportError:  # pragma: no cover
    nk = None


def ECG_assess_quality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    return {"tool": "ECG_assess_quality", "source": data.source, **signal_quality_summary(data.values)}


def ECG_detect_r_peaks(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    try:
        if nk is None:
            raise RuntimeError("neurokit2 is not installed")
        cleaned = nk.ecg_clean(values, sampling_rate=data.sampling_rate, method="pantompkins1985")
        _, info = nk.ecg_peaks(cleaned, sampling_rate=data.sampling_rate, method="pantompkins1985", correct_artifacts=True)
        peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
        details = {"method": "pantompkins1985", "median_prominence": None}
    except Exception as exc:
        peaks, details = neurokit_nabian2018_peaks(values, data.sampling_rate, low_hz=None, high_hz=None, fallback_threshold_scale=0.6)
        details["fallback_reason"] = str(exc)
    heart_rate = bpm_from_peaks(peaks, data.sampling_rate)
    confidence = 0.82 if details["method"] == "pantompkins1985" else 0.65
    if heart_rate is None or not 35 <= heart_rate <= 220:
        confidence = 0.3
    return {"tool": "ECG_detect_r_peaks", "r_peak_indices": peaks.tolist(), "num_peaks": int(len(peaks)), "heart_rate_bpm": heart_rate, "confidence": confidence, **details}


def ECG_compute_hrv(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    peak_result = ECG_detect_r_peaks(signal_path, sampling_rate, column)
    peaks = np.asarray(peak_result["r_peak_indices"], dtype=float)
    if len(peaks) < 3:
        return {"tool": "ECG_compute_hrv", "error": "not enough R peaks", "confidence": 0.1}
    rr_ms = np.diff(peaks) / float(sampling_rate) * 1000.0
    rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))) if len(rr_ms) > 1 else None
    return {"tool": "ECG_compute_hrv", "mean_rr_ms": float(np.mean(rr_ms)), "sdnn_ms": float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else 0.0, "rmssd_ms": rmssd, "confidence": peak_result["confidence"]}



def ECG_screen_arrhythmia(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    peak_result = ECG_detect_r_peaks(signal_path, sampling_rate, column)
    peaks = np.asarray(peak_result.get("r_peak_indices", []), dtype=float)
    if len(peaks) < 4:
        return {"tool": "ECG_screen_arrhythmia", "error": "not enough R peaks", "confidence": 0.1}
    rr_s = np.diff(peaks) / float(sampling_rate)
    rr_s = rr_s[(rr_s >= 0.25) & (rr_s <= 3.0)]
    if len(rr_s) < 3:
        return {"tool": "ECG_screen_arrhythmia", "error": "not enough valid RR intervals", "confidence": 0.1}
    heart_rate = float(60.0 / np.median(rr_s))
    rr_cv = float(np.std(rr_s) / np.mean(rr_s)) if np.mean(rr_s) > 0 else None
    pause_count = int(np.sum(rr_s > 2.0))
    short_long = np.abs(np.diff(rr_s)) / rr_s[:-1] if len(rr_s) > 1 else np.array([])
    ectopy_proxy_fraction = float(np.mean(short_long > 0.2)) if len(short_long) else 0.0
    flags = []
    if heart_rate < 50:
        flags.append("bradycardia_pattern")
    if heart_rate > 110:
        flags.append("tachycardia_pattern")
    if rr_cv is not None and rr_cv > 0.18:
        flags.append("irregular_rr_pattern")
    if pause_count:
        flags.append("long_pause_pattern")
    if ectopy_proxy_fraction > 0.15:
        flags.append("ectopy_proxy_pattern")
    risk = "elevated" if flags else "low"
    confidence = min(float(peak_result.get("confidence", 0.5)), 0.7)
    return {
        "tool": "ECG_screen_arrhythmia",
        "heart_rate_bpm": heart_rate,
        "rr_cv": rr_cv,
        "pause_count": pause_count,
        "ectopy_proxy_fraction": ectopy_proxy_fraction,
        "arrhythmia_flags": flags,
        "arrhythmia_risk": risk,
        "confidence": confidence,
        "method": "rr_interval_screening",
        "disclaimer": "Screening heuristic only; not a diagnostic rhythm classifier.",
    }



def ECG_screen_sleep_apnea(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    hrv = ECG_compute_hrv(signal_path, sampling_rate, column)
    arrhythmia = ECG_screen_arrhythmia(signal_path, sampling_rate, column)
    if hrv.get("error"):
        return {"tool": "ECG_screen_sleep_apnea", "error": hrv["error"], "confidence": 0.1}
    mean_rr = float(hrv.get("mean_rr_ms") or 0.0)
    rmssd = float(hrv.get("rmssd_ms") or 0.0)
    sdnn = float(hrv.get("sdnn_ms") or 0.0)
    rr_cv = arrhythmia.get("rr_cv")
    heart_rate = arrhythmia.get("heart_rate_bpm")
    score = 0
    flags = []
    if heart_rate is not None and (heart_rate < 55 or heart_rate > 95):
        score += 1
        flags.append("sleep_epoch_heart_rate_extreme")
    if rr_cv is not None and rr_cv > 0.10:
        score += 1
        flags.append("elevated_rr_variability")
    if rmssd > 80 or sdnn > 90:
        score += 1
        flags.append("high_short_term_hrv")
    if mean_rr > 1200:
        score += 1
        flags.append("bradycardic_rr_pattern")
    apnea_risk = "elevated" if score >= 2 else "low"
    return {
        "tool": "ECG_screen_sleep_apnea",
        "apnea_risk": apnea_risk,
        "apnea_proxy_score": score,
        "apnea_proxy_flags": flags,
        "heart_rate_bpm": heart_rate,
        "mean_rr_ms": mean_rr,
        "sdnn_ms": sdnn,
        "rmssd_ms": rmssd,
        "rr_cv": rr_cv,
        "confidence": 0.5,
        "method": "ecg_hrv_sleep_apnea_proxy",
        "disclaimer": "ECG-only apnea proxy for benchmarking; respiratory effort and SpO2 labels are preferred for clinical apnea detection.",
    }



def ECG_measure_morphology_intervals(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    peak_result = ECG_detect_r_peaks(signal_path, sampling_rate, column)
    peaks = np.asarray(peak_result.get("r_peak_indices", []), dtype=int)
    if len(peaks) < 3:
        return {"tool": "ECG_measure_morphology_intervals", "error": "not enough R peaks", "confidence": 0.1}
    values = data.values
    result = {
        "tool": "ECG_measure_morphology_intervals",
        "heart_rate_bpm": peak_result.get("heart_rate_bpm"),
        "confidence": min(0.55, float(peak_result.get("confidence", 0.5))),
        "method": "ecg_delineation_interval_screening",
        "disclaimer": "Screening heuristic only; ECG intervals require validated delineation and lead-specific clinical review.",
    }
    try:
        if nk is None:
            raise RuntimeError("neurokit2 is not installed")
        cleaned = nk.ecg_clean(values, sampling_rate=data.sampling_rate, method="pantompkins1985")
        _, waves = nk.ecg_delineate(cleaned, rpeaks=peaks, sampling_rate=data.sampling_rate, method="dwt", show=False, show_type="all")
        def valid(name: str) -> np.ndarray:
            arr = np.asarray(waves.get(name, []), dtype=float)
            return arr[np.isfinite(arr)]
        q = valid("ECG_Q_Peaks")
        s_peaks = valid("ECG_S_Peaks")
        p_on = valid("ECG_P_Onsets")
        qrs_on = valid("ECG_R_Onsets")
        qrs_off = valid("ECG_R_Offsets")
        t_off = valid("ECG_T_Offsets")
        qrs_ms = None
        if len(qrs_on) and len(qrs_off):
            n = min(len(qrs_on), len(qrs_off))
            qrs_ms = float(np.nanmedian((qrs_off[:n] - qrs_on[:n]) / data.sampling_rate * 1000.0))
        elif len(q) and len(s_peaks):
            n = min(len(q), len(s_peaks))
            qrs_ms = float(np.nanmedian((s_peaks[:n] - q[:n]) / data.sampling_rate * 1000.0))
        pr_ms = None
        if len(p_on) and len(qrs_on):
            n = min(len(p_on), len(qrs_on))
            pr_ms = float(np.nanmedian((qrs_on[:n] - p_on[:n]) / data.sampling_rate * 1000.0))
        qt_ms = None
        if len(qrs_on) and len(t_off):
            n = min(len(qrs_on), len(t_off))
            qt_ms = float(np.nanmedian((t_off[:n] - qrs_on[:n]) / data.sampling_rate * 1000.0))
    except Exception as exc:
        qrs_ms = None
        pr_ms = None
        qt_ms = None
        result["fallback_reason"] = str(exc)
    rr_s = np.diff(peaks) / data.sampling_rate
    rr_s = rr_s[(rr_s >= 0.3) & (rr_s <= 2.5)]
    qtc_ms = float(qt_ms / np.sqrt(np.nanmedian(rr_s))) if qt_ms is not None and len(rr_s) else None
    st_values = []
    offset = int(0.08 * data.sampling_rate)
    baseline_offset = int(0.04 * data.sampling_rate)
    for peak in peaks:
        st_idx = peak + offset
        base_idx = peak - baseline_offset
        if 0 <= st_idx < len(values) and 0 <= base_idx < len(values):
            st_values.append(values[st_idx] - values[base_idx])
    st_deviation_proxy = float(np.nanmedian(st_values)) if st_values else None
    flags = []
    if qrs_ms is not None and qrs_ms > 120:
        flags.append("wide_qrs_proxy")
    if qtc_ms is not None and qtc_ms > 470:
        flags.append("long_qtc_proxy")
    if pr_ms is not None and pr_ms > 220:
        flags.append("prolonged_pr_proxy")
    if st_deviation_proxy is not None and abs(st_deviation_proxy) > max(np.nanstd(values) * 0.25, 1e-8):
        flags.append("st_deviation_proxy")
    result.update({
        "pr_interval_ms": pr_ms,
        "qrs_duration_ms": qrs_ms,
        "qt_interval_ms": qt_ms,
        "qtc_interval_ms": qtc_ms,
        "st_deviation_proxy": st_deviation_proxy,
        "morphology_flags": flags,
        "morphology_risk": "elevated" if flags else "low",
    })
    return result
