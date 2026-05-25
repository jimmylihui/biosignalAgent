from __future__ import annotations

import numpy as np

from .ecg_tools import ECG_detect_r_peaks, ECG_screen_sleep_apnea
from .ppg_tools import PPG_assess_perfusion_variability, PPG_detect_peaks
from .resp_tools import RESP_screen_sleep_apnea_ml, RESP_summarize_event_burden
from .spo2_tools import SpO2_assess_hypoxemia_burden, SpO2_detect_desaturation, SpO2_screen_sleep_apnea_ml


def _bounded_mean(items: list[tuple[float, float]]) -> float | None:
    vals = [(float(v), float(w)) for v, w in items if v is not None and np.isfinite(v) and w > 0]
    if not vals:
        return None
    total_w = sum(w for _, w in vals)
    return float(sum(v * w for v, w in vals) / total_w) if total_w > 0 else None


def Multimodal_estimate_ecg_ppg_pat_bp_proxy(ecg_path: str, ecg_sampling_rate: float, ppg_path: str, ppg_sampling_rate: float, ecg_column: str | None = None, ppg_column: str | None = None) -> dict:
    ecg = ECG_detect_r_peaks(ecg_path, ecg_sampling_rate, ecg_column)
    ppg = PPG_detect_peaks(ppg_path, ppg_sampling_rate, ppg_column)
    r_peaks = np.asarray(ecg.get("r_peak_indices", []), dtype=float) / float(ecg_sampling_rate)
    p_peaks = np.asarray(ppg.get("peak_indices", []), dtype=float) / float(ppg_sampling_rate)
    if len(r_peaks) < 3 or len(p_peaks) < 3:
        return {"tool": "Multimodal_estimate_ecg_ppg_pat_bp_proxy", "error": "not enough ECG R-peaks or PPG peaks", "ecg_result": ecg, "ppg_result": ppg, "confidence": 0.1}
    pats = []
    j = 0
    for r in r_peaks:
        while j < len(p_peaks) and p_peaks[j] <= r + 0.08:
            j += 1
        if j < len(p_peaks):
            delay = p_peaks[j] - r
            if 0.08 <= delay <= 0.6:
                pats.append(float(delay))
    pats = np.asarray(pats, dtype=float)
    if len(pats) < 3:
        return {"tool": "Multimodal_estimate_ecg_ppg_pat_bp_proxy", "error": "no plausible PAT pairs", "ecg_result": ecg, "ppg_result": ppg, "confidence": 0.15}
    median_pat = float(np.nanmedian(pats))
    pat_cv = float(np.nanstd(pats) / (np.nanmean(pats) + 1e-12))
    perfusion = PPG_assess_perfusion_variability(ppg_path, ppg_sampling_rate, ppg_column)
    if median_pat < 0.16:
        risk = "short_pat_high_bp_or_stiffness_proxy"
    elif median_pat > 0.35:
        risk = "long_pat_low_pressure_or_low_perfusion_proxy"
    else:
        risk = "pat_within_typical_proxy_range"
    flags = []
    if pat_cv > 0.18:
        flags.append("unstable_pat_alignment")
    if perfusion.get("perfusion_level") == "low_perfusion_proxy":
        flags.append("low_ppg_perfusion_limits_pat_bp_interpretation")
    return {
        "tool": "Multimodal_estimate_ecg_ppg_pat_bp_proxy",
        "matched_beat_count": int(len(pats)),
        "median_pat_s": median_pat,
        "pat_iqr_s": float(np.nanpercentile(pats, 75) - np.nanpercentile(pats, 25)),
        "pat_cv": pat_cv,
        "pat_bp_proxy_risk": risk,
        "flags": flags,
        "ecg_heart_rate_bpm": ecg.get("heart_rate_bpm"),
        "ppg_heart_rate_bpm": ppg.get("heart_rate_bpm"),
        "perfusion_result": perfusion,
        "confidence": float(max(0.25, min(0.72, min(ecg.get("confidence", 0.3), ppg.get("confidence", 0.3)) - 0.1 * (pat_cv > 0.18)))),
        "method": "ecg_rpeak_to_ppg_peak_pulse_arrival_time_proxy",
        "disclaimer": "PAT is not calibrated BP. Use only as directional vascular/hemodynamic evidence unless subject-specific cuff/ABP calibration is available.",
    }


def Multimodal_screen_sleep_apnea_report(ecg_path: str | None = None, ecg_sampling_rate: float | None = None, resp_path: str | None = None, resp_sampling_rate: float | None = None, spo2_path: str | None = None, spo2_sampling_rate: float | None = None, ecg_column: str | None = None, resp_column: str | None = None, spo2_column: str | None = None) -> dict:
    evidence = {}
    scores: list[tuple[float, float]] = []
    flags = []
    if ecg_path and ecg_sampling_rate:
        ecg = ECG_screen_sleep_apnea(ecg_path, ecg_sampling_rate, ecg_column)
        evidence["ecg"] = ecg
        prob = ecg.get("apnea_probability") or ecg.get("sleep_apnea_probability")
        if prob is not None:
            scores.append((float(prob), 0.35))
        if ecg.get("sleep_apnea_risk") or ecg.get("apnea_risk"):
            flags.append(str(ecg.get("sleep_apnea_risk") or ecg.get("apnea_risk")))
    if resp_path and resp_sampling_rate:
        resp_burden = RESP_summarize_event_burden(resp_path, resp_sampling_rate, resp_column)
        evidence["resp_event_burden"] = resp_burden
        rei = resp_burden.get("respiratory_event_index_per_hour")
        if rei is not None:
            scores.append((float(np.clip(float(rei) / 30.0, 0.0, 1.0)), 0.3))
        if spo2_path and spo2_sampling_rate:
            resp_ml = RESP_screen_sleep_apnea_ml(resp_path, resp_sampling_rate, resp_column, spo2_path, spo2_sampling_rate, spo2_column)
            evidence["resp_spo2_ml"] = resp_ml
            if resp_ml.get("respiratory_event_probability") is not None:
                scores.append((float(resp_ml["respiratory_event_probability"]), 0.45))
    if spo2_path and spo2_sampling_rate:
        spo2_desat = SpO2_detect_desaturation(spo2_path, spo2_sampling_rate, spo2_column)
        spo2_hypox = SpO2_assess_hypoxemia_burden(spo2_path, spo2_sampling_rate, spo2_column)
        spo2_ml = SpO2_screen_sleep_apnea_ml(spo2_path, spo2_sampling_rate, spo2_column)
        evidence["spo2_desaturation"] = spo2_desat
        evidence["spo2_hypoxemia"] = spo2_hypox
        evidence["spo2_ml"] = spo2_ml
        if spo2_ml.get("respiratory_event_probability") is not None:
            scores.append((float(spo2_ml["respiratory_event_probability"]), 0.25))
        odi = spo2_desat.get("oxygen_desaturation_index_per_hour")
        if odi is not None:
            scores.append((float(np.clip(float(odi) / 30.0, 0.0, 1.0)), 0.2))
        if spo2_hypox.get("hypoxemia_risk"):
            flags.append(str(spo2_hypox.get("hypoxemia_risk")))
    if not evidence:
        return {"tool": "Multimodal_screen_sleep_apnea_report", "error": "provide at least one of ECG, RESP, or SpO2 inputs", "confidence": 0.0}
    fused = _bounded_mean(scores)
    if fused is None:
        fused = 0.0
        confidence = 0.25
    else:
        confidence = float(min(0.82, 0.35 + 0.1 * len(scores) + abs(fused - 0.5) * 0.4))
    if fused >= 0.7:
        risk = "high_sleep_apnea_evidence"
    elif fused >= 0.45:
        risk = "moderate_sleep_apnea_evidence"
    elif fused >= 0.25:
        risk = "low_to_moderate_sleep_apnea_evidence"
    else:
        risk = "low_sleep_apnea_evidence"
    return {
        "tool": "Multimodal_screen_sleep_apnea_report",
        "fused_sleep_apnea_probability": float(fused),
        "sleep_apnea_evidence_level": risk,
        "evidence_flags": sorted(set(flags)),
        "modalities_used": sorted(evidence.keys()),
        "evidence": evidence,
        "confidence": confidence,
        "method": "weighted_multimodal_sleep_apnea_evidence_fusion",
        "disclaimer": "Fusion report is screening evidence, not diagnostic AHI. Full scoring requires PSG-grade airflow/effort, SpO2/arousal rules, and sleep-time denominator.",
    }
