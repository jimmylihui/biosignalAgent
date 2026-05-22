from __future__ import annotations

import numpy as np

from biosignal_agent.session.schema import SignalInput
from biosignal_agent.tools.ecg_tools import ECG_detect_r_peaks, ECG_compute_hrv, ECG_screen_sleep_apnea
from biosignal_agent.tools.ppg_tools import PPG_detect_peaks
from biosignal_agent.tools.resp_tools import RESP_detect_apnea, RESP_detect_hypopnea, RESP_screen_rate_pattern
from biosignal_agent.tools.spo2_tools import SpO2_assess_hypoxemia_burden, SpO2_detect_desaturation


def _first_signal(signals: list[SignalInput], modality: str) -> SignalInput | None:
    for signal in signals:
        if signal.modality == modality:
            return signal
    return None


def Session_compute_ecg_ppg_pulse_arrival(signals: list[SignalInput]) -> dict:
    ecg = _first_signal(signals, "ecg")
    ppg = _first_signal(signals, "ppg")
    if ecg is None or ppg is None:
        return {"tool": "Session_compute_ecg_ppg_pulse_arrival", "error": "requires ECG and PPG signals", "confidence": 0.0}
    ecg_peaks = ECG_detect_r_peaks(ecg.path, ecg.sampling_rate, ecg.column)
    ppg_peaks = PPG_detect_peaks(ppg.path, ppg.sampling_rate, ppg.column)
    r_peaks = np.asarray(ecg_peaks.get("r_peak_indices", []), dtype=float) / float(ecg.sampling_rate)
    p_peaks = np.asarray(ppg_peaks.get("peak_indices", []), dtype=float) / float(ppg.sampling_rate)
    if len(r_peaks) < 3 or len(p_peaks) < 3:
        return {"tool": "Session_compute_ecg_ppg_pulse_arrival", "error": "not enough ECG or PPG peaks", "confidence": 0.1}
    delays = []
    ppg_idx = 0
    for r_time in r_peaks:
        while ppg_idx < len(p_peaks) and p_peaks[ppg_idx] <= r_time:
            ppg_idx += 1
        if ppg_idx >= len(p_peaks):
            break
        delay = p_peaks[ppg_idx] - r_time
        if 0.08 <= delay <= 0.6:
            delays.append(delay)
    if len(delays) < 3:
        return {"tool": "Session_compute_ecg_ppg_pulse_arrival", "error": "not enough plausible ECG-to-PPG pulse arrival pairs", "confidence": 0.2}
    delays_ms = np.asarray(delays) * 1000.0
    return {
        "tool": "Session_compute_ecg_ppg_pulse_arrival",
        "paired_pulses": int(len(delays_ms)),
        "median_pulse_arrival_time_ms": float(np.nanmedian(delays_ms)),
        "pulse_arrival_iqr_ms": float(np.nanpercentile(delays_ms, 75) - np.nanpercentile(delays_ms, 25)),
        "ecg_heart_rate_bpm": ecg_peaks.get("heart_rate_bpm"),
        "ppg_heart_rate_bpm": ppg_peaks.get("heart_rate_bpm"),
        "confidence": min(float(ecg_peaks.get("confidence", 0.5)), float(ppg_peaks.get("confidence", 0.5)), 0.65),
        "method": "nearest_following_ppg_peak_after_ecg_r_peak",
        "disclaimer": "Pulse arrival time proxy only; pulse transit time requires synchronized acquisition and pre-ejection-period considerations.",
    }


def Session_screen_sleep_apnea_multimodal(signals: list[SignalInput]) -> dict:
    ecg = _first_signal(signals, "ecg")
    resp = _first_signal(signals, "resp")
    spo2 = _first_signal(signals, "spo2")
    flags = []
    components = {}
    if resp is not None:
        apnea = RESP_detect_apnea(resp.path, resp.sampling_rate, resp.column)
        hypopnea = RESP_detect_hypopnea(resp.path, resp.sampling_rate, resp.column)
        pattern = RESP_screen_rate_pattern(resp.path, resp.sampling_rate, resp.column)
        components["resp_apnea_event_count"] = apnea.get("apnea_event_count")
        components["resp_hypopnea_event_count"] = hypopnea.get("hypopnea_event_count")
        components["respiratory_pattern_risk"] = pattern.get("respiratory_pattern_risk")
        if (apnea.get("apnea_event_count") or 0) > 0:
            flags.append("resp_apnea_events")
        if (hypopnea.get("hypopnea_event_count") or 0) > 0:
            flags.append("resp_hypopnea_events")
        if pattern.get("respiratory_pattern_risk") == "elevated":
            flags.append("resp_pattern_risk")
    if spo2 is not None:
        desat = SpO2_detect_desaturation(spo2.path, spo2.sampling_rate, spo2.column)
        hypoxemia = SpO2_assess_hypoxemia_burden(spo2.path, spo2.sampling_rate, spo2.column)
        components["desaturation_event_count"] = desat.get("desaturation_event_count")
        components["time_below_90_fraction"] = desat.get("time_below_90_fraction")
        components["hypoxemia_burden"] = hypoxemia.get("hypoxemia_burden")
        if (desat.get("desaturation_event_count") or 0) > 0:
            flags.append("spo2_desaturation_events")
        if hypoxemia.get("hypoxemia_burden") in {"moderate_hypoxemia_burden_proxy", "high_hypoxemia_burden_proxy"}:
            flags.append("spo2_hypoxemia_burden")
    if ecg is not None:
        ecg_apnea = ECG_screen_sleep_apnea(ecg.path, ecg.sampling_rate, ecg.column)
        hrv = ECG_compute_hrv(ecg.path, ecg.sampling_rate, ecg.column)
        components["ecg_apnea_risk"] = ecg_apnea.get("apnea_risk")
        components["ecg_rmssd_ms"] = hrv.get("rmssd_ms")
        if ecg_apnea.get("apnea_risk") == "elevated":
            flags.append("ecg_sleep_apnea_proxy")
    if not components:
        return {"tool": "Session_screen_sleep_apnea_multimodal", "error": "requires at least one ECG, RESP, or SpO2 signal", "confidence": 0.0}
    if any(flag in flags for flag in ["resp_apnea_events", "resp_hypopnea_events", "spo2_desaturation_events"]):
        risk = "elevated"
    elif len(flags) >= 2:
        risk = "possible"
    else:
        risk = "low"
    return {
        "tool": "Session_screen_sleep_apnea_multimodal",
        "sleep_apnea_session_risk": risk,
        "session_flags": flags,
        "components": components,
        "confidence": 0.6 if len(components) >= 2 else 0.45,
        "method": "multimodal_resp_spo2_ecg_proxy_fusion",
        "disclaimer": "Research screening proxy only; AHI diagnosis requires scored PSG respiratory events and clinical review.",
    }


SESSION_TOOLS = {
    "Session_compute_ecg_ppg_pulse_arrival": Session_compute_ecg_ppg_pulse_arrival,
    "Session_screen_sleep_apnea_multimodal": Session_screen_sleep_apnea_multimodal,
}
