from __future__ import annotations

from biosignal_agent.tools.abp_tools import ABP_assess_quality, ABP_detect_pulses, ABP_screen_pressure_events
from biosignal_agent.tools.acc_tools import ACC_assess_quality, ACC_estimate_sleep_wake, ACC_summarize_activity
from biosignal_agent.tools.bcg_tools import BCG_assess_quality, BCG_detect_j_peaks
from biosignal_agent.tools.ecg_tools import ECG_assess_quality, ECG_compute_hrv, ECG_detect_r_peaks, ECG_screen_arrhythmia, ECG_screen_sleep_apnea
from biosignal_agent.tools.eda_tools import EDA_assess_quality, EDA_detect_arousal_events, EDA_summarize
from biosignal_agent.tools.eeg_tools import EEG_assess_quality, EEG_compute_bandpower, EEG_estimate_sleep_stage_features, EEG_screen_seizure_like_activity
from biosignal_agent.tools.emg_tools import EMG_assess_quality, EMG_estimate_fatigue, EMG_summarize_activation
from biosignal_agent.tools.pcg_tools import PCG_assess_quality, PCG_detect_heart_sounds, PCG_screen_murmur_proxy
from biosignal_agent.tools.ppg_tools import PPG_assess_quality, PPG_assess_perfusion_variability, PPG_detect_peaks
from biosignal_agent.tools.resp_tools import RESP_assess_quality, RESP_detect_apnea, RESP_detect_hypopnea, RESP_estimate_rate
from biosignal_agent.tools.scg_tools import SCG_assess_quality, SCG_detect_j_peaks
from biosignal_agent.tools.spo2_tools import SpO2_assess_hypoxemia_burden, SpO2_assess_quality, SpO2_detect_desaturation, SpO2_summarize

TOOLS = {
    "ECG_assess_quality": ECG_assess_quality,
    "ECG_detect_r_peaks": ECG_detect_r_peaks,
    "ECG_compute_hrv": ECG_compute_hrv,
    "ECG_screen_arrhythmia": ECG_screen_arrhythmia,
    "ECG_screen_sleep_apnea": ECG_screen_sleep_apnea,
    "PPG_assess_quality": PPG_assess_quality,
    "PPG_detect_peaks": PPG_detect_peaks,
    "PPG_assess_perfusion_variability": PPG_assess_perfusion_variability,
    "BCG_assess_quality": BCG_assess_quality,
    "BCG_detect_j_peaks": BCG_detect_j_peaks,
    "SCG_assess_quality": SCG_assess_quality,
    "SCG_detect_j_peaks": SCG_detect_j_peaks,
    "RESP_assess_quality": RESP_assess_quality,
    "RESP_estimate_rate": RESP_estimate_rate,
    "RESP_detect_apnea": RESP_detect_apnea,
    "RESP_detect_hypopnea": RESP_detect_hypopnea,
    "SpO2_assess_quality": SpO2_assess_quality,
    "SpO2_summarize": SpO2_summarize,
    "SpO2_detect_desaturation": SpO2_detect_desaturation,
    "SpO2_assess_hypoxemia_burden": SpO2_assess_hypoxemia_burden,
    "ABP_assess_quality": ABP_assess_quality,
    "ABP_detect_pulses": ABP_detect_pulses,
    "ABP_screen_pressure_events": ABP_screen_pressure_events,
    "PCG_assess_quality": PCG_assess_quality,
    "PCG_detect_heart_sounds": PCG_detect_heart_sounds,
    "PCG_screen_murmur_proxy": PCG_screen_murmur_proxy,
    "ACC_assess_quality": ACC_assess_quality,
    "ACC_summarize_activity": ACC_summarize_activity,
    "ACC_estimate_sleep_wake": ACC_estimate_sleep_wake,
    "EDA_assess_quality": EDA_assess_quality,
    "EDA_summarize": EDA_summarize,
    "EDA_detect_arousal_events": EDA_detect_arousal_events,
    "EEG_assess_quality": EEG_assess_quality,
    "EEG_compute_bandpower": EEG_compute_bandpower,
    "EEG_estimate_sleep_stage_features": EEG_estimate_sleep_stage_features,
    "EEG_screen_seizure_like_activity": EEG_screen_seizure_like_activity,
    "EMG_assess_quality": EMG_assess_quality,
    "EMG_summarize_activation": EMG_summarize_activation,
    "EMG_estimate_fatigue": EMG_estimate_fatigue,
}

WORKFLOWS = {
    "ecg": ["ECG_assess_quality", "ECG_detect_r_peaks", "ECG_compute_hrv", "ECG_screen_arrhythmia", "ECG_screen_sleep_apnea"],
    "ppg": ["PPG_assess_quality", "PPG_detect_peaks", "PPG_assess_perfusion_variability"],
    "bcg": ["BCG_assess_quality", "BCG_detect_j_peaks"],
    "scg": ["SCG_assess_quality", "SCG_detect_j_peaks"],
    "resp": ["RESP_assess_quality", "RESP_estimate_rate", "RESP_detect_apnea", "RESP_detect_hypopnea"],
    "spo2": ["SpO2_assess_quality", "SpO2_summarize", "SpO2_detect_desaturation", "SpO2_assess_hypoxemia_burden"],
    "abp": ["ABP_assess_quality", "ABP_detect_pulses", "ABP_screen_pressure_events"],
    "pcg": ["PCG_assess_quality", "PCG_detect_heart_sounds", "PCG_screen_murmur_proxy"],
    "acc": ["ACC_assess_quality", "ACC_summarize_activity", "ACC_estimate_sleep_wake"],
    "eda": ["EDA_assess_quality", "EDA_summarize", "EDA_detect_arousal_events"],
    "eeg": ["EEG_assess_quality", "EEG_compute_bandpower", "EEG_estimate_sleep_stage_features", "EEG_screen_seizure_like_activity"],
    "emg": ["EMG_assess_quality", "EMG_summarize_activation", "EMG_estimate_fatigue"],
}
