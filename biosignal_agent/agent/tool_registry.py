from __future__ import annotations

from biosignal_agent.tools.artifact_tools import Signal_detect_artifacts
from biosignal_agent.tools.abp_tools import ABP_assess_quality, ABP_compute_hemodynamics, ABP_detect_pulses, ABP_screen_pressure_events
from biosignal_agent.tools.acc_tools import ACC_assess_quality, ACC_detect_activity_bouts, ACC_detect_fall_proxy, ACC_estimate_sleep_wake, ACC_summarize_activity
from biosignal_agent.tools.bcg_tools import BCG_assess_quality, BCG_detect_j_peaks, BCG_estimate_respiration
from biosignal_agent.tools.ecg_tools import ECG_assess_quality, ECG_compute_hrv, ECG_detect_r_peaks, ECG_measure_morphology_intervals, ECG_screen_arrhythmia, ECG_screen_sleep_apnea
from biosignal_agent.tools.eda_tools import EDA_assess_quality, EDA_detect_arousal_events, EDA_summarize
from biosignal_agent.tools.eeg_tools import EEG_assess_quality, EEG_compute_bandpower, EEG_detect_artifact_proxy, EEG_estimate_drowsiness, EEG_estimate_sleep_stage_features, EEG_screen_seizure_like_activity
from biosignal_agent.tools.emg_tools import EMG_assess_quality, EMG_detect_bursts, EMG_estimate_fatigue, EMG_summarize_activation
from biosignal_agent.tools.pcg_tools import PCG_assess_quality, PCG_detect_heart_sounds, PCG_screen_murmur_proxy, PCG_segment_s1_s2_proxy
from biosignal_agent.tools.ppg_tools import PPG_assess_quality, PPG_assess_perfusion_variability, PPG_detect_peaks, PPG_estimate_respiration_modulation
from biosignal_agent.tools.resp_tools import RESP_assess_quality, RESP_detect_apnea, RESP_detect_hypopnea, RESP_estimate_rate, RESP_screen_rate_pattern
from biosignal_agent.tools.scg_tools import SCG_assess_quality, SCG_detect_j_peaks, SCG_estimate_respiration
from biosignal_agent.tools.spo2_tools import SpO2_assess_hypoxemia_burden, SpO2_assess_quality, SpO2_detect_desaturation, SpO2_summarize

TOOLS = {
    "Signal_detect_artifacts": Signal_detect_artifacts,
    "ECG_assess_quality": ECG_assess_quality,
    "ECG_detect_r_peaks": ECG_detect_r_peaks,
    "ECG_compute_hrv": ECG_compute_hrv,
    "ECG_screen_arrhythmia": ECG_screen_arrhythmia,
    "ECG_screen_sleep_apnea": ECG_screen_sleep_apnea,
    "ECG_measure_morphology_intervals": ECG_measure_morphology_intervals,
    "PPG_assess_quality": PPG_assess_quality,
    "PPG_detect_peaks": PPG_detect_peaks,
    "PPG_assess_perfusion_variability": PPG_assess_perfusion_variability,
    "PPG_estimate_respiration_modulation": PPG_estimate_respiration_modulation,
    "BCG_assess_quality": BCG_assess_quality,
    "BCG_detect_j_peaks": BCG_detect_j_peaks,
    "BCG_estimate_respiration": BCG_estimate_respiration,
    "SCG_assess_quality": SCG_assess_quality,
    "SCG_detect_j_peaks": SCG_detect_j_peaks,
    "SCG_estimate_respiration": SCG_estimate_respiration,
    "RESP_assess_quality": RESP_assess_quality,
    "RESP_estimate_rate": RESP_estimate_rate,
    "RESP_detect_apnea": RESP_detect_apnea,
    "RESP_detect_hypopnea": RESP_detect_hypopnea,
    "RESP_screen_rate_pattern": RESP_screen_rate_pattern,
    "SpO2_assess_quality": SpO2_assess_quality,
    "SpO2_summarize": SpO2_summarize,
    "SpO2_detect_desaturation": SpO2_detect_desaturation,
    "SpO2_assess_hypoxemia_burden": SpO2_assess_hypoxemia_burden,
    "ABP_assess_quality": ABP_assess_quality,
    "ABP_detect_pulses": ABP_detect_pulses,
    "ABP_screen_pressure_events": ABP_screen_pressure_events,
    "ABP_compute_hemodynamics": ABP_compute_hemodynamics,
    "PCG_assess_quality": PCG_assess_quality,
    "PCG_detect_heart_sounds": PCG_detect_heart_sounds,
    "PCG_screen_murmur_proxy": PCG_screen_murmur_proxy,
    "PCG_segment_s1_s2_proxy": PCG_segment_s1_s2_proxy,
    "ACC_assess_quality": ACC_assess_quality,
    "ACC_summarize_activity": ACC_summarize_activity,
    "ACC_estimate_sleep_wake": ACC_estimate_sleep_wake,
    "ACC_detect_activity_bouts": ACC_detect_activity_bouts,
    "ACC_detect_fall_proxy": ACC_detect_fall_proxy,
    "EDA_assess_quality": EDA_assess_quality,
    "EDA_summarize": EDA_summarize,
    "EDA_detect_arousal_events": EDA_detect_arousal_events,
    "EEG_assess_quality": EEG_assess_quality,
    "EEG_compute_bandpower": EEG_compute_bandpower,
    "EEG_estimate_sleep_stage_features": EEG_estimate_sleep_stage_features,
    "EEG_screen_seizure_like_activity": EEG_screen_seizure_like_activity,
    "EEG_estimate_drowsiness": EEG_estimate_drowsiness,
    "EEG_detect_artifact_proxy": EEG_detect_artifact_proxy,
    "EMG_assess_quality": EMG_assess_quality,
    "EMG_summarize_activation": EMG_summarize_activation,
    "EMG_estimate_fatigue": EMG_estimate_fatigue,
    "EMG_detect_bursts": EMG_detect_bursts,
}

WORKFLOWS = {
    "ecg": ["ECG_assess_quality", "Signal_detect_artifacts", "ECG_detect_r_peaks", "ECG_compute_hrv", "ECG_screen_arrhythmia", "ECG_screen_sleep_apnea", "ECG_measure_morphology_intervals"],
    "ppg": ["PPG_assess_quality", "Signal_detect_artifacts", "PPG_detect_peaks", "PPG_assess_perfusion_variability", "PPG_estimate_respiration_modulation"],
    "bcg": ["BCG_assess_quality", "Signal_detect_artifacts", "BCG_detect_j_peaks", "BCG_estimate_respiration"],
    "scg": ["SCG_assess_quality", "Signal_detect_artifacts", "SCG_detect_j_peaks", "SCG_estimate_respiration"],
    "resp": ["RESP_assess_quality", "Signal_detect_artifacts", "RESP_estimate_rate", "RESP_detect_apnea", "RESP_detect_hypopnea", "RESP_screen_rate_pattern"],
    "spo2": ["SpO2_assess_quality", "Signal_detect_artifacts", "SpO2_summarize", "SpO2_detect_desaturation", "SpO2_assess_hypoxemia_burden"],
    "abp": ["ABP_assess_quality", "Signal_detect_artifacts", "ABP_detect_pulses", "ABP_screen_pressure_events", "ABP_compute_hemodynamics"],
    "pcg": ["PCG_assess_quality", "Signal_detect_artifacts", "PCG_detect_heart_sounds", "PCG_screen_murmur_proxy", "PCG_segment_s1_s2_proxy"],
    "acc": ["ACC_assess_quality", "Signal_detect_artifacts", "ACC_summarize_activity", "ACC_estimate_sleep_wake", "ACC_detect_activity_bouts", "ACC_detect_fall_proxy"],
    "eda": ["EDA_assess_quality", "Signal_detect_artifacts", "EDA_summarize", "EDA_detect_arousal_events"],
    "eeg": ["EEG_assess_quality", "Signal_detect_artifacts", "EEG_compute_bandpower", "EEG_estimate_sleep_stage_features", "EEG_screen_seizure_like_activity", "EEG_estimate_drowsiness", "EEG_detect_artifact_proxy"],
    "emg": ["EMG_assess_quality", "Signal_detect_artifacts", "EMG_summarize_activation", "EMG_estimate_fatigue", "EMG_detect_bursts"],
}
