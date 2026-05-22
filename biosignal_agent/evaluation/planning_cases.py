from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlanningCase:
    case_id: str
    question: str
    modality: str
    expected_tools: tuple[str, ...]



DEFAULT_PLANNING_CASES = [
    PlanningCase(
        case_id='ecg_quality',
        question='Assess ECG signal quality',
        modality='ecg',
        expected_tools=('ECG_assess_quality',),
    ),
    PlanningCase(
        case_id='ecg_hrv',
        question='Estimate ECG heart rate and HRV from this signal',
        modality='ecg',
        expected_tools=('ECG_assess_quality', 'ECG_detect_r_peaks', 'ECG_compute_hrv'),
    ),
    PlanningCase(
        case_id='ecg_r_peaks',
        question='Detect R peaks and estimate heart rate in this ECG waveform',
        modality='ecg',
        expected_tools=('ECG_assess_quality', 'ECG_detect_r_peaks'),
    ),
    PlanningCase(
        case_id='ecg_summary',
        question='Analyze this ECG signal and summarize heart rate variability features',
        modality='ecg',
        expected_tools=('ECG_assess_quality', 'ECG_detect_r_peaks', 'ECG_compute_hrv'),
    ),
    PlanningCase(
        case_id='ppg_quality',
        question='Assess PPG signal quality',
        modality='ppg',
        expected_tools=('PPG_assess_quality',),
    ),
    PlanningCase(
        case_id='ppg_pulse',
        question='Find pulse peaks and estimate heart rate from this PPG waveform',
        modality='ppg',
        expected_tools=('PPG_assess_quality', 'PPG_detect_peaks'),
    ),
    PlanningCase(
        case_id='ppg_hr',
        question='Estimate heart rate from this photoplethysmography pulse signal',
        modality='ppg',
        expected_tools=('PPG_assess_quality', 'PPG_detect_peaks'),
    ),
    PlanningCase(
        case_id='ppg_summary',
        question='Analyze this PPG signal and summarize pulse rate confidence',
        modality='ppg',
        expected_tools=('PPG_assess_quality', 'PPG_detect_peaks'),
    ),
    PlanningCase(
        case_id='bcg_quality',
        question='Assess BCG signal quality',
        modality='bcg',
        expected_tools=('BCG_assess_quality',),
    ),
    PlanningCase(
        case_id='bcg_j_peaks',
        question='Detect BCG J peaks and estimate heart rate',
        modality='bcg',
        expected_tools=('BCG_assess_quality', 'BCG_detect_j_peaks'),
    ),
    PlanningCase(
        case_id='bcg_hr',
        question='Estimate heart rate from this BCG mechanical cardiac signal',
        modality='bcg',
        expected_tools=('BCG_assess_quality', 'BCG_detect_j_peaks'),
    ),
    PlanningCase(
        case_id='bcg_summary',
        question='Analyze this BCG waveform and summarize J-peak based heart rate',
        modality='bcg',
        expected_tools=('BCG_assess_quality', 'BCG_detect_j_peaks'),
    ),
    PlanningCase(
        case_id='scg_quality',
        question='Assess SCG signal quality',
        modality='scg',
        expected_tools=('SCG_assess_quality',),
    ),
    PlanningCase(
        case_id='scg_j_peaks',
        question='Detect SCG J peaks and estimate heart rate',
        modality='scg',
        expected_tools=('SCG_assess_quality', 'SCG_detect_j_peaks'),
    ),
    PlanningCase(
        case_id='scg_hr',
        question='Estimate heart rate from this seismocardiogram mechanical cardiac signal',
        modality='scg',
        expected_tools=('SCG_assess_quality', 'SCG_detect_j_peaks'),
    ),
    PlanningCase(
        case_id='scg_summary',
        question='Analyze this SCG waveform and summarize J-peak based heart rate',
        modality='scg',
        expected_tools=('SCG_assess_quality', 'SCG_detect_j_peaks'),
    ),
    PlanningCase(
        case_id='resp_quality',
        question='Assess RESP signal quality',
        modality='resp',
        expected_tools=('RESP_assess_quality',),
    ),
    PlanningCase(
        case_id='resp_rate',
        question='Estimate respiratory rate from this breathing signal',
        modality='resp',
        expected_tools=('RESP_assess_quality', 'RESP_estimate_rate'),
    ),
    PlanningCase(
        case_id='spo2_quality',
        question='Assess SpO2 signal quality',
        modality='spo2',
        expected_tools=('SpO2_assess_quality',),
    ),
    PlanningCase(
        case_id='spo2_summary',
        question='Summarize oxygen saturation and desaturation from this SpO2 signal',
        modality='spo2',
        expected_tools=('SpO2_assess_quality', 'SpO2_summarize', 'SpO2_detect_desaturation'),
    ),
    PlanningCase(
        case_id='abp_quality',
        question='Assess ABP signal quality',
        modality='abp',
        expected_tools=('ABP_assess_quality',),
    ),
    PlanningCase(
        case_id='abp_pulses',
        question='Detect arterial blood pressure pulses and estimate heart rate',
        modality='abp',
        expected_tools=('ABP_assess_quality', 'ABP_detect_pulses'),
    ),
    PlanningCase(
        case_id='pcg_quality',
        question='Assess PCG signal quality',
        modality='pcg',
        expected_tools=('PCG_assess_quality',),
    ),
    PlanningCase(
        case_id='pcg_sounds',
        question='Detect heart sounds in this phonocardiogram and estimate heart rate',
        modality='pcg',
        expected_tools=('PCG_assess_quality', 'PCG_detect_heart_sounds'),
    ),
    PlanningCase(
        case_id='acc_quality',
        question='Assess ACC signal quality',
        modality='acc',
        expected_tools=('ACC_assess_quality',),
    ),
    PlanningCase(
        case_id='acc_activity',
        question='Summarize accelerometer activity and motion level',
        modality='acc',
        expected_tools=('ACC_assess_quality', 'ACC_summarize_activity'),
    ),
    PlanningCase(
        case_id='eda_quality',
        question='Assess EDA signal quality',
        modality='eda',
        expected_tools=('EDA_assess_quality',),
    ),
    PlanningCase(
        case_id='eda_summary',
        question='Summarize skin conductance tonic and phasic activity',
        modality='eda',
        expected_tools=('EDA_assess_quality', 'EDA_summarize'),
    ),
    PlanningCase(
        case_id='eeg_quality',
        question='Assess EEG signal quality',
        modality='eeg',
        expected_tools=('EEG_assess_quality',),
    ),
    PlanningCase(
        case_id='eeg_bandpower',
        question='Compute EEG alpha beta theta delta bandpower',
        modality='eeg',
        expected_tools=('EEG_assess_quality', 'EEG_compute_bandpower'),
    ),
    PlanningCase(
        case_id='emg_quality',
        question='Assess EMG signal quality',
        modality='emg',
        expected_tools=('EMG_assess_quality',),
    ),
    PlanningCase(
        case_id='emg_activation',
        question='Summarize EMG muscle activation and RMS',
        modality='emg',
        expected_tools=('EMG_assess_quality', 'EMG_summarize_activation'),
    ),
    PlanningCase(
        case_id='ecg_arrhythmia_screen',
        question='Screen this ECG for arrhythmia patterns such as irregular rhythm, pauses, tachycardia, or bradycardia',
        modality='ecg',
        expected_tools=('ECG_assess_quality', 'ECG_detect_r_peaks', 'ECG_compute_hrv', 'ECG_screen_arrhythmia'),
    ),
    PlanningCase(
        case_id='ecg_afib_proxy',
        question='Check whether this ECG has an irregular RR rhythm pattern suggestive of atrial fibrillation screening concern',
        modality='ecg',
        expected_tools=('ECG_assess_quality', 'ECG_detect_r_peaks', 'ECG_compute_hrv', 'ECG_screen_arrhythmia'),
    ),
    PlanningCase(
        case_id='resp_sleep_apnea',
        question='Detect sleep apnea-like breathing pauses from this respiration signal',
        modality='resp',
        expected_tools=('RESP_assess_quality', 'RESP_estimate_rate', 'RESP_detect_apnea'),
    ),
    PlanningCase(
        case_id='spo2_desaturation',
        question='Estimate oxygen desaturation burden and ODI from this SpO2 signal',
        modality='spo2',
        expected_tools=('SpO2_assess_quality', 'SpO2_summarize', 'SpO2_detect_desaturation'),
    ),
    PlanningCase(
        case_id='spo2_apnea_support',
        question='Look for SpO2 drops below 90 percent that could support sleep apnea screening',
        modality='spo2',
        expected_tools=('SpO2_assess_quality', 'SpO2_summarize', 'SpO2_detect_desaturation'),
    ),
    PlanningCase(
        case_id='eeg_sleep_stage_features',
        question='Estimate sleep stage features from this EEG epoch using delta theta alpha beta power',
        modality='eeg',
        expected_tools=('EEG_assess_quality', 'EEG_compute_bandpower', 'EEG_estimate_sleep_stage_features'),
    ),
    PlanningCase(
        case_id='acc_sleep_wake',
        question='Use accelerometer actigraphy to estimate sleep versus wake or rest versus activity',
        modality='acc',
        expected_tools=('ACC_assess_quality', 'ACC_summarize_activity', 'ACC_estimate_sleep_wake'),
    ),
    PlanningCase(
        case_id='ecg_sleep_apnea_proxy',
        question='Screen this ECG sleep segment for sleep apnea risk using heart rate variability patterns',
        modality='ecg',
        expected_tools=('ECG_assess_quality', 'ECG_detect_r_peaks', 'ECG_compute_hrv', 'ECG_screen_sleep_apnea'),
    ),

    PlanningCase(
        case_id='ppg_perfusion',
        question='Assess PPG pulse amplitude, perfusion, and pulse variability',
        modality='ppg',
        expected_tools=('PPG_assess_quality', 'PPG_detect_peaks', 'PPG_assess_perfusion_variability'),
    ),
    PlanningCase(
        case_id='abp_pressure_events',
        question='Screen ABP for hypotension or hypertension pressure events',
        modality='abp',
        expected_tools=('ABP_assess_quality', 'ABP_detect_pulses', 'ABP_screen_pressure_events'),
    ),
    PlanningCase(
        case_id='pcg_murmur_proxy',
        question='Screen this phonocardiogram for murmur or abnormal heart sound patterns',
        modality='pcg',
        expected_tools=('PCG_assess_quality', 'PCG_detect_heart_sounds', 'PCG_screen_murmur_proxy'),
    ),
    PlanningCase(
        case_id='eda_arousal_events',
        question='Detect EDA arousal events and skin conductance responses',
        modality='eda',
        expected_tools=('EDA_assess_quality', 'EDA_summarize', 'EDA_detect_arousal_events'),
    ),
    PlanningCase(
        case_id='emg_fatigue_proxy',
        question='Estimate EMG muscle fatigue using median frequency and RMS',
        modality='emg',
        expected_tools=('EMG_assess_quality', 'EMG_summarize_activation', 'EMG_estimate_fatigue'),
    ),
    PlanningCase(
        case_id='eeg_seizure_like_proxy',
        question='Screen EEG for seizure-like spikes or epileptiform abnormal EEG activity',
        modality='eeg',
        expected_tools=('EEG_assess_quality', 'EEG_compute_bandpower', 'EEG_screen_seizure_like_activity'),
    ),
    PlanningCase(
        case_id='resp_hypopnea',
        question='Detect hypopnea-like shallow breathing and reduced respiration events',
        modality='resp',
        expected_tools=('RESP_assess_quality', 'RESP_estimate_rate', 'RESP_detect_hypopnea'),
    ),
    PlanningCase(
        case_id='spo2_hypoxemia_burden',
        question='Assess SpO2 hypoxemia burden including time below 88 percent',
        modality='spo2',
        expected_tools=('SpO2_assess_quality', 'SpO2_summarize', 'SpO2_assess_hypoxemia_burden'),
    ),
]
