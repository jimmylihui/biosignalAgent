# BioSignalAgent Task Catalog

This catalog separates runnable baseline tasks from future labeled benchmarks. The current baseline tools are screening or feature-extraction utilities for agent planning research, not diagnostic models.

## Runnable Baseline Tasks

| Task | Modalities | Current Tools | Output Type | Notes |
| --- | --- | --- | --- | --- |
| Signal quality triage | ECG, PPG, BCG, SCG, RESP, SpO2, ABP, PCG, ACC, EDA, EEG, EMG | `*_assess_quality` | quality/confidence | First tool in every workflow. |
| Heart or pulse rate | ECG, PPG, BCG, SCG, ABP, PCG | peak/pulse/sound detectors | rate and event indices | ECG uses Pan-Tompkins; PPG/BCG/SCG use Nabian-style peak screening. |
| HRV/autonomic features | ECG | `ECG_compute_hrv` | RR, SDNN, RMSSD | Feature extraction, not disease classification. |
| Arrhythmia screening | ECG | `ECG_screen_arrhythmia` | brady/tachy/irregular/pause flags | RR-interval heuristic only; next step is labeled rhythm data. |
| Respiratory rate | RESP | `RESP_estimate_rate` | breaths/min | Baseline peak method. |
| Sleep apnea breathing pauses | RESP | `RESP_detect_apnea` | apnea-like event count/index | Envelope-drop heuristic; needs apnea labels for true AHI validation. |
| Oxygen desaturation burden | SpO2 | `SpO2_detect_desaturation` | ODI, time below 90%, desat events | Useful for sleep-apnea support tasks. |
| Sleep-stage EEG features | EEG | `EEG_compute_bandpower`, `EEG_estimate_sleep_stage_features` | band ratios and coarse stage hint | Not a full AASM sleep-stage classifier. |
| Sleep/wake actigraphy proxy | ACC | `ACC_estimate_sleep_wake` | rest/activity hint | Coarse proxy, useful as a second signal in sleep sessions. |
| Blood-pressure pulse summary | ABP | `ABP_detect_pulses` | pulse HR, approximate systolic/diastolic values | Future: hypotension/hypertension event rules. |
| Heart-sound timing | PCG | `PCG_detect_heart_sounds` | sound events and HR | Future: murmur/valve-screening features. |
| Stress/arousal proxy | EDA, HRV, ACC | `EDA_summarize`, `ECG_compute_hrv`, `ACC_summarize_activity` | tonic/phasic/motion features | Needs multimodal session-level task. |
| Muscle activation | EMG | `EMG_summarize_activation` | RMS/MAV | Future: gesture/fatigue/task classification. |

## High-Value Labeled Benchmarks To Add Next

| Benchmark Goal | Suggested Public Data Direction | Needed Labels | Target Metrics |
| --- | --- | --- | --- |
| Arrhythmia classification | MIT-BIH Arrhythmia, AFDB, CinC rhythm datasets | beat/rhythm labels | sensitivity, specificity, F1 by rhythm class |
| Sleep apnea detection | Apnea-ECG, Sleep-EDF/SHHS-style PSG resources where accessible | apnea/hypopnea events or AHI | event F1, AHI MAE, subject-level severity accuracy |
| Sleep staging | Sleep-EDF, PhysioNet sleep datasets | 30 s sleep stages | epoch accuracy, macro-F1, Cohen kappa |
| Desaturation detection | PSG datasets with SpO2 and scored respiratory events | desaturation/apnea events | ODI error, event precision/recall |
| Stress/arousal | WESAD or multimodal EDA/ECG/ACC datasets | stress/arousal labels | balanced accuracy, macro-F1 |
| Seizure or abnormal EEG screening | CHB-MIT or other EEG event datasets | seizure/event labels | event sensitivity, false positives/hour |
| PCG murmur screening | PhysioNet/CinC heart sound challenge data | normal/abnormal or murmur labels | AUROC, sensitivity/specificity |
| EMG gesture/fatigue | public EMG gesture/fatigue datasets | gesture/fatigue labels | accuracy, macro-F1 |

## Planning Direction

Short-term planning eval should include both single-modality tasks and cross-modality sessions, for example:

- ECG arrhythmia question routes to ECG quality, R peaks, HRV, arrhythmia screening.
- Sleep apnea question routes to RESP apnea screening and SpO2 desaturation burden.
- Sleep-stage question routes to EEG bandpower/stage features and ACC sleep/wake proxy.
- Stress question routes to EDA summary, ECG HRV, and ACC activity.

The baseline tools make these workflows executable now; labeled datasets should replace heuristic correctness with clinical or task-specific metrics.
