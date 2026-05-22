# BioSignalAgent Task Catalog

This catalog separates runnable baseline tasks from future labeled benchmarks. The current baseline tools are screening or feature-extraction utilities for agent planning research, not diagnostic models.

## Runnable Baseline Tasks

| Task | Modalities | Current Tools | Output Type | Notes |
| --- | --- | --- | --- | --- |
| Signal quality triage | ECG, PPG, BCG, SCG, RESP, SpO2, ABP, PCG, ACC, EDA, EEG, EMG | `*_assess_quality` | quality/confidence | First tool in every workflow. |
| Generic artifact gate | all single-signal modalities | `Signal_detect_artifacts` | clipping/flatline/jump/noise flags | Useful before downstream tools, especially wearable signals. |
| Heart or pulse rate | ECG, PPG, BCG, SCG, ABP, PCG | peak/pulse/sound detectors | rate and event indices | ECG uses Pan-Tompkins; PPG/BCG/SCG use Nabian-style peak screening. |
| PPG perfusion/variability proxy | PPG | `PPG_assess_perfusion_variability` | amplitude proxy, interval CV, low-perfusion flag | Needs calibrated PPG for true perfusion interpretation. |
| HRV/autonomic features | ECG | `ECG_compute_hrv` | RR, SDNN, RMSSD | Feature extraction, not disease classification. |
| Arrhythmia screening | ECG | `ECG_screen_arrhythmia` | brady/tachy/irregular/pause flags | RR-interval heuristic only; next step is labeled rhythm data. |
| ECG morphology intervals | ECG | `ECG_measure_morphology_intervals` | PR/QRS/QT/QTc/ST proxies | Delineation heuristic; useful for morphology-aware planning cases. |
| Respiratory rate | RESP | `RESP_estimate_rate` | breaths/min | Baseline peak method. |
| Respiratory pattern screen | RESP | `RESP_screen_rate_pattern` | tachypnea/bradypnea/irregular/periodic flags | Pattern heuristic for respiratory workflow routing. |
| Sleep apnea breathing pauses | RESP | `RESP_detect_apnea` | apnea-like event count/index | Envelope-drop heuristic; needs apnea labels for true AHI validation. |
| Hypopnea-like reduced breathing | RESP | `RESP_detect_hypopnea` | hypopnea-like event count/index | Reduced-envelope heuristic; useful for PSG respiratory-event benchmarks. |
| Oxygen desaturation burden | SpO2 | `SpO2_detect_desaturation` | ODI, time below 90%, desat events | Useful for sleep-apnea support tasks. |
| Hypoxemia burden | SpO2 | `SpO2_assess_hypoxemia_burden` | time below 90/88%, nadir, burden level | Threshold-burden proxy, not clinical diagnosis. |
| Sleep-stage EEG features | EEG | `EEG_compute_bandpower`, `EEG_estimate_sleep_stage_features` | band ratios and coarse stage hint | Not a full AASM sleep-stage classifier. |
| Seizure-like EEG proxy | EEG | `EEG_screen_seizure_like_activity` | robust spike count, fast-power ratio | Research heuristic only; not seizure diagnosis. |
| Sleep/wake actigraphy proxy | ACC | `ACC_estimate_sleep_wake` | rest/activity hint | Coarse proxy, useful as a second signal in sleep sessions. |
| Blood-pressure pulse summary | ABP | `ABP_detect_pulses` | pulse HR, approximate systolic/diastolic values | Baseline ABP pulse features. |
| Pressure-event proxy | ABP | `ABP_screen_pressure_events` | hypotension/hypertension proxy flags | Needs calibrated ABP and clinical context. |
| Hemodynamic summary | ABP | `ABP_compute_hemodynamics` | MAP and pulse-pressure proxies | Requires calibrated ABP units for clinical meaning. |
| Heart-sound timing | PCG | `PCG_detect_heart_sounds` | sound events and HR | Baseline PCG timing. |
| Murmur proxy | PCG | `PCG_screen_murmur_proxy` | high-frequency ratio, murmur risk proxy | Needs validated PCG segmentation and labels. |
| Stress/arousal proxy | EDA, HRV, ACC | `EDA_summarize`, `EDA_detect_arousal_events`, `ECG_compute_hrv`, `ACC_summarize_activity` | tonic/phasic/SCR/motion features | Needs multimodal session-level task. |
| Muscle activation | EMG | `EMG_summarize_activation` | RMS/MAV | Baseline activation features. |
| Muscle fatigue proxy | EMG | `EMG_estimate_fatigue` | median frequency, fatigue hint | Needs task protocol and normalization. |

## High-Value Labeled Benchmarks To Add Next

| Benchmark Goal | Suggested Public Data Direction | Needed Labels | Target Metrics |
| --- | --- | --- | --- |
| Arrhythmia classification | MIT-BIH Arrhythmia, AFDB, CinC rhythm datasets | beat/rhythm labels | sensitivity, specificity, F1 by rhythm class |
| MIT-BIH arrhythmia windows | Implemented via `scripts/prepare_labeled_arrhythmia_dataset.py` | annotation-derived normal/abnormal window labels | current RR-screen baseline: F1 0.680 on 240 windows |
| Sleep apnea detection | Apnea-ECG, Sleep-EDF/SHHS-style PSG resources where accessible | apnea/hypopnea events or AHI | event F1, AHI MAE, subject-level severity accuracy |
| Apnea-ECG minute labels | Implemented via `scripts/prepare_apnea_ecg_dataset.py` | minute-level apnea/normal ECG labels | current ECG-only proxy: F1 0.125 on 60 windows |
| UCDDB RESP/SpO2 windows | Implemented via `scripts/prepare_ucddb_resp_spo2_dataset.py` | PSG respiratory-event labels with Flow and SpO2 channels | current RESP/SpO2 baseline with hypopnea tool: F1 0.588 on 40 windows |
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
