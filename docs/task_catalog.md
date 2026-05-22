# BioSignalAgent Task Catalog

This catalog separates runnable baseline tasks from future labeled benchmarks. The current baseline tools are screening or feature-extraction utilities for agent planning research, not diagnostic models.

## Runnable Baseline Tasks

| Task | Modalities | Current Tools | Output Type | Notes |
| --- | --- | --- | --- | --- |
| Signal quality triage | ECG, PPG, BCG, SCG, RESP, SpO2, ABP, PCG, ACC, EDA, EEG, EMG | `*_assess_quality` | quality/confidence | First tool in every workflow. |
| Generic artifact gate | all single-signal modalities | `Signal_detect_artifacts` | clipping/flatline/jump/noise flags | Useful before downstream tools, especially wearable signals. |
| Heart or pulse rate | ECG, PPG, BCG, SCG, ABP, PCG | peak/pulse/sound detectors | rate and event indices | ECG uses Pan-Tompkins; PPG/BCG/SCG use Nabian-style peak screening. |
| PPG perfusion/variability proxy | PPG | `PPG_assess_perfusion_variability` | amplitude proxy, interval CV, low-perfusion flag | Needs calibrated PPG for true perfusion interpretation. |
| PPG irregular-pulse proxy | PPG | `PPG_screen_pulse_irregularity` | interval CV, normalized RMSSD, successive-change fraction | Statistical AF-like pulse irregularity baseline; ECG labels required for true AF screening. |
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
| Murmur proxy | PCG | `PCG_screen_murmur_proxy`, `PCG_extract_murmur_features` | high-frequency ratio, spectral/envelope/timing features | Feature baseline supports a small logistic-regression benchmark; needs larger PCG labels. |
| Stress/arousal proxy | EDA, HRV, ACC | `EDA_summarize`, `EDA_detect_arousal_events`, `EDA_screen_stress_proxy`, `ECG_compute_hrv`, `ACC_summarize_activity` | tonic/phasic/SCR/motion features and score | Needs WESAD-style labels for real stress classification. |
| Muscle activation | EMG | `EMG_summarize_activation` | RMS/MAV | Baseline activation features. |
| Muscle fatigue proxy | EMG | `EMG_estimate_fatigue` | median frequency, fatigue hint | Needs task protocol and normalization. |
| PPG-derived respiration proxy | PPG | `PPG_estimate_respiration_modulation` | respiratory rate, modulation index | Uses PPG envelope respiratory-band modulation; proxy only. |
| BCG/SCG-derived respiration proxy | BCG, SCG | `BCG_estimate_respiration`, `SCG_estimate_respiration` | respiratory rate, respiration power ratio | Mechanical respiratory modulation heuristic. |
| PCG S1/S2 segmentation proxy | PCG | `PCG_segment_s1_s2_proxy` | S1/S2 counts, systole/diastole timing | Alternating-peak segmentation baseline; not a validated heart-sound segmenter. |
| Activity bouts and fall/impact proxy | ACC | `ACC_detect_activity_bouts`, `ACC_detect_fall_proxy` | bout counts, impact events, fall risk proxy | Wearable activity/fall screening baseline; requires device orientation/context for real deployment. |
| EEG drowsiness and artifact proxy | EEG | `EEG_estimate_drowsiness`, `EEG_detect_artifact_proxy` | theta/alpha, slow power, artifact level | Useful for vigilance and QC planning cases; not a clinical EEG reader. |
| EMG burst/onset proxy | EMG | `EMG_detect_bursts` | burst count/rate/duration | RMS-envelope burst heuristic. |


## Current Major-Task Eval Snapshot

Latest rule-planner major-task outputs:

| Eval | Records/Sessions | Runs | Retrieval | Planning | Execution |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single-modality planning cases | n/a | 63 cases | 1.000 | 1.000 | n/a |
| Real-world manifest | 25 records | 155 case runs | 1.000 | 1.000 | 1.000 |
| Dedicated common manifest | 21 records | 96 case runs | 1.000 | 1.000 | 1.000 |
| Dedicated BCG manifest | 3 records | 15 case runs | 1.000 | 1.000 | 1.000 |
| Cross-modality session benchmark | 9 sessions | 26 signal runs | 1.000 | 1.000 | 1.000 |
| Tool-output audit | 49 records | 274 tool runs | n/a | n/a | 1.000 ok rate, 0 errors, 0 low-confidence |

Report artifacts are written under `/data1/jiahui/biosignal-agent/outputs/*major_tasks*`.

## Session-Level Tasks

| Task | Modalities | Current Tools | Output Type | Notes |
| --- | --- | --- | --- | --- |
| Pulse arrival timing | ECG + PPG | `Session_compute_ecg_ppg_pulse_arrival` | median PAT, IQR, paired pulses | Uses synchronized BIDMC ECG+PPG in the session benchmark; proxy for timing research, not BP estimation. |
| Multimodal sleep-apnea screening | ECG + RESP + SpO2 | `Session_screen_sleep_apnea_multimodal` | fused risk, flags, component evidence | Research proxy combining ECG HRV, RESP events/pattern, and SpO2 burden. |

## High-Value Labeled Benchmarks To Add Next

| Benchmark Goal | Suggested Public Data Direction | Needed Labels | Target Metrics |
| --- | --- | --- | --- |
| Arrhythmia classification | MIT-BIH Arrhythmia, AFDB, CinC rhythm datasets | beat/rhythm labels | sensitivity, specificity, F1 by rhythm class |
| MIT-BIH arrhythmia windows | Implemented via `scripts/prepare_labeled_arrhythmia_dataset.py` | annotation-derived normal/abnormal window labels | current RR-screen baseline: F1 0.680 on 240 windows |
| MIT-BIH rhythm/AF + beat windows | Implemented via `scripts/prepare_ecg_rhythm_beat_dataset.py` | rhythm aux notes and beat symbols | current AF F1 0.442 on 240 windows; beat-abnormal F1 0.308 on 18,209 beats |
| Sleep apnea detection | Apnea-ECG, Sleep-EDF/SHHS-style PSG resources where accessible | apnea/hypopnea events or AHI | event F1, AHI MAE, subject-level severity accuracy |
| Apnea-ECG minute labels | Implemented via `scripts/prepare_apnea_ecg_dataset.py` | minute-level apnea/normal ECG labels | current ECG-only proxy: F1 0.125 on 60 windows |
| UCDDB RESP/SpO2 windows | Implemented via `scripts/prepare_ucddb_resp_spo2_dataset.py` | PSG respiratory-event labels with Flow and SpO2 channels | current RESP/SpO2 baseline with hypopnea tool: F1 0.588 on 40 windows |
| UCDDB PSG sleep windows | Implemented via `scripts/prepare_psg_sleep_dataset.py` | sleep-stage labels plus respiratory-event labels using EEG/Flow/SpO2 | current coarse sleep-stage macro-F1 0.241 and respiratory-event F1 0.218 on 80 windows |
| Sleep staging | Sleep-EDF, PhysioNet sleep datasets | 30 s sleep stages | epoch accuracy, macro-F1, Cohen kappa |
| Desaturation detection | PSG datasets with SpO2 and scored respiratory events | desaturation/apnea events | ODI error, event precision/recall |
| Stress/arousal | WESAD or multimodal EDA/ECG/ACC datasets | stress/arousal labels | balanced accuracy, macro-F1 |
| Seizure or abnormal EEG screening | CHB-MIT or other EEG event datasets | seizure/event labels | event sensitivity, false positives/hour |
| PCG murmur screening | Implemented via `scripts/prepare_pcg_murmur_dataset.py` on PhysioNet/CinC 2016 | normal/abnormal labels | proxy F1 0.000; feature+logistic-regression baseline F1 0.750 on 10 balanced records |
| EMG gesture/fatigue | public EMG gesture/fatigue datasets | gesture/fatigue labels | accuracy, macro-F1 |

## Existing-Work Wrapper Sources

The next implementation layer is tracked in `biosignal_agent/tools/source_catalog.json` and summarized in `docs/tool_source_catalog.md`. The catalog maps each task to existing algorithms/libraries, public datasets, source URLs, wrapper priority, and the next wrapper to build.

## Planning Direction

Short-term planning eval should include both single-modality tasks and cross-modality sessions, for example:

- ECG arrhythmia question routes to ECG quality, R peaks, HRV, arrhythmia screening.
- Sleep apnea question routes to RESP apnea screening and SpO2 desaturation burden.
- Sleep-stage question routes to EEG bandpower/stage features and ACC sleep/wake proxy.
- Stress question routes to EDA summary, ECG HRV, and ACC activity.

The baseline tools make these workflows executable now; labeled datasets should replace heuristic correctness with clinical or task-specific metrics.
