# BioSignalAgent Task Catalog

This catalog separates runnable baseline tasks from future labeled benchmarks. The current baseline tools are screening or feature-extraction utilities for agent planning research, not diagnostic models.

## Runnable Baseline Tasks

| Task | Modalities | Current Tools | Output Type | Notes |
| --- | --- | --- | --- | --- |
| Signal quality triage | ECG, PPG, BCG, SCG, RESP, SpO2, ABP, PCG, ACC, EDA, EEG, EMG | `*_assess_quality` | quality/confidence | First tool in every workflow. |
| Generic artifact gate | all single-signal modalities | `Signal_detect_artifacts` | clipping/flatline/jump/noise flags | Useful before downstream tools, especially wearable signals. |
| Unknown signal modality routing | all single-signal modalities | `Signal_classify_modality` | predicted modality, score distribution, extracted features | Feature-based random-forest baseline trained on current public benchmark windows; falls back to heuristic scores when no model file exists. |
| Waveform image digitization | signal plot images | `Signal_digitize_waveform_image_ml` | digitized CSV waveform, pixel coverage, confidence | Current best exposed digitizer is the trained RGB pixel model: 43/43 ok with corr 0.851, NRMSE 0.056, peak-F1 0.828; rule and U-Net variants remain historical/experimental baselines and are not exposed as agent tools. ECG-Image-Kit generated benchmark adds 5 MIT-BIH-derived ECG images with bbox crops, waveform references, and plotted-pixel masks; specialized U-Net segmentation Dice 0.784, but waveform calibration is still pending. Resolution risk screening now routes low-res ECG/SCG/EEG/PCG/EMG images away from overconfident waveform reconstruction when needed. High-res ML digitization improves overall corr to 0.943 and raises ECG/SCG/EEG/PCG to 0.993/0.992/0.987/0.855. Ultra-high-res difficult-track ML reaches ECG/SCG/EEG/PCG/EMG 0.999/0.999/0.998/0.946/0.585; EMG remains image/spectrogram-task first. Added spectrogram-task baselines: PCG summary features accuracy 0.80/F1 0.75 and PCG spectrogram-image baseline accuracy 0.90/F1 0.889 on 10 records; EMG healthy/myopathy/neuropathy window-level smoke macro-F1 1.0 with summary features and 0.967 with spectrogram images on 30 windows from 3 records, marked non-subject-independent. |
| Heart or pulse rate | ECG, PPG, BCG, SCG, ABP, PCG | `ECG_estimate_heart_rate`, `PPG_estimate_heart_rate`, peak/pulse/sound detectors | rate and event indices | ECG and PPG expose dedicated HR wrappers over validated peak detectors. |
| PPG fiducials, PRV, and perfusion | PPG | `PPG_detect_fiducial_points`, `PPG_compute_prv`, `PPG_assess_perfusion_variability` | pyPPG on/sp/dn/dp fiducials with sample/time/amplitude, PRV, amplitude/perfusion flags | Fiducials use pyPPG when available and explicit fallback otherwise; PRV is not identical to ECG HRV and vascular interpretation needs calibrated/validated PPG. |
| PPG irregular-pulse / AF proxy | PPG | `PPG_screen_pulse_irregularity`, `PPG_detect_afib` | AF probability, interval CV, normalized RMSSD, successive-change fraction | Uses feature/DL interval models when available; ECG confirmation required for clinical AF. |
| PPG SpO2 proxy | PPG | `PPG_estimate_spo2` | red/IR ratio-of-ratios SpO2 proxy | Requires dual-wavelength red and infrared columns; uncalibrated unless device-specific calibration is available. |
| PPG BP, vascular, shock, sleep, stress, exercise proxies | PPG | `PPG_estimate_bp_proxy`, `PPG_assess_vascular_health`, `PPG_screen_low_perfusion_shock_risk`, `PPG_estimate_sleep_features`, `PPG_assess_stress_prv`, `PPG_estimate_exercise_intensity` | morphology/stiffness flags, low-perfusion risk, sleep/rest features, stress/recovery and intensity zones | Practical routing wrappers only; BP and vascular interpretation require calibration, sleep/stress need context/multimodal validation. |
| HRV/autonomic/stress-fatigue proxy | ECG | `ECG_compute_hrv`, `ECG_assess_stress_fatigue_hrv` | RR, SDNN, RMSSD, LF/HF, strain flags | HRV feature extraction plus nonspecific stress/fatigue proxy; needs baseline/context. |
| Arrhythmia/rhythm/AF screening | ECG | `ECG_screen_arrhythmia`, `ECG_classify_rhythm_segment`, `ECG_detect_afib`, `ECG_classify_beats` | segment rhythm, AF probability, beat rows/subtype summary, brady/tachy/irregular/pause flags | Uses existing deep/feature ECG models plus RR fallback; still screening, not diagnostic. |
| ECG morphology intervals, QT, conduction, ST proxy | ECG | `ECG_measure_morphology_intervals`, `ECG_delineate_waves_dl`, `ECG_analyze_qt_interval`, `ECG_screen_conduction_block`, `ECG_screen_ischemia_st` | PR/QRS/QT/QTc/ST proxies, QT/conduction/ST flags | Delineation heuristic plus experimental QTDB cached-90-record DL segmentation backend; QT/ST/conduction need full QTDB/LUDB/ST-T/PTB-XL validation before diagnostic use. |
| Respiratory rate | RESP | `RESP_estimate_rate` | breaths/min | Baseline peak method. |
| Respiratory pattern screen | RESP | `RESP_screen_rate_pattern` | tachypnea/bradypnea/irregular/periodic flags | Pattern heuristic for respiratory workflow routing. |
| Sleep apnea / respiratory-event screening | RESP (+ optional SpO2) | `RESP_screen_sleep_apnea_ml`, `RESP_detect_apnea` | UCDDB respiratory-event probability plus apnea-like event count/index | Flow+SpO2 UCDDB window-CV AUROC 0.724/macro-F1 0.647; not full AASM scoring or diagnostic. |
| Hypopnea-like reduced breathing | RESP | `RESP_detect_hypopnea`, `RESP_screen_sleep_apnea_ml` | reduced-envelope events plus ML respiratory-event probability | Heuristic remains fallback; ML fusion benefits from SpO2 when available. |
| Oxygen desaturation burden | SpO2 | `SpO2_detect_desaturation`, `SpO2_extract_oximetry_features` | ODI3/ODI4, time below 90/88/85%, desat depth/duration/area | Rolling-baseline ODI-style detector; useful for sleep-apnea support and overnight burden, not exact respiratory-event timing. |
| Hypoxemia burden / oximetry screening | SpO2 | `SpO2_assess_hypoxemia_burden`, `SpO2_screen_sleep_apnea_oximetry`, `SpO2_screen_sleep_apnea_ml` | T90/T88/T85, nadir, hypoxic burden area, ODI severity proxy, ML event probability | UCDDB SpO2-only 25-record GroupKFold: ML AUROC 0.542/macro-F1 0.523; heuristic AUROC 0.555. SpO2-only is weak for 30s event timing but useful for burden summaries. |
| Sleep-stage EEG features/classification | EEG | `EEG_compute_bandpower`, `EEG_estimate_sleep_stage_features`, `EEG_classify_sleep_stage_ml` | band ratios, coarse stage hint, UCDDB-trained coarse stage probabilities | UCDDB single-record window-CV baseline: accuracy 0.741/macro-F1 0.731/macro-AUROC 0.897; not AASM-equivalent or subject-independent. |
| Seizure-like EEG screening | EEG | `EEG_screen_seizure_ml`, `EEG_screen_seizure_like_activity` | CHB-MIT subset seizure probability plus robust spike/fast-power fallback | Small chb01 subset baseline: AUROC 0.959/macro-F1 0.923 on 114 windows with EDF-file grouped CV; not full CHB-MIT or clinical seizure detection. |
| Sleep/wake and activity actigraphy | ACC | `ACC_classify_activity_ml`, `ACC_extract_actigraphy_features`, `ACC_estimate_sleep_wake` | UCI-HAR activity probabilities plus rest/activity hint | Tri-axial UCI-HAR official split: accuracy 0.822/macro-F1 0.820; active/rest accuracy 1.0. Device placement can shift performance. |
| ABP fiducial detection | ABP | `ABP_detect_fiducial_points` | systolic onset/peak, dicrotic notch, diastolic peak, phase endpoint with sample/time/amplitude | Primitive ABP event detector; `ABP_detect_pulses` is retained only as a legacy summary wrapper. No beat-level reference labels yet. |
| Pressure-event proxy | ABP | `ABP_screen_pressure_events` | hypotension/hypertension proxy flags | Uses calibrated ABP units when available; still screening only and needs clinical context. |
| Hemodynamic summary | ABP | `ABP_compute_hemodynamics` | robust MAP and pulse-pressure proxies | BIDMC ABP plausibility benchmark available; requires calibrated ABP units for clinical meaning. |
| Heart-sound timing / HR | PCG | `PCG_detect_heart_sounds`, `PCG_estimate_heart_rate`, `PCG_segment_s1_s2_proxy` | S1/S2-like events, HR, systole/diastole timing | Springer-sounds supervised TCN is now default when the model file is present: fold0 S1 micro-F1 0.974, S2 micro-F1 0.952, HR MAE 0.949 bpm; falls back to duration-constrained envelope segmentation. |
| Murmur / valve / CHD proxy | PCG | `PCG_screen_murmur_proxy`, `PCG_screen_murmur_patient_multisite`, `PCG_extract_murmur_features`, `PCG_screen_valve_disease_proxy`, `PCG_screen_congenital_abnormality_proxy` | murmur probability/features, systolic/diastolic timing, broad valve-pattern candidates, pediatric structural-abnormality proxy | CirCor-style multi-location CNN plus feature proxy; valve subtype now emits BMD-HS AS/AR/MR/MS/N spectrogram-CNN probabilities plus feature-model fallback; 5-fold patient-heldout mean macro-F1 0.638 (+/-0.018) and mean macro-AUROC 0.754, still screening only. CHD/structural output now also includes CirCor outcome feature/CNN research backends: feature patient-level AUROC 0.666/F1 0.722 and CNN AUROC 0.582/F1 0.573; still screening only, not diagnosis. |
| Extra sounds / rhythm / heart-function proxy | PCG | `PCG_detect_s3_s4_proxy`, `PCG_assess_rhythm_irregularity`, `PCG_monitor_heart_function_proxy` | S3/S4 candidates, cycle variability, longitudinal HR/timing/murmur feature trends | Unvalidated auxiliary PCG feature wrappers; not replacements for ECG, echocardiography, or clinical diagnosis. |
| Stress/arousal proxy | EDA, HRV, ACC | `EDA_summarize`, `EDA_extract_tonic_phasic_features`, `EDA_detect_arousal_events`, `EDA_screen_stress_ml`, `EDA_classify_affective_state_ml`, `EDA_route_task_recommendation`, `EDA_screen_stress_proxy`, `ECG_compute_hrv`, `ACC_summarize_activity` | WESAD-trained EDA stress probability plus tonic/phasic/SCR/motion features | EDA-only WESAD subject-grouped CV: AUROC 0.870/BAcc 0.780; heuristic proxy remains fallback. Not diagnostic or standalone lie detection. |
| Muscle activation | EMG | `EMG_summarize_activation` | RMS/MAV | Baseline activation features. |
| Gesture/action recognition | EMG | `EMG_classify_gesture` | predicted gesture probabilities | UCI MYO 8-channel 6-class subject-independent baseline: accuracy 0.777, macro-F1 0.777; 7-class including sparse extended-palm accuracy 0.761/macro-F1 0.659. |
| Movement intent proxy | EMG | `EMG_predict_movement_intent` | upcoming gesture probabilities | UCI gesture onset-derived pre-onset proxy reaches 5-class accuracy/macro-F1 0.725; labels are inferred from transitions, not explicit intent. |
| Physical action screening | EMG | `EMG_classify_physical_action` | normal/aggressive-high-intensity probability | UCI Physical Action leave-one-subject-out binary AUROC 0.884, accuracy 0.827; 20-class exact action remains hard at accuracy 0.269. |
| Muscle fatigue proxy/model | EMG | `EMG_estimate_fatigue_ml`, `EMG_estimate_fatigue` | fatigue probability, median frequency, fatigue hint | Trained early-vs-late protocol model on Zenodo fatigue dataset reaches AUROC 0.774/BAcc 0.699; heuristic remains fallback. |
| Lower-limb exercise and rehab screening | EMG | `EMG_classify_lower_limb_exercise`, `EMG_classify_gait_speed`, `EMG_estimate_gait_phase`, `EMG_screen_knee_rehab_status`, `EMG_analyze_gait_activation` | exercise probabilities, knee abnormality probability, activation/coactivation features | UCI lower-limb EMG/goniometry subject-held-out baseline: exercise accuracy 0.836/macro-F1 0.835; knee status AUROC 0.930/macro-F1 0.921. GEDS full S00-S22 414-trial wearable gait baselines: speed accuracy 0.789/macro-F1 0.788; right stance/swing phase accuracy 0.974/macro-F1 0.972/AUROC 0.997. |
| Neuromuscular abnormality smoke screen | EMG | `EMG_screen_neuromuscular_abnormality` | healthy/myopathy/neuropathy probabilities, abnormal probability | EMGDB processed 10s snippets: window-level accuracy/macro-F1 1.0 on 27 windows; tiny non-subject-independent smoke benchmark only, not diagnostic. |
| PPG-derived respiration proxy | PPG | `PPG_estimate_respiration_modulation`, `PPG_estimate_sleep_features` | respiratory rate, modulation index, sleep-compatible respiration flag | Uses multi-source PPG respiratory modulation; proxy only and best validated against respiratory reference. |
| BCG HR/J-peak and HRV | BCG | `BCG_detect_j_peaks`, `BCG_compute_hrv` | J-peak indices, HR, SDNN/RMSSD/pNN50, LF/HF proxy | Figshare bed BCG 2025 paired heartbeat-reference benchmark, 46 subjects x 60s: HR MAE 10.04 bpm, median AE 2.88 bpm, within 5 bpm 0.614; harmonic/over-detection outliers are flagged. HRV remains BCG interval proxy. |
| BCG respiration, sleep/motion, rhythm, BP proxies | BCG | `BCG_estimate_respiration`, `BCG_estimate_sleep_features`, `BCG_assess_bed_presence_motion`, `BCG_screen_arrhythmia`, `BCG_estimate_bp_proxy`, `BCG_route_task_recommendation` | respiratory rate, motion burden, sleep-wake proxy, irregularity score, calibrated BP feature proxy | Respiration uses baseline plus Hilbert cardiac-envelope modulation. Sleep staging/AF/BP are conservative proxy features unless trained labels/calibration are available; current PSG/BCG benchmark audit found no directly usable public 30s PSG-labeled BCG sleep-stage dataset. Diagnostic routing should use ECG/clinical references. |
| BCG/SCG-derived respiration proxy | BCG, SCG | `BCG_estimate_respiration`, `SCG_estimate_respiration` | respiratory rate, respiration power ratio | Mechanical respiratory modulation heuristic. |
| PCG S1/S2 segmentation proxy | PCG | `PCG_segment_s1_s2_proxy` | S1/S2 counts, systole/diastole timing | Alternating-peak segmentation baseline; not a validated heart-sound segmenter. |
| Activity bouts and fall/impact proxy | ACC | `ACC_classify_activity_ml`, `ACC_detect_activity_bouts`, `ACC_detect_fall_proxy`, `ACC_detect_fall_ml` | activity class/probabilities, bout counts, impact events, fall risk proxy | UCI-HAR tri-axial activity ML is available; UniMiB fall ML is available (subject-independent BAcc 0.980/AUROC 0.998); proxy remains fallback for unmatched devices. |
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
| Pulse arrival timing | ECG + PPG | `Multimodal_estimate_ecg_ppg_pat_bp_proxy` | median PAT, IQR, paired pulses | Uses synchronized BIDMC ECG+PPG in the session benchmark; proxy for timing research, not BP estimation. |
| Multimodal sleep-apnea screening | ECG + RESP + SpO2 | `Multimodal_screen_sleep_apnea_report` | fused risk, flags, component evidence | Research proxy combining ECG HRV, RESP events/pattern, and SpO2 burden. |

## High-Value Labeled Benchmarks To Add Next

| Benchmark Goal | Suggested Public Data Direction | Needed Labels | Target Metrics |
| --- | --- | --- | --- |
| Arrhythmia classification | MIT-BIH Arrhythmia, AFDB, CinC rhythm datasets | beat/rhythm labels | sensitivity, specificity, F1 by rhythm class |
| MIT-BIH arrhythmia windows | Implemented via `scripts/prepare_labeled_arrhythmia_dataset.py` | annotation-derived normal/abnormal window labels | current RR-screen baseline: F1 0.680 on 240 windows |
| MIT-BIH rhythm/AF + beat windows | Implemented via `scripts/prepare_ecg_rhythm_beat_dataset.py` | rhythm aux notes and beat symbols | current AF F1 0.442 on 240 windows; beat-abnormal F1 0.308 on 18,209 beats |
| Sleep apnea detection | Apnea-ECG, Sleep-EDF/SHHS-style PSG resources where accessible | apnea/hypopnea events or AHI | event F1, AHI MAE, subject-level severity accuracy |
| Apnea-ECG minute labels | Implemented via `scripts/prepare_apnea_ecg_dataset.py` | minute-level apnea/normal ECG labels | current ECG-only proxy: F1 0.125 on 60 windows |
| UCDDB RESP/SpO2 windows | Implemented via `scripts/prepare_ucddb_resp_spo2_dataset.py` | PSG respiratory-event labels with Flow and SpO2 channels | current RESP/SpO2 ML fusion AUROC 0.724/macro-F1 0.647 on 2000 UCDDB windows; older hypopnea proxy F1 0.588 on 40 windows |
| UCDDB PSG sleep windows | Implemented via `scripts/prepare_psg_sleep_dataset.py` | sleep-stage labels plus respiratory-event labels using EEG/Flow/SpO2 | current EEG ML coarse sleep-stage macro-F1 0.731 on 1996 UCDDB windows; respiratory-event F1 0.218 on earlier 80-window proxy eval |
| Sleep staging | Sleep-EDF, PhysioNet sleep datasets | 30 s sleep stages | epoch accuracy, macro-F1, Cohen kappa |
| Desaturation detection | PSG datasets with SpO2 and scored respiratory events | desaturation/apnea events | ODI error, event precision/recall |
| Stress/arousal | Implemented WESAD parser/training via `scripts/train_eda_wesad_stress_models.py` | baseline/rest vs stress labels | EDA-only binary feature ensemble AUROC 0.870/BAcc 0.780; three-class raw CNN macro-AUROC 0.759/macro-F1 0.568 on subject-grouped 5-fold CV |
| Seizure or abnormal EEG screening | Implemented small CHB-MIT chb01 model via `scripts/train_eeg_chbmit_seizure_feature_model.py` | seizure vs non-seizure windows | current small-subset AUROC 0.959/macro-F1 0.923 on 114 windows; full CHB-MIT multi-subject evaluation still required |
| PCG murmur screening | Implemented via `scripts/prepare_pcg_murmur_dataset.py` on PhysioNet/CinC 2016 | normal/abnormal labels | proxy F1 0.000; feature+logistic-regression baseline F1 0.750 on 10 balanced records |
| ACC activity recognition | Implemented via `scripts/prepare_acc_activity_dataset.py` on UCI HAR | six-class activity labels | current raw tri-axial model macro-F1 0.820 on full UCI-HAR official test; older 561-feature smoke RF macro-F1 0.958 on 48 windows |
| ACC fall detection | UniMiB/SisFall parser scaffold via `scripts/prepare_acc_fall_dataset.py` | fall vs ADL labels | raw fall arrays required; logistic and impact-proxy evaluator implemented |
| PPG AF screening | Implemented via `scripts/prepare_ppg_af_dataset.py` on MIMIC PERform AF | AF/non-AF labels | current irregular-pulse proxy F1 0.857 on 8 windows |
| Signal modality classification | Implemented via `scripts/prepare_modality_classifier_dataset.py` over current public benchmark manifests | modality labels from source manifests | current feature-based RF macro-F1 0.952 on 117 signal windows across 12 modalities |
| Waveform image digitization | Implemented via `scripts/render_waveform_digitization_benchmark.py` and `scripts/prepare_ecg_image_digitization_dataset.py` | rendered waveform/ECG images with paired reference CSV and masks | current rendered benchmark: rule 31/43 ok with corr 0.855; RGB pixel model 43/43 ok with corr 0.851; tiny U-Net 43/43 ok with corr 0.843. ECG-only rendered connector smoke test: 4/4 ok, corr 0.849, NRMSE 0.087, peak-F1 0.751. ECG-Image-Kit image-only smoke: 12/12 CSV outputs, no waveform ground truth; U-Net coverage 0.947. ECG-Image-Kit generated benchmark: 5/5 mask-labeled ECG images, U-Net segmentation Dice 0.784/IoU 0.646, waveform peak-F1 currently 0.000 because physical coordinate calibration is not solved yet. |
| EMG gesture/action/fatigue/rehab/neuromuscular | UCI EMG Gestures; UCI Physical Action; UCI EMG Lower Limb; GEDS; Zenodo EMG Fatigue; EMGDB processed snippets | gesture/action/exercise/knee-status/fatigue/condition labels | gesture 6-class accuracy 0.777/macro-F1 0.777; NinaPro DB1 52-class augmented feature calibrated-user top-1 0.585/top-5 0.821, optional multi-stream CNN calibrated trial-voting top-1 0.732/top-5 0.937, and strict subject-held-out feature top-1 0.107; movement-intent proxy accuracy/macro-F1 0.725; physical-action 20-class accuracy 0.269/macro-F1 0.266 and binary AUROC 0.884/accuracy 0.827; lower-limb exercise accuracy 0.836/macro-F1 0.835; knee status AUROC 0.930/macro-F1 0.921; GEDS full S00-S22 414-trial gait speed macro-F1 0.788 and phase macro-F1 0.972; fatigue AUROC 0.774/BAcc 0.699; EMGDB neuromuscular smoke accuracy/macro-F1 1.0 on 27 windows, non-subject-independent |

## Existing-Work Wrapper Sources

The next implementation layer is tracked in `biosignal_agent/tools/source_catalog.json` and summarized in `docs/tool_source_catalog.md`. The catalog maps each task to existing algorithms/libraries, public datasets, source URLs, wrapper priority, and the next wrapper to build.

## Planning Direction

Short-term planning eval should include both single-modality tasks and cross-modality sessions, for example:

- ECG arrhythmia question routes to ECG quality, R peaks, HRV, arrhythmia screening.
- Sleep apnea question routes to RESP apnea screening and SpO2 desaturation burden.
- Sleep-stage question routes to EEG bandpower/stage features and ACC sleep/wake proxy.
- Stress question routes to EDA summary, ECG HRV, and ACC activity.

The baseline tools make these workflows executable now; labeled datasets should replace heuristic correctness with clinical or task-specific metrics.
