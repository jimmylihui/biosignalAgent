# EEG Literature Review and Reproduction Notes

## Task-to-benchmark mapping

| Task | Common benchmark | SOTA/open-source direction | Current implementation |
| --- | --- | --- | --- |
| Sleep staging | Sleep-EDF, ISRUC, SHHS/MESA, UCDDB | YASA, U-Sleep, DeepSleepNet-style CNN/RNN/sequence models | Implemented a UCDDB single-channel coarse-stage feature ensemble as a research baseline. |
| Seizure detection | CHB-MIT, TUH Seizure Corpus, Siena | CNN/Transformer/Autoformer/spiking and feature ensembles; event-level evaluation matters | Implemented a small CHB-MIT chb01 single-channel seizure screen as a research baseline. |
| Drowsiness/vigilance | driving/vigilance EEG datasets | theta/alpha and learned temporal models | Current theta/alpha proxy remains; needs labeled vigilance data. |
| Artifact detection | sleep PSG/EOG/EMG context, EEG artifact datasets | MNE/autoreject/channel-aware QC | Current artifact proxy remains; needs multi-channel artifact labels. |

## Reproduced baselines

### UCDDB coarse sleep stage

- Input: one single-channel C3A2 EEG CSV, normally 30 s.
- Classes: `wake_rem`, `n1_n2`, `n3`.
- Features: EEG band powers, relative band powers, Hjorth parameters, spectral entropy, amplitude statistics.
- Model: ExtraTrees + RandomForest soft-voting ensemble.
- Validation: 5-fold stratified window CV on `ucddb002`; this is not subject-independent.

| Metric | Value |
| --- | ---: |
| Accuracy | 0.741 |
| Balanced accuracy | 0.730 |
| Macro-F1 | 0.731 |
| Cohen kappa | 0.606 |
| Macro AUROC OVR | 0.897 |

### CHB-MIT seizure screen

- Input: single-channel `FP1-F7` EEG windows from available `chb01` seizure files.
- Classes: `seizure`, `non_seizure`.
- Features: line length, robust spikes, envelope statistics, band powers, relative fast power, Hjorth parameters, spectral entropy.
- Model: ExtraTrees + RandomForest soft-voting ensemble.
- Validation: 3-fold EDF-file grouped CV on a small `chb01` subset; this is not a full CHB-MIT benchmark.

| Metric | Value |
| --- | ---: |
| Windows | 114 |
| Accuracy | 0.947 |
| Balanced accuracy | 0.936 |
| Macro-F1 | 0.923 |
| AUROC | 0.959 |

## Tool decisions

- Expose `EEG_classify_sleep_stage_ml` as a coarse research classifier with strong caveats. It is better than the old bandpower-only sleep-stage hint but not comparable to YASA/U-Sleep subject-independent PSG staging.
- Expose `EEG_screen_seizure_ml` as a small CHB-MIT subset research screen with heuristic fallback. It is not a clinical seizure detector or alarm.
- Keep `EEG_estimate_drowsiness` and `EEG_detect_artifact_proxy` as proxies until labeled vigilance/artifact datasets are available.

## Sources checked

- YASA automatic sleep staging documentation: https://yasa-sleep.org/generated/yasa.SleepStaging.html
- YASA open-source sleep staging paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC8516415/
- U-Sleep paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC8050216/
- CHB-MIT PhysioNet dataset: https://physionet.org/content/chbmit/1.0.0/
- CHB-MIT epilepsy benchmark summary: https://epilepsybenchmarks.com/datasets/chbmit/
- Recent CHB-MIT seizure deep-learning benchmark example: https://pmc.ncbi.nlm.nih.gov/articles/PMC12839289/

## Artifacts

- Sleep training script: `scripts/train_eeg_sleep_stage_ucddb.py`
- Sleep model: `/data1/jiahui/biosignal-agent/outputs/eeg_sleep/eeg_ucddb_coarse_sleep_stage_feature_ensemble.joblib`
- Sleep report: `/data1/jiahui/biosignal-agent/outputs/eeg_sleep/eeg_ucddb_coarse_sleep_stage_report.json`
- Seizure training script: `scripts/train_eeg_chbmit_seizure_feature_model.py`
- Seizure model: `/data1/jiahui/biosignal-agent/outputs/eeg_seizure/eeg_chbmit_chb01_seizure_feature_ensemble.joblib`
- Seizure report: `/data1/jiahui/biosignal-agent/outputs/eeg_seizure/eeg_chbmit_chb01_seizure_feature_report.json`
