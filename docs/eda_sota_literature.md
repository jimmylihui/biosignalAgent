# EDA Literature Review and Reproduction Notes

## Task-to-benchmark choice

| Task | Practical EDA role | Common benchmark to prioritize | Current action |
| --- | --- | --- | --- |
| Stress detection | Strongest standalone EDA use case; sympathetic arousal response. | WESAD wrist EDA; SWELL/UBFC-Phys for external validation. | Implemented WESAD EDA-only binary stress classifier. |
| Emotion/arousal detection | EDA is useful for arousal, weak for valence without other signals. | DEAP, AMIGOS, MAHNOB-HCI; WESAD for categorical stress/amusement/baseline. | Reproduced WESAD three-class stress/baseline/amusement; not exposed as generic emotion tool. |
| Anxiety/tension monitoring | Usually operationalized as stress/arousal under task protocols. | WESAD or study-specific exposure-task data. | Covered only as stress/arousal screening, not clinical anxiety diagnosis. |
| Attention/cognitive load | EDA can help, but HR/EEG/eye/behavior context is usually needed. | WESAD is not ideal; driving/HCI workload datasets vary. | Literature route only; no single EDA-only benchmark selected yet. |
| Pain response | EDA can reflect autonomic response, but labels and stimuli matter. | BioVid / pain-monitoring datasets when accessible. | Literature route only. |
| Sleep/awakening | EDA can support arousal/awakening, but PSG/ACC dominate staging. | PSG sleep datasets with EDA are less common than EEG/EOG/EMG. | Route as auxiliary, not primary sleep-stage tool. |
| Lie detection | EDA is historically used, but standalone inference is ethically and scientifically fragile. | No standalone tool. | Explicitly not implemented as a lie detector. |
| Seizure adjunct | EDA + ACC can support generalized tonic-clonic seizure alarms. | Empatica/E4 seizure datasets; many are restricted. | Route as multimodal adjunct only. |
| UX/ad response | EDA measures physiological arousal during stimuli. | DEAP/AMIGOS/MAHNOB-HCI or domain-specific HCI studies. | Route to arousal features; no generic valence claim. |
| Exercise/recovery | EDA is confounded by sweat, motion, temperature. | Exercise physiology datasets are protocol-specific. | Prefer ACC/PPG/HRV primary, EDA auxiliary. |

## Reproduced benchmark

Dataset: WESAD wrist EDA, 15 subjects, 4 Hz Empatica E4 EDA. Windows are 60 s with 10 s stride. Validation is subject-grouped 5-fold CV, so windows from the same subject are not split across train/test.

Binary stress vs non-stress results:

| Method | Accuracy | Balanced accuracy | Macro-F1 | AUROC |
| --- | ---: | ---: | ---: | ---: |
| Feature ensemble, ExtraTrees + RF | 0.791 | 0.780 | 0.778 | 0.870 |
| Raw EDA 1D CNN / DeepConvLSTM-family baseline | 0.783 | 0.760 | 0.764 | 0.808 |

Three-class baseline/stress/amusement results:

| Method | Accuracy | Balanced accuracy | Macro-F1 | Macro AUROC OVR |
| --- | ---: | ---: | ---: | ---: |
| Feature ensemble, ExtraTrees + RF | 0.590 | 0.496 | 0.484 | 0.688 |
| Raw EDA 1D CNN / DeepConvLSTM-family baseline | 0.601 | 0.568 | 0.568 | 0.759 |

Decision: expose the binary WESAD feature ensemble as `EDA_screen_stress_ml`, because it is stronger than the raw EDA CNN on the clinically safer binary task and fits single-EDA CSV input. Expose the three-class raw CNN as `EDA_classify_affective_state_ml` only as WESAD protocol-state classification, not as generic emotion recognition.

## Sources checked

- WESAD original dataset: https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection and https://kristofvl.github.io/usi/pdf/ubi_icmi2018.pdf
- EDA-only WESAD frequency-spectrum stress detection review/benchmark: https://pmc.ncbi.nlm.nih.gov/articles/PMC9866614/
- Reproducible WESAD EDA GitHub reference: https://github.com/WJMatthew/WESAD
- WESAD CNN GitHub reference: https://github.com/peasypi/Stress-Detection-From-Wearables
- Image-encoding deep stress model with GitHub: https://pmc.ncbi.nlm.nih.gov/articles/PMC9775098/

## Implementation artifacts

- Training script: `scripts/train_eda_wesad_stress_models.py`
- Reports: `/data1/jiahui/biosignal-agent/outputs/eda_wesad/eda_wesad_*_report.json`
- Exposed tools: `EDA_screen_stress_ml`, `EDA_extract_tonic_phasic_features`, `EDA_classify_affective_state_ml`, `EDA_route_task_recommendation`
- Binary stress model bundle: `/data1/jiahui/biosignal-agent/outputs/eda_wesad/eda_wesad_binary_feature_ensemble.joblib`
- Three-class affective-state CNN bundle: `/data1/jiahui/biosignal-agent/outputs/eda_wesad/eda_wesad_three_class_raw_cnn.pt`
