# ACC Literature Review and Reproduction Notes

## Task-to-benchmark mapping

| Task | Common benchmark | Current implementation |
| --- | --- | --- |
| Human activity recognition | UCI HAR, WISDM, PAMAP2 | Implemented UCI HAR raw ACC activity classifier. |
| Actigraphy sleep/wake | MESA/SHHS actigraphy + PSG labels | Current proxy only; needs sleepdata access/labels. |
| Fall detection | UniMiB SHAR, SisFall, MobiFall, UP-Fall | Current impact proxy only; dataset parser scaffold exists. |
| Stress/activity context | WESAD ACC plus EDA/BVP/TEMP | ACC activity summary/classifier supports context, not stress by itself. |

## Reproduced UCI HAR baselines

Dataset: UCI Human Activity Recognition Using Smartphones. The full official train/test split is used, using raw `total_acc_x/y/z` windows when available.

| Input | Accuracy | Balanced accuracy | Macro-F1 | Macro AUROC OVR | Active/rest accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tri-axial raw ACC features | 0.822 | 0.821 | 0.820 | 0.975 | 1.000 |
| Magnitude-only fallback | 0.629 | 0.635 | 0.628 | 0.918 | 1.000 |

Decision: expose `ACC_classify_activity_ml`. It reads x/y/z columns when present and falls back to single-axis/magnitude mode. This is more realistic than the older 48-window smoke benchmark, but performance is still device-placement dependent.

## Sources checked

- UCI HAR dataset: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- MESA actigraphy/sleep direction: https://sleepdata.org/datasets/mesa
- SHHS actigraphy/sleep direction: https://sleepdata.org/datasets/shhs
- Fall benchmark direction: UniMiB/SisFall/MobiFall/UP-Fall, tracked in source catalog.

## Artifacts

- Tri-axial training script: `scripts/train_acc_activity_uci_har_triaxial.py`
- Magnitude fallback training script: `scripts/train_acc_activity_uci_har_raw.py`
- Tri-axial model: `/data1/jiahui/biosignal-agent/outputs/acc_activity/acc_uci_har_triaxial_activity_ensemble.joblib`
- Magnitude model: `/data1/jiahui/biosignal-agent/outputs/acc_activity/acc_uci_har_raw_magnitude_activity_ensemble.joblib`
- Reports: `/data1/jiahui/biosignal-agent/outputs/acc_activity/*activity_report.json`

## Fall detection reproduction: UniMiB SHAR

Dataset source used here: processed UniMiB SHAR CSV mirror downloaded from the public Texas State-hosted zip referenced by later UniMiB work. It contains 1,196 3-second windows at 50 Hz after available-file parsing: 710 falls and 486 ADL windows.

Evaluation protocol: subject-independent 5-fold GroupKFold by `User*`, using triaxial ACC feature ensembles (RandomForest + ExtraTrees soft voting).

| Method | Accuracy | Balanced accuracy | Fall F1 | Fall recall | Specificity | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ML triaxial ensemble | 0.982 | 0.980 | 0.985 | 0.992 | 0.969 | 0.998 |
| Impact/posture proxy | 0.576 | 0.611 | 0.543 | 0.424 | 0.798 | n/a |

Decision: expose `ACC_detect_fall_ml` as the preferred fall/ADL classifier when input resembles UniMiB-style triaxial smartphone-pocket windows, and keep `ACC_detect_fall_proxy` as a transparent fallback. This is **not** real-world elderly fall SOTA: simulated fall datasets usually overestimate deployment performance, so SisFall/UP-Fall/external validation remains the next step.

Artifacts:

- Manifest: `/data1/jiahui/biosignal-agent/datasets/processed/acc_fall_manifest.json`
- Model: `/data1/jiahui/biosignal-agent/outputs/acc_fall/acc_unimib_fall_ensemble.joblib`
- Report: `/data1/jiahui/biosignal-agent/outputs/acc_fall/acc_fall_eval.json`

## Sleep/wake status

The tool layer now has `ACC_extract_actigraphy_features`, which computes ENMO-style epoch features, restful/sedentary/activity fractions, and triaxial angle-z summaries. `ACC_estimate_sleep_wake` uses these features as a proxy. A true SOTA sleep/wake classifier should be trained on PSG-aligned actigraphy from MESA/SHHS, but those require authorized NSRR data access rather than anonymous download.
