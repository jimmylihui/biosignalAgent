# Multimodal vs Unimodal Public Benchmark Test

Dataset: UCDDB PSG windows from processed psg_sleep_manifest
Task: respiratory_event vs normal
Validation: record-level GroupKFold; records are held out, windows from the same record do not cross folds
Windows: 2000; records: 25; labels: `{'normal': 821, 'respiratory_event': 1179}`

| model | balanced accuracy | macro F1 | AUROC | average precision | accuracy |
|---|---:|---:|---:|---:|---:|
| resp_only | 0.5253 | 0.5252 | 0.5317 | 0.6111 | 0.5450 |
| spo2_only | 0.5272 | 0.5251 | 0.5384 | 0.6285 | 0.5560 |
| resp_plus_spo2 | 0.5301 | 0.5296 | 0.5598 | 0.6501 | 0.5530 |

Best by macro F1: `resp_plus_spo2`.

On this run, RESP+SpO2 fusion changed macro F1 by +0.0044 relative to the best unimodal baseline. Positive delta supports multimodal benefit; negative delta means this simple fusion did not beat the strongest single modality under this split.
