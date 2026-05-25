# SpO2 Literature Review and Reproduction Notes

## Task split

| Task | Common / SOTA direction | Current implementation | Local benchmark | Status |
| --- | --- | --- | --- | --- |
| Signal quality | Plausibility range, missing/artifact fraction, abrupt jump detection, low perfusion when available. | `SpO2_assess_quality` now reports plausible ratio, jump/artifact fractions, and dynamic range. | Smoke-tested on UCDDB SpO2 windows. | Implemented. |
| Desaturation events / ODI | ODI3/ODI4 are common oximetry screening features; event definitions vary by baseline, drop threshold, and minimum duration. | `SpO2_detect_desaturation` now uses rolling-baseline ODI3/ODI4, 10s minimum duration, event depth/duration/area, and CT-style outputs. | UCDDB windows. | Implemented interpretable evidence. |
| Hypoxemia burden | T90/T88/nadir and area below 90% are widely used; hypoxic burden can capture depth and duration better than event counts alone. | `SpO2_assess_hypoxemia_burden` adds T90/T88/T85, CT90, nadir, and area below 90 percent-min/hour. | UCDDB windows. | Implemented. |
| Oximetry-only sleep-apnea screening | ODI thresholds are useful for OSA screening but cannot replace PSG/HSAT AHI, especially for arousal-only hypopneas. | `SpO2_screen_sleep_apnea_oximetry` reports ODI-based severity/risk proxy. | Heuristic on UCDDB 30s event labels: AUROC 0.555, BAcc 0.531. | Useful as full-night burden proxy, weak as 30s event detector. |
| SpO2-only ML respiratory-event screening | Deep/ML papers use oximetry-only CNN/LSTM or multi-parameter oximetry; event-level performance improves with context and respiratory channels. | `SpO2_screen_sleep_apnea_ml` loads a UCDDB record-held-out feature ensemble from `scripts/train_spo2_ucddb_event_model.py`. | UCDDB 25 records / 1974 valid 30s windows, GroupKFold by record: AUROC 0.542, macro-F1 0.523, BAcc 0.523. | Negative result for SpO2-only 30s event detection; keep as weak evidence. |

## Interpretation

SpO2 is clinically valuable for overnight burden summaries (`ODI3/ODI4`, `T90`, nadir, hypoxic burden). It is much weaker for exact 30s respiratory-event timing because oxygen desaturation lags airflow reduction and many hypopneas/arousals have little desaturation. The local UCDDB result confirms this: the SpO2-only feature model is not close to modern multimodal SOTA and should be routed as supportive evidence, while RESP+SpO2 fusion remains the better event-screening path.

## Sources

- UCDDB / PhysioNet: https://physionet.org/content/ucddb/
- ODI diagnostic value systematic review: https://pubmed.ncbi.nlm.nih.gov/32333683/
- Novel oximetry / hypoxic burden review: https://pmc.ncbi.nlm.nih.gov/articles/PMC10649141/
- LSTM respiratory-event detection with SpO2 desaturation features: https://pmc.ncbi.nlm.nih.gov/articles/PMC7662467/
- SomnNET SpO2 deep network: https://arxiv.org/abs/2108.11468
- PhysioNet respiratory/oximetry simulated apnea dataset: https://physionet.org/content/respiratory-oximetry-apnoea/1.0.0/
