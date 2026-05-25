# RESP/SpO2 Literature Review and Reproduction Notes

## Task-to-benchmark mapping

| Task | Common benchmark | Current implementation |
| --- | --- | --- |
| Respiratory rate and standalone event burden | BIDMC, UCDDB, MIMIC waveform | Bandpass breath detector plus `RESP_summarize_event_burden` for apnea/hypopnea-like intervals and REI proxy. |
| Apnea/hypopnea respiratory events | UCDDB, SHHS, MESA PSG | Implemented UCDDB Flow+SpO2 respiratory-event ML screen. |
| Desaturation/ODI | UCDDB, SHHS, MESA | Existing 3 percent desaturation and hypoxemia burden tools remain. |
| Multimodal sleep apnea | PSG with Flow/SpO2/EEG/ECG | Current tool supports Flow+SpO2 fusion; full AASM scoring needs event-level PSG evaluation. |

## Reproduced UCDDB respiratory-event baselines

Dataset: UCDDB `ucddb002` processed into 2000 30-second windows with Flow and SpO2. Labels are window-level respiratory-event vs normal based on available event annotations.

| Input | Accuracy | Balanced accuracy | Macro-F1 | AUROC |
| --- | ---: | ---: | ---: | ---: |
| Flow + SpO2 fusion features | 0.671 | 0.645 | 0.647 | 0.724 |
| Flow-only fallback features | 0.617 | 0.587 | 0.587 | 0.639 |

Decision: expose `RESP_screen_sleep_apnea_ml` as a research respiratory-event screen. It uses Flow+SpO2 fusion when oximetry is provided and falls back to Flow-only otherwise. Keep `RESP_detect_apnea`, `RESP_detect_hypopnea`, and `SpO2_detect_desaturation` as interpretable heuristic evidence.

## Caveats

- The current benchmark is one UCDDB record with window-level CV, not subject-independent.
- Window labels collapse different respiratory-event types; this is not full AASM apnea/hypopnea scoring.
- SpO2 lag, arousal context, sleep stage, and event duration rules still need proper event-level evaluation.

## Sources checked

- UCDDB PhysioNet: https://physionet.org/content/ucddb/
- SHHS direction: https://sleepdata.org/datasets/shhs
- MESA direction: https://sleepdata.org/datasets/mesa
- Apnea-ECG related benchmark direction: https://physionet.org/content/apnea-ecg/

## Artifacts

- Training script: `scripts/train_resp_spo2_ucddb_event_model.py`
- Fusion model: `/data1/jiahui/biosignal-agent/outputs/resp_spo2/resp_spo2_ucddb_event_fusion_ensemble.joblib`
- Flow-only model: `/data1/jiahui/biosignal-agent/outputs/resp_spo2/resp_ucddb_event_flow_ensemble.joblib`
- Report: `/data1/jiahui/biosignal-agent/outputs/resp_spo2/resp_spo2_ucddb_event_report.json`

## Standalone RESP event-burden tool

Added `RESP_summarize_event_burden` to consolidate standalone airflow/respiration analysis. It returns respiratory rate, apnea-like intervals, hypopnea-like intervals, merged event intervals, longest event, respiratory-event-index proxy, and pattern flags. This gives a cleaner path to event-level scoring than only per-window ML probabilities.

Caveat: this remains a flow/envelope screen. True event-level AASM scoring needs airflow/effort plus SpO2 desaturation or arousal context and sleep-time denominator. The existing `RESP_screen_sleep_apnea_ml` remains the better UCDDB Flow+SpO2 screen when SpO2 is available.
