# ABP Literature Review and Reproduction Notes

## Task-to-benchmark mapping

| Task | Common benchmark | Current implementation |
| --- | --- | --- |
| Pulse detection / HR from ABP | MIMIC waveform, BIDMC waveform | Improved robust ABP peak-valley detector. |
| Systolic/diastolic/MAP summary | Calibrated ABP in MIMIC/BIDMC | Robust beat-level peak/valley pairing and plausibility rejection. |
| Hypotension/hypertension/shock proxy | PhysioNet/CinC 2009 AHE, MIMIC waveform + clinical labels | Beat-level MAP/SBP/pulse-pressure event burden; no outcome-label model yet. |
| Cuffless BP / PAT | ECG+PPG+ABP paired datasets | Not implemented as BP prediction; needs calibrated paired data. |

## BIDMC calibrated ABP plausibility benchmark

Dataset: local BIDMC waveform records with ABP channel. Eight records contain calibrated ABP. There are no beat-level expert labels in this local benchmark, so the metric is plausibility/stability rather than sensitivity/F1.

| Metric | Value |
| --- | ---: |
| ABP records processed | 8 |
| Plausible summaries | 8 |
| Plausible fraction | 1.000 |
| Median HR | 95 bpm |
| Median systolic | 111 mmHg |
| Median diastolic | 51 mmHg |
| Median MAP | 69 mmHg |

Implemented changes:

- Smooth ABP before peak detection.
- Detect systolic peaks with physiological refractory period.
- Pair each systolic peak with preceding diastolic valley.
- Reject beats with implausible systolic, diastolic, or pulse pressure.
- Return robust median systolic, diastolic, MAP, pulse pressure, HR, and artifact rejection fraction.

## Caveats

- This is not a beat-level labeled benchmark.
- Hypotension/hypertension flags are threshold screens, not clinical diagnoses.
- Invasive ABP units/calibration must be preserved; plotted or normalized waveforms cannot support pressure interpretation without scale recovery.

## Sources checked

- BIDMC waveform direction: https://physionet.org/content/bidmc/
- MIMIC waveform direction: https://physionet.org/content/mimic3wdb/

## Artifacts

- Evaluation script: `scripts/evaluate_abp_bidmc.py`
- Report: `/data1/jiahui/biosignal-agent/outputs/abp_bidmc/abp_bidmc_eval.json`
- Exported ABP CSV snippets: `/data1/jiahui/biosignal-agent/outputs/abp_bidmc/csv/`

## Beat-level pressure event upgrade

`ABP_classify_pressure_events` now estimates beat-level MAP/SBP/DBP/pulse pressure burden rather than only flagging whole-record medians. It reports hypotensive-beat fraction, severe MAP<55 fraction, hypertension burden, narrow/wide pulse-pressure burden, and event intervals.

BIDMC re-run after the change still has 8/8 plausible ABP summaries. Event-burden categories across those 8 records were: 3 shock/severe-hypotension proxies, 3 hypotension proxies, and 2 no-pressure-event proxies. This is a physiologic screen, not a labeled shock predictor.

For SOTA-style event prediction, the correct next benchmark is PhysioNet/CinC Challenge 2009 acute hypotensive episode prediction. The public files are available, but the waveform test sets are large (~384 MB for set A and ~1.48 GB for set B), so the current repo change adds the event-burden tool and keeps challenge download/evaluation as an explicit next benchmark step.

## PhysioNet/CinC 2009 AHE smoke benchmark

Added `scripts/evaluate_abp_challenge2009.py`, which parses official Event 1/2 answer files, streams WFDB records from PhysioNet, extracts recent ABP event-burden features, and applies the official challenge selection rule (`Event 1`: exactly 5 H; `Event 2`: configurable top-H, default 13).

Current lightweight result, avoiding full 384 MB/1.48 GB tar downloads due slow remote throughput:

| Event | Input | Rule | Score | Notes |
| --- | --- | --- | ---: | --- |
| Event 1 | a/b/c segments, last 1 min ABP each | top 5 H | 8/10 | Missed 104; false-positive 108. |
| Event 1 | c segment only, last 1-5 min ABP | top 5 H | 6/10 | Multi-segment session evidence is clearly better. |
| Event 2 | c segment only, last 1 min ABP | top 13 H | 25/40 | Lightweight smoke only; far below 37/40 open-source best. |
| Event 2 | a/b/c segments, last 1 min ABP each | top 13 H | 29/40 | Session evidence improves over c-only. |
| Event 2 | a/b/c segments, last 1 min ABP each | top 10 H | 30/40 | Best current valid top-H setting in 10-16 sweep. |

The open-source challenge SOTA reported in the official score file reached 10/10 on Event 1 and 37/40 on Event 2, so this heuristic is **not** SOTA. It is now a real labeled benchmark harness; next improvement should use full local test-set A/B windows and learned features rather than a recent-MAP/minute-MAP heuristic. The script now caches segment features under `/data1/jiahui/biosignal-agent/outputs/abp_challenge2009/segment_cache/` to make repeated top-H sweeps cheap.

## Challenge 2009 learned ranker smoke

Added `scripts/train_abp_challenge2009_ranker.py`, which builds record-level features from cached a/b/c segment summaries and evaluates a leave-one-record-out ranker. This is a small post-hoc public-label benchmark, not a deployable forecasting model.

| Model | LOO AUROC | Event 1 style | Event 2 best valid top-H | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Logistic ranker | 0.825 | 6/10 | 32/40 at top 10 H | Improves Event 2 smoke but hurts Event 1. |
| RF/ExtraTrees ensemble | 0.815 | 8/10 | 30/40 | More consistent with heuristic/Event 1. |

Decision: keep the transparent minute-MAP/event-burden heuristic as the default screening logic, and keep the logistic ranker as a benchmark artifact for Event 2 smoke improvement. A trustworthy tool-level model needs more training records or full-window local Challenge/MIMIC data.
