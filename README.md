# BioSignalAgent

Tool-first prototype for ECG, PPG, and BCG signal reasoning.

This first version keeps the LLM out of the critical path: signal analysis is done by explicit Python tools, and the agent selects a basic workflow from the requested modality. Later, the same tool schemas can be used for tool calling, ToolRAG, and instruction tuning traces.

## Current Tools

- ECG: signal quality, R-peak detection, HRV summary
- PPG: signal quality, peak detection, heart-rate estimate
- BCG: signal quality, J-peak detection, heart-rate estimate

## Run A CSV Report

```bash
python examples/run_basic_report.py --csv path/to/signal.csv --modality ecg --sampling-rate 250
```

The CSV should contain one numeric column, or a column named `signal`.

## MIT-BIH Sanity Check

Download MIT-BIH record 100, export the first 60 seconds of MLII to CSV, run the ECG agent, and evaluate R-peak detection against MIT-BIH annotations:

```bash
python scripts/prepare_mitdb_record.py --record 100 --seconds 60
python examples/run_basic_report.py \
  --csv /data1/jiahui/biosignal-agent/datasets/processed/mitdb_100_mlii_60s.csv \
  --modality ecg \
  --sampling-rate 360 \
  > /data1/jiahui/biosignal-agent/outputs/mitdb_100_mlii_60s_report.json
python scripts/evaluate_mitdb_peaks.py \
  --record 100 \
  --seconds 60 \
  --report /data1/jiahui/biosignal-agent/outputs/mitdb_100_mlii_60s_report.json \
  --out /data1/jiahui/biosignal-agent/outputs/mitdb_100_mlii_60s_peak_metrics.json
```

Observed first sanity-check result on record 100, first 60 seconds, 100 ms tolerance:

```json
{"detected_peaks": 74, "reference_beats": 74, "precision": 1.0, "recall": 1.0, "f1": 1.0}
```
