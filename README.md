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

## Algorithm Defaults

- ECG R-peak detection: NeuroKit2 `pantompkins1985`.
- PPG peak detection: NeuroKit2 ECG `nabian2018` detector after PPG bandpass preprocessing.
- BCG J-peak detection: NeuroKit2 ECG `nabian2018` detector after BCG bandpass preprocessing.

## Full MIT-BIH 60s Benchmark

Run all 48 MIT-BIH records, using the first 60 seconds of the selected lead:

```bash
python scripts/batch_evaluate_mitdb.py --seconds 60 --method-tag pantompkins
```

Latest summary:

- Output: `/data1/jiahui/biosignal-agent/outputs/mitdb_pantompkins_60s_all_summary.csv`
- Macro precision: 0.9164
- Macro recall: 0.9309
- Macro F1: 0.9224

Lowest-F1 records from the first 60-second pass:

```text
mitdb/117  F1=0.0400
mitdb/108  F1=0.4538
mitdb/102  F1=0.4966
mitdb/202  F1=0.6168
mitdb/207  F1=0.6720
mitdb/232  F1=0.7934
mitdb/104  F1=0.8980
mitdb/208  F1=0.9384
```

These are useful hard cases for quality gating, lead selection, or detector ensembles.

## Question-Driven Planning

Run the first planner agent, which selects tools from the question text:

```bash
python examples/ask_signal_agent.py   --question "Estimate ECG heart rate and HRV from this signal"   --csv /data1/jiahui/biosignal-agent/datasets/processed/mitdb_100_mlii_60s.csv   --sampling-rate 360
```

## OpenRouter LLM Agent

The LLM-backed agent uses OpenRouter chat completions with default model `openrouter/owl-alpha`.
It reads API settings from `/home/myid/jl57095/TwinMarket/openrouter_caption_with_P_wave.py` without copying keys into this repository.

```bash
python examples/ask_openrouter_agent.py   --question "Estimate ECG heart rate and HRV from this signal"   --csv /data1/jiahui/biosignal-agent/datasets/processed/mitdb_100_mlii_60s.csv   --sampling-rate 360   --model openrouter/owl-alpha
```

The OpenRouter planner returns JSON tool calls, the executor runs local signal tools, and the reporter asks the same model to summarize results. If OpenRouter fails or rate-limits, the framework falls back to the rule planner or deterministic tool-result report.
