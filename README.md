# BioSignalAgent

Tool-first prototype for ECG, PPG, and BCG signal reasoning.

This version keeps signal analysis in explicit Python tools. The agent can use either deterministic rule planning or an OpenRouter LLM planner, with local fallback when LLM output is unavailable or invalid. Tool schemas and execution traces are structured so they can support ToolRAG, tool calling, and future instruction tuning.

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

The OpenRouter planner returns JSON tool calls, and the executor runs local signal tools. Final reporting is deterministic by default for speed and reliability; add `--llm-report` when you want the model to write the prose report. If OpenRouter fails, rate-limits, or returns invalid JSON, the framework falls back to the rule planner or deterministic tool-result report.


## ToolRAG-Style Tool Retrieval

Before calling the LLM planner, the framework retrieves a small set of relevant tool schemas from the local registry instead of sending every tool. The current retriever is deterministic TF-IDF plus modality hints, so it works offline and can later be replaced with embeddings.

```bash
python examples/ask_openrouter_agent.py   --question "Estimate ECG heart rate and HRV from this signal"   --csv /data1/jiahui/biosignal-agent/datasets/processed/mitdb_100_mlii_60s.csv   --sampling-rate 360   --fallback-modality ecg   --retrieved-tool-count 3   --model openrouter/owl-alpha
```

Traces include both `retrieved_tools` and the final `tool_plan`, so planner behavior can be audited separately from tool execution.

## Trace Logging And Sessions

Every `ask_openrouter_agent.py` run saves a JSON trace under:

```text
/data1/jiahui/biosignal-agent/outputs/traces/
```

A trace contains the question, planner, model, selected tools, tool results, final report, and timestamp. These traces are the starting point for future instruction-tuning data.

Run a multi-signal session from JSON:

```json
{
  "question": "Estimate heart rate from these signals and summarize confidence.",
  "signals": [
    {"modality": "ecg", "path": "...csv", "sampling_rate": 360, "label": "ecg_example"},
    {"modality": "ppg", "path": "...csv", "sampling_rate": 100, "label": "ppg_example"},
    {"modality": "bcg", "path": "...csv", "sampling_rate": 100, "label": "bcg_example"}
  ]
}
```

```bash
python examples/run_session.py --session session.json
```

By default, multi-signal sessions use the rule planner for speed and reliability. Use `--llm-planner` to call OpenRouter for each signal.

## Framework Evaluation

The framework has a small regression harness for the TxAgent-style loop: retrieve tool schemas, plan tool calls, optionally execute tools, and write JSON/CSV metrics.

```bash
python scripts/evaluate_agent_framework.py \
  --planner rule \
  --retrieved-tool-count 3 \
  --execute \
  --ecg-csv /data1/jiahui/biosignal-agent/datasets/processed/mitdb_100_mlii_60s.csv \
  --ppg-csv /data1/jiahui/biosignal-agent/datasets/processed/synthetic_ppg.csv \
  --bcg-csv /data1/jiahui/biosignal-agent/datasets/processed/synthetic_bcg.csv \
  --ecg-sampling-rate 360 \
  --ppg-sampling-rate 100 \
  --bcg-sampling-rate 100 \
  --out-json /data1/jiahui/biosignal-agent/outputs/framework_eval_rule_execute.json \
  --out-csv /data1/jiahui/biosignal-agent/outputs/framework_eval_rule_execute.csv
```

Latest rule-planner execution eval: retrieval accuracy 1.0, planning accuracy 1.0, execution accuracy 1.0 across the default ECG/PPG/BCG planning cases.

OpenRouter planning eval is also supported, with bounded timeout/retry settings so batch runs do not hang on network/model failures:

```bash
python scripts/evaluate_agent_framework.py \
  --planner openrouter \
  --model openrouter/owl-alpha \
  --retrieved-tool-count 3 \
  --llm-timeout 20 \
  --llm-retry-max 1 \
  --out-json /data1/jiahui/biosignal-agent/outputs/framework_eval_openrouter.json \
  --out-csv /data1/jiahui/biosignal-agent/outputs/framework_eval_openrouter.csv
```

The eval summary includes `planner_backend_counts`, so OpenRouter runs can be audited for true LLM plans versus rule fallback.

## Trace-To-Training Export

Agent traces can be exported to JSONL samples for later planning/report fine-tuning or prompt evaluation:

```bash
python scripts/export_trace_dataset.py \
  --trace-dir /data1/jiahui/biosignal-agent/outputs/traces \
  --out /data1/jiahui/biosignal-agent/outputs/biosignal_trace_sft.jsonl

python scripts/export_trace_dataset.py \
  --planning-only \
  --out /data1/jiahui/biosignal-agent/outputs/biosignal_planning_sft.jsonl
```
