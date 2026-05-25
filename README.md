# BioSignalAgent

Tool-first prototype for common physiological signal reasoning: ECG, PPG, BCG, SCG, RESP, SpO2, ABP, PCG, ACC, EDA, EEG, and EMG.

This version keeps signal analysis in explicit Python tools. The agent can use either deterministic rule planning or an OpenRouter LLM planner, with local fallback when LLM output is unavailable or invalid. Tool schemas and execution traces are structured so they can support ToolRAG, tool calling, and future instruction tuning.

## Current Tools

- ECG: signal quality, artifact screen, R-peak detection, HRV summary, arrhythmia proxy, ECG-only sleep-apnea proxy, morphology/interval proxy
- PPG: signal quality, artifact screen, Nabian-style peak detection, heart-rate estimate, perfusion/variability proxy, irregular-pulse proxy, PPG-derived respiration proxy
- BCG: signal quality, Nabian-style J-peak detection, heart-rate estimate, BCG-derived respiration proxy
- SCG: signal quality, Nabian-style J-peak detection, heart-rate estimate, SCG-derived respiration proxy
- RESP: signal quality, artifact screen, respiratory-rate estimate, apnea proxy, hypopnea proxy, rate/pattern proxy
- SpO2: signal quality, artifact screen, oxygen-saturation summary, rolling-baseline ODI3/ODI4 desaturation events, hypoxemia burden, oximetry-only sleep-apnea screening/ML evidence
- ABP: signal quality, artifact screen, pulse detection, heart-rate and pressure-value summary, hypotension/hypertension proxy, MAP/pulse-pressure proxy
- PCG: signal quality, heart-sound event detection, murmur proxy, murmur feature extraction baseline, S1/S2 segmentation proxy
- ACC: signal quality, activity summary, sleep/wake proxy, activity-bout detection, fall/impact proxy
- EDA: signal quality, WESAD-compatible tonic/phasic/SCR features, arousal/SCR event proxy, WESAD-trained stress ML classifier, WESAD baseline/stress/amusement protocol-state classifier, safe task routing, heuristic stress/arousal fallback
- EEG: signal quality, artifact screen, bandpower summary, sleep-stage features, seizure-like spike proxy, drowsiness/vigilance proxy, EEG artifact proxy
- EMG: signal quality, artifact screen, activation summary, gesture classification, prosthetic gesture classification, movement-intent proxy, physical action screening, lower-limb exercise/rehab screening, gait speed/phase, neuromuscular abnormality smoke screen, fatigue ML/proxy, burst/onset proxy
- Generic routing: `Signal_classify_modality` feature-based classifier for unknown single-signal CSVs, with heuristic fallback if no trained model is present
- Image input: `Signal_digitize_waveform_image_ml` trained pixel-segmentation baseline for clean single-trace waveform plots

## Classify An Unknown Signal

```bash
python examples/classify_signal_modality.py --csv path/to/signal.csv --sampling-rate 250
```

This runs the local feature-based routing classifier and returns the top modality candidates before choosing a task-specific workflow.

## Digitize A Waveform Image

```bash
python examples/digitize_waveform_image.py \
  --image path/to/waveform.png \
  --sampling-rate 250 \
  --out-csv /data1/jiahui/biosignal-agent/outputs/digitized/example.csv
```

The baseline digitizer extracts a dark single trace from a clean plot image. If axis calibration is known, pass `--value-min`, `--value-max`, and crop bounds; otherwise the output is normalized.

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
- SCG J-peak detection: NeuroKit2 ECG `nabian2018` detector after SCG bandpass preprocessing.

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

Latest rule-planner execution eval: retrieval accuracy 1.0, planning accuracy 1.0, execution accuracy 1.0 across the default ECG/PPG/BCG/SCG planning cases.

Latest major-task rule-planner eval: 63 planning cases at retrieval/planning accuracy 1.0/1.0; real-world manifest 25 records and 155 case-runs at retrieval/planning/execution accuracy 1.0/1.0/1.0; dedicated common 21 records and 96 runs at 1.0/1.0/1.0; dedicated BCG 3 records and 15 runs at 1.0/1.0/1.0. Tool audit across 49 records and 274 tool-runs has 0 errors and 0 low-confidence outputs.

Major-task outputs:

```text
/data1/jiahui/biosignal-agent/outputs/framework_eval_rule_major_tasks.json
/data1/jiahui/biosignal-agent/outputs/real_dataset_framework_eval_rule_major_tasks.json
/data1/jiahui/biosignal-agent/outputs/dedicated_common_framework_eval_rule_major_tasks.json
/data1/jiahui/biosignal-agent/outputs/dedicated_bcg_framework_eval_rule_major_tasks.json
/data1/jiahui/biosignal-agent/outputs/tool_output_audit_major_tasks.json
```

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


## Session-Level Tools

The framework now supports session tools that consume multiple signals after per-signal planning/execution:

- ECG + PPG pulse-arrival timing: `Session_compute_ecg_ppg_pulse_arrival` estimates ECG R-peak to following PPG pulse delay.
- ECG + RESP + SpO2 sleep-apnea fusion: `Session_screen_sleep_apnea_multimodal` combines ECG HRV proxy, respiratory apnea/hypopnea, respiratory pattern, and SpO2 desaturation/hypoxemia evidence.

Latest session benchmark with session tools:

```text
9 sessions, 26 signal-runs
retrieval/planning/execution accuracy: 1.0 / 1.0 / 1.0
session tool accuracy: 1.0
PAT benchmark uses synchronized BIDMC bidmc01 ECG+PPG: median PAT 528 ms, IQR 16 ms
outputs: /data1/jiahui/biosignal-agent/outputs/session_eval_rule_major_tasks.json
```

## Existing-Work Source Catalog

BioSignalAgent tracks existing algorithms, libraries, and public benchmarks in a machine-readable source catalog:

```bash
python scripts/validate_tool_source_catalog.py
```

Catalog files:

```text
biosignal_agent/tools/source_catalog.json
docs/tool_source_catalog.md
```

The highest-priority wrappers are now larger labeled splits for PCG/PSG/ECG baselines, PSG YASA/sleep-event fusion, and ECG AF/beat classifiers.

## True Labeled Benchmarks

The framework now has three stronger labeled benchmark tracks in addition to planning/tool execution tests:

```bash
python scripts/prepare_ecg_rhythm_beat_dataset.py --max-windows-per-record 5
python scripts/evaluate_ecg_rhythm_beat.py

python scripts/prepare_psg_sleep_dataset.py --max-windows-per-record 80
python scripts/evaluate_psg_sleep.py

python scripts/prepare_pcg_murmur_dataset.py --download --max-per-class 5
python scripts/evaluate_pcg_murmur.py
python scripts/evaluate_pcg_murmur_v2.py

python scripts/prepare_ppg_af_dataset.py --download --max-per-class 4 --seconds 60
python scripts/evaluate_ppg_af.py

python scripts/prepare_acc_activity_dataset.py --download --max-per-class 4
python scripts/evaluate_acc_activity.py

# Manual raw-data placement currently required for WESAD and UniMiB/SisFall-style fall data.
python scripts/prepare_wesad_stress_dataset.py
python scripts/evaluate_wesad_stress.py
python scripts/prepare_acc_fall_dataset.py
python scripts/evaluate_acc_fall.py

# CHB-MIT EDF files can be slow to download; the script enforces both seizure and non-seizure windows.
python scripts/prepare_chbmit_seizure_dataset.py --download --max-seizure-files 1
python scripts/evaluate_chbmit_seizure.py

python scripts/prepare_modality_classifier_dataset.py
python scripts/evaluate_modality_classifier.py

python scripts/render_waveform_digitization_benchmark.py --seconds 10 --max-per-modality 4
python scripts/evaluate_waveform_digitization.py --method rule
python scripts/train_waveform_digitization_pixel_model.py --train-variant clean
python scripts/evaluate_waveform_digitization.py --method ml \
  --out-json /data1/jiahui/biosignal-agent/outputs/waveform_digitization_ml_eval.json \
  --out-csv /data1/jiahui/biosignal-agent/outputs/waveform_digitization_ml_eval.csv
python scripts/train_waveform_digitization_unet.py --epochs 8 --batch-size 4 --height 128 --width 384
python scripts/evaluate_waveform_digitization.py --method unet --probability-threshold 0.65 \
  --out-json /data1/jiahui/biosignal-agent/outputs/waveform_digitization_unet_eval.json \
  --out-csv /data1/jiahui/biosignal-agent/outputs/waveform_digitization_unet_eval.csv

python scripts/prepare_ecg_image_digitization_dataset.py
python scripts/prepare_ecg_image_kit_samples.py --limit 12
python scripts/prepare_ecg_image_kit_generated_dataset.py
python scripts/evaluate_waveform_segmentation_masks.py \
  --manifest /data1/jiahui/biosignal-agent/datasets/processed/ecg_image_kit_generated_manifest.json \
  --model-path /data1/jiahui/biosignal-agent/outputs/ecg_image_kit_generated_unet.pt \
  --out-json /data1/jiahui/biosignal-agent/outputs/ecg_image_kit_generated_segmentation_eval.json \
  --out-csv /data1/jiahui/biosignal-agent/outputs/ecg_image_kit_generated_segmentation_eval.csv
python scripts/train_waveform_digitization_unet.py \
  --manifest /data1/jiahui/biosignal-agent/datasets/processed/ecg_image_digitization_manifest.json \
  --model-path /data1/jiahui/biosignal-agent/outputs/ecg_image_digitization_unet.pt \
  --out-json /data1/jiahui/biosignal-agent/outputs/ecg_image_digitization_unet_train.json \
  --epochs 12 --batch-size 2 --height 128 --width 384
python scripts/evaluate_waveform_digitization.py \
  --manifest /data1/jiahui/biosignal-agent/datasets/processed/ecg_image_digitization_manifest.json \
  --method unet \
  --model-path /data1/jiahui/biosignal-agent/outputs/ecg_image_digitization_unet.pt \
  --probability-threshold 0.65 \
  --out-json /data1/jiahui/biosignal-agent/outputs/ecg_image_digitization_unet_eval.json \
  --out-csv /data1/jiahui/biosignal-agent/outputs/ecg_image_digitization_unet_eval.csv
python scripts/evaluate_image_digitization_smoke.py --method unet \
  --model-path /data1/jiahui/biosignal-agent/outputs/waveform_digitization_unet.pt \
  --out-json /data1/jiahui/biosignal-agent/outputs/ecg_image_kit_unet_smoke_eval.json \
  --out-csv /data1/jiahui/biosignal-agent/outputs/ecg_image_kit_unet_smoke_eval.csv
```

Latest labeled benchmark snapshot:

- MIT-BIH rhythm/AF windows: AF F1 0.442 on 240 windows.
- MIT-BIH beat abnormal screening: F1 0.308 on 18,209 annotated beats; R-peak detection recall 0.936.
- UCDDB PSG sleep staging: coarse-stage macro-F1 0.241 on 80 windows.
- UCDDB PSG respiratory events: F1 0.218 on 80 windows.
- PhysioNet/CinC 2016 PCG normal/abnormal: proxy F1 0.000 on 10 balanced records; feature+logistic-regression baseline F1 0.750.
- MIMIC PERform AF PPG AF/non-AF: irregular-pulse proxy F1 0.857 on 8 windows.
- UCI-HAR ACC activity: random-forest feature baseline macro-F1 0.958 on 48 windows; active/rest F1 1.000.
- CHB-MIT seizure-window screening: EEG seizure-like proxy F1 1.000 on 2 small windows after EDF subset download.
- Signal modality classifier: feature-based random forest macro-F1 0.952 on 117 windows across ECG, PPG, BCG, SCG, RESP, SpO2, ABP, PCG, ACC, EDA, EEG, and EMG.
- Waveform image digitization: rule baseline reaches 31/43 successful digitizations with mean correlation 0.855, NRMSE 0.056, peak-F1 0.837; trained RGB pixel-segmentation baseline reaches 43/43 with mean correlation 0.851, NRMSE 0.056, peak-F1 0.828; tiny U-Net reaches 43/43 with mean correlation 0.843, NRMSE 0.065, peak-F1 0.793 across clean/grid/color/artifact images. ECG-only rendered connector smoke test has 4/4 ok with correlation 0.849, NRMSE 0.087, peak-F1 0.751. ECG-Image-Kit image-only smoke test on 12 sample segments has no waveform ground truth; rule/RGB-pixel/U-Net all produce CSVs, with mean pixel coverage 0.153/0.673/0.947 respectively. ECG-Image-Kit generated MIT-BIH benchmark now has 5 bbox-cropped images, waveform references, and plotted-pixel masks; specialized tiny U-Net reaches segmentation Dice 0.784/IoU 0.646, while waveform calibration remains poor (peak-F1 0.000), so the next issue is coordinate/axis calibration rather than line detection alone. Added image-resolution risk screening and strategy recommendation tools so image agents can avoid under-resolved waveform reconstruction; low-res PCG improves from corr 0.237 to 0.855 under high-res aligned oracle rendering, and a high-res ML pixel digitizer improves overall corr from 0.851 to 0.943. ECG/SCG/EEG/PCG improve to 0.993/0.992/0.987/0.855 respectively. A difficult-signal ultra-high-res track further improves ECG/SCG/EEG/PCG/EMG to 0.999/0.999/0.998/0.946/0.585; EMG remains the main task/spectrogram path candidate. Added `Signal_extract_spectrogram_features` and `Signal_render_spectrogram_image`. PCG spectrogram summary baseline reaches accuracy 0.80/F1 0.75; PCG spectrogram-image PCA/logistic baseline reaches accuracy 0.90/F1 0.889 on 10 records. EMG spectrogram condition smoke reaches window-level macro-F1 1.0 with summary features and 0.967 with spectrogram-image pixels on 30 windows from 3 records; not subject-independent.
- WESAD stress and UniMiB/SisFall fall scripts are implemented; both still need local raw files before metrics are reported.

These numbers are intentionally baseline-level: they turn major tasks into measurable targets before replacing heuristics with stronger models.

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

## Python Framework Entry Point

Use `BioSignalAgentFramework` when integrating the framework from Python instead of a CLI script:

```python
from biosignal_agent.agent.framework import BioSignalAgentConfig, BioSignalAgentFramework
from biosignal_agent.session.schema import SignalInput

agent = BioSignalAgentFramework(BioSignalAgentConfig(planner="rule", retrieved_tool_count=3))
run = agent.run_signal(
    "Estimate ECG heart rate and HRV from this signal",
    SignalInput(
        modality="ecg",
        path="/data1/jiahui/biosignal-agent/datasets/processed/mitdb_100_mlii_60s.csv",
        sampling_rate=360,
    ),
)
```

The same class also runs `BioSignalSession` objects with `run_session(...)`.

## Real PPG/SCG Dataset Expansion

Prepare public real-world PPG and SCG mechanical cardiac signal examples from PhysioNet:

```bash
python scripts/prepare_real_ppg_bcg_data.py --limit 5 --seconds 60
```

This exports BIDMC PPG records and CEBSDB SCG records to CSV under:

```text
/data1/jiahui/biosignal-agent/datasets/processed/real_world/
/data1/jiahui/biosignal-agent/datasets/processed/real_world_manifest.json
```

CEBSDB provides seismocardiogram (SCG), so these records are now evaluated through the dedicated SCG agent pathway. BCG remains a separate modality for bed/load-cell ballistocardiogram data when a dedicated dataset is added.

Run systematic real-data framework evaluation across the expanded question set:

```bash
python scripts/evaluate_real_datasets.py \
  --planner rule \
  --include-ecg \
  --retrieved-tool-count 3 \
  --out-json /data1/jiahui/biosignal-agent/outputs/real_dataset_framework_eval_rule_scg.json \
  --out-csv /data1/jiahui/biosignal-agent/outputs/real_dataset_framework_eval_rule_scg.csv
```

Latest real-data rule eval: 21 records, 64 ECG/PPG/RESP/SCG case-runs, retrieval accuracy 1.0, planning accuracy 1.0, execution accuracy 1.0. BCG, SpO2, ABP, PCG, ACC, EDA, EEG, and EMG are covered in planning/smoke eval and await dedicated real datasets.

## Common Modality Expansion

The framework now includes baseline tools and planning cases for common physiological modalities in this priority order:

```text
RESP -> SpO2 -> ABP -> PCG -> ACC -> EDA -> EEG -> EMG
```

Run the expanded planning regression:

```bash
python scripts/evaluate_agent_framework.py   --planner rule   --retrieved-tool-count 3   --out-json /data1/jiahui/biosignal-agent/outputs/framework_eval_rule_common_modalities.json   --out-csv /data1/jiahui/biosignal-agent/outputs/framework_eval_rule_common_modalities.csv
```

Latest expanded planning eval: 32 cases, retrieval accuracy 1.0, planning accuracy 1.0.

Run the real-data subset evaluation with BIDMC PPG/RESP, CEBSDB RESP/SCG, and MIT-BIH ECG:

```bash
python scripts/evaluate_real_datasets.py   --planner rule   --include-ecg   --retrieved-tool-count 3   --out-json /data1/jiahui/biosignal-agent/outputs/real_dataset_framework_eval_rule_common_modalities.json   --out-csv /data1/jiahui/biosignal-agent/outputs/real_dataset_framework_eval_rule_common_modalities.csv
```

## Dedicated Public Dataset Connectors

Dedicated public-data export is available for the modalities that did not yet have real execution records in the core manifest:

```bash
python scripts/prepare_dedicated_common_datasets.py --limit 3 --seconds 60
```

This writes:

```text
/data1/jiahui/biosignal-agent/datasets/processed/dedicated_common/
/data1/jiahui/biosignal-agent/datasets/processed/dedicated_common_manifest.json
```

Current dedicated sources:

- SpO2: PhysioNet Non-EEG Dataset, `Subject*_SpO2HR`, SpO2 channel
- ACC: PhysioNet Non-EEG Dataset, `Subject*_AccTempEDA`, acceleration magnitude
- EDA: PhysioNet Non-EEG Dataset, `Subject*_AccTempEDA`, EDA channel
- ABP: MIT-BIH Polysomnographic Database, BP channel
- PCG: PhysioNet/CinC Challenge 2016 heart sound WAV records
- EEG: EEG Motor Movement/Imagery Dataset EDF records, first EEG channel
- EMG: Examples of Electromyograms, EMG channel

Run the dedicated-data evaluation:

```bash
python scripts/evaluate_real_datasets.py   --manifest /data1/jiahui/biosignal-agent/datasets/processed/dedicated_common_manifest.json   --planner rule   --retrieved-tool-count 3   --out-json /data1/jiahui/biosignal-agent/outputs/dedicated_common_framework_eval_rule.json   --out-csv /data1/jiahui/biosignal-agent/outputs/dedicated_common_framework_eval_rule.csv
```

Latest dedicated-data eval: 21 records, 42 case-runs, retrieval accuracy 1.0, planning accuracy 1.0, execution accuracy 1.0.
## Cross-Modality Session Benchmark

Evaluate TxAgent-style multi-signal sessions where one question is routed across several physiological signals:

```bash
python scripts/evaluate_sessions.py \
  --planner rule \
  --retrieved-tool-count 3 \
  --out-json /data1/jiahui/biosignal-agent/outputs/session_eval_rule.json \
  --out-csv /data1/jiahui/biosignal-agent/outputs/session_eval_rule.csv
```

Default sessions cover:

- ECG + PPG + RESP + SpO2
- ECG + ABP + SpO2
- PPG + ACC + EDA
- EEG + EMG + ACC
- SCG + RESP
- PCG + ABP + ECG

Latest session eval: 6 sessions, 18 signal-runs, retrieval accuracy 1.0, planning accuracy 1.0, execution accuracy 1.0. Each session saves a trace under `/data1/jiahui/biosignal-agent/outputs/traces/`.
## TxAgent-Style Instruction Dataset Export

Export traces into instruction JSONL with single-signal and multi-signal samples:

```bash
python scripts/export_trace_dataset.py \
  --include-negative \
  --out /data1/jiahui/biosignal-agent/outputs/biosignal_txagent_sft.jsonl

python scripts/export_trace_dataset.py \
  --planning-only \
  --include-negative \
  --out /data1/jiahui/biosignal-agent/outputs/biosignal_txagent_planning_sft.jsonl
```

The full export includes:

- `biosignal_tool_planning`
- `biosignal_tool_execution_trace`
- `biosignal_report_generation`
- `biosignal_session_tool_planning`
- `biosignal_session_tool_execution_trace`
- `biosignal_session_report_generation`
- `biosignal_negative_tool_planning`

Validate JSONL format and task counts:

```bash
python scripts/validate_instruction_dataset.py /data1/jiahui/biosignal-agent/outputs/biosignal_txagent_sft.jsonl
python scripts/validate_instruction_dataset.py /data1/jiahui/biosignal-agent/outputs/biosignal_txagent_planning_sft.jsonl
```

Latest export: 90 full samples and 32 planning-only samples, both with zero validation errors.
## Dedicated BCG Dataset Connector

A dedicated BCG connector is available for the Figshare bedside BCG dataset from the Scientific Data article "A ballistocardiogram dataset with reference ECG signals for bedside heart rhythm assessment".

```bash
python scripts/prepare_dedicated_bcg_dataset.py --limit 3 --seconds 60
```

The full Figshare dataset is 16.35 GB, so this script streams only the first requested seconds from selected BCG CSV files instead of downloading the full archive. It writes:

```text
/data1/jiahui/biosignal-agent/datasets/processed/dedicated_bcg/
/data1/jiahui/biosignal-agent/datasets/processed/dedicated_bcg_manifest.json
```

Run the BCG-specific evaluation:

```bash
python scripts/evaluate_real_datasets.py \
  --manifest /data1/jiahui/biosignal-agent/datasets/processed/dedicated_bcg_manifest.json \
  --planner rule \
  --retrieved-tool-count 3 \
  --out-json /data1/jiahui/biosignal-agent/outputs/dedicated_bcg_framework_eval_rule.json \
  --out-csv /data1/jiahui/biosignal-agent/outputs/dedicated_bcg_framework_eval_rule.csv
```

Latest dedicated BCG eval: 3 records, 12 case-runs, retrieval accuracy 1.0, planning accuracy 1.0, execution accuracy 1.0.

The cross-modality session benchmark now also includes a BCG + RESP + ECG session:

```bash
python scripts/evaluate_sessions.py \
  --planner rule \
  --retrieved-tool-count 3 \
  --out-json /data1/jiahui/biosignal-agent/outputs/session_eval_rule_with_bcg.json \
  --out-csv /data1/jiahui/biosignal-agent/outputs/session_eval_rule_with_bcg.csv
```

Latest session eval with BCG: 7 sessions, 21 signal-runs, retrieval accuracy 1.0, planning accuracy 1.0, execution accuracy 1.0.

## Tool Output Audit And First Quality Pass

Audit every local tool over the real-world, dedicated common, and dedicated BCG manifests:

```bash
python scripts/audit_tool_outputs.py \
  --manifest /data1/jiahui/biosignal-agent/datasets/processed/real_world_manifest.json \
  --manifest /data1/jiahui/biosignal-agent/datasets/processed/dedicated_common_manifest.json \
  --manifest /data1/jiahui/biosignal-agent/datasets/processed/dedicated_bcg_manifest.json \
  --out-json /data1/jiahui/biosignal-agent/outputs/tool_output_audit_optimized.json \
  --out-csv /data1/jiahui/biosignal-agent/outputs/tool_output_audit_optimized.csv
```

The first quality pass adds interval-regularity confidence to PPG, BCG, SCG, ABP, and PCG beat detectors, uses a BCG-focused 3-12 Hz preprocessing band, and replaces generic SpO2 signal quality with SpO2-specific plausibility/range/jump checks. Tool schemas expose the new audit fields.

Latest optimized tool audit: 44 records, 88 tool-runs, all modalities ok-rate 1.0, zero errors, zero low-confidence runs. Optimized regression outputs are also available at:

```text
/data1/jiahui/biosignal-agent/outputs/real_dataset_framework_eval_rule_common_modalities_optimized.json
/data1/jiahui/biosignal-agent/outputs/dedicated_common_framework_eval_rule_optimized.json
/data1/jiahui/biosignal-agent/outputs/dedicated_bcg_framework_eval_rule_optimized.json
/data1/jiahui/biosignal-agent/outputs/session_eval_rule_with_bcg_optimized.json
```

All optimized framework evals retained retrieval accuracy 1.0, planning accuracy 1.0, and execution accuracy 1.0.

## Benchmark Report Index

Build one TxAgent-style benchmark index from the current manifests, real-data evals, session eval, tool audit, and instruction JSONL validation:

```bash
python scripts/build_benchmark_report.py
```

This writes:

```text
/data1/jiahui/biosignal-agent/outputs/benchmark_report.json
/data1/jiahui/biosignal-agent/outputs/benchmark_report.md
```

The report is intended as the stable comparison surface when changing the planner backend, adding datasets, or improving individual signal tools.

## OpenRouter Planner Evaluation

The OpenRouter client loads all available keys from `/home/myid/jl57095/TwinMarket/openrouter_caption_with_P_wave.py`, including `API_KEY`, `API_KEYS`, and `candidate_keys`, plus environment-provided keys. For LLM planner benchmarking, use the checkpointed runner so long jobs can resume case by case:

```bash
python scripts/run_openrouter_planner_eval.py \
  --model openrouter/owl-alpha \
  --retrieved-tool-count 3 \
  --llm-timeout 30 \
  --llm-retry-max 2 \
  --llm-retry-delay 1
```

Compare the LLM planner against the rule planner:

```bash
python scripts/compare_planner_evals.py \
  --rule-json /data1/jiahui/biosignal-agent/outputs/framework_eval_rule_common_modalities.json \
  --candidate-json /data1/jiahui/biosignal-agent/outputs/framework_eval_openrouter_common_modalities_retry.json \
  --out-json /data1/jiahui/biosignal-agent/outputs/planner_comparison_openrouter_retry_vs_rule.json \
  --out-csv /data1/jiahui/biosignal-agent/outputs/planner_comparison_openrouter_retry_vs_rule.csv
```

Initial no-fallback LLM baseline: 32 planning cases, 30 true OpenRouter successes, 2 malformed-JSON planner errors, planning accuracy 0.9375. After adding planner-level retry for malformed or invalid JSON responses, the OpenRouter planner reached 32/32 planning accuracy with zero disagreements against the rule planner.

## Expanded Clinical And Physiological Tasks

The planning benchmark now includes broader task prompts beyond peak/rate extraction:

See `docs/ecg_ppg_sota_mapping.md` for the current ECG/PPG SOTA/GitHub provenance map, including which tools are local DL models, open-source candidates, or conservative proxy wrappers.


- Generic artifact screening: clipping, flatline/dropout, abrupt jumps, and high-frequency noise.
- ECG heart-rate, R-peak/QRS, HRV, stress/fatigue proxy, and beat-level classification wrappers.
- ECG rhythm/AF/arrhythmia screening: rhythm-segment classifier, AF-specific wrapper, irregular RR, pauses, bradycardia, tachycardia, and ectopy-proxy flags.
- ECG morphology/interval screening: PR/QRS/QT/QTc, conduction-delay, and ST-deviation proxies.
- ECG sleep-apnea proxy: HRV/RR-pattern screening for ECG-only apnea baselines.
- PPG task wrappers: HR, PRV, systolic/onset/notch fiducials, respiration, AF/irregular pulse, dual-wavelength SpO2 proxy, perfusion/low-shock proxy, BP/vascular morphology proxy, sleep/rest features, stress/recovery PRV proxy, and exercise-intensity HR proxy.
- RESP sleep-apnea, hypopnea, and rate-pattern screening: low-amplitude pauses, reduced-flow events, tachypnea/bradypnea/periodic breathing proxies.
- SpO2 desaturation and hypoxemia burden: ODI-style event count, minimum SpO2, time below 90/88 percent.
- ABP pressure-event and hemodynamic proxy: approximate hypotension/hypertension flags, MAP, and pulse pressure.
- PCG murmur proxy and feature baseline: high-frequency continuous energy plus spectral/envelope/timing features.
- EDA arousal/stress tools: WESAD-compatible tonic/phasic/SCR features, WESAD-trained binary stress classifier (AUROC 0.870/BAcc 0.780), WESAD baseline/stress/amusement CNN protocol-state classifier (macro-AUROC 0.759/macro-F1 0.568), plus heuristic fallback and safety routing for unsupported uses such as standalone lie detection.
- EEG sleep-stage features and seizure-like proxy: band ratios plus robust spike/fast-power screen.
- EMG activation and fatigue proxy: RMS/MAV plus median-frequency fatigue hint.
- EMG gesture/action/fatigue/rehab/neuromuscular models: UCI gesture 6-class feature ensemble reaches subject-independent accuracy 0.777 / macro-F1 0.777; NinaPro DB1 52-class prosthetic gesture augmented feature baseline reaches calibrated-user top-1 0.585/top-5 0.821 but strict subject-held-out top-1 0.107; optional multi-stream CNN backend reaches calibrated trial-voting top-1 0.732/top-5 0.937; UCI onset-derived movement-intent proxy reaches accuracy/macro-F1 0.725; UCI Physical Action 20-class action recognition reaches subject-held-out accuracy 0.269 / macro-F1 0.266, while normal-vs-aggressive screen reaches AUROC 0.884 / accuracy 0.827 under leave-one-subject-out; UCI lower-limb EMG/goniometry exercise classification reaches subject-held-out accuracy 0.836 / macro-F1 0.835, and knee normal-vs-abnormal rehab screen reaches AUROC 0.930 / macro-F1 0.921; GEDS wearable gait speed reaches subject-held-out accuracy 0.789 / macro-F1 0.788 on the full S00-S22 414-trial benchmark, and right stance/swing phase reaches accuracy 0.974 / macro-F1 0.972 / AUROC 0.997; Zenodo fatigue early-vs-late protocol model reaches AUROC 0.774 / balanced accuracy 0.699. Added an EMGDB healthy/myopathy/neuropathy smoke screen with window-level accuracy/macro-F1 1.0 on 27 windows, but it is not subject-independent and is explicitly research-only. A small 1D CNN underperforms the feature ensemble on cross-subject gesture recognition (macro-F1 0.411), so the tool keeps the feature model. Added `EMG_classify_gesture`, `EMG_classify_prosthetic_gesture` with `backend="feature"|"multistream_cnn"`, `EMG_predict_movement_intent`, `EMG_classify_physical_action`, `EMG_classify_lower_limb_exercise`, `EMG_classify_gait_speed`, `EMG_estimate_gait_phase`, `EMG_screen_knee_rehab_status`, `EMG_analyze_gait_activation`, `EMG_screen_neuromuscular_abnormality`, and `EMG_estimate_fatigue_ml` tools.
- ACC sleep/wake proxy: actigraphy-style rest/activity hint.

The expanded rule-planning set has 63 planning cases. Latest expanded rule evals:

```text
planning regression: 63 cases, retrieval/planning accuracy 1.0
real-world + MIT-BIH ECG: 25 records, 155 case-runs, retrieval/planning/execution accuracy 1.0
dedicated common datasets: 21 records, 96 case-runs, retrieval/planning/execution accuracy 1.0
dedicated BCG: 3 records, 15 case-runs, retrieval/planning/execution accuracy 1.0
tool audit: 49 records, 274 tool-runs, all modalities ok-rate 1.0, zero errors, zero low-confidence runs
```



Core task expansion outputs:

```text
/data1/jiahui/biosignal-agent/outputs/framework_eval_rule_core_tasks.json
/data1/jiahui/biosignal-agent/outputs/real_dataset_framework_eval_rule_session_tools.json
/data1/jiahui/biosignal-agent/outputs/dedicated_common_framework_eval_rule_core_tasks.json
/data1/jiahui/biosignal-agent/outputs/dedicated_bcg_framework_eval_rule_core_tasks.json
/data1/jiahui/biosignal-agent/outputs/tool_output_audit_session_tools.json
```

Broader task/tool expansion outputs:

```text
/data1/jiahui/biosignal-agent/outputs/framework_eval_rule_more_tasks.json
/data1/jiahui/biosignal-agent/outputs/real_dataset_framework_eval_rule_more_tasks.json
/data1/jiahui/biosignal-agent/outputs/dedicated_common_framework_eval_rule_more_tasks.json
/data1/jiahui/biosignal-agent/outputs/tool_output_audit_more_tasks.json
/data1/jiahui/biosignal-agent/outputs/ucddb_resp_spo2_eval_more_tasks.json
```

Expanded instruction exports:

```text
/data1/jiahui/biosignal-agent/outputs/biosignal_txagent_sft_expanded_tasks.jsonl              # 342 samples, 0 validation errors
/data1/jiahui/biosignal-agent/outputs/biosignal_txagent_planning_sft_expanded_tasks.jsonl     # 116 samples, 0 validation errors
```

See `docs/task_catalog.md` for the runnable task map and the next labeled benchmarks to connect.

### First Labeled Benchmark: MIT-BIH Arrhythmia Windows

MIT-BIH annotation windows provide the first labeled task benchmark for the expanded framework:

```bash
python scripts/prepare_labeled_arrhythmia_dataset.py \
  --seconds 60 \
  --stride-seconds 60 \
  --max-windows-per-record 5

python scripts/evaluate_labeled_arrhythmia.py \
  --manifest /data1/jiahui/biosignal-agent/datasets/processed/labeled_arrhythmia_manifest.json \
  --out-json /data1/jiahui/biosignal-agent/outputs/labeled_arrhythmia_eval.json \
  --out-csv /data1/jiahui/biosignal-agent/outputs/labeled_arrhythmia_eval.csv
```

Current label benchmark: 240 MIT-BIH 60-second ECG windows, 141 abnormal and 99 normal by annotation-derived beat labels. The baseline RR-heuristic arrhythmia screen reaches accuracy 0.675, precision 0.806, recall 0.589, specificity 0.798, and F1 0.680. Many false negatives are morphology-only or paced/fusion windows with regular RR, which is expected for this first screening heuristic and motivates a true rhythm/morphology classifier.

The expanded benchmark report includes this labeled benchmark:

```text
/data1/jiahui/biosignal-agent/outputs/benchmark_report_expanded_tasks.json
/data1/jiahui/biosignal-agent/outputs/benchmark_report_expanded_tasks.md
```

### Second Labeled Benchmark: Apnea-ECG Minute Labels

A small Apnea-ECG subset is available for ECG-only sleep-apnea proxy benchmarking:

```bash
python scripts/prepare_apnea_ecg_dataset.py \
  --records a01 b01 c01 \
  --max-minutes-per-record 20 \
  --download

python scripts/evaluate_apnea_ecg.py \
  --manifest /data1/jiahui/biosignal-agent/datasets/processed/apnea_ecg_manifest.json \
  --out-json /data1/jiahui/biosignal-agent/outputs/apnea_ecg_eval.json \
  --out-csv /data1/jiahui/biosignal-agent/outputs/apnea_ecg_eval.csv
```

Current Apnea-ECG subset: 60 one-minute ECG windows, 8 apnea and 52 normal. The ECG-only HRV proxy is intentionally a weak baseline: accuracy 0.533, precision 0.083, recall 0.250, specificity 0.577, and F1 0.125. This gives the framework a concrete negative result and points the next apnea work toward RESP/SpO2/PSG-labeled datasets instead of ECG-only heuristics.

### SpO2 Tool Optimization

SpO2 now exposes rolling-baseline ODI3/ODI4 desaturation detection, event depth/duration/area, T90/T88/T85, CT90, hypoxic burden area below 90%, oximetry-only sleep-apnea severity proxy, and a UCDDB-trained SpO2-only feature model. On UCDDB 25-record / 1974 valid 30s SpO2 windows with record-level GroupKFold, the SpO2-only ML model reaches AUROC 0.542, macro-F1 0.523, and balanced accuracy 0.523; the ODI/hypoxic-burden heuristic reaches AUROC 0.555. This is intentionally documented as a negative result for exact 30s respiratory-event detection: SpO2 remains valuable for overnight burden summaries but should be fused with respiratory airflow/effort for event timing.

### RESP/SpO2 PSG Benchmark: UCDDB

UCDDB provides PSG respiratory-event labels plus `Flow` and `SpO2` channels in the `.rec` EDF-like file. The connector includes a lightweight EDF reader, so no extra EDF package is required:

```bash
python scripts/prepare_ucddb_resp_spo2_dataset.py \
  --records ucddb002 \
  --max-windows-per-record 40 \
  --download

python scripts/evaluate_ucddb_resp_spo2.py \
  --manifest /data1/jiahui/biosignal-agent/datasets/processed/ucddb_resp_spo2_manifest.json \
  --out-json /data1/jiahui/biosignal-agent/outputs/ucddb_resp_spo2_eval.json \
  --out-csv /data1/jiahui/biosignal-agent/outputs/ucddb_resp_spo2_eval.csv
```

Current UCDDB subset: 40 balanced 60-second RESP/SpO2 windows, 20 respiratory-event and 20 normal. The original baseline using `RESP_detect_apnea` plus `SpO2_detect_desaturation` reached F1 0.563. After adding `RESP_detect_hypopnea`, the expanded baseline reaches accuracy 0.650, precision 0.714, recall 0.500, specificity 0.800, and F1 0.588. This is still a lightweight PSG-screening baseline; further gains should use event-level labels, desaturation/arousal coupling, and better respiratory-flow morphology features.

Key outputs:

```text
/data1/jiahui/biosignal-agent/outputs/framework_eval_openrouter_common_modalities.json
/data1/jiahui/biosignal-agent/outputs/framework_eval_openrouter_common_modalities.jsonl
/data1/jiahui/biosignal-agent/outputs/framework_eval_openrouter_common_modalities_retry.json
/data1/jiahui/biosignal-agent/outputs/planner_comparison_openrouter_vs_rule.json
/data1/jiahui/biosignal-agent/outputs/planner_comparison_openrouter_retry_vs_rule.json
```


Multimodal wrappers added: `Multimodal_estimate_ecg_ppg_pat_bp_proxy` for ECG+PPG PAT/BP proxy evidence and `Multimodal_screen_sleep_apnea_report` for ECG/RESP/SpO2 sleep-apnea evidence fusion.
