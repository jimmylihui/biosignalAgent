# BioSignalAgent Manuscript Results Draft

This draft is generated from frozen paper artifacts. It is a writing scaffold, not a new experiment.

## Abstract-Style Result Summary

BioSignalAgent instantiates a TxAgent-style tool-use agent for biosignals with BioSignalToolUniverse v1 (131 frozen tools) and BioSignalBench v1 (238 cases). On the clean held-out split (48 cases), the selected live SFT planner/controller reaches planning accuracy 0.875, Tool F1 0.960, execution success 1.000, report factuality 0.946, and overall H-mean 0.943. On the expanded held-out stress split (92 cases), raw live SFT reaches planning 0.489 and Tool F1 0.804; the timeout/metadata-guarded controller reaches planning 1.000, Tool F1 1.000, and overall 0.989.

## Methods Draft

### BioSignalToolUniverse v1

We freeze BioSignalToolUniverse v1 as 131 executable tool schemas covering ECG, PPG, PCG, SCG, BCG, EMG, EDA, EEG, RESP, SpO2, ABP, ACC, image digitization, and multimodal/session reasoning. Each tool includes structured metadata for modality, task, evidence level, data source, metric, known failure modes, and clinical limitations. Proxy tools are retained for realistic routing but explicitly labeled and not framed as diagnostic validators.

Recommended artifact: `biosignal_tool_universe_v1.json`; recommended table: `table1_tool_universe.md`.

### BioSignalBench v1

BioSignalBench v1 contains 238 cases split into train/dev/held-out = 168/22/48. It covers modality routing, image-to-signal digitization, scale/OCR extraction, tool planning, executable tool evaluation, report factuality, and multimodal session reasoning. BioSignalBench v1-expanded now contains 1,279 cases; it is a stress/scale complement, not a replacement for the clean split. The expansion adds public-signal task cases from MIT-BIH arrhythmia windows, real-world ECG/PPG/RESP/SCG/BCG segments, dedicated ACC/EDA/SpO2/ABP/PCG/EEG/EMG/BCG manifests, waveform-image digitization cases, and UCDDB EEG+RESP+SpO2 session cases.

Recommended tables: `table2_biosignalbench_composition.md`, `table27_biosignalbench_expanded_composition.md`.

### Agent Controller

The controller uses ToolRAG retrieval, SFT planner generation, executable local biosignal tools, and grounded report generation/scoring. The final live controller includes timeout-safe generation plus metadata-derived guardrails for image digitization, scale/OCR extraction, unknown-modality routing, multimodal session completion, and minimal task-specific pruning.

## Results Draft

### Main Clean Held-Out Result

- Cases: 48
- Retrieval accuracy: 1.000
- Strict/recovered planner parse: 1.000/1.000
- Planning accuracy: 0.875
- Tool F1: 0.960
- Execution success: 1.000
- Report factuality score: 0.946
- Overall H-mean: 0.943
- Failure reasons: {"planning_tool_mismatch": 6}

Use this as the primary paper result because it follows the frozen split protocol. Recommended table: `table18_split_protocol_controller_comparison.md`.

### Expanded Stress Result and Guardrail Ablation

- Expanded held-out cases: 92
- Raw live SFT planning / Tool F1 / overall: 0.489 / 0.804 / 0.724
- Guarded live SFT planning / Tool F1 / overall: 1.000 / 1.000 / 0.989
- Raw failure reasons: {"planning_tool_mismatch": 47}
- Guarded failure reasons: {}

Interpretation: retrieval is not the bottleneck on the expanded split. Raw live SFT often emits plausible but non-exact tool sets, especially for report-factuality and multimodal cases. The metadata guardrail converts these into exact executable plans, so raw and guarded rows should be reported separately. Recommended table: `table28_expanded_heldout_baseline_comparison.md`.

### External LLM Baseline

- OpenRouter/owl-alpha clean held-out planning: 0.229
- OpenRouter/owl-alpha clean held-out Tool F1: 0.747
- OpenRouter/owl-alpha clean held-out overall: 0.440
- OpenRouter failure reasons: {"planning_tool_mismatch": 37}

Expanded OpenRouter runs were not used as a final artifact because synchronous external API calls were unstable during SSL read. Treat this as an external-baseline limitation unless a stable paid endpoint is added.

### Module Ablation

The held-out ablation table isolates controller components. Removing ToolRAG gives the largest drop; removing quality gates, image digitization, OCR/scale, modality classification, or DL tools produces targeted degradations. Recommended table: `table23_live_controller_ablation_v5_guarded_heldout.md`.

### Tool Execution Evidence

The unified tool execution index contains 34 metric rows across major modalities. Strong validated examples include ECG R-peak detection, PTB-XL 12-lead classification, PPG pulse detection, PCG S1/S2 segmentation, EDA stress detection, ACC activity recognition, and ABP hypotension/shock prediction. Weaker/proxy rows remain explicitly labeled. Recommended table: `table26_tool_execution_metrics_index.md`.

## Discussion Draft

The main empirical pattern is that BioSignalAgent has strong retrieval coverage and executable tool plumbing, while exact structured planning remains the central modeling challenge. The raw-vs-guarded expanded result makes this explicit: the SFT planner often selects plausible tools but misses benchmark minimality, and metadata-aware controller repair closes that gap. This is defensible for a tool-use system, but the manuscript should avoid presenting guardrail-repaired plans as pure LLM generation.

## Limitations To State Explicitly

- Clean held-out size is 48; expanded held-out is useful stress evidence but not the primary frozen split.
- Many tools are proxy or benchmark-limited and should not be described as clinical diagnostic systems.
- The final expanded perfect planning row depends on metadata-derived guardrails; raw SFT performance should be reported alongside it.
- External LLM baseline coverage is limited to a free OpenRouter model on the clean split because expanded synchronous calls were unstable.
- Several bottom-level signal models remain below modality-specific SOTA and need larger external validation before clinical claims.

## Current TxAgent-Comparability Gaps

- Systematic agent benchmark: Clean held-out is still smaller than a full TxAgent-scale benchmark; expanded split is stress evidence, not replacement.
- SFT tool-use agent: SFT data is still small and strict-JSON parse rate remains imperfect; distinguish model generation from metadata guardrail repair in the manuscript.
- External LLM baseline: Only one free external model; stronger GPT/Claude/Gemini/Qwen baselines need stable API/budget.
- Tool execution numeric evidence: Some rows remain proxy/small-split; needs more external validation for clinical claims.

## Paper Table Checklist

- `table1_tool_universe.md`: present
- `table2_biosignalbench_composition.md`: present
- `table18_split_protocol_controller_comparison.md`: present
- `table23_live_controller_ablation_v5_guarded_heldout.md`: present
- `table24_final_failure_analysis.md`: present
- `table25_paper_artifact_index.md`: present
- `table26_tool_execution_metrics_index.md`: present
- `table27_biosignalbench_expanded_composition.md`: present
- `table28_expanded_heldout_baseline_comparison.md`: present
- `table30_txagent_gap_matrix.md`: present


### Expanded Benchmark Size Update - 2026-05-26

We increased BioSignalBench v1-expanded from the prior 461-case stress manifest to a 1,279-case stress/scale manifest while preserving the frozen clean v1 split. The new manifest contains 603 tool-execution cases, 268 multimodal-session cases, 217 report-factuality cases, 66 tool-planning cases, 63 image digitization cases, and 62 scale/OCR cases. Inputs include 885 CSV waveform cases, 268 session cases, 125 image cases, and one text negative case. Validation passes with 0 errors.

Lightweight expanded-baseline results: rule planner + TF-IDF retrieval obtains retrieval 1.000, exact planning 0.034, and Tool F1 0.536; TF-IDF ToolRAG-as-planner obtains retrieval 1.000, exact planning 0.000, and Tool F1 0.526; oracle tools remain 1.000. This confirms that the expanded benchmark is substantially harder than the clean split and should be reported as a stress benchmark rather than the primary headline score until the live SFT controller is retrained/evaluated on the larger task distribution.
