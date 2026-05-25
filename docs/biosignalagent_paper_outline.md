# BioSignalAgent Paper Outline

## Working Title

BioSignalAgent: A Tool-Using Agent and Benchmark for Multimodal Biosignal Reasoning

## Core Claim

BioSignalAgent adapts the TxAgent-style ToolUniverse + tool-use SFT + benchmark evaluation recipe to biosignals. The artifact now includes a frozen BioSignalToolUniverse, BioSignalBench, live controller, clean split protocol, external LLM baseline, module ablations, failure analysis, expanded stress testing, and a unified tool-execution metric index.

## Main Numbers

- BioSignalToolUniverse v1: `131` frozen tools.
- BioSignalBench v1: `238` cases; train/dev/held-out = `168/22/48`.
- BioSignalBench v1-expanded: `461` cases; train/dev/held-out = `324/45/92`.
- Clean held-out live v5+guardrail: planning `0.875`, Tool F1 `0.960`, execution `1.000`, report score `0.946`, overall `0.943`.
- Expanded raw live SFT: planning `0.489`, Tool F1 `0.804`, overall `0.724`.
- Expanded timeout/metadata-guarded live SFT: planning `1.000`, Tool F1 `1.000`, overall `0.989`.
- Tool execution index: `34` metric rows.

## Paper Structure

1. Introduction: biosignal reasoning needs tool-use agents with explicit modality routing, scale/image handling, executable signal tools, and grounded reporting.
2. BioSignalToolUniverse v1: frozen schemas, evidence metadata, proxy labeling, and source catalog.
3. BioSignalBench v1: benchmark tasks, JSON schema, clean split protocol, expanded stress split.
4. BioSignalAgent: ToolRAG retrieval, SFT planner, timeout-safe live controller, metadata guardrails, tool execution, grounded reporting.
5. Experiments: clean held-out, expanded stress, raw-vs-guarded ablation, module ablations, external LLM baseline, failure analysis.
6. Tool evidence: modality-level execution metrics and validated/proxy limitations.
7. Discussion and limitations: guardrails vs raw generation, small clean split, proxy tools, external API instability, non-clinical framing.

## Recommended Primary Tables

- ToolUniverse summary: `table1_tool_universe.md`
- BioSignalBench composition: `table2_biosignalbench_composition.md`
- Main split-protocol controller comparison: `table18_split_protocol_controller_comparison.md`
- Held-out ablation: `table23_live_controller_ablation_v5_guarded_heldout.md`
- Failure analysis: `table24_final_failure_analysis.md`
- Tool execution evidence: `table26_tool_execution_metrics_index.md`
- Expanded benchmark composition: `table27_biosignalbench_expanded_composition.md`
- Expanded baseline/raw-vs-guarded comparison: `table28_expanded_heldout_baseline_comparison.md`
- TxAgent comparability matrix: `table30_txagent_gap_matrix.md`

## Next Work Before Submission

1. Add stronger stable external LLM baselines if API/budget allows.
2. Add qualitative examples for success/failure cases in an appendix.
3. Convert `docs/biosignalagent_manuscript_results_draft.md` into full prose.
4. Keep proxy/clinical limitations explicit in every report and table caption.
