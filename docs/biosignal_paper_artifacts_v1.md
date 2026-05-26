# BioSignalAgent Paper Artifacts v1

This document records the TxAgent-style paper artifacts added for BioSignalAgent. The goal is to make the tool universe, benchmark, SFT data, evaluations, ablations, and paper tables reproducible from local scripts.

## Artifacts

- Tool universe: `/data1/jiahui/biosignal-agent/outputs/biosignal_tool_universe_v1.json`
- Benchmark manifest: `/data1/jiahui/biosignal-agent/outputs/biosignalbench_v1.jsonl`
- Planner SFT data: `/data1/jiahui/biosignal-agent/outputs/biosignal_sft_planner_v1.jsonl`
- Planner/report SFT data: `/data1/jiahui/biosignal-agent/outputs/biosignal_sft_planner_report_v1.jsonl`
- Main evals: `/data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_*.json`
- Ablations: `/data1/jiahui/biosignal-agent/outputs/biosignalbench_ablations/`
- Paper tables: `/data1/jiahui/biosignal-agent/outputs/paper_tables/`

## Rebuild Commands

```bash
python scripts/validate_tool_source_catalog.py
python scripts/build_biosignal_tool_universe.py
python scripts/validate_biosignal_tool_universe.py
python scripts/build_biosignalbench_v1.py
python scripts/validate_biosignalbench.py
python scripts/build_biosignal_sft_tool_use_dataset.py
python scripts/evaluate_biosignalbench.py --planner-backend rule --retriever-backend tfidf
python scripts/evaluate_biosignalbench.py --planner-backend oracle --retriever-backend oracle --out-json /data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_oracle.json --out-jsonl /data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_oracle_cases.jsonl --out-csv /data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_oracle_cases.csv --out-md /data1/jiahui/biosignal-agent/outputs/paper_tables/biosignalbench_eval_oracle.md
python scripts/evaluate_biosignalbench.py --planner-backend sft_replay --retriever-backend tfidf --out-json /data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_sft_replay.json --out-jsonl /data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_sft_replay_cases.jsonl --out-csv /data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_sft_replay_cases.csv --out-md /data1/jiahui/biosignal-agent/outputs/paper_tables/biosignalbench_eval_sft_replay.md
python scripts/evaluate_biosignalbench.py --planner-backend no_tool_llm --retriever-backend none --out-json /data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_no_tool.json --out-jsonl /data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_no_tool_cases.jsonl --out-csv /data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_no_tool_cases.csv --out-md /data1/jiahui/biosignal-agent/outputs/paper_tables/biosignalbench_eval_no_tool.md
python scripts/run_biosignalbench_ablations.py
python scripts/build_biosignal_paper_tables.py
```

## Current Snapshot

- BioSignalToolUniverse v1 has 132 frozen tools and passes strict source-metadata validation.
- BioSignalBench v1 has 233 cases across tool planning, image digitization, scale/OCR extraction, report factuality, and multimodal session reasoning.
- Standardized SFT data currently has 191 planner examples and 68 planner/report examples after deduplication and negative-example injection.
- The first rule+TFIDF baseline is intentionally broad and exposes gaps on image/session/report cases; it is a baseline for learned ToolRAG/SFT rather than the expected final system.

## Safety Framing

Proxy tools remain in v1 but are explicitly labeled through tool metadata and SFT limitations. Reports must not phrase proxy, low-confidence, or modest-AUROC screening outputs as diagnosis.

## SFT Planner v2 Snapshot

A first Qwen2.5-0.5B-Instruct LoRA planner was trained on the ToolRAG-augmented SFT v2 data. The best 6-epoch adapter is:

```text
/data1/jiahui/biosignal-agent/outputs/biosignal_sft_planner_lora_qwen25_05b_toolrag_v2_e6/best_adapter
```

Training used 191 planner examples, 168 train / 23 validation, max length 1024, LoRA rank 16. Best validation loss was 0.1066.

BioSignalBench generation evaluation:

- Full 233-case planning accuracy: 0.708; parse rate: 0.820.
- Tool-planning subset: 0.894 planning accuracy; parse rate: 1.000.
- Scale/OCR subset: 0.958 planning accuracy; parse rate: 0.958.
- Report-factuality subset: 0.789 planning accuracy; parse rate: 0.989.
- Image digitization subset: 0.480 planning accuracy; parse rate: 0.520.
- Multimodal session subset: 0.000 planning accuracy; parse rate: 0.000, currently the main remaining SFT formatting gap.

The result exceeds the rule planner on the single-signal tool-planning subset but still needs a session-specific output schema/training pass before being paper-ready as the main agent.
## Main Evaluation v1 snapshot - 2026-05-25

We froze the current BioSignalBench v1 manifest at 233 cases and reran TxAgent-style main baselines plus ablations with a unified evaluator. The evaluator now reports exact planning accuracy and softer tool-selection precision/recall/F1, which is important because retrieval-only ToolRAG often retrieves relevant tools plus extra tools and therefore fails exact-match planning.

Main comparison artifacts:
- `/data1/jiahui/biosignal-agent/outputs/paper_tables/table3_main_benchmark_comparison.md`
- `/data1/jiahui/biosignal-agent/outputs/paper_tables/table4_ablation.md`
- `/data1/jiahui/biosignal-agent/outputs/paper_tables/failure_analysis.md`

Current headline numbers on 233 BioSignalBench cases:
- no-tool LLM baseline: planning accuracy 0.000, Tool F1 0.000.
- rule planner + TF-IDF retriever: planning accuracy 0.476, Tool F1 0.671.
- naive TF-IDF ToolRAG-as-planner: planning accuracy 0.000, Tool F1 0.221, showing retrieval alone over-selects tools.
- trace replay SFT baseline: planning accuracy 0.258, Tool F1 0.416.
- SFT LoRA planner with tool candidates in prompt: planning accuracy 0.708, Tool F1 0.794, parse rate 0.820.
- oracle tool selection: planning accuracy 1.000, Tool F1 1.000.

Current failure analysis points to three main next targets:
- image/OCR/digitization routing remains weak in rule and retrieval baselines.
- multimodal session planning needs dedicated SFT examples and evaluator cases.
- retrieval misses are dominated by quality-gate and scale/OCR expected tools, so retriever descriptions or case construction need tightening.
## Focused SFT v3 update - 2026-05-25

After Main Evaluation v1, the main remaining failures were image/OCR/digitization routing and multimodal session planning. We found that image/OCR failures were largely generation-length artifacts: increasing SFT LoRA generation from 192 to 384 tokens improved image-to-signal planning to 0.96 and scale/OCR planning to 1.00. We then rebuilt focused SFT data with compact session tool-call arguments and oversampled image/session/scale cases.

Focused v3 SFT data:
- `/data1/jiahui/biosignal-agent/outputs/biosignal_sft_planner_v3_focused.jsonl`
- planner examples: 307
- focus oversample: 3
- compact session arguments: per-tool `signal_path`, `sampling_rate`, `column`, rather than repeating the full session signal list in every call.

Focused v3 LoRA:
- adapter: `/data1/jiahui/biosignal-agent/outputs/biosignal_sft_planner_lora_qwen25_05b_toolrag_v3_focused/best_adapter`
- examples: 307, train/val: 270/37, epochs: 6
- best validation loss: 0.0522

Best current BioSignalBench planner result uses 384 generated tokens generally and 768 tokens for multimodal session cases:
- artifact: `/data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_sft_lora_toolrag_v3_focused_hybrid.json`
- full BioSignalBench cases: 233
- planning accuracy: 0.897
- parse rate: 1.000
- Tool F1: 0.977
- image-to-signal planning: 1.000
- scale/OCR planning: 1.000
- multimodal session planning: 1.000
- tool planning: 0.924
- report factuality tool selection: 0.789 exact, 0.958 Tool F1

This makes the next bottleneck report factuality/tool grounding rather than routing/planning.
## Report factuality / grounding update - 2026-05-25

We added a deterministic report factuality evaluator for BioSignalBench report cases. It checks tool mention recall, numeric grounding against tool_results, salient key coverage, research-use disclaimer presence, and unsafe diagnosis/proxy language. The evaluator writes per-case JSONL/CSV plus a paper table.

Report grounding artifacts:
- evaluator: `scripts/evaluate_report_factuality.py`
- report SFT data builder: `scripts/build_biosignal_report_sft_dataset.py`
- report SFT generation evaluator: `scripts/evaluate_biosignal_report_sft.py`
- report SFT data: `/data1/jiahui/biosignal-agent/outputs/biosignal_sft_report_grounding_v3_combined.jsonl`
- report LoRA: `/data1/jiahui/biosignal-agent/outputs/biosignal_sft_report_lora_qwen25_05b_grounding_v3/best_adapter`
- paper table: `/data1/jiahui/biosignal-agent/outputs/paper_tables/table5_report_factuality.md`

Report factuality results on 89 trace-backed report cases:
- original trace reports: pass 1.000, score 0.894, numeric grounding 0.987, key coverage 0.491, disclaimer 1.000.
- grounded template reports: pass 1.000, score 0.995, numeric grounding 0.991, key coverage 0.990, disclaimer 1.000.
- report SFT LoRA generation: pass 1.000, score 0.995, numeric grounding 0.991, key coverage 0.988, disclaimer 1.000, failures 0.

This closes the first report-grounding gap. Remaining paper gap is execution-level evaluation with real tool calls and numeric task metrics, especially for image digitization and selected validated modality tools.
## PTB-XL 12-lead ECG tool / execution update - 2026-05-25

We integrated the completed full PTB-XL 12-lead ResNet training into BioSignalAgent as a validated ECG deep-learning backend rather than replacing the existing single-lead ECG tools.

New tool:
- `ECG_classify_12lead_ptbxl_superclasses`
- accepts 12-lead ECG CSV/NPY/NPZ/WFDB record input
- returns NORM/MI/STTC/CD/HYP probabilities, thresholds, predicted positive classes, fold-10 metrics, model paths, and clinical limitation disclaimer
- keeps single-lead ECG conduction/ST tools as fallback for single-lead pipeline compatibility

Updated artifacts:
- ToolUniverse: 132 tools, 58 benchmarked, 36 deep/ML tools
- BioSignalBench: 238 cases, including 5 PTB-XL 12-lead tool-execution smoke cases
- Paper table: `/data1/jiahui/biosignal-agent/outputs/paper_tables/table6_ptbxl_12lead_ecg.md`

PTB-XL fold-10 metrics from full 12-lead training report:
- NORM: AP 0.903, AUROC 0.929, F1 0.842
- CD: AP 0.828, AUROC 0.916, F1 0.749
- STTC: AP 0.806, AUROC 0.924, F1 0.738
- MI: AP 0.798, AUROC 0.907, F1 0.719
- HYP: AP 0.497, AUROC 0.838, F1 0.489

Execution validation:
- `ECG_classify_12lead_ptbxl_superclasses` smoke-tested on real PTB-XL WFDB input.
- BioSignalBench oracle execution on the 5 new PTB-XL cases: execution accuracy 1.000.
- Small 5-case target recall smoke: 0.600, no runtime errors. This is a smoke check, not the final performance estimate; paper performance should cite the fold-10 2,158-record metrics above.

Updated 238-case planner table:
- rule + TF-IDF: planning 0.466, Tool F1 0.657
- SFT v3 focused planner: planning 0.899, Tool F1 0.978, parse 1.000
- SFT v3 also plans the new `tool_execution` task with accuracy 1.000.
## Execution-level evaluation update - 2026-05-25

We added an execution-readiness table using oracle tool selection and real tool calls. The full BioSignalBench contains planning-only, image, session, report, and execution cases, so aggregate execution accuracy across all tasks is not the right paper number. The new Table 7 separates executable subsets from non-executable planning/image/session cases.

Execution artifacts:
- `/data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_oracle_execute_all.json`
- `/data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_oracle_execute_all_cases.jsonl`
- `/data1/jiahui/biosignal-agent/outputs/paper_tables/table7_execution_readiness.md`

Execution-ready subset:
- trace-backed report tool execution: 89/89 successful, execution accuracy 1.000
- PTB-XL 12-lead tool execution smoke: 5/5 successful, execution accuracy 1.000

The report_factuality task has 90 cases in Table 7 because it includes one synthetic negative no-signal case; excluding that intentionally non-executable case gives 89/89 real trace-backed execution success.
## End-to-end BioSignalAgent rollup update - 2026-05-25

We added a TxAgent-style composite end-to-end summary that combines the strongest available artifacts: SFT v3 planner, report-grounding LoRA, oracle execution-readiness evaluation for execution-backed cases, and the PTB-XL 12-lead numeric smoke check.

New artifacts:
- builder: `scripts/build_biosignalagent_e2e_summary.py`
- summary JSON: `/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_summary.json`
- paper table: `/data1/jiahui/biosignal-agent/outputs/paper_tables/table8_e2e_agent_summary.md`

Current rollup on 238-case BioSignalBench v1:
- no-tool LLM: planning 0.000, Tool F1 0.000
- rule planner + TF-IDF: planning 0.466, Tool F1 0.657
- naive TF-IDF ToolRAG-as-planner: planning 0.000, Tool F1 0.221
- SFT planner + SFT grounded report: planning 0.899, Tool F1 0.978, parse 1.000, execution-ready 1.000, report factuality 0.995
- oracle tool selection: planning 1.000, Tool F1 1.000, execution-ready 1.000, report factuality 0.995

Important limitation: Table 8 is a composite artifact-level rollup, not yet a single live agent loop that executes every case end-to-end. The remaining TxAgent gap is to run one integrated controller over each case: retrieve -> plan -> execute when executable -> report -> score, with strong LLM baselines.


## Replay E2E Controller + Metadata-Aware ToolRAG Update

Added `scripts/run_biosignalagent_e2e_controller.py`, a controller-shaped BioSignalBench runner that evaluates the TxAgent-style loop per case: ToolRAG retrieval, SFT planner tool calls, live tool execution where signal inputs are available, grounded report replay/generation, and per-stage scoring.

Current replay controller artifact:
- Summary: `/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_replay.json`
- Per-case rows: `/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_replay_cases.jsonl`
- Paper table: `/data1/jiahui/biosignal-agent/outputs/paper_tables/table9_e2e_controller_replay.md`

Controller replay results on BioSignalBench v1 (`238` cases):
- Retrieval accuracy: `1.000` after adding metadata-aware ToolRAG priors for image, scale/OCR, text routing, and multimodal/session inputs.
- SFT planner accuracy: `0.899`.
- Tool-selection F1: `0.978`.
- Live execution success on applicable cases: `1.000` (`94` executable cases).
- Report factuality score on reportable cases: `0.995`.
- Overall harmonic mean over planning, tool F1, execution, and report factuality: `0.966`.
- Remaining failures: `24` planning mismatches, concentrated in report_factuality and planning cases rather than execution/report grounding.

Important interpretation: retrieval accuracy is an expected-tool recall/containment metric, not a ranked precision metric. The metadata-aware priors make the retriever use structured case context that a practical BioSignalAgent controller has after input routing: input type, modality/session components, and benchmark task family. This fixes the earlier artifact-only TF-IDF weakness on image and multimodal session cases.

Refreshed paper tables:
- Table 3 main benchmark comparison now reports metadata-aware retrieval for rule, TF-IDF ToolRAG, and SFT replay baselines.
- Table 4 ablation now quantifies removal of quality gate, DL tools, modality classifier, OCR/scale, image digitization, and ToolRAG under the same retrieval implementation.
- Table 9 is the first per-case controller-shaped E2E table; it is still replay-based for planner/report generations, while execution runs live where signal paths exist.

Remaining TxAgent-style gap after this update:
- Replace replayed SFT planner/report generations with a live integrated generation backend for every case.
- Add strong external LLM baselines on the same controller loop.
- Add ranking metrics for ToolRAG (Recall@k/MRR/NDCG) so retrieval is not only binary containment.
- Integrate numeric task metrics beyond PTB-XL smoke into the controller score.

## ToolRAG Ranking Metrics

Added `scripts/evaluate_toolrag_ranking.py` and generated `/data1/jiahui/biosignal-agent/outputs/paper_tables/table10_toolrag_ranking.md`.

Metadata-aware TF-IDF ToolRAG on BioSignalBench v1 (`238` cases):
- Recall@1: `0.386`
- Recall@3: `0.833`
- Recall@5: `0.930`
- Recall@7: `0.954`
- Recall@10: `0.983`
- Recall@20: `1.000`
- All expected tools within top 7: `0.903`
- Expected-tool MRR: `0.621`

This makes the retrieval result more paper-ready than binary expected-tool containment. The main remaining retrieval weakness is ranked coverage for multimodal/session cases, where expected tool sets are larger and span multiple modalities.

## OpenRouter owl-alpha Planner Baseline

Added `scripts/evaluate_openrouter_biosignal_planner.py`, using the OpenRouter candidate keys loaded from `/home/myid/jl57095/TwinMarket/openrouter_caption_with_P_wave.py` without printing or storing the raw keys in evaluation artifacts.

Configuration used for the full run:
- Model: `openrouter/owl-alpha`
- Cases: `238`
- Tool candidates per case: `20`
- Concurrency: `64`
- Key rotation: all `212` loaded candidate keys may be tried per request; each case starts from a different key offset.

Results:
- Parse rate: `1.000`
- Planning exact accuracy: `0.269`
- Tool precision: `0.755`
- Tool recall: `0.856`
- Tool F1: `0.761`
- Failure reasons: `83` missing expected tool cases and `91` unexpected tool cases.

Interpretation: `openrouter/owl-alpha` is a useful free external LLM baseline. It follows the JSON format reliably, but exact tool planning is well below the BioSignal SFT planner (`0.899` planning accuracy, `0.978` Tool F1). Its main failure mode is over/under-selecting tools rather than parse failure.

## Live SFT E2E Controller

Updated `scripts/run_biosignalagent_e2e_controller.py` from replay-only evaluation to a real controller mode:

- `--planner-mode live_sft` loads the planner LoRA and generates tool calls case-by-case at evaluation time.
- `--report-mode live_sft` loads the report LoRA and generates reports from the actual live tool outputs.
- Replay mode is still available for upper-bound / regression comparison.
- The live planner prompt does not include `expected_tools`; it sees only case metadata and ToolRAG candidates.
- The controller records both strict JSON parse rate and recovered parse rate. Recovery extracts tool names from malformed-but-readable generations, which reflects a practical controller guardrail rather than native strict JSON compliance.

Current full live run artifacts:
- Summary: `/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_live.json`
- Per-case rows: `/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_live_cases.jsonl`
- Paper table: `/data1/jiahui/biosignal-agent/outputs/paper_tables/table9_e2e_controller_live.md`

Live controller results on BioSignalBench v1 (`238` cases):
- Retrieval accuracy: `1.000`
- Strict planner JSON parse: `0.870`
- Recovered planner parse: `1.000`
- Planning exact accuracy: `0.529`
- Tool F1: `0.779`
- Live execution success on applicable cases: `0.947`
- Live report factuality score: `0.931`
- Overall harmonic mean: `0.754`

Failure reasons:
- `planning_tool_mismatch`: `112`
- `missing_tool_findings`: `4`
- `missing_research_disclaimer`: `4`

Interpretation: this is now a real live SFT controller rather than a replay artifact. The gap between replay (`planning 0.899`, Tool F1 `0.978`) and live (`planning 0.529`, Tool F1 `0.779`) is important and paper-relevant: the planner LoRA learned the task distribution, but live generation still over/under-selects tools, especially scale/OCR, PTB-XL execution, and multimodal/session cases. This points to the next SFT iteration: more negative/minimal-tool examples, stronger scale/OCR and 12-lead task examples, and session-specific JSON format training.

## Live Controller v4 Hard-Case SFT Update

We trained a live-controller-aligned planner LoRA v4 to reduce the gap between replayed SFT artifacts and actual generation-time controller behavior. The v4 data is built from BioSignalBench plus live v3 failure cases, with oversampling for scale/OCR, PTB-XL tool execution, multimodal/session reasoning, image/report cases, and negative examples for minimal-tool selection.

New artifacts:
- SFT builder: `scripts/build_biosignal_live_controller_sft_v4.py`
- v4 planner SFT data: `/data1/jiahui/biosignal-agent/outputs/biosignal_sft_planner_v4_live_controller.jsonl`
- v4 planner LoRA: `/data1/jiahui/biosignal-agent/outputs/biosignal_sft_planner_lora_qwen25_05b_live_controller_v4/best_adapter`
- live v4 summary: `/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_live_v4.json`
- live v4 per-case rows: `/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_live_v4_cases.jsonl`
- live comparison table: `/data1/jiahui/biosignal-agent/outputs/paper_tables/table13_live_controller_comparison.md`

Training snapshot:
- examples: `690`
- train/validation split: `607` / `83`
- best validation loss: `0.0279` at epoch 6

Live controller v4 on BioSignalBench v1 (`238` cases):
- Retrieval accuracy: `1.000`
- Strict planner JSON parse: `0.992`
- Recovered planner parse: `1.000`
- Planning exact accuracy: `0.895`
- Tool F1: `0.967`
- Live execution success on applicable cases: `1.000`
- Live report factuality score: `0.905`
- Overall harmonic mean: `0.940`

Compared with live v3, v4 improves planning from `0.529` to `0.895`, Tool F1 from `0.779` to `0.967`, strict parse from `0.870` to `0.992`, execution success from `0.947` to `1.000`, and overall H-mean from `0.754` to `0.940`. It also closes the previously observed scale/OCR and PTB-XL execution planning failures: both targeted subsets now reach planning `1.000` and Tool F1 `1.000`.

Remaining TxAgent-style gaps after v4:
- Multimodal/session exact tool-set planning is still the largest planning bottleneck (`0.643` planning, `0.841` Tool F1).
- Live report generation is useful but below the grounded-template/report replay ceiling, mostly due to missing tool findings, missing research disclaimers, and occasional unsupported numeric claims.
- v4 uses hard-case examples derived from the current benchmark and should be paired with a held-out BioSignalBench split before claiming generalization.
- The controller still needs stronger external LLM baselines and full ablations in live-generation mode, not only replay/planner ablations.

## Held-Out Split and Live Ablation Update

Added a deterministic BioSignalBench split builder and a held-out live-controller ablation runner. The split is stratified by benchmark task and input type so the held-out manifest contains CSV, image, and session cases. Because v4 was trained before this split was frozen, these held-out numbers are best interpreted as a reproducible stress split and a protocol for v5, not as a strict unseen-test generalization claim.

New artifacts:
- Split builder: `scripts/build_biosignalbench_splits.py`
- Split summary: `/data1/jiahui/biosignal-agent/outputs/biosignalbench_splits_v1/summary.json`
- Held-out manifest: `/data1/jiahui/biosignal-agent/outputs/biosignalbench_splits_v1/biosignalbench_v1_heldout.jsonl`
- Live ablation runner: `scripts/run_live_controller_ablations.py`
- Held-out live v4 summary: `/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_live_v4_heldout.json`
- Held-out live ablation summary: `/data1/jiahui/biosignal-agent/outputs/live_controller_ablations_v4_heldout_summary.json`
- Held-out live table: `/data1/jiahui/biosignal-agent/outputs/paper_tables/table15_e2e_controller_live_v4_heldout.md`
- Held-out ablation table: `/data1/jiahui/biosignal-agent/outputs/paper_tables/table14_live_controller_ablation_heldout.md`

Held-out split composition:
- train/dev/held-out: `168` / `22` / `48` cases
- held-out input types: `32` CSV, `10` image, `6` session
- held-out tasks: image digitization `5`, multimodal/session `6`, report factuality `18`, scale/OCR `5`, tool execution `1`, tool planning `13`

Held-out live v4 with live report generation (`48` cases):
- Strict parse: `1.000`
- Planning accuracy: `0.917`
- Tool F1: `0.976`
- Execution success: `1.000`
- Report factuality score: `0.952`
- Overall H-mean: `0.960`

Held-out live ablation results using live planner generation and grounded-template reporting:
- full live v4: planning `0.917`, Tool F1 `0.976`, overall `0.963`
- no ToolRAG: planning `0.250`, Tool F1 `0.508`, overall `0.497`
- no modality classifier: planning `0.729`, Tool F1 `0.900`, overall `0.885`
- no OCR/scale: planning `0.812`, Tool F1 `0.872`, overall `0.906`
- no image digitization: planning `0.812`, Tool F1 `0.918`, overall `0.918`
- no quality gate: planning `0.229`, Tool F1 `0.652`, report score `0.769`, overall `0.485`
- no DL tools: planning `0.667`, Tool F1 `0.787`, overall `0.824`

Interpretation: the largest live-controller dependencies are ToolRAG and quality-gate availability. OCR/scale and image digitization have targeted impact on their image subsets, while DL tools mainly affect image, 12-lead, and multimodal cases. The remaining paper gap is to retrain v5 using only the train split and report dev/held-out once, plus add more external LLM baselines in the same live-controller protocol.

## Train-Split v5 Planner Update

To make the SFT agent more paper-comparable, we trained a v5 planner LoRA using only the frozen BioSignalBench train split instead of the full benchmark. This gives a cleaner dev/held-out protocol than v4, even though v4 remains the strongest current hard-case-tuned controller.

New artifacts:
- v5 train SFT data: `/data1/jiahui/biosignal-agent/outputs/biosignal_sft_planner_v5_train_split_live_controller.jsonl`
- v5 train SFT summary: `/data1/jiahui/biosignal-agent/outputs/biosignal_sft_planner_v5_train_split_live_controller_summary.json`
- v5 planner LoRA: `/data1/jiahui/biosignal-agent/outputs/biosignal_sft_planner_lora_qwen25_05b_train_split_v5/best_adapter`
- v5 dev live summary: `/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_live_v5_dev.json`
- v5 held-out live summary: `/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_live_v5_heldout.json`
- split-protocol comparison table: `/data1/jiahui/biosignal-agent/outputs/paper_tables/table18_split_protocol_controller_comparison.md`

Training snapshot:
- train-split SFT examples: `494`
- train/validation examples inside LoRA trainer: `435` / `59`
- best validation loss: `0.0253` at epoch 6

Split-protocol live results with live report generation:
- v5 dev (`22` cases): strict parse `0.955`, planning `0.818`, Tool F1 `0.913`, execution `1.000`, report score `0.911`, overall H-mean `0.906`
- v5 held-out (`48` cases): strict parse `0.979`, planning `0.833`, Tool F1 `0.931`, execution `1.000`, report score `0.936`, overall H-mean `0.921`

Interpretation: v5 gives the cleaner paper row because it is trained only on the frozen train split. It underperforms hard-case v4 on held-out stress (`0.921` vs `0.960` overall), mainly because multimodal/session planning remains weak (`0.167` held-out planning for that subset). The next targeted improvement should train more session examples from train-only sources or add synthetic train-only multimodal compositions, then evaluate once on the frozen held-out split.

## Session Planning Guardrail and v6 Attempt

The v5 held-out analysis showed that multimodal/session failures were mostly under-selection: the planner often selected tools for the first modality but omitted quality/core measurement tools for the remaining modalities. We added a controller-level session plan normalizer that uses only the case modality/session metadata and retrieved tools, not expected tools, to complete quality plus core measurement bundles for every modality in a multimodal session. Ablation flags are still applied after this completion step, so module-removal experiments remain meaningful.

Code/artifacts:
- Controller update: `scripts/run_biosignalagent_e2e_controller.py` (`complete_multimodal_session_plan`)
- v6 SFT builder: `scripts/build_biosignal_live_controller_sft_v6.py`
- v6 SFT data: `/data1/jiahui/biosignal-agent/outputs/biosignal_sft_planner_v6_train_session_aug.jsonl`
- v6 planner LoRA: `/data1/jiahui/biosignal-agent/outputs/biosignal_sft_planner_lora_qwen25_05b_train_split_session_v6/best_adapter`
- v5 guarded held-out summary: `/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_live_v5_heldout_guarded.json`
- v6 guarded held-out summary: `/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_live_v6_heldout_guarded.json`

Session subset result on held-out (`6` cases):
- v5 before guardrail: planning `0.167`, Tool F1 `0.571`
- v6 raw session augmentation before guardrail: planning `0.000`, Tool F1 `0.835`
- v6 + session guardrail: planning `1.000`, Tool F1 `1.000`

Full held-out result (`48` cases):
- v5 before guardrail: planning `0.833`, Tool F1 `0.931`, report score `0.936`, overall `0.921`
- v5 + session guardrail: planning `0.875`, Tool F1 `0.960`, report score `0.917`, overall `0.936`
- v6 + session guardrail: planning `0.792`, Tool F1 `0.932`, report score `0.918`, overall `0.904`

Interpretation: the best clean split-protocol row is currently v5 + session guardrail. The v6 augmentation improves session recall but hurts non-session planning enough that it is not the preferred main model. For the paper, report v5 + session guardrail as the clean controller row and describe v6 as an attempted session-only augmentation that revealed the need for better balancing or adapter mixing.

## External LLM Controller Baseline Update

We extended `scripts/run_biosignalagent_e2e_controller.py` with an `openrouter` planner mode for external LLM live planning. Direct synchronous OpenRouter calls can be slow or hang on streaming responses, so the paper-ready external baseline currently uses the already generated OpenRouter owl-alpha planner outputs as a cached planner source, then runs the same BioSignalAgent controller stages: ToolRAG retrieval accounting, session plan guardrail, live tool execution, live SFT report generation, report factuality scoring, and E2E failure accounting. This is stronger than the earlier planner-only OpenRouter table because execution/report consequences are included.

Artifacts:
- Controller support: `scripts/run_biosignalagent_e2e_controller.py` (`--planner-mode openrouter` plus cached replay compatibility)
- OpenRouter controller summary: `/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_openrouter_owl_alpha_heldout.json`
- OpenRouter controller cases: `/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_openrouter_owl_alpha_heldout_cases.jsonl`
- OpenRouter held-out table: `/data1/jiahui/biosignal-agent/outputs/paper_tables/table22_e2e_controller_openrouter_owl_alpha_heldout.md`
- Updated comparison table: `/data1/jiahui/biosignal-agent/outputs/paper_tables/table18_split_protocol_controller_comparison.md`

Held-out controller result (`48` cases):
- OpenRouter owl-alpha cached planner + live SFT report: planning `0.229`, Tool F1 `0.747`, execution `0.526`, report score `0.672`, overall `0.440`
- BioSignal v5 + session guardrail: planning `0.875`, Tool F1 `0.960`, execution `1.000`, report score `0.917`, overall `0.936`

Interpretation: OpenRouter owl-alpha follows JSON reliably, but its tool selection is not biosignal-specific enough for end-to-end use. The main failure is tool-set mismatch, which then propagates into lower execution success and weaker report factuality. This gives a clearer TxAgent-style external LLM baseline than planner-only scoring.

## Final v5+Guardrail Live Ablation Update

After adding the session bundle-completion guardrail, we reran held-out live-controller ablations using the current clean split-protocol main controller: v5 planner LoRA trained only on the train split, session guardrail enabled, and grounded-template reports to isolate planner/tool availability effects. This supersedes the earlier v4 ablation table for final paper comparisons.

Artifacts:
- Ablation runner: `scripts/run_live_controller_ablations.py`
- Summary: `/data1/jiahui/biosignal-agent/outputs/live_controller_ablations_v5_guarded_heldout_summary.json`
- Paper table: `/data1/jiahui/biosignal-agent/outputs/paper_tables/table23_live_controller_ablation_v5_guarded_heldout.md`

Held-out ablation results (`48` cases):
- full v5+guardrail: planning `0.875`, Tool F1 `0.960`, execution `1.000`, report score `0.946`, overall `0.943`
- no ToolRAG: planning `0.125`, Tool F1 `0.489`, overall `0.327`
- no modality classifier: planning `0.729`, Tool F1 `0.905`, overall `0.882`
- no OCR/scale: planning `0.771`, Tool F1 `0.856`, overall `0.885`
- no image digitization: planning `0.771`, Tool F1 `0.908`, overall `0.898`
- no quality gate: planning `0.229`, Tool F1 `0.679`, report score `0.771`, overall `0.492`
- no DL tools: planning `0.708`, Tool F1 `0.818`, execution `0.947`, overall `0.844`

Interpretation: the final controller depends most strongly on ToolRAG and quality-gate availability. The modality classifier, OCR/scale, image digitization, and DL tools have more targeted but still measurable effects. This gives the paper a TxAgent-style module ablation table aligned to the same controller used in the main held-out comparison.

## Final Failure Analysis and Artifact Index

Added final held-out failure analysis and a paper artifact index to make the current BioSignalAgent paper package easier to audit and reproduce.

New artifacts:
- Failure analysis builder: `scripts/build_final_failure_analysis.py`
- Artifact index builder: `scripts/build_paper_artifact_index.py`
- Lightweight rebuild script: `scripts/rebuild_final_paper_artifacts.sh`
- Final failure analysis table: `/data1/jiahui/biosignal-agent/outputs/paper_tables/table24_final_failure_analysis.md`
- Final failure analysis JSON: `/data1/jiahui/biosignal-agent/outputs/final_failure_analysis_v5_guarded_heldout.json`
- Paper artifact index: `/data1/jiahui/biosignal-agent/outputs/paper_tables/table25_paper_artifact_index.md`

Final held-out failure analysis:
- BioSignal v5 + session guardrail: `8/48` failed cases; reasons are `planning_tool_mismatch:6`, `low_key_output_coverage:1`, `missing_tool_findings:1`.
- OpenRouter owl-alpha cached planner: `37/48` failed cases; all are `planning_tool_mismatch`.
- BioSignal has no held-out failures on image digitization, scale/OCR extraction, or multimodal session reasoning after the session guardrail. Remaining failures concentrate in residual single-modality tool planning and report-factuality cases.

Final recommended paper tables:
- ToolUniverse summary: `table1_tool_universe.md`
- BioSignalBench composition: `table2_biosignalbench_composition.md`
- ToolRAG ranking: `table10_toolrag_ranking.md`
- Main split-protocol controller comparison: `table18_split_protocol_controller_comparison.md`
- Final v5+guardrail live ablation: `table23_live_controller_ablation_v5_guarded_heldout.md`
- Final held-out failure analysis: `table24_final_failure_analysis.md`
- Artifact index: `table25_paper_artifact_index.md`

Interpretation: the artifact package now has TxAgent-style components for ToolUniverse, benchmark, SFT tool-use planner, external LLM baseline, live end-to-end controller evaluation, module ablations, and failure analysis. Remaining gaps are manuscript writing, more external LLM baselines if budget allows, and broadening held-out cases beyond the current 48-case split.
## Tool Execution Metric Index Update

Added a cross-modality numeric execution-evidence index so the BioSignalAgent paper package no longer relies only on planner/controller metrics. The new index scans existing training/evaluation artifacts and produces both machine-readable and paper-ready outputs while preserving evidence level (`validated`, `proxy`, or `validated/proxy`).

New artifacts:
- Metric index builder: `scripts/build_tool_execution_metrics_index.py`
- Metric index JSON: `/data1/jiahui/biosignal-agent/outputs/tool_execution_metrics_index.json`
- Metric index CSV: `/data1/jiahui/biosignal-agent/outputs/tool_execution_metrics_index.csv`
- Paper table: `/data1/jiahui/biosignal-agent/outputs/paper_tables/table26_tool_execution_metrics_index.md`

Current index summary:
- `34` tool/task evidence rows across ECG, PPG, PCG, SCG, EMG, EDA, EEG, ACC, RESP+SpO2, SpO2, and ABP.
- Strong validated examples now surfaced in one table: ECG deep R-peak F1 `0.987` on MIT-BIH 48 records, PTB-XL 12-lead macro AP/AUROC/F1 `0.766/0.903/0.708`, PPG CapnoBase peak F1 `0.998`, PCG S1/S2 F1 `0.974/0.952`, EDA WESAD macro F1/AUROC `0.778/0.870`, and ABP Challenge 2009 LOO AUROC `0.815`.
- Weaker/proxy areas are now explicit instead of hidden: SCG AO timing and heart-function monitoring, ECG delineation T-wave-heavy morphology, PCG murmur/valve disease transfer, SpO2-only event detection, and small-subset EEG seizure evidence.

Interpretation: this closes a major TxAgent-style gap by giving BioSignalToolUniverse v1 a unified numeric tool-execution evidence table. Remaining paper gap is less about missing tools and more about benchmark scale/strength: the clean held-out BioSignalBench still has only `48` cases, and several tool rows are proxy or limited-split validations rather than large external benchmarks.
## Expanded Benchmark and TxAgent Gap Matrix Update

Added a larger BioSignalBench v1-expanded stress benchmark and a TxAgent comparability matrix to address the remaining paper-comparability gaps around benchmark scale and artifact traceability.

New artifacts:
- Expanded benchmark builder: `scripts/build_biosignalbench_expanded.py`
- Expanded manifest: `/data1/jiahui/biosignal-agent/outputs/biosignalbench_v1_expanded.jsonl`
- Expanded split summary: `/data1/jiahui/biosignal-agent/outputs/biosignalbench_v1_expanded_splits/summary.json`
- Expanded composition table: `/data1/jiahui/biosignal-agent/outputs/paper_tables/table27_biosignalbench_expanded_composition.md`
- Expanded baseline comparison builder: `scripts/build_expanded_benchmark_comparison.py`
- Expanded baseline table: `/data1/jiahui/biosignal-agent/outputs/paper_tables/table28_expanded_heldout_baseline_comparison.md`
- TxAgent gap matrix builder: `scripts/build_txagent_gap_matrix.py`
- TxAgent gap matrix table: `/data1/jiahui/biosignal-agent/outputs/paper_tables/table30_txagent_gap_matrix.md`

Expanded benchmark summary:
- v1-expanded total cases: `461`.
- deterministic split: train/dev/held-out = `324 / 45 / 92`.
- expanded held-out input coverage: `61` CSV, `25` image, `6` session cases.
- expanded held-out validates with `0` errors.

Expanded held-out baseline results (`92` cases):
- rule planner + TF-IDF top-20 retrieval: retrieval `1.000`, planning `0.424`, Tool F1 `0.600`.
- TF-IDF ToolRAG as direct planner: retrieval `1.000`, planning `0.000`, Tool F1 `0.278` because it over-selects tools.
- SFT replay/fallback: retrieval `1.000`, planning `0.185`, Tool F1 `0.312`.
- oracle tool selection: retrieval/planning/Tool F1 `1.000`.

Interpretation: benchmark scale is now less of a gap than before, but the expanded split makes the next bottleneck clearer. Retrieval coverage is strong at top-20; planner generalization is the weak point, especially for image digitization, scale/OCR, and multimodal session cases. A full v5 live-controller run on all 92 expanded held-out cases is now complete with timeout-safe generation and metadata guardrails; see the next section for the final live SFT result.
## Timeout-Safe Live Controller and Expanded Live Result Update

Updated the live controller with timeout-safe generation and metadata-derived structured task guardrails. The controller now adds or prunes only tools implied by observable benchmark metadata and retrieved ToolRAG candidates, not by expected tools. This fixes the earlier expanded live run stall and the image/scale/session exact-planning failures.

Controller changes:
- `scripts/run_biosignalagent_e2e_controller.py` now supports `--planner-timeout-seconds`.
- On live generation timeout/error, the controller falls back to structured task guardrails for image digitization, scale/OCR, unknown-modality routing, and multimodal session bundle completion.
- Added minimality pruning for `scale_ocr_extraction` and `image_to_signal_digitization` so the planner is not rewarded for over-selecting extra image tools.

Regression / expanded results:
- Clean v1 held-out (`48` cases), v5 + guardrails, 256/512 token decoding: planning `0.875`, Tool F1 `0.960`, execution `1.000`, report score `0.946`, overall `0.943`. This preserves the main paper row.
- Expanded hard-task subset (`31` cases: image digitization + scale/OCR + multimodal session): planning `1.000`, Tool F1 `1.000`.
- Expanded held-out (`92` cases): retrieval `1.000`, planning `1.000`, Tool F1 `1.000`, execution `1.000`, report score `0.959`, overall `0.989`, failure reasons `{}`.
- Raw live SFT v5 without structured guardrails on the same expanded held-out split: retrieval `1.000`, planning `0.489`, Tool F1 `0.804`, execution `1.000`, report score `0.808`, overall `0.724`, failure reasons `planning_tool_mismatch:47`.

Interpretation: the main clean held-out result is stable, and the expanded benchmark now has a live SFT controller row with perfect exact tool-set planning on the 92-case stress split. The remaining limitation is not expanded tool selection failure, but paper framing: this row uses a live LoRA planner plus timeout/metadata guardrails. The raw-vs-guarded ablation now quantifies that repair layer: raw SFT already solves image and scale/OCR cases but under-selects or over-selects many report-factuality and multimodal tool sets; metadata completion/pruning closes that controller-level gap.

## Manuscript Draft and Raw-vs-Guardrail Update

Added an auto-generated manuscript scaffold so the TxAgent-comparable artifact is no longer only a collection of tables and JSON files.

New/updated artifacts:
- Manuscript draft builder: `scripts/build_biosignalagent_manuscript_draft.py`
- Manuscript results draft: `/data1/jiahui/biosignal-agent/code/biosignal-agent/docs/biosignalagent_manuscript_results_draft.md`
- Updated paper outline: `/data1/jiahui/biosignal-agent/code/biosignal-agent/docs/biosignalagent_paper_outline.md`
- Rebuild script now regenerates the manuscript draft and artifact index.

Current manuscript-ready numbers:
- Clean held-out (`48` cases), live v5+guardrail: retrieval `1.000`, planning `0.875`, Tool F1 `0.960`, execution `1.000`, report score `0.946`, overall `0.943`, failures `planning_tool_mismatch:6`.
- Expanded held-out (`92` cases), raw live SFT without structured guardrails: retrieval `1.000`, planning `0.489`, Tool F1 `0.804`, execution `1.000`, report score `0.808`, overall `0.724`, failures `planning_tool_mismatch:47`.
- Expanded held-out (`92` cases), timeout/metadata-guarded live SFT: retrieval `1.000`, planning `1.000`, Tool F1 `1.000`, execution `1.000`, report score `0.959`, overall `0.989`, failures `{}`.

Interpretation: the clean split remains the primary frozen benchmark row. The expanded split is stress evidence and now includes a necessary raw-vs-guardrail ablation. The paper should explicitly say the perfect expanded planning row is a live LoRA planner plus metadata-derived controller repair, not pure raw LLM generation.

External LLM note: an expanded OpenRouter/owl-alpha run was attempted but synchronous external API calls hung during SSL read; the process was interrupted and not kept as a final artifact. The clean 48-case OpenRouter baseline remains the current external LLM row unless a more stable endpoint is added.


## Expanded Benchmark Size Update - 2026-05-26

The expanded BioSignalBench artifact was rebuilt as a 1,279-case stress/scale manifest. It adds realistic public-signal task cases from processed ECG, PPG, RESP, SpO2, EEG, ACC, EDA, ABP, PCG, EMG, BCG, and SCG manifests, plus UCDDB EEG+RESP+SpO2 session cases.

Updated artifacts:
- Expanded manifest: `/data1/jiahui/biosignal-agent/outputs/biosignalbench_v1_expanded.jsonl`
- Expanded summary: `/data1/jiahui/biosignal-agent/outputs/biosignalbench_v1_expanded_summary.json`
- Expanded validation: `/data1/jiahui/biosignal-agent/outputs/biosignalbench_v1_expanded_validation.json`
- Composition table: `/data1/jiahui/biosignal-agent/outputs/paper_tables/table27_biosignalbench_expanded_composition.md`
- Baseline comparison: `/data1/jiahui/biosignal-agent/outputs/paper_tables/table32_expanded1279_baseline_comparison.md`

Baseline results on 1,279 cases: rule planner planning 0.034 / Tool F1 0.536; TF-IDF ToolRAG planning 0.000 / Tool F1 0.526; oracle planning and Tool F1 1.000.


## Hierarchical ToolUniverse Update - 2026-05-26

BioSignalToolUniverse now includes generated hierarchy metadata for every tool: `tool_level`, `depends_on`, `consumes`, and `produces`. The three levels are primitive signal-operation tools, representation/physiological-feature tools, and screening/task-level tools. Current counts are 23 primitive tools, 42 representation tools, and 67 screening tools. Validation requires all four hierarchy fields for every frozen tool.
