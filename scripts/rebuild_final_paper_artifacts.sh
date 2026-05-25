#!/usr/bin/env bash
set -euo pipefail

# Rebuild lightweight final BioSignalAgent paper artifacts from existing outputs.
# This does not retrain models or rerun expensive live-controller evaluations.

python scripts/build_split_protocol_comparison.py
python scripts/build_final_failure_analysis.py
python scripts/build_tool_execution_metrics_index.py

python scripts/build_biosignalbench_expanded.py
python scripts/build_biosignalbench_splits.py --manifest /data1/jiahui/biosignal-agent/outputs/biosignalbench_v1_expanded.jsonl --out-dir /data1/jiahui/biosignal-agent/outputs/biosignalbench_v1_expanded_splits --seed biosignalbench-v1-expanded-split-2026-05-25
python scripts/validate_biosignalbench.py --manifest /data1/jiahui/biosignal-agent/outputs/biosignalbench_v1_expanded_splits/biosignalbench_v1_heldout.jsonl --out-json /data1/jiahui/biosignal-agent/outputs/biosignalbench_v1_expanded_splits/heldout_validation.json
python scripts/evaluate_biosignalbench.py --manifest /data1/jiahui/biosignal-agent/outputs/biosignalbench_v1_expanded_splits/biosignalbench_v1_heldout.jsonl --planner-backend rule --retriever-backend tfidf --retrieved-tool-count 20 --out-json /data1/jiahui/biosignal-agent/outputs/biosignalbench_expanded_heldout_eval_rule.json --out-jsonl /data1/jiahui/biosignal-agent/outputs/biosignalbench_expanded_heldout_eval_rule_cases.jsonl --out-csv /data1/jiahui/biosignal-agent/outputs/biosignalbench_expanded_heldout_eval_rule_cases.csv --out-md /data1/jiahui/biosignal-agent/outputs/paper_tables/table28a_expanded_heldout_eval_rule.md
python scripts/evaluate_biosignalbench.py --manifest /data1/jiahui/biosignal-agent/outputs/biosignalbench_v1_expanded_splits/biosignalbench_v1_heldout.jsonl --planner-backend toolrag --retriever-backend tfidf --retrieved-tool-count 20 --out-json /data1/jiahui/biosignal-agent/outputs/biosignalbench_expanded_heldout_eval_toolrag.json --out-jsonl /data1/jiahui/biosignal-agent/outputs/biosignalbench_expanded_heldout_eval_toolrag_cases.jsonl --out-csv /data1/jiahui/biosignal-agent/outputs/biosignalbench_expanded_heldout_eval_toolrag_cases.csv --out-md /data1/jiahui/biosignal-agent/outputs/paper_tables/table28b_expanded_heldout_eval_toolrag.md
python scripts/evaluate_biosignalbench.py --manifest /data1/jiahui/biosignal-agent/outputs/biosignalbench_v1_expanded_splits/biosignalbench_v1_heldout.jsonl --planner-backend sft_replay --retriever-backend tfidf --retrieved-tool-count 20 --out-json /data1/jiahui/biosignal-agent/outputs/biosignalbench_expanded_heldout_eval_sft_replay.json --out-jsonl /data1/jiahui/biosignal-agent/outputs/biosignalbench_expanded_heldout_eval_sft_replay_cases.jsonl --out-csv /data1/jiahui/biosignal-agent/outputs/biosignalbench_expanded_heldout_eval_sft_replay_cases.csv --out-md /data1/jiahui/biosignal-agent/outputs/paper_tables/table28c_expanded_heldout_eval_sft_replay.md
python scripts/evaluate_biosignalbench.py --manifest /data1/jiahui/biosignal-agent/outputs/biosignalbench_v1_expanded_splits/biosignalbench_v1_heldout.jsonl --planner-backend oracle --retriever-backend oracle --retrieved-tool-count 20 --out-json /data1/jiahui/biosignal-agent/outputs/biosignalbench_expanded_heldout_eval_oracle.json --out-jsonl /data1/jiahui/biosignal-agent/outputs/biosignalbench_expanded_heldout_eval_oracle_cases.jsonl --out-csv /data1/jiahui/biosignal-agent/outputs/biosignalbench_expanded_heldout_eval_oracle_cases.csv --out-md /data1/jiahui/biosignal-agent/outputs/paper_tables/table28d_expanded_heldout_eval_oracle.md
python scripts/build_expanded_benchmark_comparison.py
python scripts/build_txagent_gap_matrix.py
python scripts/build_biosignalagent_manuscript_draft.py
python scripts/build_paper_artifact_index.py
python scripts/validate_biosignalbench.py --manifest /data1/jiahui/biosignal-agent/outputs/biosignalbench_splits_v1/biosignalbench_v1_heldout.jsonl --out-json /data1/jiahui/biosignal-agent/outputs/biosignalbench_splits_v1/heldout_validation.json
python -m py_compile \
  scripts/run_biosignalagent_e2e_controller.py \
  scripts/run_live_controller_ablations.py \
  scripts/build_split_protocol_comparison.py \
  scripts/build_final_failure_analysis.py \
  scripts/build_tool_execution_metrics_index.py \
  scripts/build_paper_artifact_index.py \
  scripts/build_biosignalagent_manuscript_draft.py

echo "Final paper artifacts rebuilt:"
echo "  /data1/jiahui/biosignal-agent/outputs/paper_tables/table18_split_protocol_controller_comparison.md"
echo "  /data1/jiahui/biosignal-agent/outputs/paper_tables/table23_live_controller_ablation_v5_guarded_heldout.md"
echo "  /data1/jiahui/biosignal-agent/outputs/paper_tables/table24_final_failure_analysis.md"
echo "  /data1/jiahui/biosignal-agent/outputs/paper_tables/table25_paper_artifact_index.md"
echo "  /data1/jiahui/biosignal-agent/outputs/paper_tables/table26_tool_execution_metrics_index.md"
echo "  /data1/jiahui/biosignal-agent/outputs/paper_tables/table27_biosignalbench_expanded_composition.md"
echo "  /data1/jiahui/biosignal-agent/outputs/paper_tables/table28_expanded_heldout_baseline_comparison.md"
echo "  /data1/jiahui/biosignal-agent/outputs/paper_tables/table30_txagent_gap_matrix.md"
echo "  /data1/jiahui/biosignal-agent/code/biosignal-agent/docs/biosignalagent_manuscript_results_draft.md"
