#!/usr/bin/env python3
"""Build final paper artifact index for BioSignalAgent."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

CORE_TABLES = [
    ('ToolUniverse summary', 'table1_tool_universe.md', 'Frozen BioSignalToolUniverse v1 composition.'),
    ('BioSignalBench composition', 'table2_biosignalbench_composition.md', 'Benchmark case/task/input composition.'),
    ('ToolRAG ranking', 'table10_toolrag_ranking.md', 'Recall@k / MRR style retrieval metrics.'),
    ('OpenRouter planner-only baseline', 'table11_openrouter_owl_alpha_planner.md', 'External LLM planner-only reference.'),
    ('Live controller comparison', 'table18_split_protocol_controller_comparison.md', 'Main held-out controller comparison including v5+guardrail and OpenRouter.'),
    ('Final live ablation', 'table23_live_controller_ablation_v5_guarded_heldout.md', 'Final module ablation aligned to v5+guardrail.'),
    ('Final failure analysis', 'table24_final_failure_analysis.md', 'Held-out failure analysis for BioSignal and OpenRouter.'),
    ('PTB-XL ECG tool', 'table6_ptbxl_12lead_ecg.md', 'Validated 12-lead ECG execution/numeric tool evidence.'),
    ('Tool execution metric index', 'table26_tool_execution_metrics_index.md', 'Best available numeric evidence across validated/proxy signal tools.'),
    ('Expanded BioSignalBench composition', 'table27_biosignalbench_expanded_composition.md', 'Larger v1-expanded scale/stress benchmark composition.'),
    ('Expanded held-out baselines', 'table28_expanded_heldout_baseline_comparison.md', 'Rule/ToolRAG/SFT replay/oracle baselines on 92-case expanded held-out split.'),
    ('TxAgent comparability gap matrix', 'table30_txagent_gap_matrix.md', 'Status matrix for remaining TxAgent-style paper gaps and evidence.'),
    ('Report factuality', 'table5_report_factuality.md', 'Report grounding/factuality evaluator and SFT report results.'),
]

CORE_JSONS = [
    ('ToolUniverse v1', '/data1/jiahui/biosignal-agent/outputs/biosignal_tool_universe_v1.json'),
    ('BioSignalBench v1', '/data1/jiahui/biosignal-agent/outputs/biosignalbench_v1.jsonl'),
    ('BioSignalBench splits', '/data1/jiahui/biosignal-agent/outputs/biosignalbench_splits_v1/summary.json'),
    ('Main v5+guardrail controller', '/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_live_v5_heldout_guarded.json'),
    ('OpenRouter controller baseline', '/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_openrouter_owl_alpha_heldout.json'),
    ('Final ablation summary', '/data1/jiahui/biosignal-agent/outputs/live_controller_ablations_v5_guarded_heldout_summary.json'),
    ('Final failure analysis JSON', '/data1/jiahui/biosignal-agent/outputs/final_failure_analysis_v5_guarded_heldout.json'),
    ('v5 planner LoRA summary', '/data1/jiahui/biosignal-agent/outputs/biosignal_sft_planner_lora_qwen25_05b_train_split_v5/train_summary.json'),
    ('Tool execution metric index JSON', '/data1/jiahui/biosignal-agent/outputs/tool_execution_metrics_index.json'),
    ('Tool execution metric index CSV', '/data1/jiahui/biosignal-agent/outputs/tool_execution_metrics_index.csv'),
    ('BioSignalBench expanded manifest', '/data1/jiahui/biosignal-agent/outputs/biosignalbench_v1_expanded.jsonl'),
    ('BioSignalBench expanded splits', '/data1/jiahui/biosignal-agent/outputs/biosignalbench_v1_expanded_splits/summary.json'),
    ('BioSignalBench expanded held-out comparison', '/data1/jiahui/biosignal-agent/outputs/biosignalbench_expanded_heldout_baseline_comparison.json'),
    ('TxAgent comparability gap matrix JSON', '/data1/jiahui/biosignal-agent/outputs/txagent_comparability_gap_matrix.json'),
    ('Manuscript results draft', '/data1/jiahui/biosignal-agent/code/biosignal-agent/docs/biosignalagent_manuscript_results_draft.md'),
    ('Paper outline', '/data1/jiahui/biosignal-agent/code/biosignal-agent/docs/biosignalagent_paper_outline.md'),
]


def exists(path: str | Path) -> str:
    return 'yes' if Path(path).exists() else 'missing'


def table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in rows:
        out.append('| ' + ' | '.join(str(x) for x in row) + ' |')
    return '\n'.join(out) + '\n'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--paper-dir', default='/data1/jiahui/biosignal-agent/outputs/paper_tables')
    ap.add_argument('--out-md', default='/data1/jiahui/biosignal-agent/outputs/paper_tables/table25_paper_artifact_index.md')
    args = ap.parse_args()
    paper_dir = Path(args.paper_dir)
    table_rows = []
    for title, filename, purpose in CORE_TABLES:
        path = paper_dir / filename
        table_rows.append([title, str(path), exists(path), purpose])
    json_rows = []
    for title, path in CORE_JSONS:
        json_rows.append([title, path, exists(path)])
    text = '# Table 25. BioSignalAgent Paper Artifact Index\n\n'
    text += 'Core paper-ready artifacts for reproducing the BioSignalToolUniverse, BioSignalBench, SFT controller, external baselines, ablations, and failure analysis.\n\n'
    text += '## Paper Tables\n\n'
    text += table(['Artifact', 'Path', 'Exists', 'Purpose'], table_rows)
    text += '\n## Machine-Readable Outputs\n\n'
    text += table(['Artifact', 'Path', 'Exists'], json_rows)
    text += '\nRecommended main paper rows: ToolUniverse summary, BioSignalBench composition, split-protocol live controller comparison, final live ablation, and final failure analysis. Older replay/v3/v4 tables are retained for provenance but should be treated as development artifacts unless explicitly discussed.\n'
    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(out)


if __name__ == '__main__':
    main()
