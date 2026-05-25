#!/usr/bin/env python3
"""Build a TxAgent-comparability gap/status matrix for BioSignalAgent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

OUTPUTS = Path('/data1/jiahui/biosignal-agent/outputs')
PAPER_DIR = OUTPUTS / 'paper_tables'


def load(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def fmt(value: Any) -> str:
    if value is None:
        return 'NA'
    if isinstance(value, float):
        return f'{value:.3f}'
    return str(value)


def exists(path: str | Path) -> str:
    return 'yes' if Path(path).exists() else 'missing'


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join(str(x).replace('|', '/') for x in row) + ' |')
    return '\n'.join(lines) + '\n'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-md', default=str(PAPER_DIR / 'table30_txagent_gap_matrix.md'))
    ap.add_argument('--out-json', default=str(OUTPUTS / 'txagent_comparability_gap_matrix.json'))
    args = ap.parse_args()

    universe = load(OUTPUTS / 'biosignal_tool_universe_v1.json')
    bench = load(OUTPUTS / 'biosignalbench_v1_summary.json')
    split = load(OUTPUTS / 'biosignalbench_splits_v1/summary.json')
    expanded = load(OUTPUTS / 'biosignalbench_v1_expanded_summary.json')
    expanded_split = load(OUTPUTS / 'biosignalbench_v1_expanded_splits/summary.json')
    controller = load(OUTPUTS / 'biosignalagent_e2e_controller_live_v5_heldout_guarded.json')
    expanded_live = load(OUTPUTS / 'biosignalagent_e2e_controller_live_v5_expanded_heldout_timeout_guarded.json')
    openrouter = load(OUTPUTS / 'biosignalagent_e2e_controller_openrouter_owl_alpha_heldout.json')
    ablation = load(OUTPUTS / 'live_controller_ablations_v5_guarded_heldout_summary.json')
    failures = load(OUTPUTS / 'final_failure_analysis_v5_guarded_heldout.json')
    tool_metrics = load(OUTPUTS / 'tool_execution_metrics_index.json')
    expanded_cmp = load(OUTPUTS / 'biosignalbench_expanded_heldout_baseline_comparison.json')

    clean_heldout = split.get('splits', {}).get('heldout', {}).get('num_cases')
    expanded_heldout = expanded_split.get('splits', {}).get('heldout', {}).get('num_cases')
    exp_rule = next((r for r in expanded_cmp.get('rows', []) if r.get('baseline') == 'Rule planner'), {})
    expanded_live_failures = expanded_live.get('failure_reason_counts') or {}

    rows = [
        {
            'gap': 'Unified ToolUniverse framing',
            'status': 'closed for v1',
            'evidence': f"{universe.get('num_tools', 'NA')} frozen tools; metadata/source/limitation validation artifacts exist",
            'artifact': str(OUTPUTS / 'biosignal_tool_universe_v1.json'),
            'remaining': 'Keep experimental/proxy tools labeled; new tools should go to vNext.',
        },
        {
            'gap': 'Systematic agent benchmark',
            'status': 'partially closed',
            'evidence': f"v1={bench.get('num_cases', 'NA')} cases; clean held-out={clean_heldout}; expanded={expanded.get('num_cases', 'NA')} cases; expanded held-out={expanded_heldout}",
            'artifact': str(OUTPUTS / 'biosignalbench_v1.jsonl'),
            'remaining': 'Clean held-out is still smaller than a full TxAgent-scale benchmark; expanded split is stress evidence, not replacement.',
        },
        {
            'gap': 'SFT tool-use agent',
            'status': 'partially closed',
            'evidence': f"clean v5 held-out planning={fmt(controller.get('planning_accuracy'))}, Tool F1={fmt(controller.get('tool_f1'))}, overall={fmt(controller.get('overall_hmean'))}; expanded live planning={fmt(expanded_live.get('planning_accuracy'))}, Tool F1={fmt(expanded_live.get('tool_f1'))}, failures={sum(expanded_live_failures.values()) if expanded_live_failures else 0}",
            'artifact': str(OUTPUTS / 'biosignalagent_e2e_controller_live_v5_heldout_guarded.json'),
            'remaining': 'SFT data is still small and strict-JSON parse rate remains imperfect; distinguish model generation from metadata guardrail repair in the manuscript.',
        },
        {
            'gap': 'External LLM baseline',
            'status': 'partially closed',
            'evidence': f"OpenRouter/owl-alpha held-out planning={fmt(openrouter.get('planning_accuracy'))}, Tool F1={fmt(openrouter.get('tool_f1'))}, overall={fmt(openrouter.get('overall_hmean'))}",
            'artifact': str(OUTPUTS / 'biosignalagent_e2e_controller_openrouter_owl_alpha_heldout.json'),
            'remaining': 'Only one free external model; stronger GPT/Claude/Gemini/Qwen baselines need stable API/budget.',
        },
        {
            'gap': 'Systematic ablation',
            'status': 'closed for first paper version',
            'evidence': f"{ablation.get('num_runs', 'NA')} ablation rows; final table exists",
            'artifact': str(OUTPUTS / 'paper_tables/table23_live_controller_ablation_v5_guarded_heldout.md'),
            'remaining': 'Future ablations can add prompt-only vs LoRA and larger-backbone planner.',
        },
        {
            'gap': 'Tool execution numeric evidence',
            'status': 'partially closed',
            'evidence': f"{tool_metrics.get('num_rows', 'NA')} tool/task metric rows across major modalities",
            'artifact': str(OUTPUTS / 'paper_tables/table26_tool_execution_metrics_index.md'),
            'remaining': 'Some rows remain proxy/small-split; needs more external validation for clinical claims.',
        },
        {
            'gap': 'Failure analysis',
            'status': 'closed for first paper version',
            'evidence': 'Final failure table compares BioSignal v5+guardrail vs OpenRouter on held-out failures.',
            'artifact': str(OUTPUTS / 'paper_tables/table24_final_failure_analysis.md'),
            'remaining': 'Add qualitative examples in manuscript appendix.',
        },
        {
            'gap': 'Benchmark scale stress test',
            'status': 'closed for first paper version',
            'evidence': f"expanded held-out={expanded_heldout}; rule planning={fmt(exp_rule.get('planning_accuracy'))}; live SFT planning={fmt(expanded_live.get('planning_accuracy'))}; live Tool F1={fmt(expanded_live.get('tool_f1'))}",
            'artifact': str(OUTPUTS / 'paper_tables/table28_expanded_heldout_baseline_comparison.md'),
            'remaining': 'Add stronger live external LLM baselines and report the expanded split as stress evidence, not as the primary frozen split.',
        },
        {
            'gap': 'Manuscript-ready narrative',
            'status': 'closed draft',
            'evidence': f"outline={exists('/data1/jiahui/biosignal-agent/code/biosignal-agent/docs/biosignalagent_paper_outline.md')}; results_draft={exists('/data1/jiahui/biosignal-agent/code/biosignal-agent/docs/biosignalagent_manuscript_results_draft.md')}; artifact log={exists('/data1/jiahui/biosignal-agent/code/biosignal-agent/docs/biosignal_paper_artifacts_v1.md')}",
            'artifact': '/data1/jiahui/biosignal-agent/code/biosignal-agent/docs/biosignalagent_manuscript_results_draft.md',
            'remaining': 'Convert draft into polished prose and add qualitative examples in appendix.',
        },
    ]

    Path(args.out_json).write_text(json.dumps({'artifact': 'TxAgentComparabilityGapMatrix', 'rows': rows}, indent=2) + '\n')
    md = '# Table 30. TxAgent Comparability Gap Matrix\n\n'
    md += 'Status matrix mapping BioSignalAgent artifacts to the components expected in a TxAgent-style paper.\n\n'
    md += table(['Gap', 'Status', 'Evidence', 'Primary Artifact', 'Remaining Work'], [[r['gap'], r['status'], r['evidence'], r['artifact'], r['remaining']] for r in rows])
    Path(args.out_md).write_text(md)
    print(json.dumps({'out_md': args.out_md, 'out_json': args.out_json, 'rows': len(rows)}, indent=2))


if __name__ == '__main__':
    main()
