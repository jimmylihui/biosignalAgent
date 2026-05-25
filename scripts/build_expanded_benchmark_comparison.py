#!/usr/bin/env python3
"""Build expanded BioSignalBench held-out baseline comparison table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

OUTPUTS = Path('/data1/jiahui/biosignal-agent/outputs')
PAPER_DIR = OUTPUTS / 'paper_tables'

ROWS = [
    ('Rule planner', OUTPUTS / 'biosignalbench_expanded_heldout_eval_rule.json', 'deterministic workflow planner + TF-IDF top-20 retrieval'),
    ('TF-IDF ToolRAG as planner', OUTPUTS / 'biosignalbench_expanded_heldout_eval_toolrag.json', 'top-20 retrieved tools used directly as plan'),
    ('SFT replay/fallback', OUTPUTS / 'biosignalbench_expanded_heldout_eval_sft_replay.json', 'trace/SFT replay baseline with rule fallback'),
    ('Oracle tool selection', OUTPUTS / 'biosignalbench_expanded_heldout_eval_oracle.json', 'expected tool set supplied to planner and retriever'),
    ('Raw live SFT v5 (no guardrail)', OUTPUTS / 'biosignalagent_e2e_controller_live_v5_expanded_heldout_raw_no_guardrail.json', 'real-time LoRA planner without structured metadata completion/pruning'),
    ('Live SFT v5 + timeout guardrail', OUTPUTS / 'biosignalagent_e2e_controller_live_v5_expanded_heldout_timeout_guarded.json', 'real-time LoRA planner with metadata guardrails and 20s timeout fallback'),
]


def load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def fmt(value: Any) -> str:
    if value is None:
        return 'NA'
    if isinstance(value, float):
        return f'{value:.3f}'
    return str(value)


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join(str(x).replace('|', '/') for x in row) + ' |')
    return '\n'.join(lines) + '\n'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-md', default=str(PAPER_DIR / 'table28_expanded_heldout_baseline_comparison.md'))
    ap.add_argument('--out-json', default=str(OUTPUTS / 'biosignalbench_expanded_heldout_baseline_comparison.json'))
    args = ap.parse_args()

    records = []
    for name, path, note in ROWS:
        data = load(path)
        if not data:
            records.append({'baseline': name, 'path': str(path), 'exists': False, 'note': note})
            continue
        records.append({
            'baseline': name,
            'path': str(path),
            'exists': True,
            'num_cases': data.get('num_cases'),
            'retrieval_accuracy': data.get('retrieval_accuracy'),
            'planning_accuracy': data.get('planning_accuracy'),
            'tool_f1': data.get('tool_f1'),
            'execution_accuracy': data.get('execution_accuracy', data.get('execution_success')),
            'report_factuality_score': data.get('report_factuality_score'),
            'overall_hmean': data.get('overall_hmean'),
            'failure_reason_counts': data.get('failure_reason_counts'),
            'note': note,
        })
    Path(args.out_json).write_text(json.dumps({'artifact': 'ExpandedHeldoutBaselineComparison', 'rows': records}, indent=2) + '\n')
    md_rows = []
    for r in records:
        md_rows.append([
            r['baseline'],
            r.get('num_cases', 'missing'),
            fmt(r.get('retrieval_accuracy')),
            fmt(r.get('planning_accuracy')),
            fmt(r.get('tool_f1')),
            fmt(r.get('execution_accuracy')),
            fmt(r.get('overall_hmean')),
            r['note'],
        ])
    text = '# Table 28. BioSignalBench v1-Expanded Held-Out Baselines\n\n'
    text += 'Expanded held-out (`92` cases) is a scale/stress complement to frozen BioSignalBench v1. It is not used to replace the main 48-case clean split row; it tests whether retrieval and planning trends survive a larger manifest.\n\n'
    text += table(['Baseline', 'Cases', 'Retrieval Acc.', 'Planning Acc.', 'Tool F1', 'Execution Acc.', 'Overall', 'Note'], md_rows)
    live = next((r for r in records if r.get('baseline') == 'Live SFT v5 + timeout guardrail'), {})
    live_failures = live.get('failure_reason_counts') or {}
    if live.get('planning_accuracy') == 1.0 and live.get('tool_f1') == 1.0 and not live_failures:
        live_sentence = 'The live timeout-guarded controller reaches perfect exact tool planning and Tool F1 on the expanded held-out split, with no recorded failure reasons; its overall score is limited by the grounded report factuality subscore rather than tool selection.'
    else:
        live_sentence = 'The live timeout-guarded controller improves image digitization, scale/OCR, multimodal session, and trace-derived planning cases; remaining failures should be inspected from the per-case JSONL before paper reporting.'
    text += f'\nInterpretation: TF-IDF retrieval has complete top-20 coverage on the expanded held-out split, but direct ToolRAG planning over-selects tools. {live_sentence} Oracle establishes the upper bound and validates manifest consistency.\n'
    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(json.dumps({'out_md': str(out), 'out_json': args.out_json, 'rows': len(records)}, indent=2))


if __name__ == '__main__':
    main()
