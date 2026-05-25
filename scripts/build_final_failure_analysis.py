#!/usr/bin/env python3
"""Build final held-out failure analysis for BioSignalAgent paper artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def fmt(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, float):
        return f'{value:.3f}'
    return str(value)


def table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in rows:
        out.append('| ' + ' | '.join(fmt(x) for x in row) + ' |')
    return '\n'.join(out) + '\n'


def summarize_case_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task = []
    for task in sorted({r.get('benchmark_task') for r in rows}):
        sub = [r for r in rows if r.get('benchmark_task') == task]
        failures = Counter(r.get('failure_reason') for r in sub if r.get('failure_reason'))
        by_task.append([
            task,
            len(sub),
            sum(bool(r.get('planning_pass')) for r in sub) / len(sub),
            sum(float(r.get('tool_f1') or 0) for r in sub) / len(sub),
            sum(bool(r.get('e2e_pass')) for r in sub) / len(sub),
            ', '.join(f'{k}:{v}' for k, v in failures.most_common()) or 'none',
        ])
    return {
        'failure_counts': Counter(r.get('failure_reason') for r in rows if r.get('failure_reason')),
        'by_task': by_task,
        'failed_rows': [r for r in rows if r.get('failure_reason')],
    }


def failure_examples(rows: list[dict[str, Any]], max_examples: int) -> list[list[Any]]:
    examples = []
    for r in rows:
        if not r.get('failure_reason'):
            continue
        examples.append([
            r.get('case_id'),
            r.get('benchmark_task'),
            r.get('modality'),
            r.get('failure_reason'),
            ', '.join(r.get('missing_from_plan') or [])[:120],
            ', '.join(r.get('unexpected_tools') or [])[:120],
        ])
        if len(examples) >= max_examples:
            break
    return examples


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--main-cases', default='/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_live_v5_heldout_guarded_cases.jsonl')
    ap.add_argument('--openrouter-cases', default='/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_openrouter_owl_alpha_heldout_cases.jsonl')
    ap.add_argument('--out-md', default='/data1/jiahui/biosignal-agent/outputs/paper_tables/table24_final_failure_analysis.md')
    ap.add_argument('--out-json', default='/data1/jiahui/biosignal-agent/outputs/final_failure_analysis_v5_guarded_heldout.json')
    ap.add_argument('--max-examples', type=int, default=12)
    args = ap.parse_args()

    main_rows = read_jsonl(args.main_cases)
    open_rows = read_jsonl(args.openrouter_cases)
    main_summary = summarize_case_rows(main_rows)
    open_summary = summarize_case_rows(open_rows)

    overview_rows = [
        ['BioSignal v5 + session guardrail', len(main_rows), sum(main_summary['failure_counts'].values()), ', '.join(f'{k}:{v}' for k, v in main_summary['failure_counts'].most_common()) or 'none'],
        ['OpenRouter owl-alpha cached planner', len(open_rows), sum(open_summary['failure_counts'].values()), ', '.join(f'{k}:{v}' for k, v in open_summary['failure_counts'].most_common()) or 'none'],
    ]
    text = '# Table 24. Final Held-Out Failure Analysis\n\n'
    text += 'Failure analysis for the final clean split-protocol controller row and the external OpenRouter controller baseline.\n\n'
    text += table(['Method', 'Cases', 'Failed cases', 'Failure reasons'], overview_rows)
    text += '\n## BioSignal v5 + Session Guardrail by Task\n\n'
    text += table(['Task', 'Cases', 'Planning', 'Tool F1', 'E2E', 'Failure reasons'], main_summary['by_task'])
    text += '\n## OpenRouter Baseline by Task\n\n'
    text += table(['Task', 'Cases', 'Planning', 'Tool F1', 'E2E', 'Failure reasons'], open_summary['by_task'])
    text += '\n## BioSignal Failed Case Examples\n\n'
    text += table(['Case', 'Task', 'Modality', 'Failure', 'Missing tools', 'Unexpected tools'], failure_examples(main_rows, args.max_examples))
    text += '\nInterpretation: BioSignalAgent failures are concentrated in exact tool-set mismatches for residual planning/report cases, while image digitization, scale/OCR, and multimodal session routing are solved on this held-out split. OpenRouter failures are dominated by planning tool mismatch across nearly every task, which then propagates into lower execution and report factuality.\n'

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(text)
    payload = {
        'artifact': 'BioSignalAgentFinalFailureAnalysis',
        'main_cases': args.main_cases,
        'openrouter_cases': args.openrouter_cases,
        'main_failure_counts': dict(main_summary['failure_counts']),
        'openrouter_failure_counts': dict(open_summary['failure_counts']),
        'main_failed_case_ids': [r.get('case_id') for r in main_summary['failed_rows']],
        'openrouter_failed_case_ids': [r.get('case_id') for r in open_summary['failed_rows']],
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2) + '\n')
    print(args.out_md)


if __name__ == '__main__':
    main()
