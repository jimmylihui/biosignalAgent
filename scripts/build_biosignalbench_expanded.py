#!/usr/bin/env python
"""Build a larger BioSignalBench expansion from existing artifacts.

This does not replace frozen BioSignalBench v1. It creates a larger stress/scale
manifest that can be split and evaluated with the same evaluator/controller.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.agent.tool_registry import TOOLS
from biosignal_agent.evaluation.biosignalbench import validate_bench_cases, write_json, write_jsonl, markdown_table, read_jsonl
from scripts.build_biosignalbench_v1 import (
    DATA_ROOT,
    case,
    digitization_cases,
    negative_cases,
    planning_cases,
    ptbxl_12lead_cases,
    sanitize,
    session_cases,
    trace_cases,
)

OUTPUT_ROOT = Path('/data1/jiahui/biosignal-agent/outputs')


def read_json(path: str | Path) -> Any | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def tool_metric_cases(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not payload:
        return []
    rows = []
    for idx, item in enumerate(payload.get('rows', [])):
        tool = item.get('tool')
        if not tool or tool not in TOOLS:
            continue
        modality = str(item.get('modality', 'unknown')).lower()
        task = str(item.get('task') or tool)
        rows.append(case(
            f'tool_metric_{sanitize(modality)}_{sanitize(tool)}_{idx:03d}',
            'tool_execution',
            f"Select the BioSignalAgent tool for {modality.upper()} task: {task}. Existing benchmark evidence uses {item.get('dataset', 'a benchmark')} with metric {item.get('metric', 'metric')}.",
            'csv',
            modality,
            [tool],
            str(path),
            signal={'path': None, 'sampling_rate': None, 'column': None},
            expected_key_outputs=['tool_selected', 'metric_name', 'evidence_level', 'clinical_limitation'],
            ground_truth_metric={
                'type': 'tool_execution_evidence_row',
                'metric': item.get('metric'),
                'value': item.get('value'),
                'evidence_level': item.get('evidence_level'),
                'artifact': item.get('artifact'),
            },
        ))
        if limit and len(rows) >= limit:
            break
    return rows


def composition(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'artifact': 'BioSignalBenchExpanded',
        'version': 'v1-expanded',
        'num_cases': len(cases),
        'task_counts': dict(sorted(Counter(c['benchmark_task'] for c in cases).items())),
        'input_type_counts': dict(sorted(Counter(c['input_type'] for c in cases).items())),
        'modality_counts': dict(sorted(Counter(str(c['modality']).lower() for c in cases).items())),
        'source_counts': dict(sorted(Counter(str(c.get('source')) for c in cases).items())),
    }


def write_composition_md(summary: dict[str, Any], validation: dict[str, Any], path: str | Path) -> None:
    text = [
        '# Table 27. BioSignalBench v1-Expanded Composition',
        '',
        'A larger stress/scale manifest built from existing BioSignalAgent artifacts. This complements, but does not replace, frozen BioSignalBench v1.',
        '',
        f"Total cases: {summary['num_cases']}",
        f"Validation errors: {validation['num_errors']}",
        '',
        '## Cases By Task',
        markdown_table(['Task', 'Cases'], [[k, v] for k, v in summary['task_counts'].items()]),
        '## Cases By Input Type',
        markdown_table(['Input type', 'Cases'], [[k, v] for k, v in summary['input_type_counts'].items()]),
        '## Cases By Modality',
        markdown_table(['Modality', 'Cases'], [[k, v] for k, v in summary['modality_counts'].items()]),
    ]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('\n'.join(text) + '\n')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-jsonl', default=str(OUTPUT_ROOT / 'biosignalbench_v1_expanded.jsonl'))
    ap.add_argument('--out-summary', default=str(OUTPUT_ROOT / 'biosignalbench_v1_expanded_summary.json'))
    ap.add_argument('--out-validation', default=str(OUTPUT_ROOT / 'biosignalbench_v1_expanded_validation.json'))
    ap.add_argument('--out-md', default=str(OUTPUT_ROOT / 'paper_tables/table27_biosignalbench_expanded_composition.md'))
    ap.add_argument('--trace-dir', default=str(OUTPUT_ROOT / 'traces'))
    ap.add_argument('--trace-limit', type=int, default=400)
    ap.add_argument('--session-limit', type=int, default=220)
    ap.add_argument('--digitization-max-per-manifest', type=int, default=60)
    ap.add_argument('--tool-metric-limit', type=int, default=None)
    ap.add_argument('--digitization-manifest', action='append', default=[
        str(DATA_ROOT / 'digitization_benchmark_manifest.json'),
        str(DATA_ROOT / 'digitization_benchmark_more_10s_manifest.json'),
        str(DATA_ROOT / 'digitization_benchmark_one_per_modality_mixed_30s_manifest.json'),
    ])
    ap.add_argument('--sft-jsonl', action='append', default=[
        str(OUTPUT_ROOT / 'biosignal_txagent_planning_sft_expanded_tasks.jsonl'),
        str(OUTPUT_ROOT / 'biosignal_sft_planner_v5_train_split_live_controller.jsonl'),
        str(OUTPUT_ROOT / 'biosignal_sft_planner_v6_train_session_aug.jsonl'),
    ])
    ap.add_argument('--tool-metrics', default=str(OUTPUT_ROOT / 'tool_execution_metrics_index.json'))
    args = ap.parse_args()

    rows = []
    rows.extend(planning_cases())
    rows.extend(digitization_cases(args.digitization_manifest, max_per_manifest=args.digitization_max_per_manifest))
    rows.extend(trace_cases(args.trace_dir, args.trace_limit))
    rows.extend(ptbxl_12lead_cases())
    rows.extend(session_cases(args.sft_jsonl, limit=args.session_limit))
    rows.extend(tool_metric_cases(args.tool_metrics, args.tool_metric_limit))
    rows.extend(negative_cases())

    seen = set()
    unique = []
    for row in rows:
        cid = row['case_id']
        if cid in seen:
            continue
        seen.add(cid)
        unique.append(row)

    summary = composition(unique)
    validation = validate_bench_cases(unique)
    write_jsonl(args.out_jsonl, unique)
    write_json(args.out_summary, summary)
    write_json(args.out_validation, validation)
    write_composition_md(summary, validation, args.out_md)
    print(json.dumps({'out_jsonl': args.out_jsonl, 'out_summary': args.out_summary, 'out_validation': args.out_validation, 'out_md': args.out_md, **summary, 'num_validation_errors': validation['num_errors']}, indent=2))
    if validation['num_errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
