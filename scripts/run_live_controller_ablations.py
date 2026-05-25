#!/usr/bin/env python3
"""Run live-controller ablations over a BioSignalBench manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ABLATIONS: dict[str, list[str]] = {
    'full_live_v5_guarded': [],
    'no_toolrag': ['no_toolrag'],
    'no_modality_classifier': ['no_modality_classifier'],
    'no_ocr_scale': ['no_ocr_scale'],
    'no_image_digitization': ['no_image_digitization'],
    'no_quality_gate': ['no_quality_gate'],
    'no_dl_tools': ['no_dl_tools'],
}


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)


def fmt(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, float):
        return f'{value:.3f}'
    return str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in rows:
        out.append('| ' + ' | '.join(fmt(x) for x in row) + ' |')
    return '\n'.join(out) + '\n'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_splits_v1/biosignalbench_v1_heldout.jsonl')
    ap.add_argument('--out-dir', default='/data1/jiahui/biosignal-agent/outputs/live_controller_ablations_v4_heldout')
    ap.add_argument('--out-summary', default='/data1/jiahui/biosignal-agent/outputs/live_controller_ablations_v4_heldout_summary.json')
    ap.add_argument('--out-md', default='/data1/jiahui/biosignal-agent/outputs/paper_tables/table14_live_controller_ablation_heldout.md')
    ap.add_argument('--planner-adapter', default='/data1/jiahui/biosignal-agent/outputs/biosignal_sft_planner_lora_qwen25_05b_live_controller_v4/best_adapter')
    ap.add_argument('--report-mode', choices=['grounded_template', 'live_sft'], default='grounded_template')
    ap.add_argument('--only', action='append', default=None)
    args = ap.parse_args()

    selected = args.only or list(ABLATIONS)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in selected:
        flags = ABLATIONS[name]
        out_json = out_dir / f'{name}.json'
        cmd = [
            sys.executable,
            'scripts/run_biosignalagent_e2e_controller.py',
            '--manifest', args.manifest,
            '--planner-mode', 'live_sft',
            '--report-mode', args.report_mode,
            '--planner-adapter', args.planner_adapter,
            '--out-json', str(out_json),
            '--out-jsonl', str(out_dir / f'{name}_cases.jsonl'),
            '--out-csv', str(out_dir / f'{name}_cases.csv'),
            '--out-md', str(out_dir / f'{name}.md'),
        ]
        for flag in flags:
            cmd.extend(['--ablation-flag', flag])
        print('RUN', name, 'flags=', ','.join(flags) or 'none', flush=True)
        subprocess.run(cmd, check=True)
        report = read_json(out_json)
        rows.append({
            'ablation': name,
            'flags': flags,
            'num_cases': report.get('num_cases'),
            'strict_parse': report.get('planner_strict_parse_rate'),
            'parse': report.get('planner_parse_rate'),
            'planning': report.get('planning_accuracy'),
            'tool_f1': report.get('tool_f1'),
            'execution': report.get('execution_success'),
            'report_score': report.get('report_factuality_score'),
            'overall_hmean': report.get('overall_hmean'),
            'failure_reason_counts': report.get('failure_reason_counts'),
        })
    summary = {
        'artifact': 'BioSignalAgentLiveControllerAblation',
        'manifest': args.manifest,
        'planner_adapter': args.planner_adapter,
        'report_mode': args.report_mode,
        'num_runs': len(rows),
        'runs': rows,
    }
    Path(args.out_summary).write_text(json.dumps(summary, indent=2) + '\n')
    table_rows = [[r['ablation'], ','.join(r['flags']), r['num_cases'], r['strict_parse'], r['parse'], r['planning'], r['tool_f1'], r['execution'], r['report_score'], r['overall_hmean']] for r in rows]
    text = '# Table 23. Held-Out Live Controller Ablation\n\n'
    text += 'Held-out split live planner evaluation for the selected planner adapter. The default report mode is grounded_template to isolate planner/tool availability effects; the full controller live-report result is reported separately.\n\n'
    text += markdown_table(['Ablation','Flags','Cases','Strict parse','Recovered parse','Planning','Tool F1','Exec success','Report score','Overall H-mean'], table_rows)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(text)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
