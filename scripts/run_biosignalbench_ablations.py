#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.evaluation.biosignalbench import markdown_table, read_json, write_json

ABLATIONS = {
    'full_rule': {'planner':'rule', 'retriever':'tfidf', 'flags': []},
    'no_tool_llm': {'planner':'no_tool_llm', 'retriever':'none', 'flags': []},
    'oracle_tools': {'planner':'oracle', 'retriever':'oracle', 'flags': []},
    'tfidf_toolrag': {'planner':'toolrag', 'retriever':'tfidf', 'flags': []},
    'sft_replay': {'planner':'sft_replay', 'retriever':'tfidf', 'flags': []},
    'no_toolrag': {'planner':'rule', 'retriever':'tfidf', 'flags': ['no_toolrag']},
    'no_quality_gate': {'planner':'rule', 'retriever':'tfidf', 'flags': ['no_quality_gate']},
    'no_dl_tools': {'planner':'rule', 'retriever':'tfidf', 'flags': ['no_dl_tools']},
    'no_modality_classifier': {'planner':'rule', 'retriever':'tfidf', 'flags': ['no_modality_classifier']},
    'no_ocr_scale': {'planner':'rule', 'retriever':'tfidf', 'flags': ['no_ocr_scale']},
    'no_image_digitization': {'planner':'rule', 'retriever':'tfidf', 'flags': ['no_image_digitization']},
}


def main() -> None:
    ap = argparse.ArgumentParser(description='Run TxAgent-style BioSignalBench planner/retriever ablations.')
    ap.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_v1.jsonl')
    ap.add_argument('--out-dir', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_ablations')
    ap.add_argument('--out-summary', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_ablation_summary.json')
    ap.add_argument('--out-md', default='/data1/jiahui/biosignal-agent/outputs/paper_tables/biosignalbench_ablation_table.md')
    ap.add_argument('--only', action='append', default=None)
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    selected = args.only or list(ABLATIONS)
    rows = []
    for name in selected:
        spec = ABLATIONS[name]
        out_json = out_dir / f'{name}.json'
        out_jsonl = out_dir / f'{name}_cases.jsonl'
        out_csv = out_dir / f'{name}_cases.csv'
        out_md = out_dir / f'{name}.md'
        cmd = [sys.executable, 'scripts/evaluate_biosignalbench.py', '--manifest', args.manifest, '--planner-backend', spec['planner'], '--retriever-backend', spec['retriever'], '--out-json', str(out_json), '--out-jsonl', str(out_jsonl), '--out-csv', str(out_csv), '--out-md', str(out_md)]
        if spec['flags']:
            cmd.extend(['--ablation-flag', ','.join(spec['flags'])])
        subprocess.run(cmd, check=True)
        report = read_json(out_json)
        rows.append({
            'ablation': name,
            'planner_backend': report.get('planner_backend'),
            'retriever_backend': report.get('retriever_backend'),
            'flags': spec['flags'],
            'num_cases': report.get('num_cases'),
            'retrieval_accuracy': report.get('retrieval_accuracy'),
            'planning_accuracy': report.get('planning_accuracy'),
            'execution_accuracy': report.get('execution_accuracy'),
            'tool_f1': report.get('tool_f1'),
            'failure_reason_counts': report.get('failure_reason_counts'),
        })
    summary = {'artifact':'BioSignalBenchAblationSummary','num_runs':len(rows),'runs':rows}
    write_json(args.out_summary, summary)
    table = markdown_table(['Ablation','Planner','Retriever','Flags','Cases','Retrieval','Planning','Tool F1','Execution'], [[r['ablation'], r['planner_backend'], r['retriever_backend'], ','.join(r['flags']), r['num_cases'], r['retrieval_accuracy'], r['planning_accuracy'], r.get('tool_f1'), r['execution_accuracy']] for r in rows])
    p=Path(args.out_md); p.parent.mkdir(parents=True, exist_ok=True); p.write_text('# BioSignalBench Ablation Table\n\n'+table)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
