#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.evaluation.biosignalbench import evaluate_benchmark_cases, load_bench_cases, markdown_table, write_csv, write_json, write_jsonl


def write_markdown(report: dict, path: str | Path) -> None:
    rows = []
    for task, metrics in report.get('by_task', {}).items():
        rows.append([task, metrics.get('num_cases'), metrics.get('retrieval_accuracy'), metrics.get('planning_accuracy'), metrics.get('tool_f1'), metrics.get('execution_accuracy')])
    text = [
        '# BioSignalBench Evaluation', '',
        f"Planner: `{report['planner_backend']}`",
        f"Retriever: `{report['retriever_backend']}`",
        f"Cases: {report['num_cases']}",
        f"Retrieval accuracy: {report['retrieval_accuracy']:.3f}",
        f"Planning accuracy: {report['planning_accuracy']:.3f}",
        f"Tool-selection F1: {report['tool_f1']:.3f}",
        f"Execution accuracy: {report['execution_accuracy'] if report['execution_accuracy'] is not None else 'not run'}", '',
        '## By Task', markdown_table(['Task', 'Cases', 'Retrieval', 'Planning', 'Tool F1', 'Execution'], rows),
        '## Failure Reasons', markdown_table(['Reason', 'Count'], [[k, v] for k, v in report.get('failure_reason_counts', {}).items()]),
    ]
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text('\n'.join(text))


def parse_flags(values: list[str] | None) -> dict[str, bool]:
    out = {}
    for value in values or []:
        for part in value.split(','):
            part = part.strip().replace('-', '_')
            if part:
                out[part] = True
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description='Evaluate BioSignalBench v1 with a manifest-driven TxAgent-style evaluator.')
    ap.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_v1.jsonl')
    ap.add_argument('--planner-backend', choices=['rule','toolrag','oracle','none','no_tool_llm','sft_replay','sft_planner','sft_report'], default='rule')
    ap.add_argument('--retriever-backend', choices=['tfidf','oracle','none','disabled'], default='tfidf')
    ap.add_argument('--retrieved-tool-count', type=int, default=7)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--task', action='append', default=None)
    ap.add_argument('--ablation-flag', action='append', default=None, help='Comma-separated flags, e.g. no_toolrag,no_quality_gate.')
    ap.add_argument('--sft-jsonl', action='append', default=['/data1/jiahui/biosignal-agent/outputs/biosignal_txagent_planning_sft_expanded_tasks.jsonl'])
    ap.add_argument('--out-json', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_rule.json')
    ap.add_argument('--out-jsonl', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_rule_cases.jsonl')
    ap.add_argument('--out-csv', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_rule_cases.csv')
    ap.add_argument('--out-md', default='/data1/jiahui/biosignal-agent/outputs/paper_tables/biosignalbench_eval_rule.md')
    args = ap.parse_args()
    cases = load_bench_cases(args.manifest)
    if args.task:
        wanted = set(args.task)
        cases = [case for case in cases if case.get('benchmark_task') in wanted]
    if args.limit is not None:
        cases = cases[:args.limit]
    report = evaluate_benchmark_cases(
        cases,
        planner_backend=args.planner_backend,
        retriever_backend=args.retriever_backend,
        retrieved_tool_count=args.retrieved_tool_count,
        execute=args.execute,
        sft_paths=args.sft_jsonl,
        ablation_flags=parse_flags(args.ablation_flag),
    )
    write_json(args.out_json, {k:v for k,v in report.items() if k != 'cases'})
    write_jsonl(args.out_jsonl, report['cases'])
    write_csv(args.out_csv, report['cases'])
    write_markdown(report, args.out_md)
    print(json.dumps({k:v for k,v in report.items() if k != 'cases'}, indent=2))

if __name__ == '__main__':
    main()
