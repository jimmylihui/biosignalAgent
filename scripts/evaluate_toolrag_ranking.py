#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.agent.tool_retriever import ToolRetriever
from biosignal_agent.evaluation.biosignalbench import load_bench_cases, markdown_table, retrieve_tools_for_case, write_json, write_jsonl


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with p.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def evaluate(cases: list[dict[str, Any]], max_k: int = 30) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    retriever = ToolRetriever()
    rows = []
    ks = [1, 3, 5, 7, 10, 20]
    for case in cases:
        expected = list(case.get('expected_tools') or [])
        ranked = retrieve_tools_for_case(case, retriever, 'tfidf', max_k, {})
        rank_by_tool = {tool: idx + 1 for idx, tool in enumerate(ranked)}
        ranks = [rank_by_tool.get(tool) for tool in expected]
        reciprocal_ranks = [(1.0 / r) for r in ranks if r]
        row = {
            'case_id': case.get('case_id'),
            'benchmark_task': case.get('benchmark_task'),
            'input_type': case.get('input_type'),
            'modality': str(case.get('modality', '')).lower(),
            'num_expected_tools': len(expected),
            'expected_tools': expected,
            'ranked_tools': ranked,
            'ranks': ranks,
            'all_found': all(r is not None for r in ranks),
            'mrr_expected_tools': mean(reciprocal_ranks) if expected else 1.0,
        }
        for k in ks:
            row[f'recall_at_{k}'] = sum(1 for r in ranks if r is not None and r <= k) / len(expected) if expected else 1.0
            row[f'all_expected_at_{k}'] = all(r is not None and r <= k for r in ranks) if expected else True
        rows.append(row)
    summary = summarize(rows, ks)
    return summary, rows


def summarize(rows: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    summary = {
        'artifact': 'BioSignalToolRAGRankingEvaluation',
        'num_cases': len(rows),
        'mrr_expected_tools': mean([r['mrr_expected_tools'] for r in rows]),
    }
    for k in ks:
        summary[f'recall_at_{k}'] = mean([r[f'recall_at_{k}'] for r in rows])
        summary[f'all_expected_at_{k}'] = mean([float(r[f'all_expected_at_{k}']) for r in rows])
    by_task = {}
    for task in sorted({r['benchmark_task'] for r in rows}):
        sub = [r for r in rows if r['benchmark_task'] == task]
        by_task[task] = {'num_cases': len(sub), 'mrr_expected_tools': mean([r['mrr_expected_tools'] for r in sub])}
        for k in ks:
            by_task[task][f'recall_at_{k}'] = mean([r[f'recall_at_{k}'] for r in sub])
            by_task[task][f'all_expected_at_{k}'] = mean([float(r[f'all_expected_at_{k}']) for r in sub])
    summary['by_task'] = by_task
    return summary


def write_markdown(summary: dict[str, Any], path: str | Path) -> None:
    rows = [[
        'metadata-aware TF-IDF ToolRAG',
        summary['num_cases'],
        summary['recall_at_1'],
        summary['recall_at_3'],
        summary['recall_at_5'],
        summary['recall_at_7'],
        summary['recall_at_10'],
        summary['recall_at_20'],
        summary['all_expected_at_7'],
        summary['mrr_expected_tools'],
    ]]
    task_rows = []
    for task, vals in summary.get('by_task', {}).items():
        task_rows.append([task, vals['num_cases'], vals['recall_at_3'], vals['recall_at_7'], vals['recall_at_20'], vals['all_expected_at_7'], vals['mrr_expected_tools']])
    text = '# Table 10. ToolRAG Ranking Evaluation\n\n'
    text += 'Recall@k is averaged over expected tools per case. `All@7` is stricter: every expected tool for a case must appear in the top 7 retrieved tools.\n\n'
    text += markdown_table(['Retriever','Cases','R@1','R@3','R@5','R@7','R@10','R@20','All@7','MRR'], rows)
    text += '\n## By Task\n\n'
    text += markdown_table(['Task','Cases','R@3','R@7','R@20','All@7','MRR'], task_rows)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser(description='Evaluate BioSignal ToolRAG ranking metrics on BioSignalBench.')
    ap.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_v1.jsonl')
    ap.add_argument('--max-k', type=int, default=30)
    ap.add_argument('--out-json', default='/data1/jiahui/biosignal-agent/outputs/biosignal_toolrag_ranking_eval.json')
    ap.add_argument('--out-jsonl', default='/data1/jiahui/biosignal-agent/outputs/biosignal_toolrag_ranking_eval_cases.jsonl')
    ap.add_argument('--out-csv', default='/data1/jiahui/biosignal-agent/outputs/biosignal_toolrag_ranking_eval_cases.csv')
    ap.add_argument('--out-md', default='/data1/jiahui/biosignal-agent/outputs/paper_tables/table10_toolrag_ranking.md')
    args = ap.parse_args()
    summary, rows = evaluate(load_bench_cases(args.manifest), max_k=args.max_k)
    write_json(args.out_json, summary)
    write_jsonl(args.out_jsonl, rows)
    write_csv(args.out_csv, rows)
    write_markdown(summary, args.out_md)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
