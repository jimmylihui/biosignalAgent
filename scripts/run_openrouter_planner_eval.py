from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.agent.openrouter_client import DEFAULT_MODEL
from biosignal_agent.evaluation.framework_eval import evaluate_cases
from biosignal_agent.evaluation.planning_cases import DEFAULT_PLANNING_CASES, PlanningCase


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed = {}
    if not path.exists():
        return completed
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            completed[row['case_id']] = row
    return completed


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'case_id', 'modality', 'planner', 'retrieval_pass', 'planning_pass', 'execution_ok',
        'expected_tools', 'retrieved_tools', 'planned_tools', 'missing_from_retrieval',
        'missing_from_plan', 'unexpected_tools', 'execution_errors', 'planner_error'
    ]
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key)) if isinstance(row.get(key), list) else row.get(key) for key in fieldnames})


def final_report(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    retrieval_passes = sum(1 for row in rows if row.get('retrieval_pass'))
    planning_passes = sum(1 for row in rows if row.get('planning_pass'))
    executable_rows = [row for row in rows if row.get('execution_ok') is not None]
    execution_passes = sum(1 for row in executable_rows if row.get('execution_ok'))
    backend_counts: dict[str, int] = {}
    for row in rows:
        planner = row.get('planner')
        backend_counts[planner] = backend_counts.get(planner, 0) + 1
    return {
        'planner': 'openrouter',
        'model': args.model,
        'retrieved_tool_count': args.retrieved_tool_count,
        'num_cases': len(rows),
        'retrieval_accuracy': retrieval_passes / len(rows) if rows else 0.0,
        'planning_accuracy': planning_passes / len(rows) if rows else 0.0,
        'execution_accuracy': execution_passes / len(executable_rows) if executable_rows else None,
        'planner_backend_counts': backend_counts,
        'cases': rows,
    }


def selected_cases(args: argparse.Namespace) -> list[PlanningCase]:
    cases = DEFAULT_PLANNING_CASES
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case.case_id in wanted]
    if args.start_index is not None:
        cases = cases[args.start_index - 1:]
    if args.limit is not None:
        cases = cases[: args.limit]
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description='Run checkpointed OpenRouter planner evaluation, one case at a time.')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--retrieved-tool-count', type=int, default=3)
    parser.add_argument('--llm-timeout', type=int, default=30)
    parser.add_argument('--llm-retry-max', type=int, default=1)
    parser.add_argument('--llm-retry-delay', type=float, default=1.0)
    parser.add_argument('--allow-rule-fallback', action='store_true')
    parser.add_argument('--case-id', action='append', default=None)
    parser.add_argument('--start-index', type=int, default=None, help='1-based index into the selected planning case list.')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--checkpoint-jsonl', default='/data1/jiahui/biosignal-agent/outputs/framework_eval_openrouter_common_modalities.jsonl')
    parser.add_argument('--out-json', default='/data1/jiahui/biosignal-agent/outputs/framework_eval_openrouter_common_modalities.json')
    parser.add_argument('--out-csv', default='/data1/jiahui/biosignal-agent/outputs/framework_eval_openrouter_common_modalities.csv')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_jsonl)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(checkpoint_path)
    cases = selected_cases(args)
    for index, case in enumerate(cases, start=1):
        if case.case_id in completed:
            print(f'[{index}/{len(cases)}] skip {case.case_id}', flush=True)
            continue
        print(f'[{index}/{len(cases)}] openrouter {case.case_id}', flush=True)
        report = evaluate_cases(
            cases=[case],
            planner_name='openrouter',
            model=args.model,
            retrieved_tool_count=args.retrieved_tool_count,
            llm_timeout=args.llm_timeout,
            llm_retry_max=args.llm_retry_max,
            llm_retry_delay=args.llm_retry_delay,
            llm_fallback_to_rules=args.allow_rule_fallback,
        )
        row = report['cases'][0]
        with checkpoint_path.open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        completed[row['case_id']] = row
        print(f"    planner={row['planner']} pass={row['planning_pass']} planned={row['planned_tools']}", flush=True)

    ordered_rows = [completed[case.case_id] for case in DEFAULT_PLANNING_CASES if case.case_id in completed]
    report = final_report(ordered_rows, args)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    write_csv(ordered_rows, args.out_csv)
    print(json.dumps({key: report[key] for key in ['num_cases', 'planning_accuracy', 'planner_backend_counts']}, indent=2))


if __name__ == '__main__':
    main()
