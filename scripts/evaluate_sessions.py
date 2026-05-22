from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.agent.framework import BioSignalAgentConfig, BioSignalAgentFramework
from biosignal_agent.agent.openrouter_client import DEFAULT_MODEL
from biosignal_agent.evaluation.session_cases import build_default_session_cases, case_to_session, load_records


def evaluate_session_case(case, agent: BioSignalAgentFramework) -> dict[str, Any]:
    trace = agent.run_session(case_to_session(case))
    runs_by_label = {run.get('signal_label'): run for run in trace['runs']}
    rows = []
    for expectation in case.expectations:
        run = runs_by_label.get(expectation.label)
        if run is None:
            rows.append({
                'case_id': case.case_id,
                'signal_label': expectation.label,
                'modality': expectation.modality,
                'expected_tools': list(expectation.expected_tools),
                'retrieved_tools': [],
                'planned_tools': [],
                'retrieval_pass': False,
                'planning_pass': False,
                'execution_ok': False,
                'errors': ['missing run'],
            })
            continue
        retrieved = run.get('retrieved_tools') or []
        planned = [call['name'] for call in run.get('tool_plan', [])]
        executed = [item.get('tool') for item in run.get('tool_results', [])]
        expected = set(expectation.expected_tools)
        retrieval_pass = expected.issubset(set(retrieved))
        planning_pass = expected == set(planned)
        errors = []
        execution_ok = True
        for item in run.get('tool_results', []):
            result = item.get('result', {})
            if item.get('tool') not in expected or not isinstance(result, dict) or result.get('error'):
                execution_ok = False
                errors.append(f"{item.get('tool')}: {result.get('error', 'unexpected tool or invalid result')}")
        if set(executed) != expected:
            execution_ok = False
            errors.append(f'executed tools mismatch: {executed}')
        rows.append({
            'case_id': case.case_id,
            'signal_label': expectation.label,
            'modality': expectation.modality,
            'expected_tools': list(expectation.expected_tools),
            'retrieved_tools': retrieved,
            'planned_tools': planned,
            'retrieval_pass': retrieval_pass,
            'planning_pass': planning_pass,
            'execution_ok': execution_ok,
            'errors': errors,
        })
    return {
        'case_id': case.case_id,
        'question': case.question,
        'trace_path': trace.get('trace_path'),
        'num_signals': len(case.signals),
        'rows': rows,
    }


def write_outputs(report: dict[str, Any], out_json: str | Path, out_csv: str | Path) -> None:
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['case_id', 'signal_label', 'modality', 'retrieval_pass', 'planning_pass', 'execution_ok', 'expected_tools', 'retrieved_tools', 'planned_tools', 'errors']
    with out_csv.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in report['cases']:
            for row in case['rows']:
                writer.writerow({key: json.dumps(row[key]) if isinstance(row.get(key), list) else row.get(key) for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate cross-modality BioSignalAgent sessions.')
    parser.add_argument('--real-manifest', default='/data1/jiahui/biosignal-agent/datasets/processed/real_world_manifest.json')
    parser.add_argument('--dedicated-manifest', default='/data1/jiahui/biosignal-agent/datasets/processed/dedicated_common_manifest.json')
    parser.add_argument('--planner', choices=['rule', 'openrouter'], default='rule')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--retrieved-tool-count', type=int, default=3)
    parser.add_argument('--llm-timeout', type=int, default=20)
    parser.add_argument('--llm-retry-max', type=int, default=1)
    parser.add_argument('--llm-retry-delay', type=float, default=2.0)
    parser.add_argument('--out-json', default='/data1/jiahui/biosignal-agent/outputs/session_eval_rule.json')
    parser.add_argument('--out-csv', default='/data1/jiahui/biosignal-agent/outputs/session_eval_rule.csv')
    args = parser.parse_args()

    records = load_records([Path(args.real_manifest), Path(args.dedicated_manifest)])
    cases = build_default_session_cases(records)
    agent = BioSignalAgentFramework(BioSignalAgentConfig(
        planner=args.planner,
        model=args.model,
        retrieved_tool_count=args.retrieved_tool_count,
        llm_timeout=args.llm_timeout,
        llm_retry_max=args.llm_retry_max,
        llm_retry_delay=args.llm_retry_delay,
        save_traces=True,
    ))
    evaluated = [evaluate_session_case(case, agent) for case in cases]
    rows = [row for case in evaluated for row in case['rows']]
    retrieval_passes = sum(1 for row in rows if row['retrieval_pass'])
    planning_passes = sum(1 for row in rows if row['planning_pass'])
    execution_passes = sum(1 for row in rows if row['execution_ok'])
    report = {
        'planner': args.planner,
        'model': args.model if args.planner == 'openrouter' else None,
        'num_sessions': len(evaluated),
        'num_signal_runs': len(rows),
        'retrieval_accuracy': retrieval_passes / len(rows) if rows else 0.0,
        'planning_accuracy': planning_passes / len(rows) if rows else 0.0,
        'execution_accuracy': execution_passes / len(rows) if rows else 0.0,
        'cases': evaluated,
    }
    write_outputs(report, args.out_json, args.out_csv)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
