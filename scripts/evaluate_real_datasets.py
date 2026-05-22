from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.agent.openrouter_client import DEFAULT_MODEL
from biosignal_agent.evaluation.framework_eval import evaluate_cases, write_eval_outputs
from biosignal_agent.evaluation.planning_cases import DEFAULT_PLANNING_CASES


def load_manifest(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text())
    return payload.get('records', [])


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate planning and execution across real ECG/PPG/BCG records.')
    parser.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/datasets/processed/real_world_manifest.json')
    parser.add_argument('--planner', choices=['rule', 'openrouter'], default='rule')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--retrieved-tool-count', type=int, default=3)
    parser.add_argument('--llm-timeout', type=int, default=20)
    parser.add_argument('--llm-retry-max', type=int, default=1)
    parser.add_argument('--llm-retry-delay', type=float, default=2.0)
    parser.add_argument('--include-ecg', action='store_true', help='Also include MIT-BIH ECG record 100 as a real ECG execution record.')
    parser.add_argument('--out-json', default='/data1/jiahui/biosignal-agent/outputs/real_dataset_framework_eval.json')
    parser.add_argument('--out-csv', default='/data1/jiahui/biosignal-agent/outputs/real_dataset_framework_eval.csv')
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    if args.include_ecg:
        records.append({
            'dataset': 'mitdb',
            'record': '100',
            'modality': 'ecg',
            'path': '/data1/jiahui/biosignal-agent/datasets/processed/mitdb_100_mlii_60s.csv',
            'sampling_rate': 360.0,
            'source_channel': 'MLII',
        })

    all_rows = []
    summaries = []
    for record in records:
        modality = record['modality']
        cases = [case for case in DEFAULT_PLANNING_CASES if case.modality == modality]
        if not cases:
            continue
        report = evaluate_cases(
            cases=cases,
            planner_name=args.planner,
            model=args.model,
            retrieved_tool_count=args.retrieved_tool_count,
            execute=True,
            llm_timeout=args.llm_timeout,
            llm_retry_max=args.llm_retry_max,
            llm_retry_delay=args.llm_retry_delay,
            signal_paths={modality: record['path']},
            sampling_rates={modality: float(record['sampling_rate'])},
        )
        summaries.append({
            'dataset': record.get('dataset'),
            'record': record.get('record'),
            'modality': modality,
            'retrieval_accuracy': report['retrieval_accuracy'],
            'planning_accuracy': report['planning_accuracy'],
            'execution_accuracy': report['execution_accuracy'],
            'planner_backend_counts': report.get('planner_backend_counts'),
        })
        for row in report['cases']:
            row = dict(row)
            row['dataset'] = record.get('dataset')
            row['record'] = record.get('record')
            row['source_channel'] = record.get('source_channel')
            row['signal_path'] = record.get('path')
            row['sampling_rate'] = record.get('sampling_rate')
            all_rows.append(row)

    retrieval_passes = sum(1 for row in all_rows if row['retrieval_pass'])
    planning_passes = sum(1 for row in all_rows if row['planning_pass'])
    execution_passes = sum(1 for row in all_rows if row['execution_ok'])
    backend_counts: dict[str, int] = {}
    for row in all_rows:
        backend_counts[row['planner']] = backend_counts.get(row['planner'], 0) + 1
    final_report = {
        'planner': args.planner,
        'model': args.model if args.planner == 'openrouter' else None,
        'manifest': args.manifest,
        'num_records': len(records),
        'num_case_runs': len(all_rows),
        'retrieval_accuracy': retrieval_passes / len(all_rows) if all_rows else 0.0,
        'planning_accuracy': planning_passes / len(all_rows) if all_rows else 0.0,
        'execution_accuracy': execution_passes / len(all_rows) if all_rows else 0.0,
        'planner_backend_counts': backend_counts,
        'record_summaries': summaries,
        'cases': all_rows,
    }
    write_eval_outputs(final_report, args.out_json, args.out_csv)
    print(json.dumps(final_report, indent=2))


if __name__ == '__main__':
    main()
