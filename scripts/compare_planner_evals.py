from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_cases(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    return {row['case_id']: row for row in payload.get('cases', [])}


def compare(rule_path: str | Path, candidate_path: str | Path) -> dict[str, Any]:
    rule_cases = load_cases(rule_path)
    candidate_cases = load_cases(candidate_path)
    rows = []
    for case_id in sorted(set(rule_cases) | set(candidate_cases)):
        rule = rule_cases.get(case_id, {})
        candidate = candidate_cases.get(case_id, {})
        expected = candidate.get('expected_tools') or rule.get('expected_tools') or []
        row = {
            'case_id': case_id,
            'modality': candidate.get('modality') or rule.get('modality'),
            'question': candidate.get('question') or rule.get('question'),
            'expected_tools': expected,
            'rule_tools': rule.get('planned_tools', []),
            'candidate_tools': candidate.get('planned_tools', []),
            'candidate_planner': candidate.get('planner'),
            'candidate_planning_pass': bool(candidate.get('planning_pass')),
            'candidate_missing': candidate.get('missing_from_plan', []),
            'candidate_unexpected': candidate.get('unexpected_tools', []),
            'candidate_error': candidate.get('planner_error'),
        }
        row['matches_rule'] = set(row['rule_tools']) == set(row['candidate_tools'])
        rows.append(row)
    backend_counts = Counter(row.get('candidate_planner') for row in rows)
    fail_rows = [row for row in rows if not row['candidate_planning_pass']]
    return {
        'rule_path': str(rule_path),
        'candidate_path': str(candidate_path),
        'num_cases': len(rows),
        'candidate_planning_accuracy': (len(rows) - len(fail_rows)) / len(rows) if rows else 0.0,
        'backend_counts': dict(backend_counts),
        'num_disagreements_with_rule': sum(1 for row in rows if not row['matches_rule']),
        'num_candidate_failures': len(fail_rows),
        'failures': fail_rows,
        'rows': rows,
    }


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'case_id', 'modality', 'candidate_planner', 'candidate_planning_pass', 'matches_rule',
        'expected_tools', 'rule_tools', 'candidate_tools', 'candidate_missing', 'candidate_unexpected', 'candidate_error', 'question'
    ]
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key)) if isinstance(row.get(key), list) else row.get(key) for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare rule planner and candidate planner eval JSON files.')
    parser.add_argument('--rule-json', default='/data1/jiahui/biosignal-agent/outputs/framework_eval_rule_common_modalities.json')
    parser.add_argument('--candidate-json', required=True)
    parser.add_argument('--out-json', default='/data1/jiahui/biosignal-agent/outputs/planner_comparison.json')
    parser.add_argument('--out-csv', default='/data1/jiahui/biosignal-agent/outputs/planner_comparison.csv')
    args = parser.parse_args()
    report = compare(args.rule_json, args.candidate_json)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    write_csv(report['rows'], args.out_csv)
    print(json.dumps({key: report[key] for key in ['num_cases', 'candidate_planning_accuracy', 'backend_counts', 'num_disagreements_with_rule', 'num_candidate_failures']}, indent=2))


if __name__ == '__main__':
    main()
