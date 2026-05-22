from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def validate(path: str | Path) -> dict:
    path = Path(path)
    counts = Counter()
    errors = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({'line': line_no, 'error': f'json decode: {exc}'})
                continue
            task = item.get('task')
            messages = item.get('messages')
            if not task:
                errors.append({'line': line_no, 'error': 'missing task'})
            else:
                counts[task] += 1
            if not isinstance(messages, list) or len(messages) < 2:
                errors.append({'line': line_no, 'error': 'messages must be a list with at least two items'})
                continue
            for idx, message in enumerate(messages):
                if message.get('role') not in {'system', 'user', 'assistant'}:
                    errors.append({'line': line_no, 'error': f'invalid role at message {idx}'})
                if not isinstance(message.get('content'), str) or not message.get('content'):
                    errors.append({'line': line_no, 'error': f'missing content at message {idx}'})
            if messages[-1].get('role') != 'assistant':
                errors.append({'line': line_no, 'error': 'last message must be assistant'})
    return {'path': str(path), 'num_samples': sum(counts.values()), 'task_counts': dict(counts), 'num_errors': len(errors), 'errors': errors[:20]}


def main() -> None:
    parser = argparse.ArgumentParser(description='Validate BioSignalAgent instruction JSONL files.')
    parser.add_argument('jsonl')
    args = parser.parse_args()
    report = validate(args.jsonl)
    print(json.dumps(report, indent=2))
    if report['num_errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
