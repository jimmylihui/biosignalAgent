#!/usr/bin/env python3
"""Create deterministic BioSignalBench train/dev/held-out manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')


def stable_score(case: dict[str, Any], seed: str) -> str:
    key = f"{seed}:{case.get('case_id')}:{case.get('benchmark_task')}:{case.get('modality')}"
    return hashlib.sha256(key.encode()).hexdigest()


def split_cases(cases: list[dict[str, Any]], heldout_frac: float, dev_frac: float, seed: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        key = (str(case.get('benchmark_task')), str(case.get('input_type')))
        groups[key].append(case)
    out = {'train': [], 'dev': [], 'heldout': []}
    for key, rows in groups.items():
        rows = sorted(rows, key=lambda r: stable_score(r, seed))
        n = len(rows)
        held_n = 1 if n >= 5 else 0
        held_n = max(held_n, round(n * heldout_frac)) if n >= 5 else held_n
        dev_n = 1 if n >= 8 else 0
        dev_n = max(dev_n, round(n * dev_frac)) if n >= 8 else dev_n
        if held_n + dev_n >= n:
            dev_n = max(0, n - held_n - 1)
        out['heldout'].extend(rows[:held_n])
        out['dev'].extend(rows[held_n:held_n + dev_n])
        out['train'].extend(rows[held_n + dev_n:])
    for name in out:
        out[name].sort(key=lambda r: str(r.get('case_id')))
    return out


def counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(r.get(field)) for r in rows).items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_v1.jsonl')
    ap.add_argument('--out-dir', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_splits_v1')
    ap.add_argument('--heldout-frac', type=float, default=0.20)
    ap.add_argument('--dev-frac', type=float, default=0.10)
    ap.add_argument('--seed', default='biosignalbench-v1-split-2026-05-25')
    args = ap.parse_args()

    cases = read_jsonl(args.manifest)
    splits = split_cases(cases, args.heldout_frac, args.dev_frac, args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        write_jsonl(out_dir / f'biosignalbench_v1_{name}.jsonl', rows)
    summary: dict[str, Any] = {
        'artifact': 'BioSignalBenchSplits',
        'source_manifest': args.manifest,
        'seed': args.seed,
        'heldout_frac': args.heldout_frac,
        'dev_frac': args.dev_frac,
        'splits': {},
    }
    for name, rows in splits.items():
        summary['splits'][name] = {
            'num_cases': len(rows),
            'task_counts': counts(rows, 'benchmark_task'),
            'input_type_counts': counts(rows, 'input_type'),
            'modality_counts': counts(rows, 'modality'),
        }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
