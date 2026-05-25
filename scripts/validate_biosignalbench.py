#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.evaluation.biosignalbench import load_bench_cases, validate_bench_cases, write_json


def main() -> None:
    ap = argparse.ArgumentParser(description='Validate BioSignalBench JSONL manifest.')
    ap.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_v1.jsonl')
    ap.add_argument('--out-json', default=None)
    args = ap.parse_args()
    report = validate_bench_cases(load_bench_cases(args.manifest))
    if args.out_json:
        write_json(args.out_json, report)
    print(json.dumps(report, indent=2))
    if report['num_errors']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
