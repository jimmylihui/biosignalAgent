#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.evaluation.biosignalbench import read_json, validate_tool_universe, write_json


def main() -> None:
    ap = argparse.ArgumentParser(description='Validate frozen BioSignalToolUniverse artifact.')
    ap.add_argument('--universe', default='/data1/jiahui/biosignal-agent/outputs/biosignal_tool_universe_v1.json')
    ap.add_argument('--out-json', default=None)
    args = ap.parse_args()
    report = validate_tool_universe(read_json(args.universe))
    if args.out_json:
        write_json(args.out_json, report)
    print(json.dumps(report, indent=2))
    if report['num_errors']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
