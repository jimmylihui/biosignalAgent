from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.training.trace_dataset import export_trace_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Export BioSignalAgent traces as JSONL training samples.")
    parser.add_argument("--trace-dir", default="/data1/jiahui/biosignal-agent/outputs/traces")
    parser.add_argument("--out", default="/data1/jiahui/biosignal-agent/outputs/biosignal_trace_sft.jsonl")
    parser.add_argument("--planning-only", action="store_true", help="Only export tool-planning samples.")
    args = parser.parse_args()
    counts = export_trace_dataset(args.trace_dir, args.out, include_reports=not args.planning_only)
    print(json.dumps({"output": args.out, **counts}, indent=2))


if __name__ == "__main__":
    main()
