from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.agent.llm_agent import OpenRouterBioSignalAgent
from biosignal_agent.agent.openrouter_client import DEFAULT_MODEL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--question', required=True)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--sampling-rate', required=True, type=float)
    parser.add_argument('--column', default=None)
    parser.add_argument('--fallback-modality', choices=['ecg', 'ppg', 'bcg'], default=None)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--llm-report', action='store_true', help='Use OpenRouter for final report generation. Default uses deterministic reporting.')
    parser.add_argument('--retrieved-tool-count', type=int, default=5, help='Number of tool schemas to retrieve for the planner prompt.')
    args = parser.parse_args()
    agent = OpenRouterBioSignalAgent(model=args.model, use_llm_report=args.llm_report, retrieved_tool_count=args.retrieved_tool_count)
    report = agent.run(args.question, args.csv, args.sampling_rate, args.column, args.fallback_modality)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
