from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.agent.planning_agent import PlanningBioSignalAgent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--sampling-rate", required=True, type=float)
    parser.add_argument("--column", default=None)
    parser.add_argument("--fallback-modality", choices=["ecg", "ppg", "bcg"], default=None)
    args = parser.parse_args()
    report = PlanningBioSignalAgent().run(args.question, args.csv, args.sampling_rate, args.column, args.fallback_modality)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
