from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent import BasicBioSignalAgent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to a CSV containing a signal column or one numeric column.")
    parser.add_argument("--modality", required=True, choices=["ecg", "ppg", "bcg"])
    parser.add_argument("--sampling-rate", required=True, type=float)
    parser.add_argument("--column", default=None)
    args = parser.parse_args()
    report = BasicBioSignalAgent().run_report(args.csv, args.modality, args.sampling_rate, args.column)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
