from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.modality_tools import Signal_classify_modality


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify an unknown one-dimensional biosignal CSV into a likely modality.")
    parser.add_argument("--csv", required=True, help="Input CSV path with one numeric column or a named signal column.")
    parser.add_argument("--sampling-rate", type=float, required=True)
    parser.add_argument("--column", default=None)
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    result = Signal_classify_modality(args.csv, args.sampling_rate, column=args.column)
    text = json.dumps(result, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
