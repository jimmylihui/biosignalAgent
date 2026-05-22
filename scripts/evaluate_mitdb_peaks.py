from __future__ import annotations

import argparse
import json
from pathlib import Path

import wfdb


BEAT_SYMBOLS = {"N", "L", "R", "A", "a", "J", "S", "V", "F", "e", "j", "E", "/", "f", "Q"}


def evaluate(detected: list[int], truth: list[int], tolerance_samples: int) -> dict:
    matched_truth = set()
    matched_detected = set()
    for i, sample in enumerate(detected):
        candidates = [(abs(sample - ref), j) for j, ref in enumerate(truth) if j not in matched_truth and abs(sample - ref) <= tolerance_samples]
        if candidates:
            _, j = min(candidates)
            matched_truth.add(j)
            matched_detected.add(i)
    tp = len(matched_detected)
    fp = len(detected) - tp
    fn = len(truth) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", default="100")
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/raw/mitdb"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--tolerance-ms", type=float, default=100.0)
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    detected = report["tool_calls"][1]["result"]["r_peak_indices"]
    header = wfdb.rdheader(str(args.raw_dir / args.record))
    sampto = int(args.seconds * header.fs)
    annotation = wfdb.rdann(str(args.raw_dir / args.record), "atr", sampto=sampto)
    truth = [int(sample) for sample, symbol in zip(annotation.sample, annotation.symbol) if symbol in BEAT_SYMBOLS]
    tolerance_samples = int(args.tolerance_ms / 1000.0 * header.fs)
    metrics = {
        "record": f"mitdb/{args.record}",
        "duration_seconds": args.seconds,
        "sampling_rate": float(header.fs),
        "tolerance_ms": args.tolerance_ms,
        "detected_peaks": len(detected),
        "reference_beats": len(truth),
        **evaluate(detected, truth, tolerance_samples),
    }
    output = json.dumps(metrics, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    print(output)


if __name__ == "__main__":
    main()
