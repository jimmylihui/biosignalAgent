from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.pcg_tools import PCG_detect_heart_sounds, PCG_screen_murmur_proxy, PCG_segment_s1_s2_proxy


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(1 for row in rows if row["truth"] == "abnormal" and row["prediction"] == "abnormal")
    tn = sum(1 for row in rows if row["truth"] == "normal" and row["prediction"] == "normal")
    fp = sum(1 for row in rows if row["truth"] == "normal" and row["prediction"] == "abnormal")
    fn = sum(1 for row in rows if row["truth"] == "abnormal" and row["prediction"] == "normal")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(rows) if rows else 0.0
    return {"true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn, "precision": precision, "recall_sensitivity": recall, "specificity": specificity, "f1": f1, "accuracy": accuracy}


def evaluate(manifest_path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text())
    rows = []
    for record in manifest.get("records", []):
        sounds = PCG_detect_heart_sounds(record["path"], float(record["sampling_rate"]), column=None)
        murmur = PCG_screen_murmur_proxy(record["path"], float(record["sampling_rate"]), column=None)
        segment = PCG_segment_s1_s2_proxy(record["path"], float(record["sampling_rate"]), column=None)
        prediction = "abnormal" if murmur.get("murmur_risk") == "possible_murmur_proxy" else "normal"
        rows.append({
            "record": record["record"],
            "truth": record["label"],
            "prediction": prediction,
            "murmur_risk": murmur.get("murmur_risk"),
            "murmur_proxy_score": murmur.get("murmur_proxy_score"),
            "high_frequency_ratio": murmur.get("high_frequency_ratio"),
            "continuous_sound_fraction": murmur.get("continuous_sound_fraction"),
            "num_sounds": sounds.get("num_sounds"),
            "heart_rate_bpm": sounds.get("heart_rate_bpm"),
            "num_s1": segment.get("num_s1"),
            "num_s2": segment.get("num_s2"),
            "sound_error": sounds.get("error"),
            "murmur_error": murmur.get("error"),
            "segment_error": segment.get("error"),
        })
    return {"manifest": str(manifest_path), "num_records": len(rows), "truth_counts": dict(Counter(row["truth"] for row in rows)), "prediction_counts": dict(Counter(row["prediction"] for row in rows)), "metrics": metrics(rows), "rows": rows}


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PCG murmur/normal-abnormal proxy on PhysioNet/CinC 2016 labels.")
    parser.add_argument("--manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/pcg_murmur_manifest.json")
    parser.add_argument("--out-json", default="/data1/jiahui/biosignal-agent/outputs/pcg_murmur_eval.json")
    parser.add_argument("--out-csv", default="/data1/jiahui/biosignal-agent/outputs/pcg_murmur_eval.csv")
    args = parser.parse_args()
    report = evaluate(args.manifest)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    write_csv(report["rows"], args.out_csv)
    print(json.dumps({key: report[key] for key in ["num_records", "truth_counts", "prediction_counts", "metrics"]}, indent=2))


if __name__ == "__main__":
    main()
