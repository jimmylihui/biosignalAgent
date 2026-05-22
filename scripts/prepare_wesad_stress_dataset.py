from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LABELS = {1: "non_stress", 2: "stress"}
WRIST_FS = {"EDA": 4.0, "BVP": 64.0, "ACC": 32.0, "TEMP": 4.0}
CHEST_FS = 700.0


def find_subject_files(raw_dir: Path) -> list[Path]:
    return sorted(raw_dir.glob("S*/S*.pkl")) + sorted(raw_dir.glob("S*.pkl"))


def load_subject(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def majority_label(labels: np.ndarray, start_s: float, duration_s: float) -> str | None:
    start = int(start_s * CHEST_FS)
    stop = int((start_s + duration_s) * CHEST_FS)
    segment = labels[start:stop]
    segment = segment[np.isin(segment, list(LABELS))]
    if len(segment) == 0:
        return None
    value = int(Counter(segment.tolist()).most_common(1)[0][0])
    return LABELS.get(value)


def candidate_windows(labels: np.ndarray, duration_s: float, step_s: float, max_per_class: int) -> list[tuple[float, str]]:
    total_s = len(labels) / CHEST_FS
    selected = []
    counts: dict[str, int] = defaultdict(int)
    start = 0.0
    while start + duration_s <= total_s:
        label = majority_label(labels, start, duration_s)
        if label in {"stress", "non_stress"} and counts[label] < max_per_class:
            selected.append((start, label))
            counts[label] += 1
        if counts.get("stress", 0) >= max_per_class and counts.get("non_stress", 0) >= max_per_class:
            break
        start += step_s
    return selected


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    raw_dir = Path(args.raw_dir)
    subject_files = find_subject_files(raw_dir)
    if not subject_files:
        raise FileNotFoundError(
            f"No WESAD subject .pkl files found in {raw_dir}. Download WESAD manually and place S*/S*.pkl there."
        )
    rows = []
    out_dir = Path(args.out_dir)
    for subject_path in subject_files[: args.max_subjects]:
        payload = load_subject(subject_path)
        labels = np.asarray(payload["label"], dtype=int).ravel()
        eda = np.asarray(payload["signal"]["wrist"]["EDA"], dtype=float).ravel()
        subject = subject_path.stem
        for start_s, label in candidate_windows(labels, args.window_seconds, args.step_seconds, args.max_per_class_per_subject):
            start = int(start_s * WRIST_FS["EDA"])
            stop = int((start_s + args.window_seconds) * WRIST_FS["EDA"])
            values = eda[start:stop]
            if len(values) < int(args.window_seconds * WRIST_FS["EDA"] * 0.8):
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            out_csv = out_dir / f"wesad_{subject}_{int(start_s):05d}_{label}_eda.csv"
            pd.DataFrame({"signal": values}).to_csv(out_csv, index=False)
            rows.append({
                "dataset": "wesad_stress",
                "record": f"{subject}_{int(start_s):05d}",
                "subject": subject,
                "modality": "eda",
                "path": str(out_csv),
                "sampling_rate": WRIST_FS["EDA"],
                "duration_s": float(len(values) / WRIST_FS["EDA"]),
                "label": label,
                "window_start_s": start_s,
                "source_channel": "wrist_EDA",
            })
    if not rows:
        raise RuntimeError("WESAD files were found, but no stress/non-stress windows were exported.")
    return {"dataset": "wesad_stress", "records": rows, "num_windows": len(rows), "label_counts": dict(Counter(row["label"] for row in rows))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare WESAD wrist-EDA stress/non-stress windows.")
    parser.add_argument("--raw-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/raw/wesad"))
    parser.add_argument("--out-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/wesad_stress"))
    parser.add_argument("--manifest", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/wesad_stress_manifest.json"))
    parser.add_argument("--max-subjects", type=int, default=2)
    parser.add_argument("--max-per-class-per-subject", type=int, default=8)
    parser.add_argument("--window-seconds", type=float, default=60.0)
    parser.add_argument("--step-seconds", type=float, default=60.0)
    args = parser.parse_args()
    manifest = build_manifest(args)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: manifest[key] for key in ["num_windows", "label_counts"]}, indent=2))


if __name__ == "__main__":
    main()
