from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

UCI_HAR_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
ACTIVITY_NAMES = {
    1: "walking",
    2: "walking_upstairs",
    3: "walking_downstairs",
    4: "sitting",
    5: "standing",
    6: "laying",
}
ACTIVE = {"walking", "walking_upstairs", "walking_downstairs"}


def download(url: str, path: Path, enabled: bool) -> None:
    if path.exists():
        return
    if not enabled:
        raise FileNotFoundError(f"Missing {path}. Re-run with --download or place the UCI HAR zip in --raw-dir.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)


def ensure_dataset(raw_dir: Path, download_enabled: bool) -> Path:
    dataset_dir = raw_dir / "UCI HAR Dataset"
    if dataset_dir.exists():
        return dataset_dir
    zip_path = raw_dir / "uci_har.zip"
    download(UCI_HAR_URL, zip_path, download_enabled)
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(raw_dir)
    nested = raw_dir / "UCI HAR Dataset.zip"
    if nested.exists() and not dataset_dir.exists():
        with zipfile.ZipFile(nested) as archive:
            archive.extractall(raw_dir)
    if not dataset_dir.exists():
        candidates = [path for path in raw_dir.rglob("UCI HAR Dataset") if path.is_dir()]
        if candidates:
            return candidates[0]
        raise FileNotFoundError(f"Could not find extracted UCI HAR directory under {raw_dir}.")
    return dataset_dir


def load_split(dataset_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split_dir = dataset_dir / split
    inertial = split_dir / "Inertial Signals"
    labels = np.loadtxt(split_dir / f"y_{split}.txt", dtype=int)
    subjects = np.loadtxt(split_dir / f"subject_{split}.txt", dtype=int)
    features = np.loadtxt(split_dir / f"X_{split}.txt", dtype=float)
    acc = np.stack([
        np.loadtxt(inertial / f"total_acc_x_{split}.txt", dtype=float),
        np.loadtxt(inertial / f"total_acc_y_{split}.txt", dtype=float),
        np.loadtxt(inertial / f"total_acc_z_{split}.txt", dtype=float),
    ], axis=-1)
    return labels, subjects, features, acc


def select_indices(labels: np.ndarray, max_per_class: int) -> list[int]:
    selected = []
    counts: dict[int, int] = defaultdict(int)
    for idx, label in enumerate(labels.tolist()):
        if counts[label] < max_per_class:
            selected.append(idx)
            counts[label] += 1
        if len(counts) == len(ACTIVITY_NAMES) and all(counts[label] >= max_per_class for label in ACTIVITY_NAMES):
            break
    return selected


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = ensure_dataset(Path(args.raw_dir), args.download)
    out_dir = Path(args.out_dir)
    rows = []
    for split in args.splits:
        labels, subjects, features, acc = load_split(dataset_dir, split)
        for idx in select_indices(labels, args.max_per_class):
            label = ACTIVITY_NAMES[int(labels[idx])]
            mag = np.linalg.norm(acc[idx], axis=1)
            out_dir.mkdir(parents=True, exist_ok=True)
            rec_id = f"{split}_{idx:05d}_{label}"
            signal_path = out_dir / f"uci_har_{rec_id}_acc_mag.csv"
            feature_path = out_dir / f"uci_har_{rec_id}_features.json"
            pd.DataFrame({"signal": mag}).to_csv(signal_path, index=False)
            feature_path.write_text(json.dumps({"features": features[idx].tolist()}))
            rows.append({
                "dataset": "uci_har_activity",
                "record": rec_id,
                "split": split,
                "subject": int(subjects[idx]),
                "modality": "acc",
                "path": str(signal_path),
                "feature_path": str(feature_path),
                "sampling_rate": 50.0,
                "duration_s": float(len(mag) / 50.0),
                "activity_label": label,
                "coarse_activity_label": "active" if label in ACTIVE else "rest",
            })
    return {
        "dataset": "uci_har_activity",
        "records": rows,
        "num_windows": len(rows),
        "label_counts": dict(Counter(row["activity_label"] for row in rows)),
        "coarse_label_counts": dict(Counter(row["coarse_activity_label"] for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a small UCI-HAR accelerometer activity benchmark.")
    parser.add_argument("--raw-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/raw/uci_har"))
    parser.add_argument("--out-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/acc_activity"))
    parser.add_argument("--manifest", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/acc_activity_manifest.json"))
    parser.add_argument("--splits", nargs="*", default=["train", "test"])
    parser.add_argument("--max-per-class", type=int, default=8)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest(args)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: manifest[key] for key in ["num_windows", "label_counts", "coarse_label_counts"]}, indent=2))


if __name__ == "__main__":
    main()
