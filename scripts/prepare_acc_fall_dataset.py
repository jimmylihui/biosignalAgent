from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FALL_LABELS = set(range(10, 18))


def find_unimib_arrays(raw_dir: Path) -> tuple[Path, Path]:
    data_candidates = list(raw_dir.rglob("acc_data.npy")) + list(raw_dir.rglob("*data*.npy"))
    label_candidates = list(raw_dir.rglob("acc_labels.npy")) + list(raw_dir.rglob("*label*.npy"))
    if not data_candidates or not label_candidates:
        raise FileNotFoundError(
            f"No UniMiB SHAR acc_data.npy/acc_labels.npy files found in {raw_dir}. "
            "Download UniMiB SHAR and place the .npy arrays under --raw-dir."
        )
    return data_candidates[0], label_candidates[0]


def normalize_data(data: np.ndarray) -> np.ndarray:
    if data.ndim == 3 and data.shape[-1] == 3:
        return data.astype(float)
    if data.ndim == 2 and data.shape[1] % 3 == 0:
        return data.reshape(data.shape[0], data.shape[1] // 3, 3).astype(float)
    raise ValueError(f"Unsupported UniMiB acceleration shape: {data.shape}")


def label_value(row: Any) -> int:
    arr = np.asarray(row).ravel()
    return int(arr[0])


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    data_path, labels_path = find_unimib_arrays(Path(args.raw_dir))
    data = normalize_data(np.load(data_path, allow_pickle=True))
    labels = np.load(labels_path, allow_pickle=True)
    rows = []
    counts: dict[str, int] = defaultdict(int)
    out_dir = Path(args.out_dir)
    for idx in range(min(len(data), len(labels))):
        activity_code = label_value(labels[idx])
        binary = "fall" if activity_code in FALL_LABELS else "adl"
        if counts[binary] >= args.max_per_class:
            continue
        mag = np.linalg.norm(data[idx], axis=1)
        out_dir.mkdir(parents=True, exist_ok=True)
        rec_id = f"unimib_{idx:05d}_{binary}_{activity_code}"
        out_csv = out_dir / f"{rec_id}_acc_mag.csv"
        feature_path = out_dir / f"{rec_id}_features.json"
        features = {
            "mean": float(np.nanmean(mag)),
            "std": float(np.nanstd(mag)),
            "max": float(np.nanmax(mag)),
            "min": float(np.nanmin(mag)),
            "range": float(np.nanmax(mag) - np.nanmin(mag)),
            "energy": float(np.nanmean(mag ** 2)),
        }
        pd.DataFrame({"signal": mag}).to_csv(out_csv, index=False)
        feature_path.write_text(json.dumps({"features": features}))
        rows.append({
            "dataset": "unimib_shar_fall",
            "record": rec_id,
            "modality": "acc",
            "path": str(out_csv),
            "feature_path": str(feature_path),
            "sampling_rate": 50.0,
            "duration_s": float(len(mag) / 50.0),
            "activity_code": activity_code,
            "label": binary,
        })
        counts[binary] += 1
        if counts.get("fall", 0) >= args.max_per_class and counts.get("adl", 0) >= args.max_per_class:
            break
    if not rows:
        raise RuntimeError("No UniMiB fall/ADL windows were exported.")
    return {"dataset": "unimib_shar_fall", "records": rows, "num_windows": len(rows), "label_counts": dict(Counter(row["label"] for row in rows))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare UniMiB SHAR fall-vs-ADL accelerometer windows.")
    parser.add_argument("--raw-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/raw/unimib_shar"))
    parser.add_argument("--out-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/acc_fall"))
    parser.add_argument("--manifest", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/acc_fall_manifest.json"))
    parser.add_argument("--max-per-class", type=int, default=20)
    args = parser.parse_args()
    manifest = build_manifest(args)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: manifest[key] for key in ["num_windows", "label_counts"]}, indent=2))


if __name__ == "__main__":
    main()
