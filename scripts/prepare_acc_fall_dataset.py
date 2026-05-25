from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FALL_LABELS = set(range(10, 18))
FALL_NAME_PREFIXES = ("falling", "syncope")


def find_unimib_arrays(raw_dir: Path) -> tuple[Path, Path]:
    data_candidates = list(raw_dir.rglob("acc_data.npy")) + list(raw_dir.rglob("*data*.npy"))
    label_candidates = list(raw_dir.rglob("acc_labels.npy")) + list(raw_dir.rglob("*label*.npy"))
    if not data_candidates or not label_candidates:
        raise FileNotFoundError(f"No UniMiB SHAR acc_data.npy/acc_labels.npy files found in {raw_dir}.")
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


def feature_summary(arr: np.ndarray, sampling_rate: float = 50.0) -> dict[str, float]:
    mag = np.linalg.norm(arr[:, :3], axis=1)
    jerk = np.diff(mag, prepend=mag[0]) * sampling_rate
    return {
        "mag_mean": float(np.nanmean(mag)),
        "mag_std": float(np.nanstd(mag)),
        "mag_max": float(np.nanmax(mag)),
        "mag_min": float(np.nanmin(mag)),
        "mag_range": float(np.nanmax(mag) - np.nanmin(mag)),
        "mag_energy": float(np.nanmean(mag ** 2)),
        "jerk_p95": float(np.nanpercentile(np.abs(jerk), 95)),
        "axis_std_sum": float(np.nanstd(arr[:, 0]) + np.nanstd(arr[:, 1]) + np.nanstd(arr[:, 2])),
    }


def write_record(out_dir: Path, rec_id: str, arr: np.ndarray, metadata: dict[str, Any]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{rec_id}_acc.csv"
    feature_path = out_dir / f"{rec_id}_features.json"
    frame = pd.DataFrame(arr[:, :3], columns=["acc_x", "acc_y", "acc_z"])
    frame["acc_mag"] = np.linalg.norm(arr[:, :3], axis=1)
    frame.to_csv(out_csv, index=False)
    feature_path.write_text(json.dumps({"features": feature_summary(arr), "metadata": metadata}, indent=2))
    return {
        "dataset": "unimib_shar_fall",
        "record": rec_id,
        "modality": "acc",
        "path": str(out_csv),
        "feature_path": str(feature_path),
        "sampling_rate": 50.0,
        "duration_s": float(len(arr) / 50.0),
        **metadata,
    }


def iter_processed_csv(raw_dir: Path):
    for path in sorted(raw_dir.rglob("*.csv")):
        match = re.match(r"User(?P<user>\d+)_(?P<activity>.+)_(?P<trial>\d+)\.csv$", path.name)
        if not match:
            continue
        df = pd.read_csv(path)
        lower = {c.lower(): c for c in df.columns}
        cols = [lower.get("acc_x"), lower.get("acc_y"), lower.get("acc_z")]
        if not all(cols):
            continue
        activity = match.group("activity")
        label = "fall" if activity.lower().startswith(FALL_NAME_PREFIXES) else "adl"
        yield path, df[cols].to_numpy(dtype=float), {
            "activity_name": activity,
            "activity_code": int(df["activity"].iloc[0]) if "activity" in df else None,
            "subject_id": f"User{match.group('user')}",
            "trial": int(match.group("trial")),
            "label": label,
        }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    rows = []
    counts: dict[str, int] = defaultdict(int)

    csv_records = list(iter_processed_csv(raw_dir))
    if csv_records:
        for path, arr, metadata in csv_records:
            binary = metadata["label"]
            if args.max_per_class and counts[binary] >= args.max_per_class:
                continue
            rec_id = path.stem
            rows.append(write_record(out_dir, rec_id, arr, metadata))
            counts[binary] += 1
        return {"dataset": "unimib_shar_fall", "source": "processed_csv", "records": rows, "num_windows": len(rows), "label_counts": dict(Counter(row["label"] for row in rows))}

    data_path, labels_path = find_unimib_arrays(raw_dir)
    data = normalize_data(np.load(data_path, allow_pickle=True))
    labels = np.load(labels_path, allow_pickle=True)
    for idx in range(min(len(data), len(labels))):
        activity_code = label_value(labels[idx])
        binary = "fall" if activity_code in FALL_LABELS else "adl"
        if args.max_per_class and counts[binary] >= args.max_per_class:
            continue
        rec_id = f"unimib_{idx:05d}_{binary}_{activity_code}"
        metadata = {"activity_code": activity_code, "activity_name": None, "subject_id": None, "trial": None, "label": binary}
        rows.append(write_record(out_dir, rec_id, data[idx], metadata))
        counts[binary] += 1
    if not rows:
        raise RuntimeError("No UniMiB fall/ADL windows were exported.")
    return {"dataset": "unimib_shar_fall", "source": "npy_arrays", "records": rows, "num_windows": len(rows), "label_counts": dict(Counter(row["label"] for row in rows))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare UniMiB SHAR fall-vs-ADL accelerometer windows.")
    parser.add_argument("--raw-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/raw/unimib_shar"))
    parser.add_argument("--out-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/acc_fall"))
    parser.add_argument("--manifest", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/acc_fall_manifest.json"))
    parser.add_argument("--max-per-class", type=int, default=0, help="0 keeps all available windows.")
    args = parser.parse_args()
    manifest = build_manifest(args)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: manifest[key] for key in ["source", "num_windows", "label_counts"] if key in manifest}, indent=2))


if __name__ == "__main__":
    main()
