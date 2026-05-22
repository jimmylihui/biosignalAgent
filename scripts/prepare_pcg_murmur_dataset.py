from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from scipy.io import wavfile

BASE_URL = "https://physionet.org/files/challenge-2016/1.0.0"
DEFAULT_RECORDS = []


def download_file(url: str, path: Path, download: bool) -> Path:
    if path.exists():
        return path
    if not download:
        raise FileNotFoundError(f"Missing {path}. Re-run with --download.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return path


def load_reference(training_dir: Path, download: bool) -> dict[str, str]:
    ref = download_file(f"{BASE_URL}/training-a/REFERENCE.csv", training_dir / "REFERENCE.csv", download)
    labels = {}
    for line in ref.read_text(errors="replace").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        labels[parts[0]] = "abnormal" if parts[1] == "1" else "normal" if parts[1] == "-1" else "unknown"
    return labels


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    training_dir = Path(args.raw_dir) / args.training_subset
    labels = load_reference(training_dir, args.download)
    rows = []
    out_dir = Path(args.out_dir)
    records = list(args.records)
    if not records:
        normals = [record for record, label in labels.items() if label == "normal"][: args.max_per_class]
        abnormals = [record for record, label in labels.items() if label == "abnormal"][: args.max_per_class]
        records = normals + abnormals
    for record in records:
        wav_path = download_file(f"{BASE_URL}/{args.training_subset}/{record}.wav", training_dir / f"{record}.wav", args.download)
        sampling_rate, values = wavfile.read(wav_path)
        if values.ndim > 1:
            values = values[:, 0]
        if args.seconds is not None:
            values = values[: int(args.seconds * sampling_rate)]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"challenge2016_{record}_pcg_{len(values) / sampling_rate:.0f}s.csv"
        pd.DataFrame({"signal": values.astype(float)}).to_csv(out_csv, index=False)
        rows.append({
            "dataset": "physionet_cinc2016_pcg",
            "record": record,
            "training_subset": args.training_subset,
            "duration_s": float(len(values) / sampling_rate),
            "modality": "pcg",
            "path": str(out_csv),
            "sampling_rate": float(sampling_rate),
            "source_channel": "PCG",
            "label": labels.get(record, "unknown"),
        })
    rows = [row for row in rows if row["label"] != "unknown"]
    return {"dataset": "physionet_cinc2016_pcg", "records": rows, "num_records": len(rows), "label_counts": dict(Counter(row["label"] for row in rows))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PhysioNet/CinC 2016 PCG windows with normal/abnormal labels.")
    parser.add_argument("--raw-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/raw/dedicated_common/challenge-2016"))
    parser.add_argument("--out-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/pcg_murmur"))
    parser.add_argument("--manifest", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/pcg_murmur_manifest.json"))
    parser.add_argument("--training-subset", default="training-a")
    parser.add_argument("--records", nargs="*", default=DEFAULT_RECORDS)
    parser.add_argument("--max-per-class", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest(args)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: manifest[key] for key in ["num_records", "label_counts"]}, indent=2))


if __name__ == "__main__":
    main()
