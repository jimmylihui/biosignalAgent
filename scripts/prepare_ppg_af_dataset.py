from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

AF_URL = "https://zenodo.org/records/6807403/files/mimic_perform_af_csv.zip?download=1"
NON_AF_URL = "https://zenodo.org/records/6807403/files/mimic_perform_non_af_csv.zip?download=1"


def download(url: str, path: Path, enabled: bool) -> None:
    if path.exists():
        return
    if not enabled:
        raise FileNotFoundError(f"Missing {path}. Re-run with --download or place MIMIC PERform AF CSV zips in --raw-dir.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    marker = out_dir / ".extracted"
    if marker.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(out_dir)
    marker.write_text("ok")


def find_signal_column(frame: pd.DataFrame) -> str:
    lower = {str(col).lower(): col for col in frame.columns}
    for key in ["ppg", "pleth", "photoplethysmogram"]:
        for lower_name, original in lower.items():
            if key in lower_name:
                return original
    numeric_cols = [col for col in frame.columns if pd.api.types.is_numeric_dtype(frame[col])]
    if not numeric_cols:
        raise ValueError("No numeric PPG-like column found")
    return numeric_cols[0]


def csv_records(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.csv") if path.is_file())


def export_records(paths: list[Path], label: str, max_records: int, seconds: float, out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for csv_path in paths:
        if len(rows) >= max_records:
            break
        try:
            frame = pd.read_csv(csv_path)
            col = find_signal_column(frame)
            values = pd.to_numeric(frame[col], errors="coerce").dropna().to_numpy(dtype=float)
        except Exception:
            continue
        sampling_rate = 125.0
        values = values[: int(seconds * sampling_rate)]
        if len(values) < int(20 * sampling_rate):
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        record = csv_path.stem.replace(" ", "_")
        out_csv = out_dir / f"mimic_perform_{label}_{record}_ppg_{int(len(values)/sampling_rate)}s.csv"
        pd.DataFrame({"signal": values}).to_csv(out_csv, index=False)
        rows.append({
            "dataset": "mimic_perform_af",
            "record": f"{label}_{record}",
            "modality": "ppg",
            "path": str(out_csv),
            "sampling_rate": sampling_rate,
            "duration_s": float(len(values) / sampling_rate),
            "source_file": str(csv_path),
            "source_channel": str(col),
            "label": "af" if label == "af" else "non_af",
        })
    return rows


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    raw_dir = Path(args.raw_dir)
    af_zip = raw_dir / "mimic_perform_af_csv.zip"
    non_af_zip = raw_dir / "mimic_perform_non_af_csv.zip"
    download(AF_URL, af_zip, args.download)
    download(NON_AF_URL, non_af_zip, args.download)
    af_dir = raw_dir / "af_csv"
    non_af_dir = raw_dir / "non_af_csv"
    extract_zip(af_zip, af_dir)
    extract_zip(non_af_zip, non_af_dir)
    rows = []
    rows.extend(export_records(csv_records(af_dir), "af", args.max_per_class, args.seconds, Path(args.out_dir)))
    rows.extend(export_records(csv_records(non_af_dir), "non_af", args.max_per_class, args.seconds, Path(args.out_dir)))
    if not rows:
        raise RuntimeError("No usable MIMIC PERform AF CSV records found after extraction.")
    return {"dataset": "mimic_perform_af", "records": rows, "num_records": len(rows), "label_counts": dict(Counter(row["label"] for row in rows))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MIMIC PERform AF PPG windows with AF/non-AF labels.")
    parser.add_argument("--raw-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/raw/mimic_perform_af"))
    parser.add_argument("--out-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/ppg_af"))
    parser.add_argument("--manifest", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/ppg_af_manifest.json"))
    parser.add_argument("--max-per-class", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest(args)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: manifest[key] for key in ["num_records", "label_counts"]}, indent=2))


if __name__ == "__main__":
    main()
