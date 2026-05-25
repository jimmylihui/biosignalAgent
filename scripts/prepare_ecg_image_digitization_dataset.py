from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
DEFAULT_RENDERED_MANIFEST = "/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark_manifest.json"
DEFAULT_OUT = "/data1/jiahui/biosignal-agent/datasets/processed/ecg_image_digitization_manifest.json"


def numeric_range(csv_path: str | Path) -> tuple[float | None, float | None]:
    try:
        frame = pd.read_csv(csv_path)
        col = "signal" if "signal" in frame.columns else frame.select_dtypes("number").columns[-1]
        values = frame[col].to_numpy(dtype=float)
        return float(pd.Series(values).quantile(0.01)), float(pd.Series(values).quantile(0.99))
    except Exception:
        return None, None


def from_rendered_manifest(path: str | Path, source_name: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    records = []
    for row in payload.get("records", []):
        if str(row.get("modality", "")).lower() != "ecg":
            continue
        record = dict(row)
        record["dataset"] = source_name
        record["source_dataset"] = row.get("dataset", payload.get("dataset", "rendered_waveform_digitization"))
        record["task"] = "ecg_image_digitization"
        record["lead"] = row.get("lead", "unknown")
        records.append(record)
    return records


def find_match(stem: str, candidates: dict[str, Path], suffixes: list[str]) -> Path | None:
    keys = [stem] + [f"{stem}{suffix}" for suffix in suffixes]
    for key in keys:
        if key in candidates:
            return candidates[key]
    return None


def from_raw_dir(raw_dir: str | Path, source_name: str, sampling_rate: float | None, default_crop: int) -> list[dict[str, Any]]:
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"raw dir does not exist: {root}")
    images = [path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS and "mask" not in path.stem.lower()]
    csvs = {path.stem: path for path in root.rglob("*.csv")}
    masks = {path.stem: path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS and "mask" in path.stem.lower()}
    records = []
    for image in sorted(images):
        reference = find_match(image.stem, csvs, ["_reference", "_signal", "_waveform"])
        mask = find_match(image.stem, masks, ["_mask", "_segmentation"])
        if reference is None:
            continue
        value_min, value_max = numeric_range(reference)
        records.append({
            "dataset": source_name,
            "source_dataset": source_name,
            "task": "ecg_image_digitization",
            "record": image.stem,
            "modality": "ecg",
            "lead": "unknown",
            "variant": "raw_dir",
            "image_path": str(image),
            "reference_path": str(reference),
            "mask_path": str(mask) if mask else None,
            "sampling_rate": float(sampling_rate) if sampling_rate else None,
            "duration_s": None,
            "crop_left": default_crop,
            "crop_right": default_crop,
            "crop_top": default_crop,
            "crop_bottom": default_crop,
            "value_min": value_min,
            "value_max": value_max,
            "num_points": None,
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ECG image digitization manifest from rendered ECG images or a local ECG image dataset directory.")
    parser.add_argument("--rendered-manifest", default=DEFAULT_RENDERED_MANIFEST)
    parser.add_argument("--raw-dir", default=None, help="Optional local ECG image dataset directory with images plus matching CSV references and optional masks.")
    parser.add_argument("--source-name", default="ecg_image_digitization_rendered")
    parser.add_argument("--sampling-rate", type=float, default=None, help="Sampling rate for raw-dir records if metadata are not encoded elsewhere.")
    parser.add_argument("--default-crop", type=int, default=20)
    parser.add_argument("--out-json", default=DEFAULT_OUT)
    args = parser.parse_args()

    records = []
    if args.rendered_manifest and Path(args.rendered_manifest).exists():
        records.extend(from_rendered_manifest(args.rendered_manifest, args.source_name))
    if args.raw_dir:
        records.extend(from_raw_dir(args.raw_dir, args.source_name, args.sampling_rate, args.default_crop))
    seen = set()
    unique = []
    for row in records:
        key = (row.get("image_path"), row.get("reference_path"), row.get("lead"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    if not unique:
        raise SystemExit(
            "No ECG image digitization records found. Use --rendered-manifest after running render_waveform_digitization_benchmark.py, "
            "or place ECG images with matching CSV references under --raw-dir."
        )
    report = {
        "dataset": "ecg_image_digitization",
        "source_name": args.source_name,
        "num_records": len(unique),
        "variant_counts": dict(Counter(row.get("variant") for row in unique)),
        "mask_records": sum(1 for row in unique if row.get("mask_path")),
        "records": unique,
        "notes": [
            "Rendered records include masks and can train segmentation models directly.",
            "Raw ECG-Image-Kit/PhysioNet-style records need matching waveform CSV references; masks are optional unless training segmentation models.",
        ],
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ["num_records", "variant_counts", "mask_records"]}, indent=2))


if __name__ == "__main__":
    main()
