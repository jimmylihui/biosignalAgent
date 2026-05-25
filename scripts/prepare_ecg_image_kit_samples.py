from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_ROOT = "/data1/jiahui/biosignal-agent/external/ecg-image-kit/sample-data/ecg-images"
DEFAULT_OUT = "/data1/jiahui/biosignal-agent/datasets/processed/ecg_image_kit_samples_manifest.json"


def read_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        sample = handle.read(2048)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        rows = csv.DictReader(handle, dialect=dialect)
        return {row.get("fname", ""): row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare image-only manifest from ECG-Image-Kit sample images for digitizer smoke inference.")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--include-full", action="store_true", help="Include full-page sample images; default focuses on sample-segments.")
    parser.add_argument("--sampling-rate", type=float, default=500.0)
    parser.add_argument("--default-crop", type=int, default=0)
    parser.add_argument("--out-json", default=DEFAULT_OUT)
    args = parser.parse_args()

    root = Path(args.root)
    metadata = read_metadata(root / "ecg-samples-metadata.csv")
    candidates = []
    if args.include_full:
        candidates.extend(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    segment_dir = root / "sample-segments"
    if segment_dir.exists():
        candidates.extend(path for path in segment_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    records = []
    for path in sorted(candidates)[: args.limit]:
        meta = metadata.get(path.name, {})
        records.append({
            "dataset": "ecg_image_kit_samples",
            "task": "ecg_image_digitization_smoke",
            "record": path.stem,
            "modality": "ecg",
            "variant": "sample_segment" if path.parent.name == "sample-segments" else "sample_full_page",
            "image_path": str(path),
            "reference_path": None,
            "mask_path": None,
            "sampling_rate": float(args.sampling_rate),
            "crop_left": int(args.default_crop),
            "crop_right": int(args.default_crop),
            "crop_top": int(args.default_crop),
            "crop_bottom": int(args.default_crop),
            "value_min": None,
            "value_max": None,
            "metadata": meta,
        })
    if not records:
        raise SystemExit(f"No ECG-Image-Kit sample images found under {root}")
    report = {
        "dataset": "ecg_image_kit_samples",
        "root": str(root),
        "num_records": len(records),
        "variant_counts": dict(Counter(row["variant"] for row in records)),
        "has_reference_waveforms": False,
        "records": records,
        "notes": ["Image-only smoke manifest: no ground-truth waveform is included in ECG-Image-Kit sample-data/time-series directory."],
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"num_records": report["num_records"], "variant_counts": report["variant_counts"], "out_json": str(out)}, indent=2))


if __name__ == "__main__":
    main()
