from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from scipy import signal as scipy_signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.common import load_csv_signal


def resample_values(values: np.ndarray, points: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("empty signal")
    if len(values) == points:
        return values
    return scipy_signal.resample(values, points)


def render_trace(values: np.ndarray, out_png: Path, out_mask: Path, width: int, height: int, margin: int, variant: str) -> dict[str, Any]:
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin
    y_min = float(np.percentile(values, 1))
    y_max = float(np.percentile(values, 99))
    if y_max == y_min:
        y_max = y_min + 1.0
    clipped = np.clip(values, y_min, y_max)
    xs = np.linspace(margin, width - margin - 1, len(clipped))
    ys = margin + (y_max - clipped) / (y_max - y_min) * (plot_height - 1)
    image = Image.new("RGB", (width, height), "white")
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)
    if variant in {"grid", "color_grid", "artifact"}:
        grid_color = (225, 225, 225) if variant == "grid" else (244, 205, 205)
        for gx in np.linspace(margin, width - margin, 11):
            draw.line([(float(gx), margin), (float(gx), height - margin)], fill=grid_color, width=1)
        for gy in np.linspace(margin, height - margin, 7):
            draw.line([(margin, float(gy)), (width - margin, float(gy))], fill=grid_color, width=1)
    points = list(zip(xs.tolist(), ys.tolist()))
    trace_color = (0, 0, 0)
    if variant == "color_grid":
        trace_color = (28, 96, 185)
    elif variant == "artifact":
        trace_color = (45, 45, 45)
    draw.line(points, fill=trace_color, width=3, joint="curve")
    mask_draw.line(points, fill=255, width=3, joint="curve")
    if variant == "artifact":
        rng = np.random.default_rng(abs(hash(str(out_png))) % (2**32))
        noise = rng.normal(0, 8, size=(height, width, 3))
        arr = np.asarray(image, dtype=float)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr, mode="RGB").filter(ImageFilter.GaussianBlur(radius=0.45))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    if variant == "artifact":
        image.save(out_png, format="JPEG", quality=65)
    else:
        image.save(out_png)
    mask.save(out_mask)
    return {"width": width, "height": height, "margin": margin, "plot_width": plot_width, "plot_height": plot_height, "value_min": y_min, "value_max": y_max}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render clean waveform-image benchmark with known reference signals for digitization eval.")
    parser.add_argument("--source-manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/modality_classifier_manifest.json")
    parser.add_argument("--max-per-modality", type=int, default=4)
    parser.add_argument("--points", type=int, default=760)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=260)
    parser.add_argument("--margin", type=int, default=20)
    parser.add_argument("--out-dir", default="/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark")
    parser.add_argument("--out-json", default="/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark_manifest.json")
    parser.add_argument("--include-modality", action="append", default=None, help="Only render selected modality; repeat for multiple modalities.")
    args = parser.parse_args()

    manifest = json.loads(Path(args.source_manifest).read_text())
    out_dir = Path(args.out_dir)
    counts: dict[str, int] = defaultdict(int)
    include_modalities = {str(item).lower() for item in args.include_modality} if args.include_modality else None
    records = []
    for row in manifest.get("records", []):
        modality = str(row.get("modality", "")).lower()
        if include_modalities is not None and modality not in include_modalities:
            continue
        if not modality or counts[modality] >= args.max_per_modality:
            continue
        try:
            data = load_csv_signal(row["path"], float(row["sampling_rate"]))
            stop = min(len(data.values), max(1, int(float(args.seconds) * float(data.sampling_rate))))
            source_values = data.values[:stop]
            values = resample_values(source_values, args.points)
        except Exception as exc:
            print(json.dumps({"skip": row.get("record"), "error": str(exc)}))
            continue
        variants = ["clean", "grid", "color_grid", "artifact"]
        variant = variants[counts[modality] % len(variants)]
        stem = f"{modality}_{counts[modality]:02d}_{str(row.get('record') or 'record').replace('/', '_')}_{variant}"
        ext = "jpg" if variant == "artifact" else "png"
        image_path = out_dir / "images" / f"{stem}.{ext}"
        reference_path = out_dir / "references" / f"{stem}_reference.csv"
        mask_path = out_dir / "masks" / f"{stem}_mask.png"
        meta = render_trace(values, image_path, mask_path, args.width, args.height, args.margin, variant=variant)
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"signal": np.clip(values, meta["value_min"], meta["value_max"])}).to_csv(reference_path, index=False)
        duration_s = len(source_values) / float(data.sampling_rate)
        digitized_sampling_rate = args.points / duration_s if duration_s > 0 else float(args.points)
        records.append({
            "dataset": "rendered_waveform_digitization",
            "record": stem,
            "source_record": row.get("record"),
            "modality": modality,
            "variant": variant,
            "source_path": row["path"],
            "image_path": str(image_path),
            "reference_path": str(reference_path),
            "mask_path": str(mask_path),
            "sampling_rate": digitized_sampling_rate,
            "duration_s": duration_s,
            "crop_left": args.margin,
            "crop_right": args.margin,
            "crop_top": args.margin,
            "crop_bottom": args.margin,
            "value_min": meta["value_min"],
            "value_max": meta["value_max"],
            "num_points": args.points,
        })
        counts[modality] += 1
    report = {"dataset": "rendered_waveform_digitization", "source_manifest": args.source_manifest, "num_records": len(records), "modality_counts": dict(Counter(r["modality"] for r in records)), "records": records}
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    print(json.dumps({"num_records": report["num_records"], "modality_counts": report["modality_counts"], "out_json": str(out_json)}, indent=2))


if __name__ == "__main__":
    main()
