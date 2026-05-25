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


DEFAULT_VARIANTS = [
    "clean",
    "grid",
    "color_grid",
    "artifact",
    "dark_theme",
    "thin_line",
    "thick_line",
    "lowres_jpeg",
    "axes_text",
    "multi_trace",
    "multi_panel",
    "multi_panel_multitrace",
]


def _rng_for(*parts: object) -> np.random.Generator:
    seed = abs(hash("::".join(str(part) for part in parts))) % (2**32)
    return np.random.default_rng(seed)


def _normalize_to_points(values: np.ndarray, width: int, height: int, margin: int) -> tuple[list[tuple[float, float]], float, float]:
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin
    y_min = float(np.percentile(values, 1))
    y_max = float(np.percentile(values, 99))
    if y_max == y_min:
        y_max = y_min + 1.0
    clipped = np.clip(values, y_min, y_max)
    xs = np.linspace(margin, width - margin - 1, len(clipped))
    ys = margin + (y_max - clipped) / (y_max - y_min) * (plot_height - 1)
    return list(zip(xs.tolist(), ys.tolist())), y_min, y_max


def _draw_grid(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: tuple[int, int, int], nx: int = 11, ny: int = 7) -> None:
    left, top, right, bottom = box
    for gx in np.linspace(left, right, nx):
        draw.line([(float(gx), top), (float(gx), bottom)], fill=color, width=1)
    for gy in np.linspace(top, bottom, ny):
        draw.line([(left, float(gy)), (right, float(gy))], fill=color, width=1)


def _panel_points(values: np.ndarray, panel_box: tuple[int, int, int, int], margin: int, y_min: float | None = None, y_max: float | None = None) -> tuple[list[tuple[float, float]], float, float]:
    left, top, right, bottom = panel_box
    panel_width = right - left
    panel_height = bottom - top
    inner_margin = max(4, min(margin, panel_height // 5, panel_width // 12))
    if y_min is None or y_max is None:
        y_min = float(np.percentile(values, 1))
        y_max = float(np.percentile(values, 99))
        if y_max == y_min:
            y_max = y_min + 1.0
    clipped = np.clip(values, y_min, y_max)
    xs = np.linspace(left + inner_margin, right - inner_margin - 1, len(clipped))
    ys = top + inner_margin + (y_max - clipped) / (y_max - y_min) * max(1, panel_height - 2 * inner_margin - 1)
    return list(zip(xs.tolist(), ys.tolist())), float(y_min), float(y_max)


def _style_config(variant: str) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "background": "white",
        "grid": False,
        "grid_color": (225, 225, 225),
        "trace_color": (0, 0, 0),
        "line_width": 3,
        "axis_text": False,
        "blur": 0.0,
        "jpeg_quality": None,
        "noise_sigma": 0.0,
    }
    if variant == "grid":
        cfg.update({"grid": True})
    elif variant == "color_grid":
        cfg.update({"grid": True, "grid_color": (244, 205, 205), "trace_color": (28, 96, 185)})
    elif variant == "artifact":
        cfg.update({"grid": True, "trace_color": (45, 45, 45), "noise_sigma": 8.0, "blur": 0.45, "jpeg_quality": 65})
    elif variant == "dark_theme":
        cfg.update({"background": (22, 24, 28), "grid": True, "grid_color": (58, 60, 66), "trace_color": (80, 190, 255), "axis_text": True})
    elif variant == "thin_line":
        cfg.update({"grid": True, "trace_color": (15, 15, 15), "line_width": 1})
    elif variant == "thick_line":
        cfg.update({"grid": True, "trace_color": (24, 93, 173), "line_width": 5})
    elif variant == "lowres_jpeg":
        cfg.update({"grid": True, "trace_color": (20, 85, 160), "line_width": 2, "noise_sigma": 5.0, "blur": 0.7, "jpeg_quality": 45})
    elif variant == "axes_text":
        cfg.update({"grid": True, "trace_color": (31, 119, 180), "axis_text": True})
    elif variant in {"multi_trace", "multi_panel", "multi_panel_multitrace"}:
        cfg.update({"grid": True, "trace_color": (31, 119, 180), "axis_text": True, "line_width": 3})
    return cfg


def _apply_degradation(image: Image.Image, variant: str, cfg: dict[str, Any], out_png: Path) -> Image.Image:
    rng = _rng_for(out_png, variant)
    if cfg.get("noise_sigma", 0.0):
        noise = rng.normal(0, float(cfg["noise_sigma"]), size=(image.height, image.width, 3))
        arr = np.asarray(image.convert("RGB"), dtype=float)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr, mode="RGB")
    if cfg.get("blur", 0.0):
        image = image.filter(ImageFilter.GaussianBlur(radius=float(cfg["blur"])))
    if variant == "lowres_jpeg":
        small = image.resize((max(64, image.width // 2), max(48, image.height // 2)), Image.BILINEAR)
        image = small.resize((image.width, image.height), Image.BILINEAR)
    return image


def render_trace(values: np.ndarray, out_png: Path, out_mask: Path, width: int, height: int, margin: int, variant: str) -> dict[str, Any]:
    cfg = _style_config(variant)
    background = cfg["background"]
    image = Image.new("RGB", (width, height), background)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)
    rng = _rng_for(out_png, variant)
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin

    if variant in {"multi_panel", "multi_panel_multitrace"}:
        gap = max(8, margin // 2)
        n_panels = 2 if variant == "multi_panel" else 3
        panel_h = (height - 2 * margin - gap * (n_panels - 1)) // n_panels
        panels = []
        y = margin
        for _ in range(n_panels):
            panels.append((margin, y, width - margin, y + panel_h))
            y += panel_h + gap
        target_panel = n_panels - 1
        y_min = float(np.percentile(values, 1))
        y_max = float(np.percentile(values, 99))
        if y_max == y_min:
            y_max = y_min + 1.0
        for idx, panel in enumerate(panels):
            if cfg.get("grid"):
                _draw_grid(draw, panel, cfg["grid_color"], nx=9, ny=4)
            panel_values = values
            if idx != target_panel:
                panel_values = np.roll(values, int((idx + 1) * len(values) * 0.07)) * rng.uniform(0.65, 1.15)
                panel_values = panel_values + rng.normal(0.0, np.std(values) * 0.06 if np.std(values) else 0.01, len(values))
            points, _, _ = _panel_points(panel_values, panel, margin, y_min, y_max)
            color = cfg["trace_color"] if idx == target_panel else (160, 160, 160)
            draw.line(points, fill=color, width=int(cfg["line_width"]), joint="curve")
            if idx == target_panel:
                mask_draw.line(points, fill=255, width=max(3, int(cfg["line_width"])), joint="curve")
            if variant == "multi_panel_multitrace" and idx == target_panel:
                decoy = np.roll(values, int(len(values) * 0.13)) * 0.75
                decoy_points, _, _ = _panel_points(decoy, panel, margin, y_min, y_max)
                draw.line(decoy_points, fill=(210, 80, 80), width=max(1, int(cfg["line_width"]) - 1), joint="curve")
            if cfg.get("axis_text"):
                draw.text((panel[0] + 4, panel[1] + 2), f"lead {idx + 1}" if idx != target_panel else "target", fill=(100, 100, 100))
        target_box = panels[target_panel]
    else:
        target_box = (margin, margin, width - margin, height - margin)
        if cfg.get("grid"):
            _draw_grid(draw, target_box, cfg["grid_color"])
        points, y_min, y_max = _panel_points(values, target_box, margin)
        if variant == "multi_trace":
            for decoy_idx, color in enumerate([(220, 80, 80), (120, 120, 120)]):
                decoy = np.roll(values, int((decoy_idx + 1) * len(values) * 0.11)) * rng.uniform(0.65, 0.95)
                decoy = decoy + rng.normal(0.0, np.std(values) * 0.03 if np.std(values) else 0.01, len(values))
                decoy_points, _, _ = _panel_points(decoy, target_box, margin, y_min, y_max)
                draw.line(decoy_points, fill=color, width=max(1, int(cfg["line_width"]) - 1), joint="curve")
        draw.line(points, fill=cfg["trace_color"], width=int(cfg["line_width"]), joint="curve")
        mask_draw.line(points, fill=255, width=max(3, int(cfg["line_width"])), joint="curve")
        if cfg.get("axis_text"):
            text_color = (230, 230, 230) if variant == "dark_theme" else (80, 80, 80)
            draw.text((margin + 4, 4), "Filtered ECG Signal" if rng.random() < 0.5 else "Original signal", fill=text_color)
            draw.text((width // 2 - 25, height - margin + 3), "time (s)", fill=text_color)
            draw.text((2, height // 2 - 7), "amp", fill=text_color)

    image = _apply_degradation(image, variant, cfg, out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    if cfg.get("jpeg_quality"):
        image.save(out_png, format="JPEG", quality=int(cfg["jpeg_quality"]))
    else:
        image.save(out_png)
    mask.save(out_mask)
    # value_min/value_max are target trace calibration values.
    y_min = float(np.percentile(values, 1))
    y_max = float(np.percentile(values, 99))
    if y_max == y_min:
        y_max = y_min + 1.0
    return {
        "width": width,
        "height": height,
        "margin": margin,
        "plot_width": plot_width,
        "plot_height": plot_height,
        "value_min": y_min,
        "value_max": y_max,
        "target_box": {"left": target_box[0], "top": target_box[1], "right": target_box[2], "bottom": target_box[3]},
        "style": variant,
        "is_multi_panel": variant in {"multi_panel", "multi_panel_multitrace"},
        "has_decoy_trace": variant in {"multi_trace", "multi_panel_multitrace"},
    }

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
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS), help="Comma-separated plot styles to render for each source signal.")
    parser.add_argument("--legacy-one-variant-per-source", action="store_true", help="Match the old behavior: render only one style per selected source signal.")
    args = parser.parse_args()

    manifest = json.loads(Path(args.source_manifest).read_text())
    out_dir = Path(args.out_dir)
    source_counts: dict[str, int] = defaultdict(int)
    render_counts: dict[str, int] = defaultdict(int)
    include_modalities = {str(item).lower() for item in args.include_modality} if args.include_modality else None
    selected_variants = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    records = []
    for row in manifest.get("records", []):
        modality = str(row.get("modality", "")).lower()
        if include_modalities is not None and modality not in include_modalities:
            continue
        if not modality or source_counts[modality] >= args.max_per_modality:
            continue
        try:
            data = load_csv_signal(row["path"], float(row["sampling_rate"]))
            stop = min(len(data.values), max(1, int(float(args.seconds) * float(data.sampling_rate))))
            source_values = data.values[:stop]
            values = resample_values(source_values, args.points)
        except Exception as exc:
            print(json.dumps({"skip": row.get("record"), "error": str(exc)}))
            continue
        variants = [selected_variants[source_counts[modality] % len(selected_variants)]] if args.legacy_one_variant_per_source else selected_variants
        for variant in variants:
            stem = f"{modality}_{render_counts[modality]:03d}_{str(row.get('record') or 'record').replace('/', '_')}_{variant}"
            ext = "jpg" if variant in {"artifact", "lowres_jpeg"} else "png"
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
                "render_metadata": meta,
            })
            render_counts[modality] += 1
        source_counts[modality] += 1
    report = {"dataset": "rendered_waveform_digitization", "source_manifest": args.source_manifest, "num_records": len(records), "modality_counts": dict(Counter(r["modality"] for r in records)), "records": records}
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    print(json.dumps({"num_records": report["num_records"], "modality_counts": report["modality_counts"], "out_json": str(out_json)}, indent=2))


if __name__ == "__main__":
    main()
