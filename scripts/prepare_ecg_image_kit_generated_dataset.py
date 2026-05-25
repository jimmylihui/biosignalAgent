#!/usr/bin/env python3
"""Build a BioSignalAgent digitization manifest from ECG-Image-Kit generated outputs.

This is for generated ECG-Image-Kit images that include companion WFDB files.
For two-lead MIT-BIH style records, each image is kept as one benchmark item and
uses the selected reference lead exported from the generated WFDB record.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import wfdb
from PIL import Image, ImageDraw


def _read_record(record_base: Path, lead: str | None) -> tuple[np.ndarray, float, str]:
    rec = wfdb.rdrecord(str(record_base))
    names = list(rec.sig_name or [])
    if rec.p_signal is None:
        raise ValueError(f"WFDB record has no physical signal: {record_base}")
    if lead and lead in names:
        idx = names.index(lead)
    else:
        idx = 0
    values = np.asarray(rec.p_signal[:, idx], dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError(f"reference lead has no finite samples: {record_base}")
    return values, float(rec.fs), names[idx] if names else f"lead_{idx}"


def _write_reference_csv(path: Path, values: np.ndarray, fs: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sample', 'time_s', 'value'])
        writer.writeheader()
        for i, v in enumerate(values):
            writer.writerow({'sample': i, 'time_s': i / fs, 'value': float(v) if np.isfinite(v) else ''})


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--generated-dir', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/processed/ecg_image_kit_generated'))
    ap.add_argument('--out-manifest', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/processed/ecg_image_kit_generated_manifest.json'))
    ap.add_argument('--reference-dir', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/processed/ecg_image_kit_generated/references'))
    ap.add_argument('--crop-dir', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/processed/ecg_image_kit_generated/crops'))
    ap.add_argument('--mask-dir', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/processed/ecg_image_kit_generated/masks'))
    ap.add_argument('--use-lead-bbox-crop', action='store_true', default=True)
    ap.add_argument('--lead', default='MLII')
    ap.add_argument('--modality', default='ecg')
    args = ap.parse_args()

    image_paths = sorted(args.generated_dir.glob('*.png')) + sorted(args.generated_dir.glob('*.jpg')) + sorted(args.generated_dir.glob('*.jpeg'))
    if not image_paths:
        raise SystemExit(f'No ECG-Image-Kit images found in {args.generated_dir}')

    records = []
    for image_path in image_paths:
        stem_parts = image_path.stem.split('-')
        record_name = stem_parts[0]
        record_base = args.generated_dir / record_name
        if not (record_base.with_suffix('.hea')).exists():
            # Fallback for names that themselves contain dashes.
            wfdb_candidates = sorted(args.generated_dir.glob('*.hea'))
            if len(wfdb_candidates) == 1:
                record_base = wfdb_candidates[0].with_suffix('')
            else:
                raise SystemExit(f'Cannot match WFDB header for {image_path}')

        values, fs, lead_name = _read_record(record_base, args.lead)
        finite = values[np.isfinite(values)]
        ref_path = args.reference_dir / f'{image_path.stem}_{lead_name}.csv'
        _write_reference_csv(ref_path, values, fs)
        cfg = _safe_json(image_path.with_suffix('.json'))
        benchmark_image = image_path
        crop_bounds = None
        mask_path = None
        lead_cfg = None
        if cfg.get('leads'):
            lead_cfg = next((item for item in cfg.get('leads', []) if item.get('lead_name') == lead_name), None)
        if args.use_lead_bbox_crop and lead_cfg and lead_cfg.get('lead_bounding_box'):
            pts = list(lead_cfg['lead_bounding_box'].values())
            # ECG-Image-Kit stores bbox points as [row, col]. Convert to PIL [x, y].
            rows = [float(pt[0]) for pt in pts]
            cols = [float(pt[1]) for pt in pts]
            img = Image.open(image_path)
            pad = 8
            left = max(0, int(min(cols)) - pad)
            right = min(img.width, int(max(cols)) + pad)
            top = max(0, int(min(rows)) - pad)
            bottom = min(img.height, int(max(rows)) + pad)
            if right > left and bottom > top:
                args.crop_dir.mkdir(parents=True, exist_ok=True)
                crop_path = args.crop_dir / f'{image_path.stem}_{lead_name}_crop.png'
                img.crop((left, top, right, bottom)).save(crop_path)
                benchmark_image = crop_path
                crop_bounds = {'left': left, 'top': top, 'right': right, 'bottom': bottom}
        if lead_cfg and lead_cfg.get('plotted_pixels'):
            mask_base = Image.open(benchmark_image)
            mask = Image.new('L', mask_base.size, 0)
            draw = ImageDraw.Draw(mask)
            points = []
            offset_left = crop_bounds['left'] if crop_bounds else 0
            offset_top = crop_bounds['top'] if crop_bounds else 0
            for row, col in lead_cfg.get('plotted_pixels', []):
                x = float(col) - offset_left
                y = float(row) - offset_top
                if 0 <= x < mask.width and 0 <= y < mask.height:
                    points.append((x, y))
            if len(points) >= 2:
                args.mask_dir.mkdir(parents=True, exist_ok=True)
                mask_path_obj = args.mask_dir / f'{image_path.stem}_{lead_name}_mask.png'
                draw.line(points, fill=255, width=3)
                mask.save(mask_path_obj)
                mask_path = str(mask_path_obj)
        records.append({
            'id': f'ecg_image_kit_generated_{image_path.stem}_{lead_name}',
            'record': f'ecg_image_kit_generated_{image_path.stem}_{lead_name}',
            'source': 'ecg-image-kit-generated',
            'modality': args.modality,
            'image_path': str(benchmark_image),
            'original_image_path': str(image_path),
            'reference_path': str(ref_path),
            'mask_path': mask_path,
            'sampling_rate': fs,
            'lead': lead_name,
            'value_min': float(np.nanpercentile(finite, 1)),
            'value_max': float(np.nanpercentile(finite, 99)),
            'crop_left': 0,
            'crop_right': 0,
            'crop_top': 0,
            'crop_bottom': 0,
            'metadata': {
                'lead_crop_bounds': crop_bounds,
                'wfdb_record': str(record_base),
                'config_path': str(image_path.with_suffix('.json')) if image_path.with_suffix('.json').exists() else None,
                'full_mode_lead': cfg.get('full_mode_lead'),
                'has_grid': cfg.get('grid'),
                'resolution': cfg.get('resolution'),
            },
        })

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text(json.dumps({'records': records}, indent=2))
    print(json.dumps({'records': len(records), 'manifest': str(args.out_manifest)}, indent=2))


if __name__ == '__main__':
    main()
