#!/usr/bin/env python3
"""Evaluate oracle plotted-pixel masks against raw references for rendered waveform images."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


def load_signal(path: str) -> np.ndarray:
    frame = pd.read_csv(path)
    col = 'signal' if 'signal' in frame.columns else frame.select_dtypes('number').columns[-1]
    return frame[col].to_numpy(dtype=float)


def mask_to_signal(record: dict[str, Any]) -> tuple[np.ndarray, float]:
    mask = np.asarray(Image.open(record['mask_path']).convert('L'), dtype=np.uint8) > 0
    left = int(record.get('crop_left', 0))
    right = int(record.get('crop_right', 0))
    top = int(record.get('crop_top', 0))
    bottom = int(record.get('crop_bottom', 0))
    if any([left, right, top, bottom]):
        h, w = mask.shape
        mask = mask[top:h - bottom if bottom else h, left:w - right if right else w]
    y_values = np.full(mask.shape[1], np.nan, dtype=float)
    for x in range(mask.shape[1]):
        ys = np.flatnonzero(mask[:, x])
        if len(ys):
            y_values[x] = float(np.median(ys))
    coverage = float(np.isfinite(y_values).mean()) if len(y_values) else 0.0
    finite = np.isfinite(y_values)
    if finite.sum() < 2:
        raise ValueError('mask has too few plotted columns')
    xs = np.arange(len(y_values))
    y_values = np.interp(xs, xs[finite], y_values[finite])
    normalized = 1.0 - 2.0 * (y_values / max(1, mask.shape[0] - 1))
    values = float(record['value_min']) + (normalized + 1.0) / 2.0 * (float(record['value_max']) - float(record['value_min']))
    return values, coverage


def corrcoef(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or len(b) < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def nrmse(a: np.ndarray, b: np.ndarray) -> float | None:
    denom = float(np.percentile(a, 99) - np.percentile(a, 1))
    if denom < 1e-8:
        return None
    return float(np.sqrt(np.mean((a - b) ** 2)) / denom)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark_manifest.json')
    ap.add_argument('--out-json', default='/data1/jiahui/biosignal-agent/outputs/oracle_mask_digitization_eval.json')
    ap.add_argument('--out-csv', default='/data1/jiahui/biosignal-agent/outputs/oracle_mask_digitization_eval.csv')
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    rows = []
    for record in manifest.get('records', []):
        row = {'record': record.get('record'), 'modality': record.get('modality'), 'variant': record.get('variant')}
        try:
            pred, coverage = mask_to_signal(record)
            ref = load_signal(record['reference_path'])
            n = min(len(ref), len(pred))
            ref = ref[:n]
            pred = pred[:n]
            row.update({
                'num_points': int(n),
                'pixel_coverage': coverage,
                'waveform_correlation': corrcoef(ref, pred),
                'nrmse': nrmse(ref, pred),
                'mae': float(np.mean(np.abs(ref - pred))),
            })
        except Exception as exc:
            row['error'] = str(exc)
        rows.append(row)

    ok = [r for r in rows if not r.get('error')]
    def mean(vals):
        vals = [float(v) for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None
    by_modality = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ok:
        groups[str(row.get('modality'))].append(row)
    for modality, subset in sorted(groups.items()):
        by_modality[modality] = {
            'num_records': len(subset),
            'mean_waveform_correlation': mean([r.get('waveform_correlation') for r in subset]),
            'mean_nrmse': mean([r.get('nrmse') for r in subset]),
            'mean_pixel_coverage': mean([r.get('pixel_coverage') for r in subset]),
        }
    report = {
        'manifest': args.manifest,
        'method': 'oracle_mask_centerline',
        'metrics': {
            'num_records': len(rows),
            'num_ok': len(ok),
            'mean_waveform_correlation': mean([r.get('waveform_correlation') for r in ok]),
            'mean_nrmse': mean([r.get('nrmse') for r in ok]),
            'mean_pixel_coverage': mean([r.get('pixel_coverage') for r in ok]),
        },
        'by_modality': by_modality,
        'rows': rows,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    with Path(args.out_csv).open('w', newline='') as f:
        keys = sorted({k for row in rows for k in row})
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'metrics': report['metrics'], 'by_modality': by_modality, 'out_json': args.out_json}, indent=2))


if __name__ == '__main__':
    main()
