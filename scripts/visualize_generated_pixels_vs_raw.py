#!/usr/bin/env python3
"""Compare ECG-Image-Kit generated plotted pixels directly with raw WFDB reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def load_signal(path: str) -> np.ndarray:
    frame = pd.read_csv(path)
    col = 'signal' if 'signal' in frame.columns else frame.select_dtypes('number').columns[-1]
    return frame[col].to_numpy(dtype=float)


def norm01(values: np.ndarray, invert: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    out = np.full_like(values, np.nan, dtype=float)
    lo, hi = np.nanpercentile(values[finite], [1, 99])
    out[finite] = (values[finite] - lo) / max(hi - lo, 1e-12)
    out = np.clip(out, 0, 1)
    return 1.0 - out if invert else out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/datasets/processed/ecg_image_kit_generated_manifest.json')
    ap.add_argument('--record-index', type=int, default=0)
    ap.add_argument('--out-png', default='/data1/jiahui/biosignal-agent/outputs/ecg_image_kit_generated_pixels_vs_raw.png')
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    record = manifest['records'][args.record_index]
    cfg = json.loads(Path(record['metadata']['config_path']).read_text())
    lead = record['lead']
    lead_cfg = next(item for item in cfg['leads'] if item['lead_name'] == lead)
    pixels = np.asarray(lead_cfg['plotted_pixels'], dtype=float)  # row, col
    raw = load_signal(record['reference_path'])

    # Collapse ECG-Image-Kit's rendered pixels by x coordinate to an ideal pixel-trace.
    rows = pixels[:, 0]
    cols = pixels[:, 1]
    col_rounded = np.round(cols).astype(int)
    unique_cols = np.arange(col_rounded.min(), col_rounded.max() + 1)
    y_by_col = np.full(len(unique_cols), np.nan)
    for i, c in enumerate(unique_cols):
        vals = rows[col_rounded == c]
        if vals.size:
            y_by_col[i] = np.median(vals)
    finite = np.isfinite(y_by_col)
    y_by_col = np.interp(unique_cols, unique_cols[finite], y_by_col[finite])

    fs = float(record['sampling_rate'])
    raw_t = np.arange(len(raw)) / fs
    duration = raw_t[-1] if len(raw_t) else 10.0
    pixel_t = np.linspace(0.0, duration, len(y_by_col)) if len(y_by_col) else np.array([])
    raw_interp = np.interp(pixel_t, raw_t, raw) if len(raw_t) and len(pixel_t) else np.array([])
    raw_n = norm01(raw_interp, invert=False)
    pixel_n = norm01(y_by_col, invert=True)
    finite = np.isfinite(raw_n) & np.isfinite(pixel_n)
    corr = float(np.corrcoef(raw_n[finite], pixel_n[finite])[0, 1]) if finite.sum() > 2 and np.std(raw_n[finite]) > 1e-8 and np.std(pixel_n[finite]) > 1e-8 else np.nan
    mae = float(np.nanmean(np.abs(pixel_n - raw_n)))
    n = len(pixel_n)
    t = pixel_t

    img = Image.open(record['image_path']).convert('RGB')
    mask = Image.open(record['mask_path']).convert('L') if record.get('mask_path') else None

    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img)
    if mask is not None:
        ax0.imshow(mask, cmap='Reds', alpha=np.asarray(mask) / 255 * 0.4)
    ax0.set_title('Generated crop + ground-truth plotted-pixel mask')
    ax0.axis('off')

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.scatter(cols, rows, s=1, alpha=0.35)
    ax1.invert_yaxis()
    ax1.set_title('Raw plotted_pixels from ECG-Image-Kit JSON')
    ax1.set_xlabel('image x pixel')
    ax1.set_ylabel('image y pixel')
    ax1.grid(True, alpha=0.25)

    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(t, raw_n, label='raw WFDB reference normalized', linewidth=1.2)
    ax2.plot(t, pixel_n, label='ideal generated plotted_pixels mapped to waveform', linewidth=1.0, alpha=0.85)
    ax2.set_title(f"Generated plotted pixels vs raw: corr={corr:.3f}, normalized MAE={mae:.3f}")
    ax2.set_xlabel('Time (s), index-aligned')
    ax2.set_ylabel('Normalized amplitude')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.25)

    out = Path(args.out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(json.dumps({'out_png': str(out), 'record': record['record'], 'correlation_normalized': corr, 'mae_normalized': mae, 'num_points': int(n)}, indent=2))


if __name__ == '__main__':
    main()
