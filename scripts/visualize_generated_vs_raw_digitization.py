#!/usr/bin/env python3
"""Visualize ECG-Image-Kit generated image/crop against raw reference and digitized output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.digitize_tools import Signal_digitize_waveform_image, Signal_digitize_waveform_image_ml
from biosignal_agent.tools.digitize_unet_tools import Signal_digitize_waveform_image_unet


def load_signal(path: str) -> np.ndarray:
    frame = pd.read_csv(path)
    col = 'signal' if 'signal' in frame.columns else frame.select_dtypes('number').columns[-1]
    return frame[col].to_numpy(dtype=float)


def norm01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    out = np.full_like(values, np.nan, dtype=float)
    if not finite.any():
        return out
    lo, hi = np.nanpercentile(values[finite], [1, 99])
    if abs(hi - lo) < 1e-12:
        out[finite] = 0.5
    else:
        out[finite] = (values[finite] - lo) / (hi - lo)
    return np.clip(out, 0, 1)


def run_digitizer(record: dict, method: str, out_csv: Path, model_path: str | None, threshold: float) -> dict:
    args = dict(
        image_path=record['image_path'],
        sampling_rate=float(record['sampling_rate']),
        out_csv=str(out_csv),
        value_min=float(record['value_min']),
        value_max=float(record['value_max']),
        crop_left=int(record.get('crop_left', 0)),
        crop_right=int(record.get('crop_right', 0)),
        crop_top=int(record.get('crop_top', 0)),
        crop_bottom=int(record.get('crop_bottom', 0)),
        smooth_window=1,
    )
    if method == 'unet':
        return Signal_digitize_waveform_image_unet(**args, model_path=model_path, probability_threshold=threshold)
    if method == 'ml':
        return Signal_digitize_waveform_image_ml(**args, model_path=model_path, probability_threshold=threshold)
    return Signal_digitize_waveform_image(**args, threshold=80)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/datasets/processed/ecg_image_kit_generated_manifest.json')
    ap.add_argument('--record-index', type=int, default=0)
    ap.add_argument('--method', choices=['rule', 'ml', 'unet'], default='unet')
    ap.add_argument('--model-path', default='/data1/jiahui/biosignal-agent/outputs/ecg_image_kit_generated_unet.pt')
    ap.add_argument('--probability-threshold', type=float, default=0.65)
    ap.add_argument('--out-png', default='/data1/jiahui/biosignal-agent/outputs/ecg_image_kit_generated_vs_raw.png')
    ap.add_argument('--out-csv', default='/data1/jiahui/biosignal-agent/outputs/digitized/ecg_image_kit_generated_vs_raw_digitized.csv')
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    record = manifest['records'][args.record_index]
    digitized = run_digitizer(record, args.method, Path(args.out_csv), args.model_path, args.probability_threshold)
    if digitized.get('error'):
        raise SystemExit(digitized['error'])

    raw = load_signal(record['reference_path'])
    pred = load_signal(digitized['out_csv'])
    fs = float(record['sampling_rate'])
    raw_t = np.arange(len(raw)) / fs
    duration = raw_t[-1] if len(raw_t) else len(pred) / fs
    pred_t = np.linspace(0.0, duration, len(pred)) if len(pred) else np.array([])
    raw_interp = np.interp(pred_t, raw_t, raw) if len(raw_t) and len(pred_t) else np.array([])
    raw_n = norm01(raw_interp)
    pred_n = norm01(pred)
    err = pred_n - raw_n
    finite = np.isfinite(raw_n) & np.isfinite(pred_n)
    corr = float(np.corrcoef(raw_n[finite], pred_n[finite])[0, 1]) if finite.sum() > 2 and np.std(raw_n[finite]) > 1e-8 and np.std(pred_n[finite]) > 1e-8 else np.nan
    mae = float(np.nanmean(np.abs(err)))
    n = len(pred_n)
    t = pred_t

    original_path = record.get('original_image_path') or record['image_path']
    crop_img = Image.open(record['image_path']).convert('RGB')
    original_img = Image.open(original_path).convert('RGB')
    mask_img = Image.open(record['mask_path']).convert('L') if record.get('mask_path') else None

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 0.75])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(original_img)
    ax0.set_title('Generated ECG-Image-Kit page')
    ax0.axis('off')

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(crop_img)
    if mask_img is not None:
        ax1.imshow(mask_img, cmap='Reds', alpha=np.asarray(mask_img) / 255 * 0.45)
    ax1.set_title('Lead crop with plotted-pixel mask overlay')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(t, raw_n, label='raw WFDB reference (normalized)', linewidth=1.2)
    ax2.plot(t, pred_n, label=f'{args.method} digitized waveform (normalized)', linewidth=1.0, alpha=0.85)
    ax2.set_title(f"Raw vs digitized waveform: {record.get('record')} | corr={corr:.3f}, normalized MAE={mae:.3f}")
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Normalized amplitude')
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc='upper right')

    ax3 = fig.add_subplot(gs[2, :])
    ax3.plot(t, err, color='tab:red', linewidth=0.9)
    ax3.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    ax3.set_title('Digitized - raw normalized error')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Error')
    ax3.grid(True, alpha=0.25)

    out = Path(args.out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(json.dumps({
        'out_png': str(out),
        'digitized_csv': digitized['out_csv'],
        'record': record.get('record'),
        'method': args.method,
        'correlation_normalized': corr,
        'mae_normalized': mae,
        'num_points': int(n),
    }, indent=2))


if __name__ == '__main__':
    main()
