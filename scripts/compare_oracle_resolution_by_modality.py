#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--low-json', default='/data1/jiahui/biosignal-agent/outputs/oracle_mask_digitization_eval.json')
    ap.add_argument('--high-json', default='/data1/jiahui/biosignal-agent/outputs/oracle_mask_digitization_highres_aligned_eval.json')
    ap.add_argument('--out-png', default='/data1/jiahui/biosignal-agent/outputs/oracle_resolution_by_modality.png')
    ap.add_argument('--out-json', default='/data1/jiahui/biosignal-agent/outputs/oracle_resolution_by_modality.json')
    args = ap.parse_args()
    low = json.loads(Path(args.low_json).read_text()).get('by_modality', {})
    high = json.loads(Path(args.high_json).read_text()).get('by_modality', {})
    modalities = sorted(set(low) | set(high))
    rows = []
    for modality in modalities:
        low_corr = low.get(modality, {}).get('mean_waveform_correlation')
        high_corr = high.get(modality, {}).get('mean_waveform_correlation')
        rows.append({'modality': modality, 'lowres_oracle_corr': low_corr, 'highres_oracle_corr': high_corr, 'delta': None if low_corr is None or high_corr is None else high_corr - low_corr})
    Path(args.out_json).write_text(json.dumps({'rows': rows}, indent=2))
    x = np.arange(len(modalities))
    width = 0.38
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    ax.bar(x - width / 2, [low.get(m, {}).get('mean_waveform_correlation', np.nan) for m in modalities], width=width, label='low-res oracle mask', color='#4c78a8')
    ax.bar(x + width / 2, [high.get(m, {}).get('mean_waveform_correlation', np.nan) for m in modalities], width=width, label='high-res oracle mask', color='#f58518')
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('Mean waveform correlation')
    ax.set_title('Effect of image time-axis resolution on oracle mask-to-waveform reconstruction')
    ax.set_xticks(x)
    ax.set_xticklabels(modalities, rotation=35, ha='right')
    ax.grid(axis='y', alpha=0.25)
    ax.legend()
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=160)
    plt.close(fig)
    print(json.dumps({'out_png': args.out_png, 'out_json': args.out_json, 'rows': len(rows)}, indent=2))


if __name__ == '__main__':
    main()
