#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def corr_by_modality(path: str) -> dict[str, float]:
    rows = json.loads(Path(path).read_text())['rows']
    out = {}
    for modality in ['ecg', 'scg', 'eeg', 'pcg', 'emg']:
        vals = [r['waveform_correlation'] for r in rows if r.get('modality') == modality and not r.get('digitizer_error') and r.get('waveform_correlation') is not None]
        if vals:
            out[modality] = float(np.mean(vals))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--low-json', default='/data1/jiahui/biosignal-agent/outputs/waveform_digitization_ml_eval.json')
    ap.add_argument('--high-json', default='/data1/jiahui/biosignal-agent/outputs/waveform_digitization_ml_highres_eval.json')
    ap.add_argument('--ultra-json', default='/data1/jiahui/biosignal-agent/outputs/waveform_digitization_ml_ultrahighres_difficult_eval.json')
    ap.add_argument('--out-png', default='/data1/jiahui/biosignal-agent/outputs/difficult_digitization_resolution_comparison.png')
    ap.add_argument('--out-json', default='/data1/jiahui/biosignal-agent/outputs/difficult_digitization_resolution_comparison.json')
    args = ap.parse_args()
    modalities = ['ecg', 'scg', 'eeg', 'pcg', 'emg']
    low = corr_by_modality(args.low_json)
    high = corr_by_modality(args.high_json)
    ultra = corr_by_modality(args.ultra_json)
    rows = []
    for m in modalities:
        rows.append({'modality': m, 'lowres_ml_corr': low.get(m), 'highres_ml_corr': high.get(m), 'ultrahighres_ml_corr': ultra.get(m)})
    Path(args.out_json).write_text(json.dumps({'rows': rows}, indent=2))
    x = np.arange(len(modalities))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.bar(x - width, [low.get(m, np.nan) for m in modalities], width=width, label='low-res', color='#4c78a8')
    ax.bar(x, [high.get(m, np.nan) for m in modalities], width=width, label='high-res', color='#54a24b')
    ax.bar(x + width, [ultra.get(m, np.nan) for m in modalities], width=width, label='ultra-high-res', color='#f58518')
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel('Mean waveform correlation')
    ax.set_title('Resolution scaling improves difficult waveform digitization')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in modalities])
    ax.grid(axis='y', alpha=0.25)
    ax.legend()
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=160)
    plt.close(fig)
    print(json.dumps({'out_png': args.out_png, 'out_json': args.out_json, 'rows': len(rows)}, indent=2))


if __name__ == '__main__':
    main()
