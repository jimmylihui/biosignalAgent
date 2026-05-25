#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def modality_corr(path: str) -> dict[str, float]:
    rows = json.loads(Path(path).read_text())['rows']
    out = {}
    for modality in sorted({r['modality'] for r in rows if not r.get('digitizer_error')}):
        vals = [r['waveform_correlation'] for r in rows if r.get('modality') == modality and not r.get('digitizer_error') and r.get('waveform_correlation') is not None]
        if vals:
            out[modality] = float(np.mean(vals))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--low-json', default='/data1/jiahui/biosignal-agent/outputs/waveform_digitization_ml_eval.json')
    ap.add_argument('--high-json', default='/data1/jiahui/biosignal-agent/outputs/waveform_digitization_ml_highres_eval.json')
    ap.add_argument('--out-png', default='/data1/jiahui/biosignal-agent/outputs/ml_digitization_low_vs_highres_by_modality.png')
    ap.add_argument('--out-json', default='/data1/jiahui/biosignal-agent/outputs/ml_digitization_low_vs_highres_by_modality.json')
    args = ap.parse_args()
    low = modality_corr(args.low_json)
    high = modality_corr(args.high_json)
    modalities = sorted(set(low) | set(high))
    rows = [{'modality': m, 'lowres_ml_corr': low.get(m), 'highres_ml_corr': high.get(m), 'delta': None if low.get(m) is None or high.get(m) is None else high[m] - low[m]} for m in modalities]
    Path(args.out_json).write_text(json.dumps({'rows': rows}, indent=2))
    x = np.arange(len(modalities))
    width = 0.38
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    ax.bar(x - width / 2, [low.get(m, np.nan) for m in modalities], width=width, label='low-res ML digitizer', color='#4c78a8')
    ax.bar(x + width / 2, [high.get(m, np.nan) for m in modalities], width=width, label='high-res ML digitizer', color='#54a24b')
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('Mean waveform correlation')
    ax.set_title('ML digitizer improvement from higher horizontal resolution')
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
