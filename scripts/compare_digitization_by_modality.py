#!/usr/bin/env python3
"""Compare oracle/rule/ML/U-Net waveform digitization metrics by modality."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(path: str, method: str) -> list[dict]:
    payload = json.loads(Path(path).read_text())
    rows = payload.get('rows') or []
    out = []
    for row in rows:
        if row.get('digitizer_error') or row.get('error'):
            continue
        out.append({**row, 'method': method})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--oracle-json', default='/data1/jiahui/biosignal-agent/outputs/oracle_mask_digitization_eval.json')
    ap.add_argument('--rule-json', default='/data1/jiahui/biosignal-agent/outputs/waveform_digitization_eval.json')
    ap.add_argument('--ml-json', default='/data1/jiahui/biosignal-agent/outputs/waveform_digitization_ml_eval.json')
    ap.add_argument('--unet-json', default='/data1/jiahui/biosignal-agent/outputs/waveform_digitization_unet_eval.json')
    ap.add_argument('--out-json', default='/data1/jiahui/biosignal-agent/outputs/digitization_by_modality_comparison.json')
    ap.add_argument('--out-csv', default='/data1/jiahui/biosignal-agent/outputs/digitization_by_modality_comparison.csv')
    ap.add_argument('--out-png', default='/data1/jiahui/biosignal-agent/outputs/digitization_by_modality_comparison.png')
    args = ap.parse_args()

    rows = []
    rows.extend(load_rows(args.oracle_json, 'oracle_mask'))
    rows.extend(load_rows(args.rule_json, 'rule'))
    rows.extend(load_rows(args.ml_json, 'ml_pixel'))
    rows.extend(load_rows(args.unet_json, 'tiny_unet'))

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get('modality'), row.get('method'))].append(row)

    summary = []
    for (modality, method), subset in sorted(grouped.items()):
        def mean(key):
            vals = [r.get(key) for r in subset if r.get(key) is not None]
            return float(np.mean([float(v) for v in vals])) if vals else None
        summary.append({
            'modality': modality,
            'method': method,
            'num_ok': len(subset),
            'mean_waveform_correlation': mean('waveform_correlation'),
            'mean_nrmse': mean('nrmse'),
            'mean_peak_f1': mean('peak_f1'),
            'mean_pixel_coverage': mean('pixel_coverage'),
        })

    report = {'rows': summary, 'source_rows': len(rows)}
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    with Path(args.out_csv).open('w', newline='') as f:
        keys = ['modality', 'method', 'num_ok', 'mean_waveform_correlation', 'mean_nrmse', 'mean_peak_f1', 'mean_pixel_coverage']
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary)

    modalities = sorted({row['modality'] for row in summary})
    methods = ['oracle_mask', 'rule', 'ml_pixel', 'tiny_unet']
    values = {method: [] for method in methods}
    for modality in modalities:
        by_method = {row['method']: row for row in summary if row['modality'] == modality}
        for method in methods:
            values[method].append(by_method.get(method, {}).get('mean_waveform_correlation') or np.nan)

    x = np.arange(len(modalities))
    width = 0.2
    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)
    colors = {'oracle_mask': '#222222', 'rule': '#4c78a8', 'ml_pixel': '#f58518', 'tiny_unet': '#54a24b'}
    for i, method in enumerate(methods):
        ax.bar(x + (i - 1.5) * width, values[method], width=width, label=method, color=colors[method])
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('Mean waveform correlation')
    ax.set_title('Generated/oracle vs digitizer waveform reconstruction by modality')
    ax.set_xticks(x)
    ax.set_xticklabels(modalities, rotation=35, ha='right')
    ax.grid(axis='y', alpha=0.25)
    ax.legend(ncol=4, loc='upper center')
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=160)
    plt.close(fig)

    print(json.dumps({'out_json': args.out_json, 'out_csv': args.out_csv, 'out_png': args.out_png, 'rows': len(summary)}, indent=2))


if __name__ == '__main__':
    main()
