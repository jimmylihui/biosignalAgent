#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.tools.digitize_tools import estimate_digitization_resolution_risk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark_manifest.json')
    ap.add_argument('--out-json', default='/data1/jiahui/biosignal-agent/outputs/digitization_resolution_risk_eval.json')
    ap.add_argument('--out-csv', default='/data1/jiahui/biosignal-agent/outputs/digitization_resolution_risk_eval.csv')
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    rows = []
    for rec in manifest.get('records', []):
        img = Image.open(rec['image_path'])
        width = img.width - int(rec.get('crop_left', 0)) - int(rec.get('crop_right', 0))
        risk = estimate_digitization_resolution_risk(width, rec.get('duration_s'), modality=rec.get('modality'))
        rows.append({
            'record': rec.get('record'),
            'modality': rec.get('modality'),
            'variant': rec.get('variant'),
            'image_width': width,
            'duration_s': rec.get('duration_s'),
            'expected_signal_bandwidth_hz': risk['expected_signal_bandwidth_hz'],
            'pixels_per_second': risk['pixels_per_second'],
            'pixels_per_cycle_at_bandwidth': risk['pixels_per_cycle_at_bandwidth'],
            'risk': risk['risk'],
            'recommendation': risk['recommendation'],
        })
    by_modality = {}
    groups = defaultdict(list)
    for row in rows:
        groups[row['modality']].append(row)
    for modality, subset in sorted(groups.items()):
        by_modality[modality] = {
            'num_records': len(subset),
            'risk_counts': dict(Counter(r['risk'] for r in subset)),
            'mean_pixels_per_cycle': float(np.mean([r['pixels_per_cycle_at_bandwidth'] for r in subset if r['pixels_per_cycle_at_bandwidth'] is not None])),
        }
    report = {'manifest': args.manifest, 'num_records': len(rows), 'risk_counts': dict(Counter(r['risk'] for r in rows)), 'by_modality': by_modality, 'rows': rows}
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    with Path(args.out_csv).open('w', newline='') as f:
        keys = list(rows[0]) if rows else []
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'risk_counts': report['risk_counts'], 'by_modality': by_modality, 'out_json': args.out_json}, indent=2))


if __name__ == '__main__':
    main()
