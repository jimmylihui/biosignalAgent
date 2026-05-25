from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_image_scale_prior import build_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/datasets/processed/modality_classifier_manifest.json')
    ap.add_argument('--out-dir', default='/data1/jiahui/biosignal-agent/outputs/image_scale_prior_aug')
    ap.add_argument('--samples-per-duration', type=int, default=12)
    ap.add_argument('--max-records', type=int, default=0)
    ap.add_argument('--seed', type=int, default=29)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    rows_path = out_dir / 'scale_prior_dataset.json'
    if rows_path.exists() and not args.force:
        rows = json.loads(rows_path.read_text())
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = build_dataset(Path(args.manifest), out_dir, args.samples_per_duration, args.max_records, args.seed)
        rows_path.write_text(json.dumps(rows, indent=2))
    by = {}
    for row in rows:
        by.setdefault(str(row.get('modality')), 0)
        by[str(row.get('modality'))] += 1
    print(json.dumps({'rows_path': str(rows_path), 'num_rows': len(rows), 'by_modality': by}, indent=2))


if __name__ == '__main__':
    main()
