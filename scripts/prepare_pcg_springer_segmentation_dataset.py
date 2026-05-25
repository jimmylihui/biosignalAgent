from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

OUT_DIR = Path('/data1/jiahui/biosignal-agent/datasets/processed/pcg_springer_segmentation')
MANIFEST = Path('/data1/jiahui/biosignal-agent/datasets/processed/pcg_springer_segmentation_manifest.json')


def parse_csv_blob(blob: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(blob))
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df.rename(columns={'timestep': 'time_s', 'amplitude': 'signal', 'label': 'state_label'})


def infer_fs(time_s: np.ndarray) -> float:
    dt = np.diff(time_s.astype(float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return 1000.0
    return float(round(1.0 / float(np.median(dt)), 3))


def main() -> None:
    ap = argparse.ArgumentParser(description='Prepare Springer/Schmidt PCG state-segmentation benchmark from Hugging Face.')
    ap.add_argument('--out-dir', type=Path, default=OUT_DIR)
    ap.add_argument('--manifest', type=Path, default=MANIFEST)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    ds = load_dataset('alvgaona/springer-sounds', split='train')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    label_counts = Counter()
    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    for i in range(n):
        item = ds[i]
        key = str(item.get('__key__') or f'{i:04d}')
        df = parse_csv_blob(item['csv'])
        fs = infer_fs(df['time_s'].to_numpy(dtype=float))
        out_csv = args.out_dir / f'springer_{key}_pcg.csv'
        df[['signal', 'state_label']].to_csv(out_csv, index=False)
        counts = Counter(map(int, df['state_label'].to_numpy(dtype=int)))
        label_counts.update(counts)
        rows.append({
            'dataset': 'springer-sounds',
            'record_id': key,
            'path': str(out_csv),
            'sampling_rate': fs,
            'num_samples': int(len(df)),
            'duration_s': float(len(df) / fs),
            'state_label_mapping': {'1': 'S1', '2': 'systole', '3': 'S2', '4': 'diastole'},
            'label_counts': {str(k): int(v) for k, v in sorted(counts.items())},
            'modality': 'pcg',
        })
    manifest = {
        'dataset': 'springer-sounds',
        'source': 'https://huggingface.co/datasets/alvgaona/springer-sounds',
        'num_records': len(rows),
        'rows': rows,
        'state_label_mapping': {'1': 'S1', '2': 'systole', '3': 'S2', '4': 'diastole'},
        'global_label_counts': {str(k): int(v) for k, v in sorted(label_counts.items())},
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: manifest[k] for k in ['dataset', 'num_records', 'global_label_counts']}, indent=2))


if __name__ == '__main__':
    main()
