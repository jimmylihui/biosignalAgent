from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from scipy.io import wavfile

RAW = Path('/data1/jiahui/biosignal-agent/datasets/raw/pcg/BMD-HS-Dataset')
MANIFEST = Path('/data1/jiahui/biosignal-agent/datasets/processed/pcg_bmdhs_valve_manifest.json')
LABELS = ['AS', 'AR', 'MR', 'MS', 'N']


def parse_recording_name(name: str) -> dict:
    parts = str(name).split('_')
    disease = parts[0] if len(parts) > 0 else None
    posture = parts[2] if len(parts) > 3 else None
    site = parts[3] if len(parts) > 3 else None
    return {'recording_stem': name, 'prefix_label': disease, 'posture': posture, 'site': site}


def main() -> None:
    ap = argparse.ArgumentParser(description='Prepare BMD-HS PCG valve disease recording-level manifest.')
    ap.add_argument('--raw-dir', type=Path, default=RAW)
    ap.add_argument('--manifest', type=Path, default=MANIFEST)
    args = ap.parse_args()
    df = pd.read_csv(args.raw_dir / 'train.csv')
    rows = []
    skipped = {}
    for _, r in df.iterrows():
        patient_id = str(r['patient_id'])
        labels = {lab: int(float(r[lab]) > 0.5) for lab in LABELS}
        for col in [c for c in df.columns if c.startswith('recording_')]:
            stem = str(r[col])
            if not stem or stem == 'nan':
                continue
            path = args.raw_dir / 'train' / f'{stem}.wav'
            if not path.exists():
                skipped[stem] = 'missing_wav'
                continue
            try:
                fs, values = wavfile.read(path)
                n = int(values.shape[0])
            except Exception as exc:
                skipped[stem] = f'{type(exc).__name__}:{str(exc)[:120]}'
                continue
            meta = parse_recording_name(stem)
            row = {
                'dataset': 'BMD-HS',
                'patient_id': patient_id,
                'group': patient_id,
                'recording': stem,
                'path': str(path),
                'sampling_rate': float(fs),
                'num_samples': n,
                'duration_s': float(n / float(fs)),
                'modality': 'pcg',
                **meta,
            }
            for lab, val in labels.items():
                row[f'label_{lab.lower()}'] = val
            rows.append(row)
    label_counts = {lab: int(sum(x[f'label_{lab.lower()}'] for x in rows)) for lab in LABELS}
    patient_label_counts = {lab: int(sum(float(x) > 0.5 for x in df[lab])) for lab in LABELS}
    manifest = {
        'dataset': 'BMD-HS',
        'source': 'https://github.com/sani002/BMD-HS-Dataset',
        'num_records': len(rows),
        'num_patients': int(df['patient_id'].nunique()),
        'labels': LABELS,
        'label_counts_recording_level': label_counts,
        'label_counts_patient_level': patient_label_counts,
        'site_counts': dict(Counter(x.get('site') for x in rows)),
        'posture_counts': dict(Counter(x.get('posture') for x in rows)),
        'rows': rows,
        'skipped': skipped,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: manifest[k] for k in ['num_records','num_patients','label_counts_patient_level','site_counts','skipped']}, indent=2))


if __name__ == '__main__':
    main()
