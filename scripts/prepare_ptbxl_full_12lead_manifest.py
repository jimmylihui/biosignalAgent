from __future__ import annotations
import argparse, ast, json
from collections import Counter
from pathlib import Path

import pandas as pd
import wfdb

PTB_URL = 'https://physionet.org/files/ptb-xl/1.0.3/'
RAW = Path('/data1/jiahui/biosignal-agent/datasets/raw/ptb-xl')
OUT = Path('/data1/jiahui/biosignal-agent/datasets/processed/ptbxl_superclass_12lead_full_manifest.json')
CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

def ensure_metadata(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    import urllib.request
    for name in ['ptbxl_database.csv', 'scp_statements.csv']:
        path = raw_dir / name
        if not path.exists():
            urllib.request.urlretrieve(PTB_URL + name, path)

def labels_for(code_text: str, scp: pd.DataFrame) -> list[str]:
    codes = ast.literal_eval(code_text)
    labels = set()
    for code in codes:
        if code in scp.index:
            row = scp.loc[code]
            cls = row.get('diagnostic_class')
            if bool(row.get('diagnostic', False)) and isinstance(cls, str) and cls in CLASSES:
                labels.add(str(cls))
    return sorted(labels)

def local_record_exists(raw_dir: Path, rec: str) -> bool:
    return (raw_dir / f'{rec}.hea').exists() and (raw_dir / f'{rec}.dat').exists()

def maybe_download(raw_dir: Path, records: list[str]) -> None:
    missing = [r for r in records if not local_record_exists(raw_dir, r)]
    print(json.dumps({'records_requested': len(records), 'missing': len(missing)}), flush=True)
    for i, rec in enumerate(missing, 1):
        print(f'download {i}/{len(missing)} {rec}', flush=True)
        wfdb.dl_database('ptb-xl', dl_dir=str(raw_dir), records=[rec])

def main() -> None:
    ap = argparse.ArgumentParser(description='Prepare full PTB-XL 12-lead diagnostic-superclass manifest.')
    ap.add_argument('--raw-dir', type=Path, default=RAW)
    ap.add_argument('--manifest', type=Path, default=OUT)
    ap.add_argument('--folds', nargs='*', type=int, default=list(range(1, 11)))
    ap.add_argument('--no-download', action='store_true')
    args = ap.parse_args()
    ensure_metadata(args.raw_dir)
    db = pd.read_csv(args.raw_dir / 'ptbxl_database.csv')
    scp = pd.read_csv(args.raw_dir / 'scp_statements.csv', index_col=0)
    db['diagnostic_classes'] = db['scp_codes'].apply(lambda s: labels_for(s, scp))
    db = db[db['strat_fold'].isin(args.folds)].copy()
    db = db[db['diagnostic_classes'].apply(bool)].sort_values('ecg_id')
    records = [str(x) for x in db['filename_lr'].tolist()]
    if not args.no_download:
        maybe_download(args.raw_dir, records)
    rows = []
    skipped = {}
    for _, row in db.iterrows():
        rec = str(row['filename_lr'])
        if not local_record_exists(args.raw_dir, rec):
            skipped[rec] = 'missing_record'
            continue
        labels = list(row['diagnostic_classes'])
        item = {
            'dataset': 'ptb-xl',
            'ecg_id': int(row['ecg_id']),
            'record': rec,
            'group': int(row['patient_id']),
            'strat_fold': int(row['strat_fold']),
            'path': '',
            'sampling_rate': 100.0,
            'lead': '12lead',
            'diagnostic_classes': labels,
            'modality': 'ecg',
        }
        for cls in CLASSES:
            item[f'label_{cls.lower()}'] = int(cls in labels)
        rows.append(item)
    label_counts = {cls: int(sum(r[f'label_{cls.lower()}'] for r in rows)) for cls in CLASSES}
    fold_counts = dict(Counter(str(r['strat_fold']) for r in rows))
    manifest = {'dataset': 'ptb-xl-superclass-12lead-full', 'classes': CLASSES, 'rows': rows, 'num_records': len(rows), 'label_counts': label_counts, 'fold_counts': fold_counts, 'skipped': skipped}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: manifest[k] for k in ['dataset', 'num_records', 'label_counts', 'fold_counts']}, indent=2), flush=True)
    print('skipped', len(skipped), flush=True)

if __name__ == '__main__':
    main()
