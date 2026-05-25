from __future__ import annotations
import argparse, ast, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

PTB_URL = 'https://physionet.org/files/ptb-xl/1.0.3/'
RAW = Path('/data1/jiahui/biosignal-agent/datasets/raw/ptb-xl')
OUT = Path('/data1/jiahui/biosignal-agent/datasets/processed/ptbxl_superclass_lead2')


def labels_for(code_text: str, scp: pd.DataFrame) -> set[str]:
    codes = ast.literal_eval(code_text)
    out = set()
    for code in codes:
        if code in scp.index:
            row = scp.loc[code]
            if bool(row.get('diagnostic', False)) and isinstance(row.get('diagnostic_class'), str):
                out.add(str(row['diagnostic_class']))
    return out


def ensure_metadata(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    import urllib.request
    for name in ['ptbxl_database.csv', 'scp_statements.csv']:
        path = raw_dir / name
        if not path.exists():
            urllib.request.urlretrieve(PTB_URL + name, path)


def local_record_exists(raw_dir: Path, rec: str) -> bool:
    return (raw_dir / f'{rec}.hea').exists() and (raw_dir / f'{rec}.dat').exists()


def maybe_download(raw_dir: Path, records: list[str]) -> None:
    missing = [r for r in records if not local_record_exists(raw_dir, r)]
    for i, rec in enumerate(missing, 1):
        print('download', i, '/', len(missing), rec, flush=True)
        wfdb.dl_database('ptb-xl', dl_dir=str(raw_dir), records=[rec])


def main() -> None:
    ap = argparse.ArgumentParser(description='Prepare a PTB-XL diagnostic-superclass lead-II benchmark subset.')
    ap.add_argument('--raw-dir', type=Path, default=RAW)
    ap.add_argument('--out-dir', type=Path, default=OUT)
    ap.add_argument('--manifest', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/processed/ptbxl_superclass_lead2_manifest.json'))
    ap.add_argument('--max-per-class', type=int, default=350)
    ap.add_argument('--folds', nargs='*', type=int, default=list(range(1, 11)))
    ap.add_argument('--no-download', action='store_true')
    args = ap.parse_args()
    ensure_metadata(args.raw_dir)
    db = pd.read_csv(args.raw_dir / 'ptbxl_database.csv')
    scp = pd.read_csv(args.raw_dir / 'scp_statements.csv', index_col=0)
    db['diagnostic_classes'] = db['scp_codes'].apply(lambda s: sorted(labels_for(s, scp)))
    db = db[db['strat_fold'].isin(args.folds)].copy()
    targets = ['NORM', 'CD', 'STTC']
    chosen = []
    counts = Counter()
    used = set()
    per_fold_target = max(1, int(np.ceil(args.max_per_class / max(1, len(args.folds)))))
    for target in targets:
        per_fold_counts = Counter()
        sub = db[db['diagnostic_classes'].apply(lambda xs: target in xs)].sort_values(['ecg_id'])
        for fold in args.folds:
            fold_sub = sub[sub['strat_fold'] == fold]
            for _, row in fold_sub.iterrows():
                if counts[target] >= args.max_per_class or per_fold_counts[fold] >= per_fold_target:
                    break
                rec = str(row['filename_lr'])
                if int(row['ecg_id']) in used:
                    continue
                chosen.append(row)
                used.add(int(row['ecg_id']))
                per_fold_counts[fold] += 1
                for cls in row['diagnostic_classes']:
                    if cls in targets:
                        counts[cls] += 1
    chosen_df = pd.DataFrame(chosen)
    records = [str(x) for x in chosen_df['filename_lr'].tolist()]
    if not args.no_download:
        maybe_download(args.raw_dir, records)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped = {}
    for i, row in chosen_df.iterrows():
        rec = str(row['filename_lr'])
        try:
            wr = wfdb.rdrecord(str(args.raw_dir / rec))
            names = list(wr.sig_name)
            lead_idx = names.index('II') if 'II' in names else min(1, wr.p_signal.shape[1] - 1)
            signal = wr.p_signal[:, lead_idx].astype(float)
            out_csv = args.out_dir / f"ptbxl_{int(row['ecg_id']):05d}_leadII_ecg.csv"
            pd.DataFrame({'signal': signal}).to_csv(out_csv, index=False)
            labels = list(row['diagnostic_classes'])
            rows.append({
                'dataset': 'ptb-xl',
                'ecg_id': int(row['ecg_id']),
                'record': rec,
                'group': int(row['patient_id']),
                'strat_fold': int(row['strat_fold']),
                'path': str(out_csv),
                'sampling_rate': float(wr.fs),
                'lead': names[lead_idx],
                'diagnostic_classes': labels,
                'label_norm': int('NORM' in labels),
                'label_cd': int('CD' in labels),
                'label_sttc': int('STTC' in labels),
                'modality': 'ecg',
            })
        except Exception as exc:
            skipped[rec] = f'{type(exc).__name__}:{str(exc)[:160]}'
            print('skip', rec, skipped[rec], flush=True)
    label_counts = {k: int(sum(r[f'label_{k.lower()}'] for r in rows)) for k in ['NORM','CD','STTC']}
    manifest = {'dataset': 'ptb-xl-superclass-leadII', 'rows': rows, 'num_records': len(rows), 'label_counts': label_counts, 'fold_counts': dict(Counter(str(r['strat_fold']) for r in rows)), 'skipped': skipped}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: manifest[k] for k in ['num_records','label_counts','fold_counts','skipped']}, indent=2))


if __name__ == '__main__':
    main()
