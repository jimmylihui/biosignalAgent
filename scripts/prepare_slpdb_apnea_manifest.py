from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import wfdb

APNEA_TOKENS = {'H', 'HA', 'OA', 'X', 'CA', 'CAA'}

def _subject_group(record: str) -> str:
    if record.startswith('slp01'):
        return 'slp01'
    if record.startswith('slp02'):
        return 'slp02'
    return record

def _annotation_epoch_labels(record: str) -> dict[int, dict]:
    ann = wfdb.rdann(record, 'st')
    labels = {}
    for sample, aux in zip(ann.sample, ann.aux_note):
        tokens = str(aux).strip().split()
        event_tokens = [tok for tok in tokens[1:] if tok in APNEA_TOKENS]
        epoch30 = int(round(sample / 7500.0))  # slpdb is 250 Hz, annotations every 30 s.
        labels[epoch30] = {
            'sleep_stage': tokens[0] if tokens else '',
            'event_tokens': event_tokens,
            'label': 'apnea' if event_tokens else 'normal',
        }
    return labels

def _read_ecg(record: str):
    rec = wfdb.rdrecord(record)
    sig_names = [str(s).lower() for s in rec.sig_name]
    try:
        idx = sig_names.index('ecg')
    except ValueError:
        idx = next((i for i, name in enumerate(sig_names) if 'ecg' in name), None)
        if idx is None:
            raise RuntimeError(f'No ECG channel in {record}: {rec.sig_name}')
    values = np.asarray(rec.p_signal[:, idx], dtype=np.float32)
    return values, float(rec.fs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-dir', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/raw/slpdb'))
    ap.add_argument('--out-dir', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/processed/slpdb_ecg_minutes'))
    ap.add_argument('--manifest', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/processed/slpdb_apnea_ecg_manifest.json'))
    ap.add_argument('--epoch-s', type=int, default=60)
    ap.add_argument('--limit-records', type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = [x.strip() for x in (args.raw_dir / 'RECORDS').read_text().splitlines() if x.strip()]
    if args.limit_records:
        records = records[:args.limit_records]
    rows = []
    skipped = []
    old_cwd = Path.cwd()
    try:
        import os
        os.chdir(args.raw_dir)
        for i, record in enumerate(records, start=1):
            try:
                values, fs = _read_ecg(record)
                ann = _annotation_epoch_labels(record)
            except Exception as exc:
                skipped.append({'record': record, 'error': f'{type(exc).__name__}:{str(exc)[:160]}'})
                continue
            samples_per_epoch = int(round(args.epoch_s * fs))
            n_epochs = min(len(values) // samples_per_epoch, (max(ann) + 1) // 2 if ann else len(values) // samples_per_epoch)
            print(f'{i}/{len(records)} {record}: fs={fs} minutes={n_epochs} ann30={len(ann)}', flush=True)
            rec_out = args.out_dir / record
            rec_out.mkdir(parents=True, exist_ok=True)
            for minute in range(n_epochs):
                start = minute * samples_per_epoch
                stop = (minute + 1) * samples_per_epoch
                chunk = values[start:stop]
                e0 = ann.get(2 * minute, {'label': 'normal', 'event_tokens': [], 'sleep_stage': ''})
                e1 = ann.get(2 * minute + 1, {'label': 'normal', 'event_tokens': [], 'sleep_stage': ''})
                events = sorted(set(e0.get('event_tokens', []) + e1.get('event_tokens', [])))
                label = 'apnea' if events else 'normal'
                out_path = rec_out / f'{record}_{minute:04d}.csv'
                pd.DataFrame({'value': chunk}).to_csv(out_path, index=False)
                rows.append({
                    'record': _subject_group(record),
                    'segment': record,
                    'minute': int(minute),
                    'path': str(out_path),
                    'sampling_rate': fs,
                    'label': label,
                    'source': 'slpdb',
                    'event_types': events,
                    'sleep_stages': [e0.get('sleep_stage', ''), e1.get('sleep_stage', '')],
                })
    finally:
        import os
        os.chdir(old_cwd)
    manifest = {
        'dataset': 'slpdb',
        'epoch_s': args.epoch_s,
        'num_records': len(set(r['record'] for r in rows)),
        'num_segments': len(set(r['segment'] for r in rows)),
        'num_windows': len(rows),
        'label_counts': dict(Counter(r['label'] for r in rows)),
        'records': rows,
        'skipped_records': skipped,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ['num_records', 'num_segments', 'num_windows', 'label_counts', 'skipped_records']}, indent=2))
if __name__ == '__main__':
    main()
