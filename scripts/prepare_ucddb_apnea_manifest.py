from __future__ import annotations
import argparse, json, re, sys, warnings
from datetime import date, datetime, time, timedelta
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

try:
    import edfio
except ImportError as exc:
    raise SystemExit('edfio is required: pip install edfio') from exc

EVENT_TYPES = ('HYP', 'O', 'C', 'M')
TIME_RE = re.compile(r'^(\d{2}:\d{2}:\d{2})\s+(\S+)\s+(?:\S*\s+)?(\d+(?:\.\d+)?)')

def _seconds_of_day(text: str) -> int:
    h, m, s = [int(x) for x in text.split(':')]
    return h * 3600 + m * 60 + s

def parse_events(path: Path, start_seconds: int, min_duration_s: float = 5.0) -> list[tuple[float, float, str]]:
    events=[]
    if not path.exists():
        return events
    for raw in path.read_text(errors='ignore').splitlines():
        raw=raw.strip()
        m=TIME_RE.match(raw)
        if not m:
            continue
        tstr, typ, dur = m.group(1), m.group(2), float(m.group(3))
        typ=typ.upper()
        if dur < min_duration_s or not typ.startswith(EVENT_TYPES):
            continue
        sec=_seconds_of_day(tstr) - start_seconds
        if sec < -3600:
            sec += 24 * 3600
        if sec < 0:
            continue
        events.append((float(sec), float(sec + dur), typ))
    return events

def event_overlap_label(start: float, stop: float, events: list[tuple[float,float,str]], min_overlap_s: float) -> tuple[str, list[str], float]:
    max_overlap=0.0; types=[]
    for ev_start, ev_stop, typ in events:
        overlap=max(0.0, min(stop, ev_stop) - max(start, ev_start))
        if overlap > 0:
            max_overlap=max(max_overlap, overlap); types.append(typ)
    return ('apnea' if max_overlap >= min_overlap_s else 'normal', sorted(set(types)), max_overlap)

def read_ecg_from_rec(path: Path) -> tuple[np.ndarray, float, int, float]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        edf=edfio.read_edf(path)
    warning_text = ' '.join(str(w.message) for w in caught)
    if 'Incomplete data record' in warning_text or 'file contains' in warning_text:
        raise RuntimeError(f'incomplete_edf:{warning_text[:120]}')
    labels=[s.label.strip().lower() for s in edf.signals]
    try:
        idx=labels.index('ecg')
    except ValueError:
        idx=next((i for i,label in enumerate(labels) if 'ecg' in label), None)
        if idx is None:
            raise RuntimeError(f'No ECG channel in {path}')
    sig=edf.signals[idx]
    data=np.asarray(sig.data, dtype=np.float32)
    fs=float(sig.sampling_frequency)
    st=edf.starttime
    start_seconds=st.hour * 3600 + st.minute * 60 + st.second
    return data, fs, start_seconds, float(edf.duration)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw-dir', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/raw/ucddb'))
    ap.add_argument('--out-dir', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/processed/ucddb_ecg_minutes'))
    ap.add_argument('--manifest', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/processed/ucddb_apnea_ecg_manifest.json'))
    ap.add_argument('--epoch-s', type=int, default=60)
    ap.add_argument('--min-overlap-s', type=float, default=10.0)
    ap.add_argument('--limit-records', type=int, default=0)
    args=ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    recs=sorted(args.raw_dir.glob('ucddb*.rec'))
    if args.limit_records:
        recs=recs[:args.limit_records]
    rows=[]; skipped=[]
    for rec_i, rec_path in enumerate(recs, start=1):
        stem=rec_path.stem
        event_path=args.raw_dir / f'{stem}_respevt.txt'
        try:
            values, fs, start_seconds, duration_s = read_ecg_from_rec(rec_path)
            events=parse_events(event_path, start_seconds)
        except Exception as exc:
            skipped.append({'record': stem, 'error': f'{type(exc).__name__}:{str(exc)[:120]}'})
            continue
        n_epochs=int(len(values) // int(args.epoch_s * fs))
        print(f'{rec_i}/{len(recs)} {stem}: fs={fs} epochs={n_epochs} events={len(events)}', flush=True)
        rec_out=args.out_dir / stem
        rec_out.mkdir(parents=True, exist_ok=True)
        for minute in range(n_epochs):
            start_idx=int(minute * args.epoch_s * fs)
            stop_idx=int((minute + 1) * args.epoch_s * fs)
            chunk=values[start_idx:stop_idx]
            if len(chunk) < int(0.9 * args.epoch_s * fs):
                continue
            label, types, overlap=event_overlap_label(minute * args.epoch_s, (minute + 1) * args.epoch_s, events, args.min_overlap_s)
            out_path=rec_out / f'{stem}_{minute:04d}.csv'
            pd.DataFrame({'value': chunk}).to_csv(out_path, index=False)
            rows.append({'record': stem, 'minute': int(minute), 'path': str(out_path), 'sampling_rate': fs, 'label': label, 'source': 'ucddb', 'event_types': types, 'event_overlap_s': overlap})
    manifest={'dataset':'ucddb','epoch_s':args.epoch_s,'min_overlap_s':args.min_overlap_s,'num_records':len(set(r['record'] for r in rows)),'num_windows':len(rows),'label_counts':dict(Counter(r['label'] for r in rows)),'records':rows,'skipped_records':skipped}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k:manifest[k] for k in ['num_records','num_windows','label_counts','skipped_records']}, indent=2))
if __name__ == '__main__':
    main()
