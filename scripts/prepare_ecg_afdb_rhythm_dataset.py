from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_labeled_arrhythmia_dataset import BEAT_SYMBOLS  # noqa: E402

AFDB_RECORDS = [
    '04015', '04043', '04048', '04126', '04746', '04908', '04936', '05091',
    '05121', '05261', '06426', '06453', '06995', '07162', '07859', '07879',
    '07910', '08215', '08219', '08378', '08405', '08434', '08455'
]
AF_RHYTHMS = {'AFIB', 'AFL'}
NORMAL_RHYTHMS = {'N', 'NSR'}


def clean_rhythm(note: str) -> str | None:
    note = (note or '').replace('\x00', '').strip()
    if not note.startswith('('):
        return None
    token = re.sub(r'[^A-Za-z0-9_+-]', '', note[1:])
    return token or None


def maybe_download(raw_dir: Path, records: list[str]) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    missing = [r for r in records if not (raw_dir / f'{r}.hea').exists()]
    if missing:
        wfdb.dl_database('afdb', dl_dir=str(raw_dir), records=missing)


def rhythm_intervals(raw_dir: Path, record: str, annotator: str, total_samples: int) -> list[dict[str, Any]]:
    ann = wfdb.rdann(str(raw_dir / record), annotator)
    starts = []
    for sample, note in zip(ann.sample, ann.aux_note):
        rhythm = clean_rhythm(note)
        if rhythm:
            starts.append((int(sample), rhythm))
    if not starts:
        starts = [(0, 'UNKNOWN')]
    if starts[0][0] > 0:
        starts.insert(0, (0, starts[0][1]))
    out = []
    for i, (start, rhythm) in enumerate(starts):
        stop = starts[i + 1][0] if i + 1 < len(starts) else total_samples
        out.append({'start_sample': int(start), 'end_sample': int(stop), 'rhythm': rhythm})
    return out


def rhythm_for_window(intervals: list[dict[str, Any]], start: int, stop: int) -> dict[str, Any]:
    overlap: Counter[str] = Counter()
    for item in intervals:
        n = max(0, min(stop, item['end_sample']) - max(start, item['start_sample']))
        if n:
            overlap[item['rhythm']] += n
    rhythm = overlap.most_common(1)[0][0] if overlap else 'UNKNOWN'
    purity = float(overlap[rhythm] / max(1, stop - start)) if overlap else 0.0
    if rhythm in AF_RHYTHMS:
        coarse = 'af'
    elif rhythm in NORMAL_RHYTHMS:
        coarse = 'normal'
    else:
        coarse = 'other_rhythm'
    return {'rhythm_label': rhythm, 'coarse_rhythm_label': coarse, 'rhythm_overlap_samples': dict(overlap), 'rhythm_purity': purity}


def beat_label(symbol: str) -> str:
    if symbol in {'N', 'L', 'R', 'e', 'j'}:
        return 'normal'
    if symbol in {'A', 'a', 'J', 'S'}:
        return 'supraventricular'
    if symbol in {'V', 'E'}:
        return 'ventricular'
    if symbol in {'F', '/', 'f', 'Q'}:
        return 'fusion_paced_unknown'
    return 'other'


def export_record_windows(raw_dir: Path, out_dir: Path, record: str, seconds: int, stride_seconds: int, max_windows_per_record: int | None, min_purity: float, annotator: str, beat_annotator: str) -> list[dict[str, Any]]:
    header = wfdb.rdheader(str(raw_dir / record))
    intervals = rhythm_intervals(raw_dir, record, annotator, int(header.sig_len))
    total_seconds = int(header.sig_len / header.fs)
    rows: list[dict[str, Any]] = []
    kept = 0
    lead = None
    for start_s in range(0, max(0, total_seconds - seconds + 1), stride_seconds):
        if max_windows_per_record is not None and kept >= max_windows_per_record:
            break
        start = int(round(start_s * header.fs)); stop = int(round((start_s + seconds) * header.fs))
        rhythm = rhythm_for_window(intervals, start, stop)
        if rhythm['rhythm_purity'] < min_purity or rhythm['coarse_rhythm_label'] == 'other_rhythm':
            continue
        rec = wfdb.rdrecord(str(raw_dir / record), sampfrom=start, sampto=stop)
        if lead is None:
            lead = rec.sig_name[0]
        lead_idx = rec.sig_name.index(lead) if lead in rec.sig_name else 0
        beats = []
        try:
            ann = wfdb.rdann(str(raw_dir / record), beat_annotator, sampfrom=start, sampto=stop)
            for sample, symbol in zip(ann.sample, ann.symbol):
                if symbol in BEAT_SYMBOLS or beat_annotator == 'qrs':
                    sym = symbol if symbol in BEAT_SYMBOLS else 'N'
                    beats.append({'sample': int(sample - start), 'absolute_sample': int(sample), 'symbol': sym, 'label': beat_label(sym)})
        except Exception:
            beats = []
        if len(beats) < 5:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f'afdb_{record}_{start_s:06d}_{seconds}s_ecg.csv'
        pd.DataFrame({'signal': rec.p_signal[:, lead_idx].astype(float)}).to_csv(out_csv, index=False)
        counts = Counter(b['label'] for b in beats)
        rows.append({
            'dataset': 'afdb_rhythm_windows', 'source_database': 'afdb', 'record': f'afdb_{record}',
            'window_start_s': int(start_s), 'duration_s': int(seconds), 'modality': 'ecg', 'path': str(out_csv),
            'sampling_rate': float(header.fs), 'source_channel': lead or rec.sig_name[lead_idx],
            'num_beats': len(beats), 'beat_label_counts': dict(sorted(counts.items())), 'beat_annotations': beats, **rhythm,
        })
        kept += 1
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description='Prepare MIT-BIH AFDB rhythm windows for AF/non-AF training.')
    ap.add_argument('--raw-dir', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/raw/afdb'))
    ap.add_argument('--out-dir', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/processed/ecg_afdb_rhythm'))
    ap.add_argument('--manifest', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/processed/ecg_afdb_rhythm_manifest.json'))
    ap.add_argument('--records', nargs='*', default=AFDB_RECORDS)
    ap.add_argument('--seconds', type=int, default=60)
    ap.add_argument('--stride-seconds', type=int, default=30)
    ap.add_argument('--max-windows-per-record', type=int, default=300)
    ap.add_argument('--min-purity', type=float, default=0.80)
    ap.add_argument('--annotator', default='atr')
    ap.add_argument('--beat-annotator', default='qrs')
    ap.add_argument('--no-download', action='store_true')
    args = ap.parse_args()
    if not args.no_download:
        maybe_download(args.raw_dir, list(args.records))
    rows = []
    skipped = {}
    for r in args.records:
        try:
            part = export_record_windows(args.raw_dir, args.out_dir, r, args.seconds, args.stride_seconds, args.max_windows_per_record, args.min_purity, args.annotator, args.beat_annotator)
            rows.extend(part)
            print(r, len(part), flush=True)
        except Exception as exc:
            skipped[r] = f'{type(exc).__name__}:{str(exc)[:120]}'
            print('skip', r, skipped[r], flush=True)
    manifest = {
        'dataset': 'afdb_rhythm_windows', 'source_database': 'afdb', 'records': rows, 'num_windows': len(rows),
        'rhythm_counts': dict(Counter(x['coarse_rhythm_label'] for x in rows)),
        'detailed_rhythm_counts': dict(Counter(x['rhythm_label'] for x in rows)),
        'beat_counts': dict(sum((Counter(x['beat_label_counts']) for x in rows), Counter())),
        'window_seconds': args.seconds, 'stride_seconds': args.stride_seconds, 'max_windows_per_record': args.max_windows_per_record,
        'min_purity': args.min_purity, 'skipped_records': skipped,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ['num_windows','rhythm_counts','detailed_rhythm_counts','skipped_records']}, indent=2))


if __name__ == '__main__':
    main()
