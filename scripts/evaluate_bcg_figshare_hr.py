
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import requests
from scipy import signal as scipy_signal

from biosignal_agent.tools.common import bandpass_filter, bpm_from_peaks, interval_regularity
from biosignal_agent.tools.peak_detectors import neurokit_nabian2018_peaks, scipy_adaptive_peaks, robust_std
from scripts.prepare_dedicated_bcg_dataset import figshare_files

RAW_DIR = Path('/data1/jiahui/biosignal-agent/datasets/raw/dedicated_bcg_figshare')
OUT_DIR = Path('/data1/jiahui/biosignal-agent/datasets/processed/dedicated_bcg_eval')
FS = 125.0

def _parse_hms(text: str) -> datetime:
    text = str(text).strip()
    fmt = '%H:%M:%S.%f' if '.' in text else '%H:%M:%S'
    return datetime.strptime(text, fmt)

def _seconds_since(start: datetime, t: datetime) -> float:
    delta = (t - start).total_seconds()
    if delta < -12 * 3600:
        delta += 24 * 3600
    elif delta > 12 * 3600:
        delta -= 24 * 3600
    return delta

def ensure_file(name: str, url: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / name
    if path.exists() and path.stat().st_size > 0:
        return path
    with requests.get(url, stream=True, timeout=90) as r:
        r.raise_for_status()
        with path.open('wb') as fh:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    fh.write(chunk)
    return path

def stream_bcg_segment(name: str, url: str, seconds: float) -> tuple[np.ndarray, datetime]:
    n = int(seconds * FS)
    values = []
    start = None
    # Use local full file when available; otherwise stream just enough rows.
    path = RAW_DIR / name
    source = path.open('r', newline='') if path.exists() else None
    try:
        if source is None:
            resp = requests.get(url, stream=True, timeout=90)
            resp.raise_for_status()
            lines = (line.decode('utf-8') for line in resp.iter_lines() if line)
        else:
            lines = source
        reader = csv.DictReader(lines)
        for row in reader:
            if start is None:
                start = _parse_hms(row['time'])
            values.append(float(row['value']))
            if len(values) >= n:
                break
    finally:
        if source is not None:
            source.close()
        else:
            resp.close()
    if start is None:
        raise RuntimeError(f'no rows read from {name}')
    return np.asarray(values, dtype=float), start

def reference_hr_from_heartbeat(name: str, url: str, start: datetime, seconds: float) -> dict:
    path = ensure_file(name, url)
    df = pd.read_csv(path)
    times = df['Time'].map(_parse_hms)
    offsets = times.map(lambda t: _seconds_since(start, t)).to_numpy(dtype=float)
    rr = pd.to_numeric(df['RR Interval(ms)'], errors='coerce').to_numpy(dtype=float)
    symbols = df['Beat Symbol'].astype(str).to_numpy()
    mask = (offsets >= 0) & (offsets < seconds) & np.isfinite(rr) & (rr > 250) & (rr < 2000) & (symbols != 'X')
    ref_offsets = offsets[mask]
    ref_rr = rr[mask]
    if len(ref_rr) < 3:
        return {'reference_hr_bpm': None, 'reference_beats': int(len(ref_rr)), 'reference_method': 'heartbeat_annotation_median_rr'}
    return {
        'reference_hr_bpm': float(60000.0 / np.median(ref_rr)),
        'reference_beats': int(len(ref_rr)),
        'reference_method': 'heartbeat_annotation_median_rr',
    }


from biosignal_agent.tools.bcg_tools import BCG_detect_j_peaks


def evaluate_subject(record: str, files: dict, seconds: float) -> dict:
    bcg_name = f'{record}_bcg.csv'
    hb_name = f'{record}_heartbeat.csv'
    values, start = stream_bcg_segment(bcg_name, files[bcg_name]['download_url'], seconds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    segment_path = OUT_DIR / f'{record.lower()}_bcg_{int(seconds)}s.csv'
    pd.DataFrame({'signal': values}).to_csv(segment_path, index=False)
    ref = reference_hr_from_heartbeat(hb_name, files[hb_name]['download_url'], start, seconds)
    result = BCG_detect_j_peaks(str(segment_path), FS)
    pred_hr = result.get('heart_rate_bpm')
    row = {
        'record': record,
        'seconds': seconds,
        'start_time': start.strftime('%H:%M:%S.%f'),
        'predicted_hr_bpm': pred_hr,
        'predicted_beats': int(result.get('num_peaks', 0)),
        'selected_candidate': result.get('peak_detector_selected'),
        'interval_cv': result.get('interval_cv'),
        'confidence': result.get('confidence'),
        **ref,
    }
    if pred_hr is not None and ref.get('reference_hr_bpm') is not None:
        row['hr_abs_error_bpm'] = float(abs(pred_hr - ref['reference_hr_bpm']))
    else:
        row['hr_abs_error_bpm'] = None
    row['details'] = {k: v for k, v in result.items() if k not in {'j_peak_indices'}}
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects', type=int, default=12)
    parser.add_argument('--seconds', type=float, default=60.0)
    parser.add_argument('--output', type=Path, default=OUT_DIR / 'bcg_figshare_hr_eval.json')
    args = parser.parse_args()
    files = {f['name']: f for f in figshare_files()}
    records = [f'Sub{i:02d}' for i in range(1, args.subjects + 1) if f'Sub{i:02d}_bcg.csv' in files and f'Sub{i:02d}_heartbeat.csv' in files]
    rows = []
    for record in records:
        print('evaluate', record, flush=True)
        rows.append(evaluate_subject(record, files, args.seconds))
    valid = [r['hr_abs_error_bpm'] for r in rows if r.get('hr_abs_error_bpm') is not None]
    summary = {
        'dataset': 'figshare_bed_bcg_2025',
        'reference': 'heartbeat annotations derived from paired ECG',
        'subjects_requested': args.subjects,
        'subjects_evaluated': len(rows),
        'seconds_per_subject': args.seconds,
        'hr_mae_bpm': float(np.mean(valid)) if valid else None,
        'hr_median_ae_bpm': float(np.median(valid)) if valid else None,
        'hr_within_5_bpm': float(np.mean(np.asarray(valid) <= 5.0)) if valid else None,
        'rows': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2))
    print('wrote', args.output)


if __name__ == '__main__':
    main()
