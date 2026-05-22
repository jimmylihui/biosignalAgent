from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from urllib.request import urlretrieve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import soundfile as sf
import wfdb
from scipy.signal import resample_poly


def clean_name(name: str) -> str:
    return name.strip().strip(',').upper()


def select_channel(sig_names: list[str], candidates: list[str]) -> int:
    cleaned = [clean_name(name) for name in sig_names]
    for candidate in candidates:
        candidate = candidate.upper()
        for idx, name in enumerate(cleaned):
            if name == candidate or candidate in name:
                return idx
    raise ValueError(f'Could not find any of {candidates} in {sig_names}')


def maybe_resample(values: np.ndarray, source_fs: float, target_fs: float | None) -> tuple[np.ndarray, float]:
    if target_fs is None or abs(float(source_fs) - float(target_fs)) < 1e-9:
        return values, float(source_fs)
    source_i = int(round(source_fs))
    target_i = int(round(target_fs))
    gcd = math.gcd(source_i, target_i)
    return resample_poly(values, target_i // gcd, source_i // gcd), float(target_i)


def write_csv(values: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'signal': values.astype(float)}).to_csv(path, index=False)


def export_wfdb_channel(db: str, record: str, modality: str, candidates: list[str], seconds: float, out_dir: Path, target_fs: float | None = None) -> dict:
    header = wfdb.rdheader(record, pn_dir=db)
    source_fs = float(header.fs)
    sampto = int(round(seconds * source_fs))
    rec = wfdb.rdrecord(record, pn_dir=db, sampto=sampto)
    idx = select_channel(rec.sig_name, candidates)
    values = rec.p_signal[:, idx].astype(float)
    values = values[np.isfinite(values)]
    values, fs = maybe_resample(values, source_fs, target_fs)
    safe_record = record.replace('/', '_')
    out_path = out_dir / f'{db}_{safe_record}_{modality}_{int(seconds)}s.csv'
    write_csv(values, out_path)
    return {
        'dataset': db,
        'record': record,
        'modality': modality,
        'path': str(out_path),
        'sampling_rate': fs,
        'source_sampling_rate': source_fs,
        'seconds': seconds,
        'source_channel': rec.sig_name[idx],
        'num_samples': int(len(values)),
    }


def export_noneeg_acc_eda(record: str, seconds: float, out_dir: Path) -> list[dict]:
    rec = wfdb.rdrecord(record, pn_dir='noneeg', sampto=int(round(seconds * 8)))
    names = [clean_name(name) for name in rec.sig_name]
    rows = []
    axes = []
    for name in ['AX', 'AY', 'AZ']:
        axes.append(rec.p_signal[:, names.index(name)].astype(float))
    magnitude = np.sqrt(sum(axis ** 2 for axis in axes))
    safe_record = record.replace('/', '_')
    acc_path = out_dir / f'noneeg_{safe_record}_acc_{int(seconds)}s.csv'
    write_csv(magnitude, acc_path)
    rows.append({'dataset': 'noneeg', 'record': record, 'modality': 'acc', 'path': str(acc_path), 'sampling_rate': float(rec.fs), 'source_sampling_rate': float(rec.fs), 'seconds': seconds, 'source_channel': 'sqrt(ax^2+ay^2+az^2)', 'num_samples': int(len(magnitude))})
    eda = rec.p_signal[:, names.index('EDA')].astype(float)
    eda_path = out_dir / f'noneeg_{safe_record}_eda_{int(seconds)}s.csv'
    write_csv(eda, eda_path)
    rows.append({'dataset': 'noneeg', 'record': record, 'modality': 'eda', 'path': str(eda_path), 'sampling_rate': float(rec.fs), 'source_sampling_rate': float(rec.fs), 'seconds': seconds, 'source_channel': 'EDA', 'num_samples': int(len(eda))})
    return rows


def export_pcg(record_id: str, seconds: float, out_dir: Path, raw_dir: Path) -> dict:
    url = f'https://physionet.org/files/challenge-2016/1.0.0/training-a/{record_id}.wav'
    wav_path = raw_dir / 'challenge-2016' / 'training-a' / f'{record_id}.wav'
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    if not wav_path.exists():
        urlretrieve(url, wav_path)
    values, fs = sf.read(wav_path)
    if values.ndim > 1:
        values = values[:, 0]
    values = values[: int(round(seconds * fs))]
    out_path = out_dir / f'challenge2016_{record_id}_pcg_{int(seconds)}s.csv'
    write_csv(np.asarray(values), out_path)
    return {'dataset': 'challenge-2016', 'record': record_id, 'modality': 'pcg', 'path': str(out_path), 'sampling_rate': float(fs), 'source_sampling_rate': float(fs), 'seconds': min(seconds, len(values) / fs), 'source_channel': 'wav', 'num_samples': int(len(values))}


def read_edf_first_channel(path: Path, seconds: float) -> tuple[np.ndarray, float, str]:
    with path.open('rb') as handle:
        fixed = handle.read(256)
        header_bytes = int(fixed[184:192].decode('ascii').strip())
        num_records = int(float(fixed[236:244].decode('ascii').strip()))
        duration = float(fixed[244:252].decode('ascii').strip())
        ns = int(fixed[252:256].decode('ascii').strip())
        signal_header = handle.read(header_bytes - 256)
        offset = 0
        labels = [signal_header[offset + i * 16: offset + (i + 1) * 16].decode('ascii').strip() for i in range(ns)]
        offset += 16 * ns + 80 * ns
        phys_dims = [signal_header[offset + i * 8: offset + (i + 1) * 8].decode('ascii').strip() for i in range(ns)]
        offset += 8 * ns
        phys_min = np.array([float(signal_header[offset + i * 8: offset + (i + 1) * 8].decode('ascii').strip()) for i in range(ns)])
        offset += 8 * ns
        phys_max = np.array([float(signal_header[offset + i * 8: offset + (i + 1) * 8].decode('ascii').strip()) for i in range(ns)])
        offset += 8 * ns
        dig_min = np.array([float(signal_header[offset + i * 8: offset + (i + 1) * 8].decode('ascii').strip()) for i in range(ns)])
        offset += 8 * ns
        dig_max = np.array([float(signal_header[offset + i * 8: offset + (i + 1) * 8].decode('ascii').strip()) for i in range(ns)])
        offset += 8 * ns + 80 * ns
        samples_per_record = np.array([int(signal_header[offset + i * 8: offset + (i + 1) * 8].decode('ascii').strip()) for i in range(ns)])
        fs = float(samples_per_record[0] / duration)
        records_to_read = min(num_records, int(math.ceil(seconds / duration)))
        first = []
        for _ in range(records_to_read):
            for sig_idx in range(ns):
                raw = np.frombuffer(handle.read(samples_per_record[sig_idx] * 2), dtype='<i2').astype(float)
                if sig_idx == 0:
                    scale = (phys_max[0] - phys_min[0]) / (dig_max[0] - dig_min[0])
                    first.append((raw - dig_min[0]) * scale + phys_min[0])
        values = np.concatenate(first)[: int(round(seconds * fs))]
        return values, fs, labels[0]


def export_eeg(record: str, seconds: float, out_dir: Path, raw_dir: Path) -> dict:
    url = f'https://physionet.org/files/eegmmidb/1.0.0/{record}'
    edf_path = raw_dir / 'eegmmidb' / record
    edf_path.parent.mkdir(parents=True, exist_ok=True)
    if not edf_path.exists():
        urlretrieve(url, edf_path)
    values, fs, channel = read_edf_first_channel(edf_path, seconds)
    safe_record = record.replace('/', '_').replace('.edf', '')
    out_path = out_dir / f'eegmmidb_{safe_record}_eeg_{int(seconds)}s.csv'
    write_csv(values, out_path)
    return {'dataset': 'eegmmidb', 'record': record, 'modality': 'eeg', 'path': str(out_path), 'sampling_rate': fs, 'source_sampling_rate': fs, 'seconds': seconds, 'source_channel': channel, 'num_samples': int(len(values))}


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare dedicated public datasets for common BioSignalAgent modalities.')
    parser.add_argument('--out-dir', default='/data1/jiahui/biosignal-agent/datasets/processed/dedicated_common')
    parser.add_argument('--raw-dir', default='/data1/jiahui/biosignal-agent/datasets/raw/dedicated_common')
    parser.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/datasets/processed/dedicated_common_manifest.json')
    parser.add_argument('--seconds', type=float, default=60.0)
    parser.add_argument('--limit', type=int, default=3)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    raw_dir = Path(args.raw_dir)
    manifest = []

    noneeg_subjects = [f'Subject{i}' for i in range(10, 10 + args.limit)]
    for subject in noneeg_subjects:
        manifest.extend(export_noneeg_acc_eda(f'{subject}_AccTempEDA', args.seconds, out_dir))
        manifest.append(export_wfdb_channel('noneeg', f'{subject}_SpO2HR', 'spo2', ['SpO2'], args.seconds, out_dir))
    for record in ['slp01a', 'slp01b', 'slp02a'][:args.limit]:
        manifest.append(export_wfdb_channel('slpdb', record, 'abp', ['BP'], args.seconds, out_dir))
    for record in ['a0001', 'a0002', 'a0003'][:args.limit]:
        manifest.append(export_pcg(record, args.seconds, out_dir, raw_dir))
    for record in ['S001/S001R01.edf', 'S002/S002R01.edf', 'S003/S003R01.edf'][:args.limit]:
        manifest.append(export_eeg(record, args.seconds, out_dir, raw_dir))
    for record in wfdb.get_record_list('emgdb')[:args.limit]:
        manifest.append(export_wfdb_channel('emgdb', record, 'emg', ['EMG'], min(args.seconds, 10.0), out_dir, target_fs=1000.0))

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({'records': manifest}, indent=2))
    pd.DataFrame(manifest).to_csv(manifest_path.with_suffix('.csv'), index=False)
    print(json.dumps({'manifest': str(manifest_path), 'csv_manifest': str(manifest_path.with_suffix('.csv')), 'num_records': len(manifest)}, indent=2))


if __name__ == '__main__':
    main()
