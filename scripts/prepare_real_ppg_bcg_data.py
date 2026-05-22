from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
import wfdb


def clean_signal_name(name: str) -> str:
    return name.strip().strip(',').upper()


def select_channel(sig_names: list[str], candidates: list[str]) -> int:
    cleaned = [clean_signal_name(name) for name in sig_names]
    for candidate in candidates:
        candidate = candidate.upper()
        for idx, name in enumerate(cleaned):
            if name == candidate or candidate in name:
                return idx
    raise ValueError(f'Could not find any of {candidates} in {sig_names}')


def maybe_resample(values: np.ndarray, source_fs: float, target_fs: float | None) -> tuple[np.ndarray, float]:
    if target_fs is None or abs(source_fs - target_fs) < 1e-9:
        return values, float(source_fs)
    source_i = int(round(source_fs))
    target_i = int(round(target_fs))
    gcd = math.gcd(source_i, target_i)
    return resample_poly(values, target_i // gcd, source_i // gcd), float(target_i)


def export_record(db: str, record: str, channel_candidates: list[str], modality: str, seconds: float, out_dir: Path, target_fs: float | None = None) -> dict:
    header = wfdb.rdheader(record, pn_dir=db)
    source_fs = float(header.fs)
    sampto = int(round(seconds * source_fs))
    wfdb_record = wfdb.rdrecord(record, pn_dir=db, sampto=sampto)
    channel_idx = select_channel(wfdb_record.sig_name, channel_candidates)
    values = wfdb_record.p_signal[:, channel_idx].astype(float)
    values = values[np.isfinite(values)]
    values, sampling_rate = maybe_resample(values, source_fs, target_fs)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{db}_{record}_{modality}_{int(seconds)}s.csv'
    pd.DataFrame({'signal': values}).to_csv(out_path, index=False)
    return {
        'dataset': db,
        'record': record,
        'modality': modality,
        'path': str(out_path),
        'sampling_rate': sampling_rate,
        'source_sampling_rate': source_fs,
        'seconds': seconds,
        'source_channel': wfdb_record.sig_name[channel_idx],
        'num_samples': int(len(values)),
        'note': 'CEBSDB SCG channel is used as a BCG-like mechanical cardiac signal proxy.' if db == 'cebsdb' else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Export real public PPG and BCG-like mechanical signal CSVs from PhysioNet.')
    parser.add_argument('--out-dir', default='/data1/jiahui/biosignal-agent/datasets/processed/real_world')
    parser.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/datasets/processed/real_world_manifest.json')
    parser.add_argument('--seconds', type=float, default=60.0)
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--bcg-target-fs', type=float, default=250.0, help='Downsample CEBSDB SCG proxy to this sampling rate for faster tool execution.')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    manifest = []
    bidmc_records = wfdb.get_record_list('bidmc')[:args.limit]
    cebs_records = wfdb.get_record_list('cebsdb')[:args.limit]
    for record in bidmc_records:
        manifest.append(export_record('bidmc', record, ['PLETH', 'PPG'], 'ppg', args.seconds, out_dir, target_fs=None))
    for record in cebs_records:
        manifest.append(export_record('cebsdb', record, ['SCG'], 'bcg', args.seconds, out_dir, target_fs=args.bcg_target_fs))

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({'records': manifest}, indent=2))
    pd.DataFrame(manifest).to_csv(manifest_path.with_suffix('.csv'), index=False)
    print(json.dumps({'manifest': str(manifest_path), 'csv_manifest': str(manifest_path.with_suffix('.csv')), 'num_records': len(manifest)}, indent=2))


if __name__ == '__main__':
    main()
