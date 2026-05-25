from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

CANDIDATES = {
    'lowres_direct': '/data1/jiahui/biosignal-agent/outputs/waveform_digitization_ml_eval.json',
    'lowres_trace_path': '/data1/jiahui/biosignal-agent/outputs/waveform_digitization_ml_trace_path_eval.json',
    'lowres_trace_momentum': '/data1/jiahui/biosignal-agent/outputs/waveform_digitization_ml_trace_momentum_eval.json',
    'x4_median': '/data1/jiahui/biosignal-agent/outputs/lowres_recovery_digitization_lanczos_eval.json',
    'x4_path': '/data1/jiahui/biosignal-agent/outputs/lowres_recovery_digitization_lanczos_trace_path_eval.json',
    'x4_momentum': '/data1/jiahui/biosignal-agent/outputs/lowres_recovery_digitization_lanczos_trace_momentum_eval.json',
    'signal_sr_pchip': '/data1/jiahui/biosignal-agent/outputs/lowres_signal_sr_pchip_eval.json',
    'ecg_unet_t08': '/data1/jiahui/biosignal-agent/outputs/lowres_recovery_digitization_ecg_unet_t0.8_eval.json',
    'scg_unet_t08': '/data1/jiahui/biosignal-agent/outputs/lowres_recovery_digitization_scg_unet_t0.8_eval.json',
    'pcg_unet_t07': '/data1/jiahui/biosignal-agent/outputs/lowres_recovery_digitization_pcg_unet_t0.7_eval.json',
    'emg_unet_t07': '/data1/jiahui/biosignal-agent/outputs/lowres_recovery_digitization_emg_unet_t0.7_eval.json',
}

BANDS = [None, (0.3, 8), (0.5, 12), (0.8, 25), (1, 35), (2, 40), (5, 80), (20, 150), (20, 250)]
DISTANCES = [0.18, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
PROMINENCES = [0.15, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5]


def load_signal(path: str):
    frame = pd.read_csv(path)
    col = 'signal' if 'signal' in frame.columns else frame.select_dtypes('number').columns[-1]
    values = frame[col].to_numpy(dtype=float)
    return values[np.isfinite(values)]


def peaks(values: np.ndarray, fs: float, distance_s: float, prominence_scale: float, band: tuple[float, float] | None):
    z = values - np.nanmedian(values)
    if band is not None:
        low, high = band
        high = min(float(high), fs * 0.45)
        if low < high:
            sos = scipy_signal.butter(2, [float(low), high], btype='bandpass', fs=fs, output='sos')
            z = scipy_signal.sosfiltfilt(sos, z)
    prom = max(float(np.nanstd(z)) * float(prominence_scale), 1e-8)
    p, _ = scipy_signal.find_peaks(z, distance=max(1, int(fs * distance_s)), prominence=prom)
    return p.astype(int)


def hr_from_peaks(p: np.ndarray, fs: float):
    if len(p) < 2:
        return None
    intervals = np.diff(p) / fs
    intervals = intervals[intervals > 0]
    if len(intervals) == 0:
        return None
    return float(60.0 / np.median(intervals))


def output_path(row: dict[str, Any]) -> str | None:
    return row.get('digitized_csv') or row.get('sr_csv') or row.get('out_csv')


def main() -> None:
    ap = argparse.ArgumentParser(description='Tune HR/peak detector over existing low-res digitized outputs.')
    ap.add_argument('--modality', required=True)
    ap.add_argument('--highres-manifest', default='/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark_highres_aligned_manifest.json')
    ap.add_argument('--out-json', default=None)
    args = ap.parse_args()
    high = {r['record']: r for r in json.loads(Path(args.highres_manifest).read_text())['records']}
    candidate_rows = []
    for name, path in CANDIDATES.items():
        p = Path(path)
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        for row in data.get('rows', []):
            if row.get('modality') != args.modality or row.get('digitizer_error') or row.get('error'):
                continue
            out = output_path(row)
            if out and Path(out).exists() and row.get('record') in high:
                candidate_rows.append((name, row, out))
    results = []
    for candidate_name in sorted({name for name, _, _ in candidate_rows}):
        subset = [(row, out) for name, row, out in candidate_rows if name == candidate_name]
        for distance_s in DISTANCES:
            for prominence in PROMINENCES:
                for band in BANDS:
                    errs = []
                    details = []
                    for row, out in subset:
                        rec = row['record']
                        fs = float(high[rec]['sampling_rate'])
                        ref = load_signal(high[rec]['reference_path'])
                        pred = load_signal(out)
                        ref_hr = hr_from_peaks(peaks(ref, fs, 0.25, 0.5, None), fs)
                        pred_hr = hr_from_peaks(peaks(pred, fs, distance_s, prominence, band), fs)
                        if ref_hr is None or pred_hr is None:
                            continue
                        err = abs(ref_hr - pred_hr)
                        errs.append(err)
                        details.append({'record': rec, 'reference_hr_bpm': ref_hr, 'predicted_hr_bpm': pred_hr, 'hr_abs_error_bpm': err, 'generated_path': out})
                    if errs:
                        results.append({
                            'candidate': candidate_name,
                            'distance_s': distance_s,
                            'prominence_scale': prominence,
                            'band': band,
                            'num_records': len(errs),
                            'mean_hr_abs_error_bpm': float(np.mean(errs)),
                            'median_hr_abs_error_bpm': float(np.median(errs)),
                            'max_hr_abs_error_bpm': float(np.max(errs)),
                            'details': details,
                        })
    results.sort(key=lambda x: (x['mean_hr_abs_error_bpm'], x['max_hr_abs_error_bpm']))
    out = Path(args.out_json or f'/data1/jiahui/biosignal-agent/outputs/{args.modality}_hr_detector_grid.json')
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps({'out_json': str(out), 'top': results[:12]}, indent=2))

if __name__ == '__main__':
    main()
