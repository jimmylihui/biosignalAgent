from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy import signal as scipy_signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biosignal_agent.tools.scg_tools import SCG_estimate_respiration

RAW_DIR = Path('/data1/jiahui/biosignal-agent/datasets/raw/cebsdb')
OUT = Path('/data1/jiahui/biosignal-agent/outputs/scg_respiration_cebs_benchmark.json')


def complete_records(raw_dir: Path) -> list[str]:
    return [p.with_suffix('').name for p in sorted(raw_dir.glob('b*.hea')) if p.with_suffix('.dat').exists() and p.with_suffix('.atr').exists()]


def preprocess(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x - np.nanmedian(x)
    scale = np.nanpercentile(np.abs(x), 95) + 1e-8
    return np.clip(x / scale, -8, 8)


def read_record(rec: str, target_fs: float = 250.0):
    r = wfdb.rdrecord(str(RAW_DIR / rec))
    names = list(r.sig_name)
    fs = float(r.fs)
    scg = preprocess(r.p_signal[:, names.index('SCG')])
    resp = preprocess(r.p_signal[:, names.index('RESP')])
    if fs != target_fs:
        n = int(round(len(scg) * target_fs / fs))
        scg = scipy_signal.resample(scg, n)
        resp = scipy_signal.resample(resp, n)
    return scg.astype(np.float32), resp.astype(np.float32), target_fs


def band_limited_rate(x: np.ndarray, fs: float, low: float = 0.08, high: float = 0.7) -> tuple[float | None, float]:
    high = min(high, 0.45 * fs)
    if len(x) < fs * 10 or high <= low:
        return None, 0.0
    sos = scipy_signal.butter(3, [low / (0.5 * fs), high / (0.5 * fs)], btype='bandpass', output='sos')
    y = scipy_signal.sosfiltfilt(sos, x - np.nanmedian(x))
    nperseg = min(len(y), int(fs * 32))
    freqs, psd = scipy_signal.welch(y, fs=fs, nperseg=nperseg)
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return None, 0.0
    sub_f, sub_p = freqs[mask], psd[mask]
    rate = float(sub_f[int(np.argmax(sub_p))] * 60.0)
    ratio = float(np.trapezoid(sub_p, sub_f) / (np.trapezoid(psd, freqs) + 1e-12))
    return rate, ratio


def envelope_rate(scg: np.ndarray, fs: float, cardiac_low: float, cardiac_high: float, resp_low: float = 0.08, resp_high: float = 0.7):
    hi = min(cardiac_high, 0.45 * fs)
    if hi <= cardiac_low:
        return None, 0.0
    sos = scipy_signal.butter(3, [cardiac_low / (0.5 * fs), hi / (0.5 * fs)], btype='bandpass', output='sos')
    y = scipy_signal.sosfiltfilt(sos, scg - np.nanmedian(scg))
    env = np.abs(scipy_signal.hilbert(y))
    return band_limited_rate(env, fs, resp_low, resp_high)


def tool_rate(scg: np.ndarray, fs: float, rec: str):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / f'{rec}.csv'
        pd.DataFrame({'signal': scg}).to_csv(p, index=False)
        out = SCG_estimate_respiration(str(p), fs)
    return out.get('respiratory_rate_bpm'), out


def mae(rows, key):
    vals = [abs(r[key] - r['ref_rr_bpm']) for r in rows if r.get(key) is not None and r.get('ref_rr_bpm') is not None]
    return float(np.mean(vals)) if vals else None


def main():
    rows=[]
    for rec in complete_records(RAW_DIR):
        scg, resp, fs = read_record(rec)
        ref, ref_power = band_limited_rate(resp, fs)
        direct, direct_power = band_limited_rate(scg, fs)
        env_08_20, env_08_20_power = envelope_rate(scg, fs, 0.8, 20.0)
        env_5_35, env_5_35_power = envelope_rate(scg, fs, 5.0, 35.0)
        env_10_35, env_10_35_power = envelope_rate(scg, fs, 10.0, 35.0)
        tool, tool_out = tool_rate(scg, fs, rec)
        rows.append({
            'record': rec,
            'ref_rr_bpm': ref,
            'ref_resp_power_ratio': ref_power,
            'tool_rr_bpm': tool,
            'direct_rr_bpm': direct,
            'direct_power_ratio': direct_power,
            'env_0p8_20_rr_bpm': env_08_20,
            'env_0p8_20_power_ratio': env_08_20_power,
            'env_5_35_rr_bpm': env_5_35,
            'env_5_35_power_ratio': env_5_35_power,
            'env_10_35_rr_bpm': env_10_35,
            'env_10_35_power_ratio': env_10_35_power,
            'tool_confidence': tool_out.get('confidence'),
        })
    summary={
        'n_records': len(rows),
        'tool_mae_bpm': mae(rows, 'tool_rr_bpm'),
        'direct_mae_bpm': mae(rows, 'direct_rr_bpm'),
        'env_0p8_20_mae_bpm': mae(rows, 'env_0p8_20_rr_bpm'),
        'env_5_35_mae_bpm': mae(rows, 'env_5_35_rr_bpm'),
        'env_10_35_mae_bpm': mae(rows, 'env_10_35_rr_bpm'),
    }
    report={'summary': summary, 'per_record': rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
