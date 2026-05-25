from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biosignal_agent.tools.scg_tools import (  # noqa: E402
    SCG_assess_sensor_placement,
    SCG_compute_cardiac_time_intervals,
    SCG_detect_fiducial_points,
    SCG_detect_j_peaks,
    SCG_screen_mechanical_abnormality,
)

RAW = Path('/data1/jiahui/biosignal-agent/datasets/raw/scg_vhd_zenodo')
MAT = RAW / 'MAT_Files/MAT_Files'
JSON_DIR = RAW / 'JSON_Files/JSON_Files'
SUMMARY = RAW / 'Summary_Pub_Deidentified.xlsx'
OUT = Path('/data1/jiahui/biosignal-agent/outputs/scg_vhd_zenodo_common_benchmark.json')
TMP = Path('/data1/jiahui/biosignal-agent/outputs/scg_vhd_zenodo_csv_cache')


def time_to_seconds(x: str) -> float | None:
    if not isinstance(x, str) or not x:
        return None
    parts = x.split(':')
    if len(parts) != 3:
        return None
    h, m, s = parts
    return int(h) * 3600 + int(m) * 60 + float(s)


def times_to_indices(times: list[str], fs: float, n: int) -> np.ndarray:
    vals = []
    for t in times:
        sec = time_to_seconds(t)
        if sec is None:
            continue
        idx = int(round(sec * fs))
        if 0 <= idx < n:
            vals.append(idx)
    return np.asarray(vals, dtype=int)


def load_vectors(pid: str) -> dict[str, np.ndarray]:
    d = loadmat(MAT / f'{pid}-Vectors.mat', squeeze_me=True, struct_as_record=False)
    def arr(*names: str) -> np.ndarray:
        for name in names:
            if name in d:
                return np.asarray(d[name], dtype=float).reshape(-1)
        raise KeyError(names)
    return {
        'scg_x': arr('Shimmer_D0CD_Accel_LN_X_CAL1', 'ECG_Accel_LN_X_CAL1'),
        'scg_y': arr('Shimmer_D0CD_Accel_LN_Y_CAL1', 'ECG_Accel_LN_Y_CAL1'),
        'scg_z': arr('Shimmer_D0CD_Accel_LN_Z_CAL1', 'ECG_Accel_LN_Z_CAL1'),
        'ecg_lara': arr('Shimmer_D0CD_ECG_LARA_24BIT_CAL1', 'ECG_ECG_LARA_24BIT_CAL1'),
    }


def reconstruct_from_first_and_rr(first_time: str | None, rr_ms: np.ndarray, fs: float, n: int) -> np.ndarray:
    first = time_to_seconds(first_time) if first_time else None
    if first is None:
        return np.asarray([], dtype=int)
    rr_s = np.asarray(rr_ms, dtype=float).reshape(-1) / 1000.0
    times = np.concatenate([[first], first + np.cumsum(rr_s)])
    idx = np.asarray(np.round(times * fs), dtype=int)
    return idx[(idx >= 0) & (idx < n)]


def load_refs(pid: str, fs: float, n: int) -> dict[str, np.ndarray]:
    scg_json = json.load(open(JSON_DIR / f'{pid}-SCG.json'))
    ecg_json = json.load(open(JSON_DIR / f'{pid}-ECG.json'))
    scg_mat = loadmat(MAT / f'{pid}-SCG.mat', squeeze_me=True, struct_as_record=False)['data']
    ecg_mat = loadmat(MAT / f'{pid}-ECG.mat', squeeze_me=True, struct_as_record=False)['data']
    scg_rr = list(scg_mat.RR_int)
    ecg_rr = list(ecg_mat.RR_int)
    return {
        'ao_x': reconstruct_from_first_and_rr((scg_json.get('SCG_X_Peaks') or [None])[0], scg_rr[0], fs, n),
        'ao_y': reconstruct_from_first_and_rr((scg_json.get('SCG_Y_Peaks') or [None])[0], scg_rr[1], fs, n),
        'ao_z': reconstruct_from_first_and_rr((scg_json.get('SCG_Z_Peaks') or [None])[0], scg_rr[2], fs, n),
        'r_lara': reconstruct_from_first_and_rr((ecg_json.get('LARA_R_Peaks') or [None])[0], ecg_rr[0], fs, n),
    }



def ecg_guided_ao_from_r(scg: np.ndarray, r_peaks: np.ndarray, fs: float, start_s: float = 0.04, end_s: float = 0.30, low_hz: float = 0.8, high_hz: float = 35.0) -> np.ndarray:
    if len(scg) < fs * 3 or len(r_peaks) == 0:
        return np.asarray([], dtype=int)
    high = min(high_hz, 0.45 * fs)
    centered = scg - np.nanmedian(scg)
    sos = __import__('scipy').signal.butter(3, [low_hz / (0.5 * fs), high / (0.5 * fs)], btype='bandpass', output='sos')
    filtered = __import__('scipy').signal.sosfiltfilt(sos, centered)
    env = np.abs(filtered)
    out = []
    for r in np.asarray(r_peaks, dtype=int):
        lo = max(0, int(r + round(start_s * fs)))
        hi = min(len(env), int(r + round(end_s * fs)))
        if hi <= lo:
            continue
        out.append(lo + int(np.argmax(env[lo:hi])))
    # De-duplicate rare overlapping windows.
    dedup = []
    min_dist = int(round(0.25 * fs))
    for idx in out:
        if not dedup or idx - dedup[-1] >= min_dist:
            dedup.append(idx)
        elif env[idx] > env[dedup[-1]]:
            dedup[-1] = idx
    return np.asarray(dedup, dtype=int)

def match_peaks(ref: np.ndarray, pred: np.ndarray, fs: float, tol_s: float) -> dict[str, Any]:
    ref = np.asarray(ref, dtype=int)
    pred = np.asarray(pred, dtype=int)
    tol = int(round(tol_s * fs))
    used = np.zeros(len(ref), dtype=bool)
    tp = 0
    errs = []
    for p in pred:
        if len(ref) == 0:
            continue
        j = int(np.argmin(np.abs(ref - p)))
        if not used[j] and abs(ref[j] - p) <= tol:
            used[j] = True
            tp += 1
            errs.append((p - ref[j]) / fs)
    fp = len(pred) - tp
    fn = len(ref) - tp
    sens = tp / (tp + fn) if tp + fn else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * sens * ppv / (sens + ppv) if sens + ppv else 0.0
    return {
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
        'sensitivity': float(sens), 'ppv': float(ppv), 'f1': float(f1),
        'timing_mae_ms': float(np.mean(np.abs(errs)) * 1000.0) if errs else None,
    }


def hr_from_peaks(peaks: np.ndarray, fs: float) -> float | None:
    peaks = np.asarray(peaks, dtype=int)
    if len(peaks) < 3:
        return None
    rr = np.diff(peaks) / fs
    rr = rr[(rr >= 0.3) & (rr <= 2.0)]
    return float(60.0 / np.median(rr)) if len(rr) else None


def aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    total = {'tp': 0, 'fp': 0, 'fn': 0}
    maes = []
    for row in rows:
        m = row[key]
        total['tp'] += m['tp']; total['fp'] += m['fp']; total['fn'] += m['fn']
        if m.get('timing_mae_ms') is not None:
            maes.append(m['timing_mae_ms'])
    sens = total['tp'] / (total['tp'] + total['fn']) if total['tp'] + total['fn'] else 0.0
    ppv = total['tp'] / (total['tp'] + total['fp']) if total['tp'] + total['fp'] else 0.0
    f1 = 2 * sens * ppv / (sens + ppv) if sens + ppv else 0.0
    return {**total, 'sensitivity': sens, 'ppv': ppv, 'f1': f1, 'timing_mae_ms': float(np.mean(maes)) if maes else None}


def abnormal_score(risk: str | None) -> float | None:
    return {'low': 0.0, 'borderline_proxy': 0.5, 'elevated_proxy': 1.0}.get(risk)


def classification_metrics(rows: list[dict[str, Any]], label_key: str) -> dict[str, Any]:
    y, score = [], []
    for r in rows:
        label = r.get(label_key)
        s = r.get('mechanical_abnormality_score')
        if label is None or (isinstance(label, float) and math.isnan(label)) or s is None:
            continue
        y.append(int(label)); score.append(float(s))
    if len(set(y)) < 2:
        return {'n': len(y), 'auroc': None, 'auprc': None, 'f1_at_borderline': None}
    pred = [1 if s >= 0.5 else 0 for s in score]
    return {
        'n': len(y),
        'positive_rate': float(np.mean(y)),
        'auroc': float(roc_auc_score(y, score)),
        'auprc': float(average_precision_score(y, score)),
        'f1_at_borderline': float(f1_score(y, pred, zero_division=0)),
    }


def main() -> None:
    TMP.mkdir(parents=True, exist_ok=True)
    summary = pd.read_excel(SUMMARY)
    summary = summary[summary['Patient ID'].astype(str).str.startswith(('CP-', 'UP-'))]
    rows = []
    for _, meta in summary.iterrows():
        pid = str(meta['Patient ID'])
        vec_path = MAT / f'{pid}-Vectors.mat'
        if not vec_path.exists():
            continue
        fs = float(meta['Sampling rate(Hz)'])
        try:
            vectors = load_vectors(pid)
            n = len(vectors['scg_z'])
            refs = load_refs(pid, fs, n)
        except Exception as exc:
            rows.append({'patient_id': pid, 'error': repr(exc)})
            continue
        scg_csv = TMP / f'{pid}_scg_z.csv'
        ecg_csv = TMP / f'{pid}_ecg_lara.csv'
        if not scg_csv.exists():
            pd.DataFrame({'signal': vectors['scg_z']}).to_csv(scg_csv, index=False)
        if not ecg_csv.exists():
            pd.DataFrame({'signal': vectors['ecg_lara']}).to_csv(ecg_csv, index=False)
        try:
            j = SCG_detect_j_peaks(str(scg_csv), fs)
            fid = SCG_detect_fiducial_points(str(scg_csv), fs, ecg_path=str(ecg_csv), ecg_sampling_rate=fs)
            cti = SCG_compute_cardiac_time_intervals(str(scg_csv), fs, ecg_path=str(ecg_csv), ecg_sampling_rate=fs)
            placement = SCG_assess_sensor_placement(str(scg_csv), fs)
            abnormal = SCG_screen_mechanical_abnormality(str(scg_csv), fs, ecg_path=str(ecg_csv), ecg_sampling_rate=fs)
        except Exception as exc:
            rows.append({'patient_id': pid, 'error': repr(exc)})
            continue
        pred_j = np.asarray(j.get('j_peak_indices', []), dtype=int)
        pred_ao = np.asarray(fid.get('ao_indices', []), dtype=int)
        pred_ecg_guided_ao = ecg_guided_ao_from_r(vectors['scg_z'], refs['r_lara'], fs)
        ref_ao = refs['ao_z']
        ref_hr = hr_from_peaks(ref_ao, fs)
        pred_hr = hr_from_peaks(pred_ao, fs)
        label_any_valve = int(max(float(meta.get(c, 0) or 0) for c in ['Moderate or greater MS','Moderate or greater MR','Moderate or greater AR','Moderate or greater AS','moderate or greater TR']))
        row = {
            'patient_id': pid,
            'fs': fs,
            'n_samples': n,
            'ref_ao_count': int(len(ref_ao)),
            'pred_ao_count': int(len(pred_ao)),
            'j_peak_match_100ms': match_peaks(ref_ao, pred_j, fs, 0.10),
            'fiducial_ao_match_100ms': match_peaks(ref_ao, pred_ao, fs, 0.10),
            'fiducial_ao_match_50ms': match_peaks(ref_ao, pred_ao, fs, 0.05),
            'ecg_guided_ao_match_100ms': match_peaks(ref_ao, pred_ecg_guided_ao, fs, 0.10),
            'ecg_guided_ao_match_50ms': match_peaks(ref_ao, pred_ecg_guided_ao, fs, 0.05),
            'ecg_guided_ao_count': int(len(pred_ecg_guided_ao)),
            'ref_hr_bpm': ref_hr,
            'pred_hr_bpm': pred_hr,
            'hr_abs_error_bpm': abs(pred_hr - ref_hr) if pred_hr is not None and ref_hr is not None else None,
            'r_to_ao_ms': cti.get('r_to_ao_ms', {}).get('median_ms'),
            'lvet_ms': cti.get('lvet_ms', {}).get('median_ms'),
            'r_to_ao_over_lvet': cti.get('r_to_ao_over_lvet'),
            'placement_quality': placement.get('placement_quality'),
            'mechanical_abnormality_risk': abnormal.get('mechanical_abnormality_risk'),
            'mechanical_abnormality_score': abnormal_score(abnormal.get('mechanical_abnormality_risk')),
            'mechanical_flags': abnormal.get('flags'),
            'label_any_moderate_valve': label_any_valve,
            'label_as': int(float(meta.get('Moderate or greater AS', 0) or 0)),
            'label_ms': int(float(meta.get('Moderate or greater MS', 0) or 0)),
            'label_mr': int(float(meta.get('Moderate or greater MR', 0) or 0)),
            'label_ar': int(float(meta.get('Moderate or greater AR', 0) or 0)),
            'ef': None if pd.isna(meta.get('Ejection fraction (%)')) else float(meta.get('Ejection fraction (%)')),
        }
        rows.append(row)
        print(pid, {k: row[k] for k in ['ref_ao_count','pred_ao_count','hr_abs_error_bpm','mechanical_abnormality_risk']}, flush=True)
    ok = [r for r in rows if 'error' not in r]
    hr_err = [r['hr_abs_error_bpm'] for r in ok if r.get('hr_abs_error_bpm') is not None]
    report = {
        'dataset': 'Zenodo 5279448 VHD cardio-mechanical signals; SCG_Z AO annotations and ECG LARA R annotations',
        'n_total_rows': len(rows),
        'n_ok': len(ok),
        'ao_detection': {
            'j_peak_100ms': aggregate(ok, 'j_peak_match_100ms'),
            'fiducial_ao_100ms': aggregate(ok, 'fiducial_ao_match_100ms'),
            'fiducial_ao_50ms': aggregate(ok, 'fiducial_ao_match_50ms'),
            'ecg_guided_ao_100ms': aggregate(ok, 'ecg_guided_ao_match_100ms'),
            'ecg_guided_ao_50ms': aggregate(ok, 'ecg_guided_ao_match_50ms'),
            'hr_mae_bpm': float(np.mean(hr_err)) if hr_err else None,
        },
        'mechanical_abnormality_proxy': {
            'any_moderate_valve': classification_metrics(ok, 'label_any_moderate_valve'),
            'moderate_or_greater_as': classification_metrics(ok, 'label_as'),
            'moderate_or_greater_ms': classification_metrics(ok, 'label_ms'),
            'moderate_or_greater_mr': classification_metrics(ok, 'label_mr'),
            'moderate_or_greater_ar': classification_metrics(ok, 'label_ar'),
        },
        'per_patient': rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ['n_total_rows','n_ok','ao_detection','mechanical_abnormality_proxy']}, indent=2))

if __name__ == '__main__':
    main()
