from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.common import bpm_from_peaks
from biosignal_agent.tools.pcg_tools import _duration_constrained_pcg_events


def state_centers(labels: np.ndarray, target: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    mask = labels == int(target)
    if not np.any(mask):
        return np.asarray([], dtype=int)
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    centers = ((starts + ends - 1) / 2.0).round().astype(int)
    return centers


def match_events(true_idx: np.ndarray, pred_idx: np.ndarray, tol: int) -> tuple[int, list[float]]:
    true_idx = np.asarray(true_idx, dtype=int)
    pred_idx = np.asarray(pred_idx, dtype=int)
    used = np.zeros(len(pred_idx), dtype=bool)
    errors = []
    tp = 0
    for t in true_idx:
        if len(pred_idx) == 0:
            break
        distances = np.abs(pred_idx - int(t)).astype(float)
        distances[used] = np.inf
        j = int(np.argmin(distances))
        if distances[j] <= tol:
            used[j] = True
            tp += 1
            errors.append(float(pred_idx[j] - int(t)))
    return tp, errors


def event_metrics(true_idx: np.ndarray, pred_idx: np.ndarray, fs: float, tolerance_ms: float) -> dict:
    tol = max(1, int(round(float(tolerance_ms) * fs / 1000.0)))
    tp, err_samples = match_events(true_idx, pred_idx, tol)
    fp = int(max(0, len(pred_idx) - tp))
    fn = int(max(0, len(true_idx) - tp))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    err_ms = np.asarray(err_samples, dtype=float) * 1000.0 / float(fs)
    return {
        'true': int(len(true_idx)),
        'pred': int(len(pred_idx)),
        'tp': int(tp),
        'fp': fp,
        'fn': fn,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'mae_ms': float(np.mean(np.abs(err_ms))) if len(err_ms) else None,
        'bias_ms': float(np.mean(err_ms)) if len(err_ms) else None,
    }


def hr_from_centers(idx: np.ndarray, fs: float) -> float | None:
    if len(idx) < 2:
        return None
    return bpm_from_peaks(np.asarray(idx, dtype=int), float(fs))


def main() -> None:
    ap = argparse.ArgumentParser(description='Evaluate current PCG S1/S2 proxy on Springer state labels.')
    ap.add_argument('--manifest', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/processed/pcg_springer_segmentation_manifest.json'))
    ap.add_argument('--report-path', type=Path, default=Path('/data1/jiahui/biosignal-agent/outputs/pcg_springer_segmentation_proxy_report.json'))
    ap.add_argument('--max-records', type=int, default=None)
    ap.add_argument('--tolerance-ms', type=float, default=80.0)
    args = ap.parse_args()
    manifest = json.load(open(args.manifest))
    rows = manifest['rows'][:args.max_records]
    per_record = []
    agg = {'s1': [], 's2': [], 'hr_abs_error': []}
    for row in rows:
        df = pd.read_csv(row['path'])
        values = df['signal'].to_numpy(dtype=float)
        labels = df['state_label'].to_numpy(dtype=int)
        fs = float(row['sampling_rate'])
        events = _duration_constrained_pcg_events(values, fs)
        true_s1 = state_centers(labels, 1)
        true_s2 = state_centers(labels, 3)
        pred_s1 = np.asarray(events.get('s1_indices', []), dtype=int)
        pred_s2 = np.asarray(events.get('s2_indices', []), dtype=int)
        s1 = event_metrics(true_s1, pred_s1, fs, args.tolerance_ms)
        s2 = event_metrics(true_s2, pred_s2, fs, args.tolerance_ms)
        true_hr = hr_from_centers(true_s1, fs)
        pred_hr = hr_from_centers(pred_s1, fs)
        hr_abs_error = abs(float(pred_hr) - float(true_hr)) if true_hr is not None and pred_hr is not None else None
        if hr_abs_error is not None:
            agg['hr_abs_error'].append(hr_abs_error)
        agg['s1'].append(s1)
        agg['s2'].append(s2)
        per_record.append({
            'record_id': row['record_id'],
            'duration_s': row.get('duration_s'),
            'segmentation_confidence': events.get('segmentation_confidence'),
            'true_hr_bpm': true_hr,
            'pred_hr_bpm': pred_hr,
            'hr_abs_error_bpm': hr_abs_error,
            's1': s1,
            's2': s2,
        })
    def summarize(items: list[dict]) -> dict:
        keys = ['precision', 'recall', 'f1']
        out = {k: float(np.mean([x[k] for x in items])) if items else 0.0 for k in keys}
        maes = [x['mae_ms'] for x in items if x.get('mae_ms') is not None]
        out['mae_ms_mean_record'] = float(np.mean(maes)) if maes else None
        tp = sum(x['tp'] for x in items); fp = sum(x['fp'] for x in items); fn = sum(x['fn'] for x in items)
        p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
        out.update({'micro_precision': float(p), 'micro_recall': float(r), 'micro_f1': float(2*p*r/max(1e-12, p+r)), 'tp': int(tp), 'fp': int(fp), 'fn': int(fn)})
        return out
    report = {
        'dataset': manifest.get('dataset'),
        'num_records': len(rows),
        'tolerance_ms': args.tolerance_ms,
        's1_summary': summarize(agg['s1']),
        's2_summary': summarize(agg['s2']),
        'heart_rate_mae_bpm': float(np.mean(agg['hr_abs_error'])) if agg['hr_abs_error'] else None,
        'heart_rate_num_records': int(len(agg['hr_abs_error'])),
        'per_record': per_record,
        'method': 'duration_constrained_hilbert_envelope_s1_s2_proxy_vs_springer_labels',
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['num_records', 'tolerance_ms', 's1_summary', 's2_summary', 'heart_rate_mae_bpm']}, indent=2))


if __name__ == '__main__':
    main()
