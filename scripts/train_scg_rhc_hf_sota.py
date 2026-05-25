from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    Dataset = object

try:
    import wfdb
except Exception as exc:  # pragma: no cover
    raise SystemExit("wfdb is required for SCG-RHC WFDB records") from exc

RAW = Path('/data1/jiahui/biosignal-agent/datasets/raw/scg_rhc_physionet')
PROCESSED = RAW / 'processed_data'
META = RAW / 'meta_information'
OUT_MODEL = Path('/data1/jiahui/biosignal-agent/outputs/scg_rhc_hf_feature_ensemble.joblib')
OUT_CNN = Path('/data1/jiahui/biosignal-agent/outputs/scg_rhc_hf_raw_cnn.pt')
OUT_REPORT = Path('/data1/jiahui/biosignal-agent/outputs/scg_rhc_hf_sota_report.json')

TARGETS = {
    'elevated_pcwp': ('PCWM', lambda x: numeric(x) is not None and numeric(x) >= 15.0),
    'elevated_pam': ('PAM', lambda x: numeric(x) is not None and numeric(x) >= 20.0),
    'decompensated_physiology': ('Physiological Score', lambda x: str(x).strip().lower().startswith('decomp')),
}


def norm_record_id(x: str) -> str:
    return str(x).replace('.', '-')


def numeric(v) -> float | None:
    try:
        y = float(v)
    except Exception:
        return None
    return y if math.isfinite(y) else None


def bandpower(x: np.ndarray, fs: float, lo: float, hi: float) -> float:
    if len(x) < fs * 2:
        return 0.0
    hi = min(hi, 0.45 * fs)
    if hi <= lo:
        return 0.0
    x = x - np.nanmedian(x)
    f, p = scipy_signal.welch(x, fs=fs, nperseg=min(len(x), int(fs * 8)))
    m = (f >= lo) & (f <= hi)
    return float(np.trapezoid(p[m], f[m])) if np.any(m) else 0.0


def preprocess_scg(x: np.ndarray, fs: float, target_fs: float = 100.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.nanmedian(x)
    hi = min(35.0, 0.45 * fs)
    if len(x) > fs * 3 and hi > 0.8:
        sos = scipy_signal.butter(3, [0.8 / (0.5 * fs), hi / (0.5 * fs)], btype='bandpass', output='sos')
        x = scipy_signal.sosfiltfilt(sos, x).astype(np.float32)
    if fs != target_fs:
        x = scipy_signal.resample(x, int(round(len(x) * target_fs / fs))).astype(np.float32)
    scale = np.nanpercentile(np.abs(x), 95) + 1e-6
    return np.clip(x / scale, -8, 8).astype(np.float32)


def summarize_channel(prefix: str, x: np.ndarray, fs: float) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {}
    xc = x - np.nanmedian(x)
    total = bandpower(xc, fs, 0.05, min(45.0, 0.45 * fs)) + 1e-12
    return {
        f'{prefix}_std': float(np.nanstd(xc)),
        f'{prefix}_iqr': float(np.percentile(xc, 75) - np.percentile(xc, 25)),
        f'{prefix}_p95_abs': float(np.percentile(np.abs(xc), 95)),
        f'{prefix}_cardiac_power_ratio': bandpower(xc, fs, 0.8, 35.0) / total,
        f'{prefix}_resp_power_ratio': bandpower(xc, fs, 0.08, 0.7) / total,
        f'{prefix}_hf_noise_ratio': bandpower(xc, fs, 35.0, min(80.0, 0.45 * fs)) / total,
        f'{prefix}_systolic_band_ratio': bandpower(xc, fs, 5.0, 25.0) / total,
    }


def detect_ecg_hr(ecg: np.ndarray, fs: float) -> dict[str, float]:
    if len(ecg) < fs * 5:
        return {'ecg_hr_bpm': 0.0, 'ecg_rr_cv': 0.0, 'ecg_peak_count': 0.0}
    x = ecg - np.nanmedian(ecg)
    hi = min(20.0, 0.45 * fs)
    if hi > 5.0:
        sos = scipy_signal.butter(3, [5.0 / (0.5 * fs), hi / (0.5 * fs)], btype='bandpass', output='sos')
        x = scipy_signal.sosfiltfilt(sos, x)
    env = np.abs(x)
    peaks, _ = scipy_signal.find_peaks(env, distance=max(1, int(0.30 * fs)), height=np.percentile(env, 95))
    rr = np.diff(peaks) / fs if len(peaks) > 2 else np.asarray([])
    rr = rr[(rr >= 0.3) & (rr <= 2.0)]
    return {
        'ecg_hr_bpm': float(60.0 / np.median(rr)) if len(rr) else 0.0,
        'ecg_rr_cv': float(np.std(rr) / (np.mean(rr) + 1e-12)) if len(rr) > 1 else 0.0,
        'ecg_peak_count': float(len(peaks)),
    }


def available_records(max_records: int | None = None) -> list[str]:
    recs = []
    list_path = META / 'list_exported_recs.txt'
    candidates = [norm_record_id(x.strip()) for x in list_path.read_text().splitlines() if x.strip()] if list_path.exists() else []
    local_records = sorted(
        p.stem for p in PROCESSED.glob('*.hea')
        if (PROCESSED / f'{p.stem}.dat').exists()
    )
    candidates = list(dict.fromkeys(candidates + local_records))
    for rec in candidates:
        if (PROCESSED / f'{rec}.hea').exists() and (PROCESSED / f'{rec}.dat').exists():
            recs.append(rec)
        if max_records and len(recs) >= max_records:
            break
    return recs


def load_labels() -> pd.DataFrame:
    df = pd.read_csv(META / 'RHC_values.csv')
    df['record_id'] = df['Study ID'].map(norm_record_id)
    for name, (col, fn) in TARGETS.items():
        df[name] = df[col].map(lambda v: None if pd.isna(v) else int(fn(v)))
    # Some records have multiple RHC phase rows. Use Baseline when available so each
    # WFDB record maps to one stable label row.
    df['_phase_rank'] = df['RHC Phase'].apply(lambda x: 0 if 'baseline' in str(x).lower() else 1)
    df = df.sort_values(['record_id', '_phase_rank']).groupby('record_id', as_index=False).first()
    return df


def read_record(rec: str):
    r = wfdb.rdrecord(str(PROCESSED / rec))
    names = list(r.sig_name)
    fs = float(r.fs)
    data = np.asarray(r.p_signal, dtype=float)
    return data, names, fs


def _rhc_event_seconds(rec: str) -> list[float]:
    path = PROCESSED / f"{rec}.json"
    if not path.exists():
        return []
    try:
        events = json.loads(path.read_text()).get("ChamEvents_in_s", {})
    except Exception:
        return []
    out = []
    for name, value in events.items():
        if ("PA" in name or "PCW" in name) and value is not None:
            try:
                out.append(float(value))
            except Exception:
                pass
    return out


def build_windows(max_records: int | None, seconds: float, stride_s: float, target_fs: float, alignment_window_s: float | None = None) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    labels = load_labels()
    label_by_rec = labels.set_index('record_id')
    rows = []
    raw_windows = []
    groups = []
    y_by_target = {k: [] for k in TARGETS}
    for rec in available_records(max_records):
        if rec not in label_by_rec.index:
            continue
        try:
            data, names, fs = read_record(rec)
        except Exception as exc:
            print('skip wfdb', rec, repr(exc), flush=True)
            continue
        acc_cols = [i for i, n in enumerate(names) if n in {'patch_ACC_lat', 'patch_ACC_hf', 'patch_ACC_dv'}]
        ecg_cols = [i for i, n in enumerate(names) if n == 'patch_ECG']
        if len(acc_cols) < 3:
            continue
        scg_axes = [preprocess_scg(data[:, i], fs, target_fs) for i in acc_cols]
        min_len = min(map(len, scg_axes))
        scg = np.stack([a[:min_len] for a in scg_axes], axis=0)
        ecg = data[:, ecg_cols[0]] if ecg_cols else np.asarray([])
        n = int(round(seconds * target_fs)); step = int(round(stride_s * target_fs))
        event_seconds = _rhc_event_seconds(rec) if alignment_window_s else []
        for start in range(0, max(1, min_len - n + 1), max(1, step)):
            end = start + n
            center_s = (start + n / 2.0) / target_fs
            if event_seconds and min(abs(center_s - t) for t in event_seconds) > float(alignment_window_s):
                continue
            seg = scg[:, start:end]
            if seg.shape[1] != n:
                continue
            feat = {'record_id': rec, 'start_s': start / target_fs}
            mag = np.sqrt(np.sum(seg ** 2, axis=0))
            feat.update(summarize_channel('scg_mag', mag, target_fs))
            for axis, name in enumerate(['lat', 'hf', 'dv']):
                feat.update(summarize_channel(f'scg_{name}', seg[axis], target_fs))
            if len(ecg):
                e0 = int(round(start * fs / target_fs)); e1 = int(round(end * fs / target_fs))
                feat.update(detect_ecg_hr(ecg[e0:e1], fs))
            rows.append(feat)
            raw_windows.append(seg.astype(np.float32))
            groups.append(rec)
            for target in TARGETS:
                y_by_target[target].append(int(label_by_rec.loc[rec, target]))
    return pd.DataFrame(rows), {k: np.asarray(v, dtype=int) for k, v in y_by_target.items()}, np.asarray(raw_windows, dtype=np.float32), np.asarray(groups)


class SCGWindowDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y.astype(np.float32))
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]


class SCGCNN(nn.Module):
    def __init__(self, n_targets: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 32, 11, padding=5, bias=False), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 9, padding=4, bias=False), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 96, 7, padding=3, bias=False), nn.BatchNorm1d(96), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(96, 128, 5, padding=2, bias=False), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(0.25), nn.Linear(128, n_targets),
        )
    def forward(self, x): return self.net(x)


def eval_scores(y: np.ndarray, score: np.ndarray, orient: bool = False) -> dict[str, Any]:
    out = {'n': int(len(y)), 'positive_rate': float(np.mean(y)) if len(y) else None}
    if len(set(y.tolist())) < 2:
        return {**out, 'auroc': None, 'auprc': None, 'f1_at_0p5': None, 'best_f1': None, 'best_threshold': None, 'flip_probability': False}
    raw_auc = float(roc_auc_score(y, score))
    flip = bool(orient and raw_auc < 0.5)
    used_score = 1.0 - score if flip else score
    pred = (used_score >= 0.5).astype(int)
    thresholds = np.unique(np.clip(used_score, 0.05, 0.95))
    if len(thresholds) > 512:
        thresholds = np.quantile(thresholds, np.linspace(0.0, 1.0, 512))
    thresholds = np.unique(np.concatenate([thresholds, np.asarray([0.5])]))
    best_f1 = -1.0
    best_threshold = 0.5
    for threshold in thresholds:
        f1 = f1_score(y, (used_score >= threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return {
        **out,
        'auroc': float(roc_auc_score(y, used_score)),
        'raw_auroc': raw_auc,
        'auprc': float(average_precision_score(y, used_score)),
        'f1_at_0p5': float(f1_score(y, pred, zero_division=0)),
        'best_f1': best_f1,
        'best_threshold': best_threshold,
        'flip_probability': flip,
    }


def train_feature_models(df: pd.DataFrame, labels: dict[str, np.ndarray], groups: np.ndarray, out_model: Path) -> dict[str, Any]:
    feature_names = [c for c in df.columns if c not in {'record_id'}]
    x = df[feature_names].to_numpy(dtype=float)
    reports = {}
    models = {}
    for target, y in labels.items():
        if len(np.unique(y)) < 2 or len(np.unique(groups)) < 3:
            reports[target] = {'error': 'insufficient classes or groups', **eval_scores(y, np.full(len(y), np.mean(y) if len(y) else 0.0))}
            continue
        sgkf = StratifiedGroupKFold(n_splits=min(5, len(np.unique(groups))), shuffle=True, random_state=113)
        candidates = {
            'logreg': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=2000)),
            'rf': make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(n_estimators=400, min_samples_leaf=4, class_weight='balanced', random_state=113)),
            'extra': make_pipeline(SimpleImputer(strategy='median'), ExtraTreesClassifier(n_estimators=500, min_samples_leaf=4, class_weight='balanced', random_state=113)),
        }
        cand_reports = []
        for name, proto in candidates.items():
            score = np.zeros(len(y), dtype=float)
            for tr, te in sgkf.split(x, y, groups):
                model = candidates[name]
                model.fit(x[tr], y[tr])
                score[te] = model.predict_proba(x[te])[:, 1]
            cand_reports.append({'model': name, **eval_scores(y, score, orient=True)})
        raw_candidates = [r for r in cand_reports if r.get('raw_auroc') is not None and r.get('raw_auroc') >= 0.55 and not r.get('flip_probability')]
        if not raw_candidates:
            oriented_best = max([r for r in cand_reports if r['auroc'] is not None], key=lambda r: (r['auroc'], r['auprc']))
            reports[target] = {
                'selected': None,
                'diagnostic_oriented_best': oriented_best,
                'candidates': cand_reports,
                'deployable': False,
                'reason': 'no raw-direction grouped-CV model reached AUROC >= 0.55; oriented/flip scores are diagnostic only',
            }
            continue
        best = max(raw_candidates, key=lambda r: (r['raw_auroc'], r['auprc']))
        model = candidates[best['model']]
        model.fit(x, y)
        models[target] = model
        reports[target] = {'selected': best, 'candidates': cand_reports, 'deployable': True, 'flip_probability': False}
    joblib.dump({
        'models': models,
        'feature_names': feature_names,
        'targets': list(models.keys()),
        'target_columns': {k: v[0] for k, v in TARGETS.items()},
        'deployment_flip_probability': {k: False for k in models.keys()},
        'decision_thresholds': {k: float(reports[k]['selected'].get('best_threshold', 0.5)) for k in models.keys()},
        'cv_flip_probability': {k: bool((v.get('selected') or {}).get('flip_probability', False)) for k, v in reports.items()},
        'reports': reports,
        'note': 'SCG-RHC SOTA-style feature ensemble for HF/RHC screening; use subject/group CV. cv_flip_probability is evaluation-only and is not applied at deployment.',
    }, out_model)
    return reports


def train_cnn(raw: np.ndarray, labels: dict[str, np.ndarray], groups: np.ndarray, epochs: int, out_model: Path) -> dict[str, Any]:
    if torch is None or len(raw) < 40 or len(np.unique(groups)) < 4:
        return {'skipped': 'torch unavailable or too few windows/groups'}
    target_names = [k for k, v in labels.items() if len(np.unique(v)) >= 2]
    if not target_names:
        return {'skipped': 'no target has two classes'}
    y = np.stack([labels[k] for k in target_names], axis=1)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_splits = min(5, len(np.unique(groups)))
    primary_y = y[:, 0]
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=113)

    def fit_cnn(train_idx: np.ndarray) -> SCGCNN:
        model = SCGCNN(y.shape[1]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        pos = np.maximum(y[train_idx].sum(axis=0), 1)
        neg = np.maximum(len(train_idx) - pos, 1)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, dtype=torch.float32, device=dev))
        loader = DataLoader(SCGWindowDataset(raw[train_idx], y[train_idx]), batch_size=64, shuffle=True)
        for _ in range(epochs):
            model.train()
            for xb, yb in loader:
                xb = xb.to(dev); yb = yb.to(dev); opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
        return model

    fold_reports = []
    for fold, (tr, te) in enumerate(splitter.split(raw, primary_y, groups), 1):
        model = fit_cnn(tr)
        model.eval()
        with torch.no_grad():
            score = torch.sigmoid(model(torch.from_numpy(raw[te]).to(dev))).cpu().numpy()
        fold_reports.append({target_names[i]: eval_scores(y[te, i], score[:, i]) for i in range(y.shape[1])})

    full_idx = np.arange(len(raw))
    model = fit_cnn(full_idx)
    torch.save({'state_dict': model.state_dict(), 'target_names': target_names, 'architecture': 'SCGCNN_3axis_multitask', 'epochs': epochs}, out_model)
    return {'fold_reports': fold_reports, 'model_out': str(out_model)}


def main() -> None:
    ap = argparse.ArgumentParser(description='Train SOTA-style SCG-RHC HF/hemodynamic screening models from wearable SCG.')
    ap.add_argument('--max-records', type=int, default=None)
    ap.add_argument('--seconds', type=float, default=30.0)
    ap.add_argument('--stride-s', type=float, default=30.0)
    ap.add_argument('--target-fs', type=float, default=100.0)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--alignment-window-s', type=float, default=None, help='Restrict SCG-RHC training windows to this many seconds around PA/PCW chamber events when record JSON contains event times.')
    ap.add_argument('--out-model', default=str(OUT_MODEL))
    ap.add_argument('--out-cnn', default=str(OUT_CNN))
    ap.add_argument('--report', default=str(OUT_REPORT))
    args = ap.parse_args()
    df, labels, raw, groups = build_windows(args.max_records, args.seconds, args.stride_s, args.target_fs, args.alignment_window_s)
    dataset_summary = {
        'available_records': available_records(args.max_records),
        'n_windows': int(len(df)),
        'n_records': int(len(set(groups.tolist()))) if len(groups) else 0,
        'alignment_window_s': args.alignment_window_s,
        'targets': {k: {'n': int(len(v)), 'positive_rate': float(np.mean(v)) if len(v) else None, 'classes': sorted(set(v.tolist())) if len(v) else []} for k, v in labels.items()},
    }
    report = {'dataset': 'PhysioNet SCG-RHC wearable database', 'dataset_summary': dataset_summary, 'feature_model': None, 'raw_cnn': None}
    if len(df) and len(set(groups.tolist())) >= 2:
        report['feature_model'] = train_feature_models(df, labels, groups, Path(args.out_model))
    else:
        report['feature_model'] = {'skipped': 'not enough downloaded WFDB records for grouped training'}
    report['raw_cnn'] = train_cnn(raw, labels, groups, args.epochs, Path(args.out_cnn))
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2)[:6000])


if __name__ == '__main__':
    main()
