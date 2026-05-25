from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_values(path: str | Path) -> np.ndarray:
    df = pd.read_csv(path)
    col = df.columns[0]
    return pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)


def clean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    med = float(np.nanmedian(x[finite]))
    x = np.where(finite, x, med)
    lo, hi = np.percentile(x, [0.5, 99.5])
    if hi > lo:
        x = np.clip(x, lo, hi)
    x = x - np.median(x)
    return x


def features(x: np.ndarray, fs: float) -> list[float]:
    x = clean(x)
    n = len(x)
    if n == 0:
        return [0.0] * 64
    freqs, psd = scipy_signal.welch(x, fs=fs, nperseg=min(n, int(fs * 4)))
    bands = {
        'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
        'sigma': (12, 16), 'beta': (13, 30), 'gamma': (30, min(45, fs * 0.45)),
        'slow': (0.5, 1.5), 'spindle': (11, 16),
    }
    def band(lo: float, hi: float) -> float:
        m = (freqs >= lo) & (freqs < hi)
        return float(np.trapezoid(psd[m], freqs[m])) if m.any() else 0.0
    total = band(0.5, min(45, fs * 0.45)) + 1e-12
    bp = {name: band(*rng) for name, rng in bands.items()}
    rel = {name: val / total for name, val in bp.items()}
    dx = np.diff(x, prepend=x[0]) * fs
    zcr = float(np.mean(np.diff(np.signbit(x)).astype(float))) if len(x) > 1 else 0.0
    hjorth_activity = float(np.var(x))
    hjorth_mobility = float(np.sqrt(np.var(dx) / (np.var(x) + 1e-12)))
    ddx = np.diff(dx, prepend=dx[0]) * fs
    hjorth_complexity = float(np.sqrt(np.var(ddx) / (np.var(dx) + 1e-12)) / (hjorth_mobility + 1e-12))
    q = np.percentile(x, [1, 5, 25, 50, 75, 95, 99])
    # Spectral entropy.
    p = psd[(freqs >= 0.5) & (freqs <= min(45, fs * 0.45))]
    p = p / (p.sum() + 1e-12)
    sent = float(-np.sum(p * np.log2(p + 1e-12)) / np.log2(len(p) + 1e-12)) if len(p) else 0.0
    ratios = [
        rel['delta'] / (rel['theta'] + 1e-12), rel['theta'] / (rel['alpha'] + 1e-12),
        rel['alpha'] / (rel['delta'] + 1e-12), rel['sigma'] / (rel['delta'] + 1e-12),
        (rel['delta'] + rel['theta']) / (rel['alpha'] + rel['beta'] + 1e-12),
    ]
    feats = [
        float(np.mean(x)), float(np.std(x)), float(np.min(x)), float(np.max(x)), float(np.ptp(x)),
        *[float(v) for v in q], float(skew(x)), float(kurtosis(x)),
        float(np.mean(np.abs(x))), float(np.sqrt(np.mean(x * x))), zcr,
        hjorth_activity, hjorth_mobility, hjorth_complexity, sent,
        *[float(bp[k]) for k in sorted(bp)], *[float(rel[k]) for k in sorted(rel)], *ratios,
        float(np.mean(dx)), float(np.std(dx)), float(np.percentile(np.abs(dx), 95)),
    ]
    return [0.0 if not np.isfinite(v) else float(v) for v in feats]


def metrics(y_true, y_pred, y_prob, labels):
    out = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted')),
        'cohen_kappa': float(cohen_kappa_score(y_true, y_pred)),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        'labels': labels,
    }
    try:
        out['macro_auroc_ovr'] = float(roc_auc_score(y_true, y_prob, labels=labels, multi_class='ovr', average='macro'))
    except Exception as exc:
        out['macro_auroc_error'] = str(exc)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/processed/psg_sleep_manifest.json'))
    ap.add_argument('--out-dir', type=Path, default=Path('/data1/jiahui/biosignal-agent/outputs/eeg_sleep'))
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    X, y, rows = [], [], []
    for rec in manifest['records']:
        x = load_values(rec['eeg_path'])
        label = rec['coarse_sleep_stage']
        if label == 'unknown':
            continue
        X.append(features(x, float(rec['eeg_sampling_rate'])))
        y.append(label)
        rows.append({'record': rec['record'], 'window_start_s': rec['window_start_s'], 'label': label})
    X = np.asarray(X, dtype=np.float32); y = np.asarray(y)
    labels = sorted(set(y.tolist()))
    clf = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('ensemble', VotingClassifier([
            ('extra', ExtraTreesClassifier(n_estimators=700, min_samples_leaf=2, class_weight='balanced', random_state=21, n_jobs=-1)),
            ('rf', RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight='balanced', random_state=23, n_jobs=-1)),
        ], voting='soft')),
    ])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=31)
    all_true=[]; all_pred=[]; all_prob=[]; folds=[]
    from sklearn.base import clone
    for fold,(tr,te) in enumerate(skf.split(X,y),1):
        model=clone(clf); model.fit(X[tr],y[tr])
        pred=model.predict(X[te]); prob=model.predict_proba(X[te])
        all_true.extend(y[te]); all_pred.extend(pred); all_prob.extend(prob.tolist())
        folds.append({'fold':fold, **metrics(y[te], pred, prob, labels)})
    clf.fit(X,y)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path=args.out_dir/'eeg_ucddb_coarse_sleep_stage_feature_ensemble.joblib'
    joblib.dump({'model':clf,'labels':labels,'feature':'single_channel_eeg_band_hjorth_spectral_entropy','sampling_rate_hint':128.0}, model_path)
    report={
        'dataset':'UCDDB ucddb002 processed PSG windows',
        'task':'coarse sleep stage: wake_rem / n1_n2 / n3',
        'model':'single-channel EEG feature ensemble ExtraTrees+RF',
        'validation':'5-fold stratified window CV on one UCDDB record; not subject-independent',
        'num_windows':int(len(y)),
        'label_counts':dict(Counter(y.tolist())),
        'overall':metrics(np.asarray(all_true), np.asarray(all_pred), np.asarray(all_prob), labels),
        'folds':folds,
        'model_path':str(model_path),
    }
    (args.out_dir/'eeg_ucddb_coarse_sleep_stage_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'num_windows':report['num_windows'],'label_counts':report['label_counts'],'overall':report['overall'],'model_path':str(model_path)}, indent=2))

if __name__ == '__main__':
    main()
