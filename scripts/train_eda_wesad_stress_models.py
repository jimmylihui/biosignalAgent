from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

WRIST_FS = 4.0
LABEL_FS = 700.0
BINARY_LABELS = {1: 'non_stress', 2: 'stress'}
THREE_LABELS = {1: 'baseline', 2: 'stress', 3: 'amusement'}


def find_subject_files(raw_dir: Path) -> list[Path]:
    root = raw_dir / 'WESAD' if (raw_dir / 'WESAD').exists() else raw_dir
    return sorted(root.glob('S*/S*.pkl')) + sorted(root.glob('S*.pkl'))


def load_subject(path: Path) -> dict[str, Any]:
    with path.open('rb') as handle:
        return pickle.load(handle, encoding='latin1')


def majority_label(labels: np.ndarray, start_s: float, stop_s: float, label_map: dict[int, str], min_fraction: float) -> str | None:
    start = int(start_s * LABEL_FS)
    stop = int(stop_s * LABEL_FS)
    seg = labels[start:stop]
    seg = seg[np.isin(seg, list(label_map))]
    if len(seg) == 0:
        return None
    counts = Counter(seg.astype(int).tolist())
    label, n = counts.most_common(1)[0]
    if n / max(1, len(seg)) < min_fraction:
        return None
    return label_map.get(int(label))


def normalize_window(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=float)
    med = float(np.nanmedian(x[finite]))
    x = np.where(finite, x, med)
    lo, hi = np.percentile(x, [1, 99])
    if hi > lo:
        x = np.clip(x, lo, hi)
    return x


def eda_features(x: np.ndarray, fs: float = WRIST_FS) -> list[float]:
    x = normalize_window(x)
    n = len(x)
    if n == 0:
        return [0.0] * 44
    t = np.arange(n) / fs
    if n >= int(fs * 5):
        k = max(5, int(fs * 15) | 1)
        if k >= n:
            k = max(3, (n // 2) * 2 - 1)
        tonic = scipy_signal.medfilt(x, kernel_size=k) if k >= 3 else np.full_like(x, np.median(x))
    else:
        tonic = np.full_like(x, np.median(x))
    phasic = x - tonic
    dx = np.diff(x, prepend=x[0]) * fs
    dph = np.diff(phasic, prepend=phasic[0]) * fs
    duration_min = max(n / fs / 60.0, 1e-9)
    prom = max(float(np.nanstd(phasic)) * 0.5, 0.01)
    peaks, props = scipy_signal.find_peaks(phasic, distance=max(1, int(fs)), prominence=prom)
    rises = np.diff(peaks) / fs if len(peaks) > 1 else np.array([])
    slope = np.polyfit(t, x, 1)[0] if n > 3 else 0.0
    tonic_slope = np.polyfit(t, tonic, 1)[0] if n > 3 else 0.0
    freqs, pxx = scipy_signal.welch(x - np.mean(x), fs=fs, nperseg=min(n, 128)) if n >= 16 else (np.array([0.0]), np.array([0.0]))
    def band(lo: float, hi: float) -> float:
        m = (freqs >= lo) & (freqs < hi)
        return float(np.trapezoid(pxx[m], freqs[m])) if m.any() else 0.0
    q = np.percentile(x, [5, 25, 50, 75, 95])
    tq = np.percentile(tonic, [5, 50, 95])
    pq = np.percentile(phasic, [5, 50, 95])
    amp = props.get('prominences', np.array([], dtype=float))
    feats = [
        float(np.mean(x)), float(np.std(x)), float(np.min(x)), float(np.max(x)), float(np.ptp(x)),
        *[float(v) for v in q], float(skew(x)), float(kurtosis(x)), float(slope),
        float(np.mean(tonic)), float(np.std(tonic)), *[float(v) for v in tq], float(tonic_slope),
        float(np.mean(phasic)), float(np.std(phasic)), float(np.max(phasic)), float(np.min(phasic)), *[float(v) for v in pq],
        float(np.mean(dx)), float(np.std(dx)), float(np.percentile(np.abs(dx), 95)),
        float(np.mean(dph)), float(np.std(dph)), float(np.percentile(np.abs(dph), 95)),
        float(len(peaks)), float(len(peaks) / duration_min), float(np.mean(amp) if len(amp) else 0.0),
        float(np.max(amp) if len(amp) else 0.0), float(np.std(amp) if len(amp) else 0.0),
        float(np.mean(rises) if len(rises) else 0.0), float(np.std(rises) if len(rises) else 0.0),
        band(0.00, 0.045), band(0.045, 0.15), band(0.15, 0.40), band(0.40, 1.0),
        float(np.mean(np.abs(phasic)) / (abs(np.mean(x)) + 1e-6)),
        float(np.sum(dx > np.percentile(dx, 90)) / max(1, n)),
    ]
    return [0.0 if not np.isfinite(v) else float(v) for v in feats]


def build_windows(raw_dir: Path, task: str, window_s: float, step_s: float, min_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    label_map = BINARY_LABELS if task == 'binary' else THREE_LABELS
    X_feat, X_raw, y, groups, meta = [], [], [], [], []
    for p in find_subject_files(raw_dir):
        d = load_subject(p)
        labels = np.asarray(d['label'], dtype=int).ravel()
        eda = np.asarray(d['signal']['wrist']['EDA'], dtype=float).ravel()
        subject = p.stem
        total_s = min(len(eda) / WRIST_FS, len(labels) / LABEL_FS)
        nwin = int(window_s * WRIST_FS)
        start_s = 0.0
        while start_s + window_s <= total_s:
            lab = majority_label(labels, start_s, start_s + window_s, label_map, min_fraction)
            if lab is not None:
                start = int(start_s * WRIST_FS)
                seg = eda[start:start + nwin]
                if len(seg) == nwin:
                    seg = normalize_window(seg)
                    X_feat.append(eda_features(seg))
                    z = (seg - np.mean(seg)) / (np.std(seg) + 1e-6)
                    X_raw.append(z.astype(np.float32))
                    y.append(lab)
                    groups.append(subject)
                    meta.append({'subject': subject, 'start_s': float(start_s), 'label': lab})
            start_s += step_s
    return np.asarray(X_feat, dtype=np.float32), np.asarray(X_raw, dtype=np.float32), np.asarray(y), np.asarray(groups), meta


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None, labels: list[str]) -> dict[str, Any]:
    out = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted')),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        'labels': labels,
    }
    if y_prob is not None:
        try:
            if len(labels) == 2:
                pos = labels.index('stress') if 'stress' in labels else 1
                out['auroc'] = float(roc_auc_score((y_true == labels[pos]).astype(int), y_prob[:, pos]))
            else:
                out['macro_auroc_ovr'] = float(roc_auc_score(y_true, y_prob, labels=labels, multi_class='ovr', average='macro'))
        except Exception as exc:
            out['auroc_error'] = str(exc)
    return out


def train_feature_model(X: np.ndarray, y: np.ndarray, groups: np.ndarray, task: str) -> tuple[Pipeline, dict[str, Any]]:
    labels = sorted(set(y.tolist()))
    clf = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('ensemble', VotingClassifier([
            ('extra', ExtraTreesClassifier(n_estimators=700, min_samples_leaf=2, class_weight='balanced', random_state=13, n_jobs=-1)),
            ('rf', RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight='balanced', random_state=17, n_jobs=-1)),
        ], voting='soft')),
    ])
    n_splits = min(5, len(set(groups.tolist())))
    fold_reports = []
    all_true, all_pred, all_prob = [], [], []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups), start=1):
        model = clone(clf)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        prob = model.predict_proba(X[te])
        all_true.extend(y[te].tolist())
        all_pred.extend(pred.tolist())
        all_prob.extend(prob.tolist())
        fold_reports.append({'fold': fold, 'test_subjects': sorted(set(groups[te].tolist())), **metric_dict(y[te], pred, prob, labels)})
    clf.fit(X, y)
    report = {
        'task': task,
        'model': 'EDA feature ensemble: ExtraTrees + RandomForest soft voting',
        'validation': f'{n_splits}-fold subject-grouped CV',
        'num_windows': int(len(y)),
        'num_subjects': int(len(set(groups.tolist()))),
        'label_counts': dict(Counter(y.tolist())),
        'overall': metric_dict(np.asarray(all_true), np.asarray(all_pred), np.asarray(all_prob), labels),
        'folds': fold_reports,
    }
    return clf, report


def train_cnn_model(X_raw: np.ndarray, y: np.ndarray, groups: np.ndarray, task: str, epochs: int) -> dict[str, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        return {'task': task, 'model': 'raw_eda_cnn', 'error': f'torch unavailable: {exc}'}
    labels = sorted(set(y.tolist()))
    enc = LabelEncoder().fit(labels)
    yy = enc.transform(y)
    class Net(nn.Module):
        def __init__(self, n_classes: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, 32, 9, padding=4), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.25), nn.Linear(128, n_classes),
            )
        def forward(self, x): return self.net(x)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_splits = min(5, len(set(groups.tolist())))
    all_true, all_pred, all_prob, folds = [], [], [], []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X_raw, yy, groups), start=1):
        torch.manual_seed(100 + fold)
        model = Net(len(labels)).to(device)
        counts = np.bincount(yy[tr], minlength=len(labels)).astype(np.float32)
        weights = counts.sum() / np.maximum(counts, 1.0)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        tr_ds = TensorDataset(torch.tensor(X_raw[tr, None, :], dtype=torch.float32), torch.tensor(yy[tr], dtype=torch.long))
        loader = DataLoader(tr_ds, batch_size=128, shuffle=True)
        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_raw[te, None, :], dtype=torch.float32, device=device))
            prob = torch.softmax(logits, dim=1).cpu().numpy()
        pred_i = prob.argmax(axis=1)
        true_l = enc.inverse_transform(yy[te])
        pred_l = enc.inverse_transform(pred_i)
        all_true.extend(true_l.tolist())
        all_pred.extend(pred_l.tolist())
        all_prob.extend(prob.tolist())
        folds.append({'fold': fold, 'test_subjects': sorted(set(groups[te].tolist())), **metric_dict(true_l, pred_l, prob, labels)})
    return {
        'task': task,
        'model': 'raw EDA 1D CNN / DeepConvLSTM-family baseline',
        'validation': f'{n_splits}-fold subject-grouped CV',
        'device': device,
        'epochs': int(epochs),
        'num_windows': int(len(y)),
        'num_subjects': int(len(set(groups.tolist()))),
        'label_counts': dict(Counter(y.tolist())),
        'overall': metric_dict(np.asarray(all_true), np.asarray(all_pred), np.asarray(all_prob), labels),
        'folds': folds,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description='Train WESAD EDA stress/arousal classifiers.')
    ap.add_argument('--raw-dir', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/raw/wesad'))
    ap.add_argument('--out-dir', type=Path, default=Path('/data1/jiahui/biosignal-agent/outputs/eda_wesad'))
    ap.add_argument('--window-seconds', type=float, default=60.0)
    ap.add_argument('--step-seconds', type=float, default=10.0)
    ap.add_argument('--min-label-fraction', type=float, default=0.9)
    ap.add_argument('--cnn-epochs', type=int, default=10)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {'benchmark': 'WESAD wrist EDA', 'window_seconds': args.window_seconds, 'step_seconds': args.step_seconds, 'tasks': {}}
    for task in ['binary', 'three_class']:
        Xf, Xr, y, groups, meta = build_windows(args.raw_dir, task, args.window_seconds, args.step_seconds, args.min_label_fraction)
        model, feat_report = train_feature_model(Xf, y, groups, task)
        model_path = args.out_dir / f'eda_wesad_{task}_feature_ensemble.joblib'
        joblib.dump({'model': model, 'task': task, 'window_seconds': args.window_seconds, 'sampling_rate': WRIST_FS, 'labels': sorted(set(y.tolist()))}, model_path)
        cnn_report = train_cnn_model(Xr, y, groups, task, args.cnn_epochs)
        (args.out_dir / f'eda_wesad_{task}_feature_report.json').write_text(json.dumps(feat_report, indent=2))
        (args.out_dir / f'eda_wesad_{task}_cnn_report.json').write_text(json.dumps(cnn_report, indent=2))
        summary['tasks'][task] = {'feature': feat_report, 'cnn': cnn_report, 'model_path': str(model_path)}
        print(json.dumps({'task': task, 'feature': feat_report['overall'], 'cnn': cnn_report.get('overall')}, indent=2))
    (args.out_dir / 'eda_wesad_training_summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
