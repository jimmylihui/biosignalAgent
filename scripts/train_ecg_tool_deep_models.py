from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy import signal as scipy_signal
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.tools.common import load_csv_signal

OUT = Path('/data1/jiahui/biosignal-agent/outputs')
ARRHYTHMIA_DEEP_PATH = OUT / 'ecg_arrhythmia_1dcnn_model.pt'
APNEA_DEEP_PATH = OUT / 'ecg_apnea_1dcnn_model.pt'


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def robust_resample(values: np.ndarray, target_len: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.zeros(target_len, dtype=np.float32)
    med = float(np.median(values))
    q75, q25 = np.percentile(values, [75, 25])
    scale = float(q75 - q25)
    if scale < 1e-8:
        scale = float(np.std(values)) + 1e-8
    values = np.clip((values - med) / scale, -8.0, 8.0)
    if len(values) == target_len:
        return values.astype(np.float32)
    return scipy_signal.resample(values, target_len).astype(np.float32)


def load_manifest(manifest_path: str, label_fn: Callable[[dict[str, Any]], int], target_len: int) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, Any]]]:
    manifest = json.loads(Path(manifest_path).read_text())
    X, y, groups, records = [], [], [], []
    for rec in manifest.get('records', []):
        try:
            data = load_csv_signal(rec['path'], float(rec['sampling_rate']), column=None)
            X.append(robust_resample(data.values, target_len))
            y.append(int(label_fn(rec)))
            groups.append(str(rec.get('record') or rec.get('path')))
            records.append(rec)
        except Exception as exc:
            print(f'skip {rec.get("path")}: {type(exc).__name__}: {exc}', file=sys.stderr)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), groups, records


class ECGTinyCNN(nn.Module):
    def __init__(self, dropout: float = 0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(24, 48, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(48, 96, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(96, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x[:, None, :]
        return self.net(x).squeeze(-1)


def make_splits(X: np.ndarray, y: np.ndarray, groups: list[str], seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    uniq_groups = len(set(groups))
    if uniq_groups >= 3:
        splits = list(GroupKFold(n_splits=min(5, uniq_groups)).split(X, y, groups=groups))
        if all(len(set(y[tr])) == 2 and len(set(y[va])) == 2 for tr, va in splits):
            return splits
    n_pos = int(np.sum(y == 1)); n_neg = int(np.sum(y == 0))
    n_splits = max(2, min(5, n_pos, n_neg))
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y))


def train_fold(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray | None, y_val: np.ndarray | None, *, epochs: int, seed: int, device: str) -> tuple[ECGTinyCNN, float]:
    seed_all(seed)
    model = ECGTinyCNN().to(device)
    pos = max(float(np.sum(y_train == 1)), 1.0)
    neg = max(float(np.sum(y_train == 0)), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(train_ds, batch_size=min(32, len(train_ds)), shuffle=True)
    best_state = None; best_loss = math.inf; stale = 0
    max_epochs = epochs if X_val is not None else max(epochs, 80)
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            opt.step()
        if X_val is None:
            continue
        model.eval()
        with torch.no_grad():
            xv = torch.tensor(X_val, dtype=torch.float32, device=device)
            yv = torch.tensor(y_val, dtype=torch.float32, device=device)
            val_loss = float(criterion(model(xv), yv).detach().cpu())
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 12:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_loss


def predict_proba(model: ECGTinyCNN, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    probs = []
    loader = DataLoader(torch.tensor(X, dtype=torch.float32), batch_size=64, shuffle=False)
    with torch.no_grad():
        for xb in loader:
            logits = model(xb.to(device))
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probs).astype(float)


def metrics_for(y: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    out = {
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'average_precision': float(average_precision_score(y, proba)),
    }
    try:
        out['roc_auc'] = float(roc_auc_score(y, proba))
    except Exception:
        out['roc_auc'] = 0.0
    return out


def choose_threshold(y: np.ndarray, proba: np.ndarray) -> float:
    thresholds = np.linspace(0.15, 0.85, 71)
    scored = [(f1_score(y, proba >= t, zero_division=0), t) for t in thresholds]
    return float(max(scored, key=lambda item: item[0])[1])


def train_task(task: str, manifest: str, out_path: Path, label_fn, *, target_len: int, epochs: int, seed: int, device: str) -> dict[str, Any]:
    X, y, groups, _ = load_manifest(manifest, label_fn, target_len)
    splits = make_splits(X, y, groups, seed)
    cv_proba = np.zeros(len(y), dtype=float)
    fold_reports = []
    for fold, (tr, va) in enumerate(splits):
        model, val_loss = train_fold(X[tr], y[tr], X[va], y[va], epochs=epochs, seed=seed + fold, device=device)
        cv_proba[va] = predict_proba(model, X[va], device)
        fold_reports.append({'fold': fold, 'train_size': int(len(tr)), 'val_size': int(len(va)), 'val_loss': float(val_loss), 'val_label_counts': dict(Counter(map(int, y[va])))})
    threshold = choose_threshold(y, cv_proba)
    cv_metrics = metrics_for(y, cv_proba, threshold)
    final_model, _ = train_fold(X, y, None, None, epochs=epochs, seed=seed + 999, device=device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'state_dict': final_model.cpu().state_dict(),
        'architecture': 'ECGTinyCNN',
        'target_len': int(target_len),
        'threshold': float(threshold),
        'task': task,
        'cv_metrics': cv_metrics,
        'fold_reports': fold_reports,
        'label_counts': dict(Counter(map(int, y))),
    }, out_path)
    return {'task': task, 'manifest': manifest, 'num_rows': int(len(y)), 'label_counts': dict(Counter(map(int, y))), 'model_path': str(out_path), 'target_len': target_len, 'threshold': threshold, 'cv_metrics': cv_metrics, 'fold_reports': fold_reports}


def main() -> None:
    parser = argparse.ArgumentParser(description='Train lightweight deep ECG 1D CNN tool models.')
    parser.add_argument('--target-len', type=int, default=8192)
    parser.add_argument('--arrhythmia-manifest', default='/data1/jiahui/biosignal-agent/datasets/processed/labeled_arrhythmia_manifest.json')
    parser.add_argument('--apnea-manifest', default='/data1/jiahui/biosignal-agent/datasets/processed/apnea_ecg_manifest.json')
    parser.add_argument('--arrhythmia-model-path', default=str(ARRHYTHMIA_DEEP_PATH))
    parser.add_argument('--apnea-model-path', default=str(APNEA_DEEP_PATH))
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    seed_all(args.seed)
    report = {
        'device': args.device,
        'arrhythmia': train_task(
            'arrhythmia',
            args.arrhythmia_manifest,
            Path(args.arrhythmia_model_path),
            lambda rec: 1 if rec.get('binary_label') == 'abnormal' else 0,
            target_len=args.target_len,
            epochs=args.epochs,
            seed=args.seed,
            device=args.device,
        ),
        'apnea': train_task(
            'apnea',
            args.apnea_manifest,
            Path(args.apnea_model_path),
            lambda rec: 1 if rec.get('label') == 'apnea' else 0,
            target_len=args.target_len,
            epochs=args.epochs,
            seed=args.seed + 100,
            device=args.device,
        ),
    }
    out = OUT / 'ecg_tool_deep_model_train_report.json'
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
