from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path('/data1/jiahui/biosignal-agent')
NINAPRO = ROOT / 'datasets/raw/emg_ninapro_db1'
OUT = ROOT / 'outputs'


def build_ninapro_cache(path: Path, window: int = 30, step: int = 10, max_per_class: int = 2500, seed: int = 13) -> None:
    rng = np.random.default_rng(seed)
    offsets = {1: 0, 2: 12, 3: 29}
    buckets: dict[int, list[tuple[np.ndarray, str, int]]] = {}
    for mat_path in sorted(NINAPRO.glob('s*/S*_A1_E*.mat')):
        mat = loadmat(mat_path)
        emg = np.asarray(mat['emg'], dtype=np.float32)
        stim = np.asarray(mat['restimulus']).ravel().astype(int)
        rep = np.asarray(mat['rerepetition']).ravel().astype(int)
        subject = f"s{int(np.asarray(mat['subject']).ravel()[0]):02d}"
        exercise = int(np.asarray(mat['exercise']).ravel()[0])
        off = offsets[exercise]
        mean = emg.mean(axis=0, keepdims=True)
        std = emg.std(axis=0, keepdims=True) + 1e-6
        emg = (emg - mean) / std
        for start in range(0, len(emg) - window + 1, step):
            local = int(np.bincount(stim[start:start + window], minlength=60).argmax())
            if local == 0 or np.mean(stim[start:start + window] == local) < 0.8:
                continue
            label = local + off
            repetition = int(np.bincount(rep[start:start + window]).argmax())
            buckets.setdefault(label, []).append((emg[start:start + window].copy(), subject, repetition))
    xs, ys, subjects, reps = [], [], [], []
    for label, items in sorted(buckets.items()):
        if len(items) > max_per_class:
            idx = rng.choice(len(items), size=max_per_class, replace=False)
            items = [items[int(i)] for i in idx]
        for x, subject, repetition in items:
            xs.append(x.T)  # channels, time
            ys.append(label)
            subjects.append(subject)
            reps.append(repetition)
    X = np.stack(xs).astype(np.float32)
    y = np.asarray(ys)
    np.savez_compressed(path, X=X, y=y, subjects=np.asarray(subjects), repetitions=np.asarray(reps))
    print(json.dumps({'cache': str(path), 'shape': X.shape, 'classes': len(set(y)), 'counts': {str(k): len(v) for k, v in buckets.items()}}, indent=2))


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class TinyTCN(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, width: int = 96, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_channels, width, kernel_size=5, padding=2), nn.BatchNorm1d(width), nn.ReLU())
        self.blocks = nn.Sequential(TCNBlock(width, 1, dropout), TCNBlock(width, 2, dropout), TCNBlock(width, 4, dropout))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(width, n_classes))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def split_indices(subjects: np.ndarray, reps: np.ndarray, protocol: str):
    if protocol == 'calibrated':
        train = reps <= 7
        val = reps == 8
        test = reps >= 9
    elif protocol == 'subject':
        order = sorted(set(subjects))
        test_subjects = set(order[-5:])
        val_subjects = set(order[-8:-5])
        test = np.asarray([s in test_subjects for s in subjects])
        val = np.asarray([s in val_subjects for s in subjects])
        train = ~(test | val)
    else:
        raise ValueError(protocol)
    return np.where(train)[0], np.where(val)[0], np.where(test)[0]


def evaluate(model, loader, device):
    model.eval(); ys=[]; probs=[]
    with torch.no_grad():
        for xb, yb in loader:
            p = torch.softmax(model(xb.to(device)), dim=1).cpu().numpy()
            probs.append(p); ys.append(yb.numpy())
    y = np.concatenate(ys); proba = np.concatenate(probs); pred = proba.argmax(axis=1)
    return {
        'accuracy': float(accuracy_score(y, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, average='macro')),
        'weighted_f1': float(f1_score(y, pred, average='weighted')),
        'top3_accuracy': float(top_k_accuracy_score(y, proba, k=3, labels=np.arange(proba.shape[1]))),
        'top5_accuracy': float(top_k_accuracy_score(y, proba, k=5, labels=np.arange(proba.shape[1]))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=str(OUT / 'emg_ninapro_db1_raw_windows_tcn.npz'))
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--protocol', choices=['calibrated', 'subject'], default='calibrated')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--max-per-class', type=int, default=2500)
    args = ap.parse_args()
    cache = Path(args.cache)
    if args.prepare or not cache.exists():
        build_ninapro_cache(cache, max_per_class=args.max_per_class)
    data = np.load(cache, allow_pickle=True)
    X = data['X'].astype(np.float32); y_raw = data['y']; subjects = data['subjects']; reps = data['repetitions']
    le = LabelEncoder(); y = le.fit_transform(y_raw)
    tr, va, te = split_indices(subjects, reps, args.protocol)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TinyTCN(X.shape[1], len(le.classes_)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(y[tr]).long()), batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X[va]), torch.from_numpy(y[va]).long()), batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X[te]), torch.from_numpy(y[te]).long()), batch_size=args.batch_size, shuffle=False, num_workers=2)
    best = None; best_state = None; t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; n = 0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
            total += float(loss.item()) * len(yb); n += len(yb)
        val = evaluate(model, val_loader, device)
        print(json.dumps({'epoch': epoch, 'train_loss': total / max(1, n), 'val': val}, indent=2), flush=True)
        if best is None or val['macro_f1'] > best['macro_f1']:
            best = val; best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device)
    report = {
        'task': f'emg_ninapro_db1_52class_tcn_{args.protocol}',
        'dataset': 'NinaPro DB1 raw normalized windows',
        'protocol': args.protocol,
        'n_total': int(len(y)), 'n_train': int(len(tr)), 'n_val': int(len(va)), 'n_test': int(len(te)),
        'n_classes': int(len(le.classes_)), 'labels': [int(v) for v in le.classes_],
        'best_val': best, 'test': test, 'epochs': args.epochs,
        'elapsed_sec': round(time.time() - t0, 2),
        'note': 'Raw-sequence TCN SOTA-style baseline; compare against feature ensemble with same protocol before wiring tools.'
    }
    out_json = OUT / f'emg_ninapro_db1_52class_tcn_{args.protocol}_report.json'
    out_pt = OUT / f'emg_ninapro_db1_52class_tcn_{args.protocol}.pt'
    out_json.write_text(json.dumps(report, indent=2))
    torch.save({'state_dict': best_state, 'labels': le.classes_.tolist(), 'in_channels': int(X.shape[1])}, out_pt)
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()
