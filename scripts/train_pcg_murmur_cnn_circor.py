from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import signal as scipy_signal
from scipy.io import wavfile
from scipy.ndimage import zoom
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_wav(path: str) -> tuple[int, np.ndarray]:
    fs, values = wavfile.read(path)
    if values.ndim > 1:
        values = values[:, 0]
    values = values.astype(np.float32)
    values = values - np.nanmedian(values)
    scale = np.nanpercentile(np.abs(values), 95) + 1e-6
    values = np.clip(values / scale, -5.0, 5.0)
    return int(fs), values


def resample_signal(values: np.ndarray, fs: int, target_fs: int) -> np.ndarray:
    if fs == target_fs:
        return values.astype(np.float32)
    n = max(8, int(round(len(values) * target_fs / float(fs))))
    return scipy_signal.resample(values, n).astype(np.float32)


def crop_or_pad(values: np.ndarray, length: int, train: bool, rng: np.random.Generator) -> np.ndarray:
    if len(values) >= length:
        if train:
            start = int(rng.integers(0, len(values) - length + 1))
        else:
            start = max(0, (len(values) - length) // 2)
        return values[start:start + length]
    out = np.zeros(length, dtype=np.float32)
    start = (length - len(values)) // 2
    out[start:start + len(values)] = values
    return out


def log_spectrogram(values: np.ndarray, fs: int, image_shape: tuple[int, int]) -> np.ndarray:
    freqs, times, spec = scipy_signal.spectrogram(values, fs=fs, window='hann', nperseg=256, noverlap=192, mode='magnitude', scaling='density')
    mask = (freqs >= 20.0) & (freqs <= 500.0)
    spec = spec[mask]
    log_spec = np.log1p(spec ** 2)
    if log_spec.size == 0:
        return np.zeros(image_shape, dtype=np.float32)
    lo, hi = np.percentile(log_spec, [2, 98])
    log_spec = np.clip((log_spec - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    zoom_f = image_shape[0] / log_spec.shape[0]
    zoom_t = image_shape[1] / log_spec.shape[1]
    img = zoom(log_spec, (zoom_f, zoom_t), order=1)
    return img.astype(np.float32)


class PCGSpecDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], indices: np.ndarray, train: bool, target_fs: int, seconds: float, image_shape: tuple[int, int], seed: int):
        self.records = records
        self.indices = np.asarray(indices, dtype=int)
        self.train = train
        self.target_fs = target_fs
        self.length = int(target_fs * seconds)
        self.image_shape = image_shape
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        rec = self.records[int(self.indices[item])]
        fs, values = load_wav(rec['path'])
        values = resample_signal(values, fs, self.target_fs)
        values = crop_or_pad(values, self.length, self.train, self.rng)
        if self.train:
            values = values + self.rng.normal(0.0, 0.01, size=values.shape).astype(np.float32)
            values = values * float(self.rng.uniform(0.85, 1.15))
        img = log_spectrogram(values, self.target_fs, self.image_shape)
        x = torch.from_numpy(img[None, :, :])
        y = torch.tensor(1 if rec['label'] == 'abnormal' else 0, dtype=torch.long)
        return x, y


class SmallPCGCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.BatchNorm2d(16), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1), nn.BatchNorm2d(96), nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Dropout(0.25), nn.Linear(96, 2),
        )

    def forward(self, x):
        return self.net(x)


def metrics(y_true: list[int], prob: list[float], threshold: float = 0.5) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    pred = (p >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        'true_positive': int(tp), 'true_negative': int(tn), 'false_positive': int(fp), 'false_negative': int(fn),
        'accuracy': float(accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall_sensitivity': float(recall_score(y, pred, zero_division=0)),
        'specificity': float(tn / (tn + fp)) if tn + fp else 0.0,
        'f1': float(f1_score(y, pred, zero_division=0)),
        'auroc': float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else None,
        'threshold': float(threshold),
    }


def best_f1_threshold(y_true: list[int], prob: list[float]) -> tuple[float, dict[str, Any]]:
    best_t = 0.5
    best_m = metrics(y_true, prob, 0.5)
    for threshold in np.linspace(0.1, 0.9, 81):
        m = metrics(y_true, prob, float(threshold))
        if (m['f1'], m['accuracy'], m['specificity']) > (best_m['f1'], best_m['accuracy'], best_m['specificity']):
            best_t = float(threshold)
            best_m = m
    return best_t, best_m


def train_one(model, loader, optimizer, device, class_weights=None):
    model.train()
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    total = 0.0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * len(y)
    return total / max(1, len(loader.dataset))


def predict(model, loader, device):
    model.eval()
    y_true = []
    probs = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist()
            probs.extend(prob)
            y_true.extend(y.numpy().tolist())
    return y_true, probs


def run_fold(records, train_idx, test_idx, args, seed):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    train_ds = PCGSpecDataset(records, train_idx, True, args.target_fs, args.seconds, (args.freq_bins, args.time_bins), seed)
    test_ds = PCGSpecDataset(records, test_idx, False, args.target_fs, args.seconds, (args.freq_bins, args.time_bins), seed + 1000)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = SmallPCGCNN().to(device)
    train_labels = np.asarray([1 if records[int(i)]['label'] == 'abnormal' else 0 for i in train_idx], dtype=int)
    counts = np.bincount(train_labels, minlength=2).astype(float)
    class_weights = torch.tensor([len(train_labels) / max(1.0, 2.0 * counts[0]), len(train_labels) / max(1.0, 2.0 * counts[1])], dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = None
    for epoch in range(args.epochs):
        loss = train_one(model, train_loader, opt, device, class_weights)
        y, p = predict(model, test_loader, device)
        m = metrics(y, p)
        if best is None or (m['f1'], m['auroc'] or 0.0) > (best['metrics']['f1'], best['metrics']['auroc'] or 0.0):
            best = {'epoch': epoch + 1, 'loss': loss, 'metrics': m, 'y': y, 'prob': p, 'test_idx': test_idx.tolist(), 'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    assert best is not None
    return best


def train_full(records, args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    indices = np.arange(len(records))
    ds = PCGSpecDataset(records, indices, True, args.target_fs, args.seconds, (args.freq_bins, args.time_bins), args.seed + 999)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = SmallPCGCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for _ in range(args.epochs):
        train_labels = np.asarray([1 if records[int(i)]['label'] == 'abnormal' else 0 for i in indices], dtype=int)
        counts = np.bincount(train_labels, minlength=2).astype(float)
        class_weights = torch.tensor([len(train_labels) / max(1.0, 2.0 * counts[0]), len(train_labels) / max(1.0, 2.0 * counts[1])], dtype=torch.float32)
        train_one(model, loader, opt, device, class_weights)
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/datasets/processed/circor_pcg_murmur_manifest.json')
    ap.add_argument('--out-model', default='/data1/jiahui/biosignal-agent/outputs/pcg_murmur_circor_spec_cnn.pt')
    ap.add_argument('--report', default='/data1/jiahui/biosignal-agent/outputs/pcg_murmur_circor_spec_cnn_report.json')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=24)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--target-fs', type=int, default=1000)
    ap.add_argument('--seconds', type=float, default=10.0)
    ap.add_argument('--freq-bins', type=int, default=64)
    ap.add_argument('--time-bins', type=int, default=128)
    ap.add_argument('--seed', type=int, default=23)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()
    set_seed(args.seed)
    manifest = json.loads(Path(args.manifest).read_text())
    records = [r for r in manifest['records'] if r.get('label') in {'normal', 'abnormal'}]
    y = np.asarray([1 if r['label'] == 'abnormal' else 0 for r in records], dtype=int)
    groups = np.asarray([r['patient_id'] for r in records])
    cv = GroupKFold(n_splits=min(args.folds, len(set(groups))))
    all_y = []
    all_prob = []
    all_indices = []
    fold_reports = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(np.arange(len(records)), y, groups), start=1):
        best = run_fold(records, train_idx, test_idx, args, args.seed + fold)
        all_y.extend(best['y'])
        all_prob.extend(best['prob'])
        all_indices.extend(best['test_idx'])
        fold_best_t, fold_best_m = best_f1_threshold(best['y'], best['prob'])
        rep = {'fold': fold, 'epoch': best['epoch'], 'num_train': int(len(train_idx)), 'num_test': int(len(test_idx)), 'metrics': best['metrics'], 'best_threshold': fold_best_t, 'best_threshold_metrics': fold_best_m}
        fold_reports.append(rep)
        print(json.dumps(rep), flush=True)
    cv_metrics = metrics(all_y, all_prob)
    best_threshold, cv_best_threshold_metrics = best_f1_threshold(all_y, all_prob)
    patient_prob_lists = {}
    patient_truth = {}
    for idx, prob in zip(all_indices, all_prob):
        rec = records[int(idx)]
        pid = rec['patient_id']
        patient_prob_lists.setdefault(pid, []).append(float(prob))
        patient_truth[pid] = 1 if rec.get('patient_label', rec['label']) == 'abnormal' else 0
    patient_ids = sorted(patient_prob_lists)
    patient_y = [patient_truth[pid] for pid in patient_ids]
    patient_prob = [float(np.mean(patient_prob_lists[pid])) for pid in patient_ids]
    patient_metrics = metrics(patient_y, patient_prob)
    patient_best_threshold, patient_best_threshold_metrics = best_f1_threshold(patient_y, patient_prob)
    full_state = train_full(records, args)
    payload = {
        'model_state_dict': full_state,
        'architecture': 'SmallPCGCNN_log_spectrogram',
        'target_fs': args.target_fs, 'seconds': args.seconds, 'freq_bins': args.freq_bins, 'time_bins': args.time_bins,
        'cv_metrics': cv_metrics, 'best_threshold': best_threshold, 'cv_best_threshold_metrics': cv_best_threshold_metrics, 'patient_cv_metrics': patient_metrics, 'patient_best_threshold': patient_best_threshold, 'patient_best_threshold_metrics': patient_best_threshold_metrics,
        'reference': 'CirCor heart sound 1.0.3, patient murmur present/absent, record-level predictions with patient-group CV.',
    }
    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out_model)
    report = {'manifest': args.manifest, 'num_records': len(records), 'num_patients': len(set(groups.tolist())), 'label_counts': dict(Counter(['abnormal' if v else 'normal' for v in y.tolist()])), 'model_out': args.out_model, 'folds': fold_reports, 'cv_metrics': cv_metrics, 'best_threshold': best_threshold, 'cv_best_threshold_metrics': cv_best_threshold_metrics, 'patient_cv_metrics': patient_metrics, 'patient_best_threshold': patient_best_threshold, 'patient_best_threshold_metrics': patient_best_threshold_metrics}
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
