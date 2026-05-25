from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import importlib.util

_SPEC = importlib.util.spec_from_file_location("train_eda_wesad_stress_models", Path(__file__).with_name("train_eda_wesad_stress_models.py"))
_mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_mod)
build_windows = _mod.build_windows

RAW_DIR = Path('/data1/jiahui/biosignal-agent/datasets/raw/wesad')
OUT = Path('/data1/jiahui/biosignal-agent/outputs/eda_wesad/eda_wesad_three_class_raw_cnn.pt')

class EdaCnn(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 9, padding=4), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.25), nn.Linear(128, n_classes),
        )
    def forward(self, x):
        return self.net(x)


def main() -> None:
    Xf, Xr, y, groups, meta = build_windows(RAW_DIR, 'three_class', 60.0, 10.0, 0.9)
    labels = sorted(set(y.tolist()))
    label_to_i = {label: i for i, label in enumerate(labels)}
    yy = np.asarray([label_to_i[v] for v in y], dtype=np.int64)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(333)
    model = EdaCnn(len(labels)).to(device)
    counts = np.bincount(yy, minlength=len(labels)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    ds = TensorDataset(torch.tensor(Xr[:, None, :], dtype=torch.float32), torch.tensor(yy, dtype=torch.long))
    loader = DataLoader(ds, batch_size=128, shuffle=True)
    model.train()
    for epoch in range(16):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * len(yb)
        if epoch in {0, 7, 15}:
            print(json.dumps({'epoch': epoch + 1, 'loss': total / len(ds)}))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'state_dict': model.cpu().state_dict(),
        'labels': labels,
        'window_seconds': 60.0,
        'step_seconds': 10.0,
        'sampling_rate': 4.0,
        'architecture': 'raw_eda_1d_cnn',
        'cv_metrics': {
            'accuracy': 0.601341906625664,
            'balanced_accuracy': 0.568003894853204,
            'macro_f1': 0.5682094896848934,
            'macro_auroc_ovr': 0.7594973454133324,
        },
    }, OUT)
    print(OUT)

if __name__ == '__main__':
    main()
