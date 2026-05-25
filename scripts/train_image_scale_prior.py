from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DURATIONS = [2.5, 5.0, 10.0, 15.0, 30.0, 60.0]
OUT_ROOT = Path('/data1/jiahui/biosignal-agent/outputs/image_scale_prior')


def load_signal(path: str | Path) -> np.ndarray:
    df = pd.read_csv(path)
    col = 'signal' if 'signal' in df.columns else df.select_dtypes('number').columns[-1]
    values = df[col].to_numpy(dtype=float)
    return values[np.isfinite(values)]


def render_axisless(values: np.ndarray, fs: float, duration_s: float, out_path: Path, rng: random.Random) -> None:
    n = max(8, int(round(duration_s * fs)))
    if len(values) > n:
        start = rng.randint(0, len(values) - n)
        y = values[start:start+n]
    else:
        y = np.interp(np.linspace(0, len(values)-1, n), np.arange(len(values)), values)
    t = np.arange(len(y)) / fs
    width = rng.choice([360, 480, 640, 800, 960, 1200, 1600, 2000])
    height = rng.choice([160, 200, 220, 260, 300, 380, 480])
    dpi = 100
    fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)
    color = rng.choice(['black', '#1f2937', '#0f172a'])
    lw = rng.choice([0.7, 0.9, 1.0, 1.25, 1.5, 1.75, 2.2])
    ax.plot(t, y, color=color, linewidth=lw)
    ax.set_xlim(0, duration_s)
    pad = max(1e-6, float(np.nanpercentile(y, 99) - np.nanpercentile(y, 1)) * 0.12)
    ax.set_ylim(float(np.nanmin(y) - pad), float(np.nanmax(y) + pad))
    ax.set_xticks([])
    ax.set_yticks([])
    if rng.random() < 0.65:
        # grid without numeric labels; enough to mimic user screenshots but not reveal units.
        ax.grid(True, color=rng.choice(['#d6d6d6', '#e4e4e7', '#cbd5e1']), linewidth=rng.choice([0.5, 0.7, 0.9]), alpha=rng.uniform(0.45, 0.85))
        ax.set_xticks(np.linspace(0, duration_s, 6), minor=False)
        ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 5), minor=False)
        ax.tick_params(labelbottom=False, labelleft=False, length=0)
    else:
        ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(rng.random() < 0.7)
        spine.set_color('#111111')
        spine.set_linewidth(0.8)

    if rng.random() < 0.35:
        ax.set_facecolor(rng.choice(['#ffffff', '#fafafa', '#f8fafc']))
    fig.tight_layout(pad=rng.uniform(0.0, 0.35))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def build_dataset(manifest_path: Path, out_dir: Path, samples_per_duration: int, max_records: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    manifest = json.loads(manifest_path.read_text())
    records = [r for r in manifest.get('records', []) if Path(r.get('path', '')).exists() and float(r.get('sampling_rate', 0)) > 0]
    rng.shuffle(records)
    if max_records:
        records = records[:max_records]
    rows = []
    for rec_idx, rec in enumerate(records):
        try:
            values = load_signal(rec['path'])
            fs = float(rec['sampling_rate'])
        except Exception:
            continue
        if len(values) < int(min(DURATIONS) * fs * 0.9):
            continue
        for duration in DURATIONS:
            if len(values) < int(duration * fs * 0.5):
                continue
            for k in range(samples_per_duration):
                duration_tag = str(duration).replace('.', 'p')
                name = f"{rec_idx:04d}_{rec.get('modality','unk')}_{duration_tag}_{k}.png"
                img = out_dir / 'images' / name
                render_axisless(values, fs, duration, img, rng)
                rows.append({'image_path': str(img), 'duration_s': duration, 'label': DURATIONS.index(duration), 'modality': rec.get('modality'), 'source_path': rec['path'], 'sampling_rate': fs})
    return rows


class ScaleImageDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], image_size: tuple[int, int] = (192, 64)):
        self.rows = rows
        self.image_size = image_size
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(row['image_path']).convert('L').resize(self.image_size, Image.Resampling.BILINEAR)
        arr = 1.0 - np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr[None, :, :])
        y = torch.tensor(int(row['label']), dtype=torch.long)
        return x, y


class ScalePriorCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(0.25), nn.Linear(96, num_classes),
        )
    def forward(self, x):
        return self.net(x)


def train(rows: list[dict[str, Any]], out_dir: Path, epochs: int, batch_size: int, seed: int) -> dict[str, Any]:
    train_rows, test_rows = train_test_split(rows, test_size=0.25, random_state=seed, stratify=[r['label'] for r in rows])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ScalePriorCNN(len(DURATIONS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(ScaleImageDataset(train_rows), batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(ScaleImageDataset(test_rows), batch_size=batch_size, shuffle=False, num_workers=2)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        preds, truth = [], []
        with torch.no_grad():
            for x, y in test_loader:
                logits = model(x.to(device))
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
                truth.extend(y.numpy().tolist())
        acc = accuracy_score(truth, preds) if truth else 0.0
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), 'val_accuracy': float(acc)})
        print(json.dumps(history[-1]), flush=True)
    cm = confusion_matrix(truth, preds, labels=list(range(len(DURATIONS)))).tolist()
    model_path = out_dir / 'image_scale_prior_cnn.pt'
    torch.save({'state_dict': model.cpu().state_dict(), 'durations': DURATIONS, 'image_size': [192, 64], 'history': history}, model_path)
    report = {'num_rows': len(rows), 'num_train': len(train_rows), 'num_test': len(test_rows), 'durations': DURATIONS, 'history': history, 'confusion_matrix': cm, 'model_path': str(model_path)}
    (out_dir / 'image_scale_prior_train_report.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/datasets/processed/modality_classifier_manifest.json')
    ap.add_argument('--out-dir', default=str(OUT_ROOT))
    ap.add_argument('--samples-per-duration', type=int, default=2)
    ap.add_argument('--max-records', type=int, default=80)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--seed', type=int, default=17)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / 'scale_prior_dataset.json'
    if rows_path.exists():
        rows = json.loads(rows_path.read_text())
    else:
        rows = build_dataset(Path(args.manifest), out_dir, args.samples_per_duration, args.max_records, args.seed)
        rows_path.write_text(json.dumps(rows, indent=2))
    print(json.dumps({'num_rows': len(rows), 'rows_path': str(rows_path)}, indent=2), flush=True)
    report = train(rows, out_dir, args.epochs, args.batch_size, args.seed)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
