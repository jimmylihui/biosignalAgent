#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class WindowDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx: int):
        arr = np.load(self.rows[idx]["path"])
        x = arr["signal"].astype(np.float32)[None, :]
        y = arr["mask"].astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, 9, padding=4), nn.BatchNorm1d(c_out), nn.ReLU(),
            nn.Conv1d(c_out, c_out, 9, padding=4), nn.BatchNorm1d(c_out), nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)


class TinyUNet1D(nn.Module):
    def __init__(self, classes: int = 4, base: int = 16):
        super().__init__()
        self.e1 = ConvBlock(1, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool1d(2)
        self.mid = ConvBlock(base * 4, base * 8)
        self.u3 = nn.ConvTranspose1d(base * 8, base * 4, 2, stride=2)
        self.d3 = ConvBlock(base * 8, base * 4)
        self.u2 = nn.ConvTranspose1d(base * 4, base * 2, 2, stride=2)
        self.d2 = ConvBlock(base * 4, base * 2)
        self.u1 = nn.ConvTranspose1d(base * 2, base, 2, stride=2)
        self.d1 = ConvBlock(base * 2, base)
        self.head = nn.Conv1d(base, classes, 1)
    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        z = self.mid(self.pool(e3))
        z = self.u3(z)
        if z.shape[-1] != e3.shape[-1]: z = torch.nn.functional.pad(z, (0, e3.shape[-1] - z.shape[-1]))
        z = self.d3(torch.cat([z, e3], 1))
        z = self.u2(z)
        if z.shape[-1] != e2.shape[-1]: z = torch.nn.functional.pad(z, (0, e2.shape[-1] - z.shape[-1]))
        z = self.d2(torch.cat([z, e2], 1))
        z = self.u1(z)
        if z.shape[-1] != e1.shape[-1]: z = torch.nn.functional.pad(z, (0, e1.shape[-1] - z.shape[-1]))
        z = self.d1(torch.cat([z, e1], 1))
        return self.head(z)


def split_rows(rows: list[dict], val_fraction: float, seed: int):
    records = sorted({r["record"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(records)
    n_val = max(1, int(round(len(records) * val_fraction)))
    val_records = set(records[:n_val])
    train = [r for r in rows if r["record"] not in val_records]
    val = [r for r in rows if r["record"] in val_records]
    return train, val, sorted(val_records)


def class_weights(rows: list[dict], classes: int = 4):
    counts = np.ones(classes, dtype=np.float64)
    for r in rows[: min(len(rows), 5000)]:
        y = np.load(r["path"])["mask"]
        for c in range(classes): counts[c] += np.sum(y == c)
    freq = counts / counts.sum()
    w = 1.0 / np.sqrt(freq)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def dice_loss(logits, target, classes: int = 4, include_background: bool = False):
    probs = torch.softmax(logits, dim=1)
    one_hot = torch.nn.functional.one_hot(target, num_classes=classes).permute(0, 2, 1).float()
    start = 0 if include_background else 1
    probs = probs[:, start:, :]
    one_hot = one_hot[:, start:, :]
    dims = (0, 2)
    inter = torch.sum(probs * one_hot, dims)
    denom = torch.sum(probs + one_hot, dims)
    dice = (2.0 * inter + 1.0) / (denom + 1.0)
    return 1.0 - dice.mean()


def evaluate(model, loader, device, classes: int = 4):
    model.eval()
    cm = np.zeros((classes, classes), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu().numpy().reshape(-1)
            truth = y.numpy().reshape(-1)
            valid = (truth >= 0) & (truth < classes)
            idx = truth[valid] * classes + pred[valid]
            cm += np.bincount(idx, minlength=classes * classes).reshape(classes, classes)
    rows = {}
    for c, name in enumerate(["background", "p", "qrs", "t"]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        iou = tp / (tp + fp + fn + 1e-12)
        rows[name] = {"precision": float(precision), "recall": float(recall), "f1": float(f1), "iou": float(iou), "support_pixels": int(cm[c, :].sum())}
    rows["macro_wave_f1"] = float(np.mean([rows[k]["f1"] for k in ["p", "qrs", "t"]]))
    rows["macro_wave_iou"] = float(np.mean([rows[k]["iou"] for k in ["p", "qrs", "t"]]))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Train a lightweight ECG P/QRS/T delineation U-Net on prepared QTDB windows.")
    ap.add_argument("--manifest", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/qtdb_delineation_manifest.json"))
    ap.add_argument("--model-path", type=Path, default=Path("/data1/jiahui/biosignal-agent/outputs/ecg_delineation_qtdb_unet.pt"))
    ap.add_argument("--report", type=Path, default=Path("/data1/jiahui/biosignal-agent/outputs/ecg_delineation_qtdb_unet_report.json"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--base", type=int, default=12)
    ap.add_argument("--val-fraction", type=float, default=0.25)
    ap.add_argument("--dice-weight", type=float, default=0.5, help="Weight of soft Dice loss added to weighted cross entropy.")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    payload = json.loads(args.manifest.read_text())
    rows = payload["rows"]
    train_rows, val_rows, val_records = split_rows(rows, args.val_fraction, args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyUNet1D(classes=4, base=args.base).to(device)
    weights = class_weights(train_rows).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader = DataLoader(WindowDataset(train_rows), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(WindowDataset(val_rows), batch_size=args.batch_size, shuffle=False, num_workers=0)
    history = []
    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            if args.dice_weight > 0:
                loss = loss + args.dice_weight * dice_loss(logits, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate(model, val_loader, device)
        item = {"epoch": epoch, "loss": float(np.mean(losses)), "val": metrics}
        history.append(item)
        if best is None or metrics["macro_wave_f1"] > best["val"]["macro_wave_f1"]:
            best = item
            args.model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "class_to_id": payload.get("class_to_id"), "base": args.base, "val_records": val_records, "best_epoch": epoch}, args.model_path)
        print(json.dumps({"epoch": epoch, "loss": item["loss"], "macro_wave_f1": metrics["macro_wave_f1"]}), flush=True)
    report = {"model": "TinyUNet1D_QTDB_delineation", "manifest": str(args.manifest), "model_path": str(args.model_path), "device": device, "num_train_windows": len(train_rows), "num_val_windows": len(val_rows), "val_records": val_records, "class_weights": [float(x) for x in weights.detach().cpu().numpy()], "dice_weight": args.dice_weight, "best": best, "history": history}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(args.report), "best_macro_wave_f1": best["val"]["macro_wave_f1"]}, indent=2))


if __name__ == "__main__":
    main()
