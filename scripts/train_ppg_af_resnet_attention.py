from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def find_ppg_column(frame: pd.DataFrame) -> str:
    lower = {str(col).lower(): col for col in frame.columns}
    for key in ["ppg", "pleth", "photoplethysmogram", "signal"]:
        for lower_name, original in lower.items():
            if key in lower_name:
                return original
    numeric = frame.select_dtypes("number").columns
    if len(numeric) == 0:
        raise ValueError("no numeric PPG column")
    return str(numeric[0])


def normalize_window(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.all():
        median = np.nanmedian(x[finite]) if finite.any() else 0.0
        x = np.where(finite, x, median)
    x = x - np.median(x)
    scale = np.percentile(np.abs(x), 90) + 1e-6
    x = x / scale
    d1 = np.gradient(x).astype(np.float32)
    d2 = np.gradient(d1).astype(np.float32)
    channels = np.stack([x, d1, d2], axis=0)
    channels = np.clip(channels, -8.0, 8.0)
    return channels.astype(np.float32)


def load_manifest_windows(manifest_path: Path, window_s: float, stride_s: float, max_windows_per_record: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text())
    xs, ys, groups, meta = [], [], [], []
    for rec in manifest.get("records", []):
        source_path = Path(rec.get("source_file") or rec["path"])
        if not source_path.exists():
            source_path = Path(rec["path"])
        frame = pd.read_csv(source_path)
        col = find_ppg_column(frame)
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float32)
        fs = float(rec.get("sampling_rate", 125.0))
        win = int(round(window_s * fs))
        stride = int(round(stride_s * fs))
        starts = list(range(0, max(0, len(values) - win + 1), stride))
        if max_windows_per_record is not None and len(starts) > max_windows_per_record:
            idx = np.linspace(0, len(starts) - 1, max_windows_per_record).round().astype(int)
            starts = [starts[i] for i in idx]
        label = 1 if rec["label"] == "af" else 0
        for start in starts:
            window = values[start:start + win]
            if len(window) != win:
                continue
            xs.append(normalize_window(window))
            ys.append(label)
            groups.append(rec["record"])
            meta.append({"record": rec["record"], "label": rec["label"], "start_s": float(start / fs), "source_file": str(source_path)})
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64), np.asarray(groups), meta


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation * 3
        self.main = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=7, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_ch),
            SEBlock(out_ch),
        )
        self.skip = nn.Identity() if in_ch == out_ch and stride == 1 else nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.skip(x))


class ResNetSEAttention(nn.Module):
    def __init__(self, in_ch: int = 3, width: int = 32, dropout: float = 0.25) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, width, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResBlock(width, width, stride=1),
            ResBlock(width, width * 2, stride=2),
            ResBlock(width * 2, width * 2, stride=1, dilation=2),
            ResBlock(width * 2, width * 4, stride=2),
            ResBlock(width * 4, width * 4, stride=1, dilation=2),
        )
        feat = width * 4
        self.attn = nn.Sequential(
            nn.Conv1d(feat, feat // 2, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(feat // 2, 1, kernel_size=1),
        )
        self.head = nn.Sequential(nn.LayerNorm(feat), nn.Dropout(dropout), nn.Linear(feat, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.blocks(self.stem(x))
        weights = torch.softmax(self.attn(z), dim=-1)
        pooled = torch.sum(z * weights, dim=-1)
        return self.head(pooled).squeeze(-1)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(torch.tensor(x), torch.tensor(y, dtype=torch.float32)), batch_size=batch_size, shuffle=shuffle)


def predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for xb, _ in make_loader(x, np.zeros(len(x)), batch_size, False):
            logits = model(xb.to(device))
            outs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outs) if outs else np.asarray([])


def train_one_fold(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, args: argparse.Namespace, pos_weight: float, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(args.seed)
    model = ResNetSEAttention(width=args.width, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    best_state, best_auc, best_epoch, bad_epochs = None, -1.0, -1, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in make_loader(x_train, y_train, args.batch_size, True):
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_prob = predict(model, x_val, device, args.batch_size)
        try:
            auc = roc_auc_score(y_val, val_prob)
        except ValueError:
            auc = 0.5
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_auc": float(auc)})
        if auc > best_auc + 1e-4:
            best_auc = float(auc)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_auc": best_auc, "best_epoch": best_epoch, "history": history}


def aggregate_record_predictions(prob: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[list[int], list[int], list[float]]:
    y_true, y_pred, y_prob = [], [], []
    for group in sorted(set(groups)):
        idx = np.where(groups == group)[0]
        p = float(np.mean(prob[idx]))
        label = int(np.round(np.mean(y[idx])))
        y_true.append(label)
        y_prob.append(p)
        y_pred.append(1 if p >= 0.5 else 0)
    return y_true, y_pred, y_prob


def metrics(y_true: list[int], y_pred: list[int], y_prob: list[float]) -> dict[str, Any]:
    labels = [0, 1]
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = None
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "auroc": float(auc) if auc is not None else None,
        "labels": ["non_af", "af"],
        "per_class": {
            "non_af": {"precision": float(precision[0]), "recall": float(recall[0]), "f1": float(f1[0]), "support": int(support[0])},
            "af": {"precision": float(precision[1]), "recall": float(recall[1]), "f1": float(f1[1]), "support": int(support[1])},
        },
        "confusion_matrix_labels": ["non_af", "af"],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def train_cv(args: argparse.Namespace) -> dict[str, Any]:
    x, y, groups, meta = load_manifest_windows(Path(args.manifest), args.window_s, args.stride_s, args.max_windows_per_record)
    if len(set(groups)) < 4 or len(set(y.tolist())) < 2:
        raise RuntimeError("Need more grouped records/classes for DL training")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    cv = GroupKFold(n_splits=min(args.folds, len(set(groups))))
    all_prob = np.zeros(len(y), dtype=float)
    fold_reports = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(x, y, groups), start=1):
        y_train = y[train_idx]
        pos_weight = float((len(y_train) - y_train.sum()) / max(1, y_train.sum()))
        model, train_info = train_one_fold(x[train_idx], y[train_idx], x[val_idx], y[val_idx], args, pos_weight, device)
        prob = predict(model, x[val_idx], device, args.batch_size)
        all_prob[val_idx] = prob
        yt, yp, ypr = aggregate_record_predictions(prob, y[val_idx], groups[val_idx])
        fold_reports.append({"fold": fold, "train_info": train_info, "record_metrics": metrics(yt, yp, ypr), "val_records": sorted(set(groups[val_idx].tolist()))})
        print(json.dumps({"fold": fold, "records": len(set(groups[val_idx])), "record_metrics": fold_reports[-1]["record_metrics"]}, indent=2))
    record_true, record_pred, record_prob = aggregate_record_predictions(all_prob, y, groups)
    cv_metrics = metrics(record_true, record_pred, record_prob)

    final_pos_weight = float((len(y) - y.sum()) / max(1, y.sum()))
    final_model, final_info = train_one_fold(x, y, x, y, args, final_pos_weight, device)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": final_model.state_dict(),
        "architecture": "ResNetSEAttention",
        "window_s": args.window_s,
        "stride_s": args.stride_s,
        "sampling_rate": 125.0,
        "input_channels": ["ppg_z", "first_derivative", "second_derivative"],
        "width": args.width,
        "dropout": args.dropout,
        "cv_metrics": cv_metrics,
        "final_train_info": final_info,
    }, args.model_out)
    report = {
        "model_name": "ppg_raw_resnet_se_attention",
        "method_note": "SOTA-aligned raw waveform DL baseline: PPG + first/second derivatives, 1D ResNet with squeeze-excitation and temporal attention pooling. Evaluated with record-level GroupKFold.",
        "manifest": str(args.manifest),
        "num_windows": int(len(y)),
        "num_records": int(len(set(groups))),
        "window_s": args.window_s,
        "stride_s": args.stride_s,
        "label_counts_windows": {"non_af": int((y == 0).sum()), "af": int((y == 1).sum())},
        "label_counts_records": dict(Counter([meta_i["label"] for meta_i in meta if meta_i["start_s"] == 0.0])),
        "device": str(device),
        "cv_metrics_record_level": cv_metrics,
        "fold_reports": fold_reports,
        "model_out": str(args.model_out),
        "caveat": "Dataset is small (35 records), so this is a SOTA-style DL baseline, not a SOTA claim. More external PPG AF data is needed before replacing the feature model.",
    }
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SOTA-aligned raw PPG ResNet-SE-attention AF classifier.")
    parser.add_argument("--manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/ppg_af_manifest.json")
    parser.add_argument("--model-out", default="/data1/jiahui/biosignal-agent/outputs/ppg_af_resnet_se_attention.pt")
    parser.add_argument("--report-out", default="/data1/jiahui/biosignal-agent/outputs/ppg_af_resnet_se_attention_report.json")
    parser.add_argument("--window-s", type=float, default=30.0)
    parser.add_argument("--stride-s", type=float, default=15.0)
    parser.add_argument("--max-windows-per-record", type=int, default=40)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    report = train_cv(args)
    print(json.dumps({
        "model_name": report["model_name"],
        "num_windows": report["num_windows"],
        "num_records": report["num_records"],
        "device": report["device"],
        "cv_metrics_record_level": report["cv_metrics_record_level"],
    }, indent=2))


if __name__ == "__main__":
    main()
