from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.peak_detectors import ppg_multiscale_systolic_peaks  # noqa: E402
from biosignal_agent.tools.ppg_tools import _ppg_artifact_metrics  # noqa: E402


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


def interval_features(intervals: np.ndarray) -> np.ndarray:
    intervals = np.asarray(intervals, dtype=np.float32)
    intervals = intervals[np.isfinite(intervals)]
    intervals = intervals[(intervals >= 0.25) & (intervals <= 3.0)]
    if len(intervals) < 4:
        return np.zeros(10, dtype=np.float32)
    mean = float(np.mean(intervals))
    med = float(np.median(intervals))
    diff = np.abs(np.diff(intervals))
    rmssd = float(np.sqrt(np.mean(np.diff(intervals) ** 2))) if len(intervals) > 1 else 0.0
    robust_cv = float((1.4826 * np.median(np.abs(intervals - med))) / (med + 1e-8))
    return np.asarray([
        mean,
        med,
        float(np.std(intervals) / (mean + 1e-8)),
        robust_cv,
        rmssd / (mean + 1e-8),
        float(np.mean(diff > 0.08)) if len(diff) else 0.0,
        float(np.mean(diff > 0.12)) if len(diff) else 0.0,
        float(np.mean(diff > 0.20)) if len(diff) else 0.0,
        float(np.mean(intervals < 0.5)),
        float(np.mean(intervals > 1.5)),
    ], dtype=np.float32)


def normalize_interval_sequence(intervals: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    intervals = np.asarray(intervals, dtype=np.float32)
    intervals = intervals[np.isfinite(intervals)]
    intervals = intervals[(intervals >= 0.25) & (intervals <= 3.0)]
    mask = np.zeros(seq_len, dtype=np.float32)
    seq = np.zeros((seq_len, 4), dtype=np.float32)
    if len(intervals) == 0:
        return seq, mask
    intervals = intervals[:seq_len]
    med = np.median(intervals)
    mad = np.median(np.abs(intervals - med)) + 1e-4
    z = np.clip((intervals - med) / (1.4826 * mad), -8.0, 8.0)
    d = np.concatenate([[0.0], np.diff(intervals)])
    dz = np.clip(d / (med + 1e-4), -4.0, 4.0)
    seq[: len(intervals), 0] = z
    seq[: len(intervals), 1] = dz
    seq[: len(intervals), 2] = np.clip(intervals / (med + 1e-4), 0.2, 3.0)
    seq[: len(intervals), 3] = np.arange(len(intervals), dtype=np.float32) / max(1, seq_len - 1)
    mask[: len(intervals)] = 1.0
    return seq, mask


def load_windows(manifest_path: Path, window_s: float, stride_s: float, seq_len: int, max_windows_per_record: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text())
    xs, masks, feats, ys, groups, meta = [], [], [], [], [], []
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
            try:
                peaks, _ = ppg_multiscale_systolic_peaks(window, fs)
            except Exception:
                peaks = np.asarray([], dtype=int)
            if len(peaks) < 8:
                continue
            intervals = np.diff(peaks) / fs
            seq, mask = normalize_interval_sequence(intervals, seq_len)
            if mask.sum() < 6:
                continue
            artifact = _ppg_artifact_metrics(window, fs)
            feat = np.concatenate([interval_features(intervals), np.asarray([
                artifact.get("artifact_score") or 0.0,
                artifact.get("baseline_wander_ratio") or 0.0,
                artifact.get("high_frequency_noise_ratio") or 0.0,
                len(peaks) / max(1e-6, window_s / 60.0),
            ], dtype=np.float32)])
            xs.append(seq)
            masks.append(mask)
            feats.append(feat)
            ys.append(label)
            groups.append(rec["record"])
            meta.append({"record": rec["record"], "label": rec["label"], "start_s": float(start / fs), "num_peaks": int(len(peaks))})
    return np.asarray(xs, np.float32), np.asarray(masks, np.float32), np.asarray(feats, np.float32), np.asarray(ys, np.int64), np.asarray(groups), meta


class IntervalAttentionBiLSTM(nn.Module):
    def __init__(self, seq_features: int = 4, tab_features: int = 14, hidden: int = 48, dropout: float = 0.3) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(seq_features, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.lstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.attn = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.tab = nn.Sequential(nn.LayerNorm(tab_features), nn.Linear(tab_features, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, seq: torch.Tensor, mask: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(seq)
        z, _ = self.lstm(z)
        scores = self.attn(z).squeeze(-1)
        scores = scores.masked_fill(mask <= 0, -1e4)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(z * weights.unsqueeze(-1), dim=1)
        tab = self.tab(feat)
        return self.head(torch.cat([pooled, tab], dim=-1)).squeeze(-1)


def make_loader(seq, mask, feat, y, batch_size, shuffle) -> DataLoader:
    return DataLoader(TensorDataset(torch.tensor(seq), torch.tensor(mask), torch.tensor(feat), torch.tensor(y, dtype=torch.float32)), batch_size=batch_size, shuffle=shuffle)


def predict(model, seq, mask, feat, device, batch_size) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for xb, mb, fb, _ in make_loader(seq, mask, feat, np.zeros(len(seq)), batch_size, False):
            outs.append(torch.sigmoid(model(xb.to(device), mb.to(device), fb.to(device))).cpu().numpy())
    return np.concatenate(outs) if outs else np.asarray([])


def train_fold(train, val, args, device):
    seq_tr, mask_tr, feat_tr, y_tr = train
    seq_va, mask_va, feat_va, y_va = val
    torch.manual_seed(args.seed)
    model = IntervalAttentionBiLSTM(hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = float((len(y_tr) - y_tr.sum()) / max(1, y_tr.sum()))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    best_state, best_auc, best_epoch, bad = None, -1.0, -1, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses=[]
        for xb, mb, fb, yb in make_loader(seq_tr, mask_tr, feat_tr, y_tr, args.batch_size, True):
            xb, mb, fb, yb = xb.to(device), mb.to(device), fb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb, mb, fb), yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
            losses.append(float(loss.detach().cpu()))
        prob = predict(model, seq_va, mask_va, feat_va, device, args.batch_size)
        try: auc = roc_auc_score(y_va, prob)
        except ValueError: auc = 0.5
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_auc": float(auc)})
        if auc > best_auc + 1e-4:
            best_auc, best_epoch = float(auc), epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= args.patience: break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_auc": best_auc, "best_epoch": best_epoch, "history": history}


def aggregate(prob, y, groups):
    yt, yp, yprob = [], [], []
    for group in sorted(set(groups)):
        idx = np.where(groups == group)[0]
        p = float(np.mean(prob[idx])); label = int(np.round(np.mean(y[idx])))
        yt.append(label); yprob.append(p); yp.append(1 if p >= 0.5 else 0)
    return yt, yp, yprob


def metric_report(yt, yp, yprob):
    precision, recall, f1, support = precision_recall_fscore_support(yt, yp, labels=[0,1], zero_division=0)
    try: auc = roc_auc_score(yt, yprob)
    except ValueError: auc = None
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro")),
        "weighted_f1": float(f1_score(yt, yp, average="weighted")),
        "auroc": float(auc) if auc is not None else None,
        "labels": ["non_af", "af"],
        "per_class": {
            "non_af": {"precision": float(precision[0]), "recall": float(recall[0]), "f1": float(f1[0]), "support": int(support[0])},
            "af": {"precision": float(precision[1]), "recall": float(recall[1]), "f1": float(f1[1]), "support": int(support[1])},
        },
        "confusion_matrix_labels": ["non_af", "af"],
        "confusion_matrix": confusion_matrix(yt, yp, labels=[0,1]).tolist(),
    }


def run(args):
    seq, mask, feat, y, groups, meta = load_windows(Path(args.manifest), args.window_s, args.stride_s, args.seq_len, args.max_windows_per_record)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    cv = GroupKFold(n_splits=min(args.folds, len(set(groups))))
    all_prob = np.zeros(len(y), dtype=float)
    folds=[]
    for fold, (tr, va) in enumerate(cv.split(seq, y, groups), start=1):
        model, info = train_fold((seq[tr], mask[tr], feat[tr], y[tr]), (seq[va], mask[va], feat[va], y[va]), args, device)
        prob = predict(model, seq[va], mask[va], feat[va], device, args.batch_size)
        all_prob[va] = prob
        yt, yp, yprob = aggregate(prob, y[va], groups[va])
        rep = metric_report(yt, yp, yprob)
        folds.append({"fold": fold, "train_info": info, "record_metrics": rep, "val_records": sorted(set(groups[va].tolist()))})
        print(json.dumps({"fold": fold, "record_metrics": rep}, indent=2))
    yt, yp, yprob = aggregate(all_prob, y, groups)
    cv_metrics = metric_report(yt, yp, yprob)
    model, final_info = train_fold((seq, mask, feat, y), (seq, mask, feat, y), args, device)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "architecture": "IntervalAttentionBiLSTM", "seq_len": args.seq_len, "window_s": args.window_s, "stride_s": args.stride_s, "cv_metrics": cv_metrics, "final_train_info": final_info}, args.model_out)
    report = {
        "model_name": "ppg_interval_attention_bilstm",
        "method_note": "SOTA-style rhythm DL: pulse interval sequence with attention BiLSTM plus compact rhythm/artifact features. Record-level GroupKFold.",
        "num_windows": int(len(y)), "num_records": int(len(set(groups))), "device": str(device),
        "window_s": args.window_s, "stride_s": args.stride_s, "seq_len": args.seq_len,
        "label_counts_windows": {"non_af": int((y==0).sum()), "af": int((y==1).sum())},
        "cv_metrics_record_level": cv_metrics, "fold_reports": folds, "model_out": str(args.model_out),
        "caveat": "Uses PPG peak-derived intervals, so it is a deep rhythm model rather than fully raw waveform DL. Still limited by 35-record MIMIC PERform AF size.",
    }
    Path(args.report_out).write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description="Train PPG pulse-interval attention BiLSTM AF classifier.")
    parser.add_argument("--manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/ppg_af_manifest.json")
    parser.add_argument("--model-out", default="/data1/jiahui/biosignal-agent/outputs/ppg_af_interval_attention_bilstm.pt")
    parser.add_argument("--report-out", default="/data1/jiahui/biosignal-agent/outputs/ppg_af_interval_attention_bilstm_report.json")
    parser.add_argument("--window-s", type=float, default=60.0)
    parser.add_argument("--stride-s", type=float, default=15.0)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-windows-per-record", type=int, default=60)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"model_name": report["model_name"], "num_windows": report["num_windows"], "num_records": report["num_records"], "cv_metrics_record_level": report["cv_metrics_record_level"]}, indent=2))


if __name__ == "__main__":
    main()
