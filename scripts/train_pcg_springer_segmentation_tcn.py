from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, accuracy_score, classification_report

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.common import bpm_from_peaks
from scripts.evaluate_pcg_springer_segmentation import event_metrics, state_centers

OUT = Path('/data1/jiahui/biosignal-agent/outputs/pcg_springer_segmentation_tcn.pt')
REPORT = Path('/data1/jiahui/biosignal-agent/outputs/pcg_springer_segmentation_tcn_report.json')


def robust_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    med = float(np.nanmedian(x))
    scale = float(np.nanpercentile(np.abs(x - med), 95)) + 1e-6
    return np.clip((x - med) / scale, -6.0, 6.0).astype(np.float32)


class ChunkDataset(Dataset):
    def __init__(self, rows: list[dict], chunk_len: int, stride: int):
        self.rows = rows
        self.chunk_len = int(chunk_len)
        self.index = []
        self.cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for i, row in enumerate(rows):
            n = int(row['num_samples'])
            starts = list(range(0, max(1, n - self.chunk_len + 1), int(stride)))
            if not starts or starts[-1] + self.chunk_len < n:
                starts.append(max(0, n - self.chunk_len))
            self.index.extend((i, s) for s in starts)
    def __len__(self):
        return len(self.index)
    def _load(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        if i not in self.cache:
            df = pd.read_csv(self.rows[i]['path'])
            x = robust_norm(df['signal'].to_numpy(dtype=np.float32))
            y = df['state_label'].to_numpy(dtype=np.int64) - 1
            self.cache[i] = (x, y)
        return self.cache[i]
    def __getitem__(self, idx: int):
        row_i, start = self.index[idx]
        x, y = self._load(row_i)
        end = start + self.chunk_len
        xx = np.zeros(self.chunk_len, dtype=np.float32)
        yy = np.full(self.chunk_len, 3, dtype=np.int64)
        seg_x = x[start:min(end, len(x))]
        seg_y = y[start:min(end, len(y))]
        xx[:len(seg_x)] = seg_x
        yy[:len(seg_y)] = seg_y
        return torch.from_numpy(xx[None, :]), torch.from_numpy(yy)


class DilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.05):
        super().__init__()
        pad = dilation * 3
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 7, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(x + self.net(x))


class PCGStateTCN(nn.Module):
    def __init__(self, channels: int = 48):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, channels, 15, padding=7, bias=False), nn.BatchNorm1d(channels), nn.SiLU())
        self.blocks = nn.Sequential(*[DilatedBlock(channels, d) for d in [1, 2, 4, 8, 16, 32, 64, 1, 2, 4]])
        self.head = nn.Conv1d(channels, 4, 1)
    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def split_rows(rows: list[dict], val_fold: int) -> tuple[list[dict], list[dict]]:
    val = [r for r in rows if int(r.get('record_id', '0')) % 5 == int(val_fold) % 5]
    train = [r for r in rows if r not in val]
    return train, val


def predict_labels(model: nn.Module, values: np.ndarray, device: str, chunk_len: int) -> np.ndarray:
    x = robust_norm(values)
    pad = (-len(x)) % chunk_len
    if pad:
        x = np.pad(x, (0, pad))
    probs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), chunk_len):
            xb = torch.from_numpy(x[start:start + chunk_len][None, None, :]).to(device)
            probs.append(torch.softmax(model(xb), dim=1).cpu().numpy()[0])
    pred = np.concatenate(probs, axis=1)[:, :len(values)].argmax(axis=0) + 1
    return pred.astype(int)


def evaluate(model: nn.Module, rows: list[dict], device: str, chunk_len: int, tolerance_ms: float) -> dict:
    y_true_all = []
    y_pred_all = []
    s1_items = []
    s2_items = []
    hr_errors = []
    per_record = []
    for row in rows:
        df = pd.read_csv(row['path'])
        values = df['signal'].to_numpy(dtype=np.float32)
        labels = df['state_label'].to_numpy(dtype=int)
        pred = predict_labels(model, values, device, chunk_len)
        y_true_all.append(labels)
        y_pred_all.append(pred)
        fs = float(row['sampling_rate'])
        true_s1 = state_centers(labels, 1); true_s2 = state_centers(labels, 3)
        pred_s1 = state_centers(pred, 1); pred_s2 = state_centers(pred, 3)
        s1 = event_metrics(true_s1, pred_s1, fs, tolerance_ms)
        s2 = event_metrics(true_s2, pred_s2, fs, tolerance_ms)
        s1_items.append(s1); s2_items.append(s2)
        true_hr = bpm_from_peaks(true_s1, fs) if len(true_s1) >= 2 else None
        pred_hr = bpm_from_peaks(pred_s1, fs) if len(pred_s1) >= 2 else None
        hr_err = abs(float(pred_hr) - float(true_hr)) if true_hr is not None and pred_hr is not None else None
        if hr_err is not None:
            hr_errors.append(hr_err)
        per_record.append({'record_id': row['record_id'], 's1': s1, 's2': s2, 'true_hr_bpm': true_hr, 'pred_hr_bpm': pred_hr, 'hr_abs_error_bpm': hr_err})
    yt = np.concatenate(y_true_all)
    yp = np.concatenate(y_pred_all)
    def summarize(items: list[dict]) -> dict:
        tp = sum(x['tp'] for x in items); fp = sum(x['fp'] for x in items); fn = sum(x['fn'] for x in items)
        p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
        maes = [x['mae_ms'] for x in items if x.get('mae_ms') is not None]
        return {'micro_precision': float(p), 'micro_recall': float(r), 'micro_f1': float(2*p*r/max(1e-12, p+r)), 'mean_record_f1': float(np.mean([x['f1'] for x in items])), 'mae_ms_mean_record': float(np.mean(maes)) if maes else None, 'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}
    return {
        'state_accuracy': float(accuracy_score(yt, yp)),
        'state_macro_f1': float(f1_score(yt, yp, average='macro')),
        'state_weighted_f1': float(f1_score(yt, yp, average='weighted')),
        'state_report': classification_report(yt, yp, labels=[1,2,3,4], target_names=['S1','systole','S2','diastole'], zero_division=0, output_dict=True),
        's1_summary': summarize(s1_items),
        's2_summary': summarize(s2_items),
        'heart_rate_mae_bpm': float(np.mean(hr_errors)) if hr_errors else None,
        'heart_rate_num_records': int(len(hr_errors)),
        'per_record': per_record,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description='Train a lightweight TCN PCG state-segmentation baseline on Springer labels.')
    ap.add_argument('--manifest', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/processed/pcg_springer_segmentation_manifest.json'))
    ap.add_argument('--model-path', type=Path, default=OUT)
    ap.add_argument('--report-path', type=Path, default=REPORT)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--chunk-len', type=int, default=4096)
    ap.add_argument('--stride', type=int, default=2048)
    ap.add_argument('--val-fold', type=int, default=0)
    ap.add_argument('--tolerance-ms', type=float, default=80.0)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rows = json.load(open(args.manifest))['rows']
    train_rows, val_rows = split_rows(rows, args.val_fold)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_ds = ChunkDataset(train_rows, args.chunk_len, args.stride)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    counts = np.zeros(4, dtype=float)
    for r in train_rows:
        for k, v in r['label_counts'].items():
            counts[int(k)-1] += float(v)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    model = PCGStateTCN().to(device)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    history = []
    best = None
    best_score = -1.0
    for ep in range(1, args.epochs + 1):
        model.train(); losses = []
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val = evaluate(model, val_rows, device, args.chunk_len, args.tolerance_ms)
        score = float((val['s1_summary']['micro_f1'] + val['s2_summary']['micro_f1']) / 2.0)
        row = {'epoch': ep, 'train_loss': float(np.mean(losses)), 'val_state_macro_f1': val['state_macro_f1'], 'val_s1_micro_f1': val['s1_summary']['micro_f1'], 'val_s2_micro_f1': val['s2_summary']['micro_f1'], 'val_hr_mae_bpm': val['heart_rate_mae_bpm']}
        history.append(row)
        print(json.dumps(row), flush=True)
        if score > best_score:
            best_score = score
            best = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best is not None:
        model.load_state_dict(best)
    val = evaluate(model, val_rows, device, args.chunk_len, args.tolerance_ms)
    bundle = {'model_state_dict': model.cpu().state_dict(), 'model': 'PCGStateTCN', 'sampling_rate': 1000.0, 'state_label_mapping': {'1': 'S1', '2': 'systole', '3': 'S2', '4': 'diastole'}, 'chunk_len': args.chunk_len, 'val_fold': args.val_fold, 'history': history, 'val_metrics': {k: v for k, v in val.items() if k != 'per_record'}}
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.model_path)
    report = {'model_path': str(args.model_path), 'num_train_records': len(train_rows), 'num_val_records': len(val_rows), 'history': history, **val}
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['num_train_records','num_val_records','state_macro_f1','s1_summary','s2_summary','heart_rate_mae_bpm']}, indent=2))


if __name__ == '__main__':
    main()
