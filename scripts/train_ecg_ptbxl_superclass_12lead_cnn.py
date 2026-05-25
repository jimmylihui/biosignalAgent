from __future__ import annotations
import argparse, json, sys, random
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import wfdb
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path('/data1/jiahui/biosignal-agent/outputs')


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def robust_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.zeros(1000, dtype=np.float32)
    med = float(np.median(x)); iqr = float(np.percentile(x, 75) - np.percentile(x, 25))
    if iqr < 1e-6:
        iqr = float(np.std(x)) + 1e-6
    x = np.clip((x - med) / iqr, -8, 8).astype(np.float32)
    if len(x) != 1000:
        from scipy import signal as scipy_signal
        x = scipy_signal.resample(x, 1000).astype(np.float32)
    return x


class ECGDataset(Dataset):
    def __init__(self, rows: list[dict], target: str, raw_dir: str = '/data1/jiahui/biosignal-agent/datasets/raw/ptb-xl'):
        self.rows = rows
        self.target = target
        self.raw_dir = Path(raw_dir)
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        try:
            rec = wfdb.rdrecord(str(self.raw_dir / r['record']))
            arr = rec.p_signal.astype(np.float32).T
        except Exception:
            x = pd.read_csv(r['path'])['signal'].to_numpy(dtype=np.float32)
            arr = np.tile(x[None, :], (12, 1))
        chans = []
        for c in range(min(12, arr.shape[0])):
            chans.append(robust_norm(arr[c]))
        while len(chans) < 12:
            chans.append(np.zeros(1000, dtype=np.float32))
        x = np.stack(chans[:12], axis=0).astype(np.float32)
        y = float(r[f'label_{self.target}'])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, 7, stride=stride, padding=3, bias=False), nn.BatchNorm1d(c_out), nn.ReLU(inplace=True),
            nn.Conv1d(c_out, c_out, 7, padding=3, bias=False), nn.BatchNorm1d(c_out),
        )
        self.skip = nn.Identity() if c_in == c_out and stride == 1 else nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm1d(c_out))
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.net(x) + self.skip(x))


class TwelveLeadResNet(nn.Module):
    def __init__(self, dropout: float = 0.25):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(12, 32, 15, padding=7, bias=False), nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(2))
        self.blocks = nn.Sequential(
            ResBlock(32, 48, 2),
            ResBlock(48, 64, 2),
            ResBlock(64, 96, 2),
            ResBlock(96, 128, 2),
            ResBlock(128, 160, 2),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(160, 1))
    def forward(self, x):
        return self.head(self.blocks(self.stem(x))).squeeze(-1)


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    return {
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'average_precision': float(average_precision_score(y, p)) if int(y.sum()) else 0.0,
        'roc_auc': float(roc_auc_score(y, p)) if len(set(map(int, y))) > 1 else 0.0,
        'class_report': classification_report(y, pred, labels=[0,1], zero_division=0, output_dict=True),
    }


def train_one(rows: list[dict], target: str, model_path: Path, epochs: int, seed: int, batch_size: int, device: str, num_workers: int = 0, eval_folds: list[int] | None = None) -> dict:
    seed_all(seed)
    folds = eval_folds or sorted({int(r['strat_fold']) for r in rows})
    y_all_full = np.asarray([int(r[f'label_{target}']) for r in rows], dtype=int)
    oof = np.zeros(len(rows), dtype=float)
    evaluated = np.zeros(len(rows), dtype=bool)
    history = []
    for fold in folds:
        train_rows = [r for r in rows if int(r['strat_fold']) != fold]
        val_idx = [i for i, r in enumerate(rows) if int(r['strat_fold']) == fold]
        val_rows = [rows[i] for i in val_idx]
        train_ds = ECGDataset(train_rows, target); val_ds = ECGDataset(val_rows, target)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        model = TwelveLeadResNet().to(device)
        train_y = np.asarray([int(r[f'label_{target}']) for r in train_rows])
        pos_weight = torch.tensor([(len(train_y) - train_y.sum()) / max(1, train_y.sum())], dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        best_p = None; best_ap = -1.0; best_epoch = 0
        for ep in range(1, epochs + 1):
            model.train(); losses=[]
            for xb, yb in train_loader:
                xb = xb.to(device); yb = yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)
                loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
            model.eval(); probs=[]; ys=[]
            with torch.no_grad():
                for xb, yb in val_loader:
                    prob = torch.sigmoid(model(xb.to(device))).detach().cpu().numpy()
                    probs.extend(prob.tolist()); ys.extend(yb.numpy().tolist())
            probs = np.asarray(probs, dtype=float); ys = np.asarray(ys, dtype=int)
            ap = float(average_precision_score(ys, probs)) if int(ys.sum()) else 0.0
            if ap > best_ap:
                best_ap = ap; best_p = probs.copy(); best_epoch = ep
        for j, idx in enumerate(val_idx):
            oof[idx] = float(best_p[j])
            evaluated[idx] = True
        history.append({'fold': fold, 'best_epoch': best_epoch, 'best_average_precision': best_ap, 'val_size': len(val_rows), 'val_positive': int(sum(int(r[f'label_{target}']) for r in val_rows))})
        print(target, 'fold', fold, 'best_ap', round(best_ap,4), flush=True)
    y_eval = y_all_full[evaluated]
    p_eval = oof[evaluated]
    best_thr = float(max(((f1_score(y_eval, p_eval >= t, zero_division=0), t) for t in np.linspace(0.05, 0.95, 91)), key=lambda z: z[0])[1])
    cv = metrics(y_eval, p_eval, best_thr)
    cv['eval_folds'] = list(map(int, folds))
    cv['eval_records'] = int(evaluated.sum())

    # Fit final model on all rows for tool use.
    final_ds = ECGDataset(rows, target)
    final_loader = DataLoader(final_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    final_model = TwelveLeadResNet().to(device)
    pos_weight = torch.tensor([(len(y_all_full) - y_all_full.sum()) / max(1, y_all_full.sum())], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(final_model.parameters(), lr=1e-3, weight_decay=1e-3)
    for ep in range(1, epochs + 1):
        final_model.train()
        for xb, yb in final_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True); loss = loss_fn(final_model(xb), yb); loss.backward(); opt.step()
    bundle = {'model_state_dict': final_model.cpu().state_dict(), 'target': target, 'threshold': best_thr, 'cv_metrics': cv, 'history': history, 'model': 'TwelveLeadResNet', 'input_len': 1000, 'label_counts': dict(Counter(map(int, y_all_full)))}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, model_path)
    return {'model_path': str(model_path), 'target': target, 'threshold': best_thr, 'cv_metrics': cv, 'history': history, 'label_counts': bundle['label_counts']}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/processed/ptbxl_superclass_lead2_balanced360_manifest.json'))
    ap.add_argument('--out-dir', type=Path, default=OUT)
    ap.add_argument('--report-path', type=Path, default=OUT/'ecg_ptbxl_superclass_12lead_resnet_train_report.json')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=24)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--targets', nargs='+', default=['cd', 'sttc'])
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--eval-folds', nargs='*', type=int, default=None)
    args = ap.parse_args()
    rows = json.load(open(args.manifest))['rows']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    report = {'manifest': str(args.manifest), 'device': device, 'num_records': len(rows), 'targets': {}}
    for target in [t.lower() for t in args.targets]:
        report['targets'][target] = train_one(rows, target, args.out_dir / f'ecg_ptbxl_{target}_12lead_resnet.pt', args.epochs, args.seed, args.batch_size, device, args.num_workers, args.eval_folds)
    args.report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
