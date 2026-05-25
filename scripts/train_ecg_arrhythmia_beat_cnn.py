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
import wfdb
from scipy import signal as scipy_signal
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

MITDB_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]
BEAT_SYMBOLS = {"N", "L", "R", "A", "a", "J", "S", "V", "F", "e", "j", "E", "/", "f", "Q"}
NORMAL_SYMBOLS = {"N", "L", "R", "e", "j"}
OUT = Path('/data1/jiahui/biosignal-agent/outputs')
MODEL_PATH = OUT / 'ecg_arrhythmia_beat_cnn_model.pt'


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def robust_segment(values: np.ndarray, center: int, pre: int, post: int, beat_len: int) -> np.ndarray:
    start = center - pre; stop = center + post
    seg = np.zeros(pre + post, dtype=np.float32)
    src_start = max(start, 0); src_stop = min(stop, len(values))
    dst_start = src_start - start; dst_stop = dst_start + max(0, src_stop - src_start)
    if src_stop > src_start:
        seg[dst_start:dst_stop] = values[src_start:src_stop]
    med = float(np.median(seg)); q75, q25 = np.percentile(seg, [75, 25]); scale = float(q75 - q25)
    if scale < 1e-8:
        scale = float(np.std(seg)) + 1e-8
    seg = np.clip((seg - med) / scale, -8.0, 8.0)
    if len(seg) != beat_len:
        seg = scipy_signal.resample(seg, beat_len)
    return seg.astype(np.float32)


def load_beats(raw_dir: Path, records: list[str], beat_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    X, F, y, groups = [], [], [], []
    for record in records:
        header = wfdb.rdheader(str(raw_dir / record))
        rec = wfdb.rdrecord(str(raw_dir / record))
        lead = 'MLII' if 'MLII' in rec.sig_name else rec.sig_name[0]
        values = rec.p_signal[:, rec.sig_name.index(lead)].astype(np.float32)
        ann = wfdb.rdann(str(raw_dir / record), 'atr')
        beat_positions = []
        beat_labels = []
        for sample, symbol in zip(ann.sample, ann.symbol):
            if symbol not in BEAT_SYMBOLS:
                continue
            beat_positions.append(int(sample))
            beat_labels.append(0 if symbol in NORMAL_SYMBOLS else 1)
        if len(beat_positions) < 3:
            continue
        beat_positions = np.asarray(beat_positions, dtype=int)
        rr_prev = np.r_[np.nan, np.diff(beat_positions) / float(header.fs)]
        rr_next = np.r_[np.diff(beat_positions) / float(header.fs), np.nan]
        med_rr = np.nanmedian(np.r_[rr_prev, rr_next])
        rr_prev = np.nan_to_num(rr_prev, nan=med_rr, posinf=med_rr, neginf=med_rr)
        rr_next = np.nan_to_num(rr_next, nan=med_rr, posinf=med_rr, neginf=med_rr)
        pre = int(0.25 * header.fs); post = int(0.45 * header.fs)
        for i, (sample, label) in enumerate(zip(beat_positions, beat_labels)):
            X.append(robust_segment(values, int(sample), pre, post, beat_len))
            local_hr = 60.0 / max(float((rr_prev[i] + rr_next[i]) / 2.0), 0.25)
            F.append([float(np.clip(rr_prev[i], 0.25, 3.0)), float(np.clip(rr_next[i], 0.25, 3.0)), float(np.clip(local_hr / 100.0, 0.2, 2.5))])
            y.append(int(label)); groups.append(record)
    return np.asarray(X, dtype=np.float32), np.asarray(F, dtype=np.float32), np.asarray(y, dtype=np.int64), groups


class BeatCNN(nn.Module):
    def __init__(self, feature_dim: int = 3, dropout: float = 0.25):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 9, padding=4, bias=False), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 7, padding=3, bias=False), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2, bias=False), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 160, 3, padding=1, bias=False), nn.BatchNorm1d(160), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(160 + feature_dim, 96), nn.ReLU(), nn.Dropout(dropout), nn.Linear(96, 1))

    def forward(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x[:, None, :]
        z = self.cnn(x)
        return self.head(torch.cat([z, f], dim=1)).squeeze(-1)


def predict(model: BeatCNN, X: np.ndarray, F: np.ndarray, device: str) -> np.ndarray:
    model.eval(); out=[]
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(F, dtype=torch.float32))
    with torch.no_grad():
        for xb, fb in DataLoader(ds, batch_size=512):
            out.append(torch.sigmoid(model(xb.to(device), fb.to(device))).cpu().numpy())
    return np.concatenate(out).astype(float)


def train_fold(X, F, y, tr, va, epochs, seed, device):
    seed_all(seed)
    model = BeatCNN(feature_dim=F.shape[1]).to(device)
    pos=max(float(np.sum(y[tr]==1)),1.0); neg=max(float(np.sum(y[tr]==0)),1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos], dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    ds = TensorDataset(torch.tensor(X[tr], dtype=torch.float32), torch.tensor(F[tr], dtype=torch.float32), torch.tensor(y[tr], dtype=torch.float32))
    loader = DataLoader(ds, batch_size=512, shuffle=True)
    best_state=None; best_loss=math.inf; stale=0
    for epoch in range(epochs):
        model.train()
        for xb, fb, yb in loader:
            xb=xb.to(device); fb=fb.to(device); yb=yb.to(device)
            opt.zero_grad(set_to_none=True); loss=criterion(model(xb, fb), yb); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 4.0); opt.step()
        if va is None:
            continue
        model.eval()
        with torch.no_grad():
            vv = torch.tensor(X[va], dtype=torch.float32, device=device); vf=torch.tensor(F[va], dtype=torch.float32, device=device); vy=torch.tensor(y[va], dtype=torch.float32, device=device)
            val_loss=float(criterion(model(vv, vf), vy).cpu())
        if val_loss < best_loss - 1e-4:
            best_loss=val_loss; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else:
            stale += 1
        if stale >= 8:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_loss


def metrics(y, proba, threshold):
    pred=(proba>=threshold).astype(int)
    d={'threshold':float(threshold),'accuracy':float(accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'average_precision':float(average_precision_score(y,proba))}
    try: d['roc_auc']=float(roc_auc_score(y,proba))
    except Exception: d['roc_auc']=0.0
    return d


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw-dir', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/raw/mitdb'))
    parser.add_argument('--records', nargs='*', default=MITDB_RECORDS)
    parser.add_argument('--beat-len', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--seed', type=int, default=31)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model-path', type=Path, default=MODEL_PATH)
    args=parser.parse_args(); seed_all(args.seed)
    X,F,y,groups=load_beats(args.raw_dir,args.records,args.beat_len)
    print(json.dumps({'num_beats':int(len(y)),'label_counts':dict(Counter(map(int,y))),'records':len(set(groups))}, indent=2))
    cv=GroupKFold(n_splits=5); proba=np.zeros(len(y)); folds=[]
    for fold,(tr,va) in enumerate(cv.split(X,y,groups=groups)):
        model,val_loss=train_fold(X,F,y,tr,va,args.epochs,args.seed+fold,args.device)
        proba[va]=predict(model,X[va],F[va],args.device)
        folds.append({'fold':fold,'train_size':int(len(tr)),'val_size':int(len(va)),'val_loss':float(val_loss),'val_label_counts':dict(Counter(map(int,y[va])))})
        print('fold', fold, folds[-1], flush=True)
    thresholds=np.linspace(0.05,0.95,91)
    threshold=float(max(((f1_score(y,proba>=t,zero_division=0),t) for t in thresholds), key=lambda z:z[0])[1])
    cv_metrics=metrics(y,proba,threshold)
    all_idx=np.arange(len(y)); final_model,_=train_fold(X,F,y,all_idx,None,max(args.epochs,30),args.seed+999,args.device)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict':final_model.cpu().state_dict(),'architecture':'BeatCNN','beat_len':args.beat_len,'feature_dim':int(F.shape[1]),'threshold':threshold,'cv_metrics':cv_metrics,'fold_reports':folds,'label_counts':dict(Counter(map(int,y)))}, args.model_path)
    report={'model_path':str(args.model_path),'num_beats':int(len(y)),'label_counts':dict(Counter(map(int,y))),'threshold':threshold,'cv_metrics':cv_metrics,'fold_reports':folds}
    out=OUT/'ecg_arrhythmia_beat_cnn_train_report.json'; out.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
