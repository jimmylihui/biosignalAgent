from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import wfdb
from scipy import signal as scipy_signal
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
OUT = Path('/data1/jiahui/biosignal-agent/outputs')
MODEL_PATH = OUT / 'ecg_rpeak_segmentation_cnn_model.pt'


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def complete(raw: Path, record: str) -> bool:
    return all((raw / f'{record}.{ext}').exists() for ext in ['hea', 'dat', 'atr'])


def robust_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    med = float(np.median(x)); q75, q25 = np.percentile(x, [75, 25]); scale = float(q75 - q25)
    if scale < 1e-8: scale = float(np.std(x)) + 1e-8
    return np.clip((x - med) / scale, -8.0, 8.0).astype(np.float32)


def load_records(specs, target_fs: float = 250.0):
    rows = []
    for db_name, raw_dir, records in specs:
        for record in records:
            if not complete(raw_dir, record): continue
            try:
                header = wfdb.rdheader(str(raw_dir / record))
                rec = wfdb.rdrecord(str(raw_dir / record))
                lead = 'MLII' if 'MLII' in rec.sig_name else rec.sig_name[0]
                values = rec.p_signal[:, rec.sig_name.index(lead)].astype(np.float32)
                ann = wfdb.rdann(str(raw_dir / record), 'atr')
            except Exception as exc:
                print(f'skip {db_name}/{record}: {type(exc).__name__}: {exc}', flush=True); continue
            peak_samples = np.asarray([int(s) for s, sym in zip(ann.sample, ann.symbol) if sym in BEAT_SYMBOLS], dtype=int)
            if len(peak_samples) < 5: continue
            fs = float(header.fs)
            if abs(fs - target_fs) > 1e-6:
                new_len = int(round(len(values) * target_fs / fs))
                values_rs = scipy_signal.resample(values, new_len).astype(np.float32)
                peaks_rs = np.rint(peak_samples * target_fs / fs).astype(int)
                fs_out = target_fs
            else:
                values_rs = values; peaks_rs = peak_samples; fs_out = fs
            peaks_rs = peaks_rs[(peaks_rs >= 0) & (peaks_rs < len(values_rs))]
            rows.append({'name': f'{db_name}:{record}', 'values': values_rs, 'peaks': peaks_rs, 'fs': fs_out})
            print(f'loaded {db_name}/{record}: samples={len(values_rs)} peaks={len(peaks_rs)}', flush=True)
    return rows


def make_windows(records, win_len: int, stride: int, radius: int):
    X, Y, groups = [], [], []
    for rec in records:
        values = rec['values']; peaks = rec['peaks']
        for start in range(0, max(1, len(values) - win_len + 1), stride):
            stop = start + win_len
            seg = values[start:stop]
            if len(seg) < win_len: continue
            local = peaks[(peaks >= start) & (peaks < stop)] - start
            if len(local) == 0 and random.random() > 0.10:
                continue
            y = np.zeros(win_len, dtype=np.float32)
            for p in local:
                lo = max(0, int(p) - radius); hi = min(win_len, int(p) + radius + 1)
                y[lo:hi] = 1.0
            X.append(robust_norm(seg)); Y.append(y); groups.append(rec['name'])
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), groups


class RPeakSegCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 9, padding=4, bias=False), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 48, 9, padding=8, dilation=2, bias=False), nn.BatchNorm1d(48), nn.ReLU(),
            nn.Conv1d(48, 64, 9, padding=16, dilation=4, bias=False), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 7, padding=18, dilation=6, bias=False), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 32, 5, padding=2, bias=False), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 1, 1),
        )
    def forward(self, x):
        if x.ndim == 2: x = x[:, None, :]
        return self.net(x).squeeze(1)


def train_fold(X, Y, tr, va, epochs, seed, device):
    seed_all(seed); model=RPeakSegCNN().to(device)
    pos = float(Y[tr].sum()); neg = float(Y[tr].size - pos)
    pos_weight = torch.tensor([min(neg / max(pos, 1.0), 200.0)], dtype=torch.float32, device=device)
    criterion=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt=torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    ds=TensorDataset(torch.tensor(X[tr],dtype=torch.float32), torch.tensor(Y[tr],dtype=torch.float32))
    loader=DataLoader(ds,batch_size=64,shuffle=True)
    best=None; best_loss=math.inf; stale=0
    for epoch in range(epochs):
        model.train()
        for xb,yb in loader:
            xb=xb.to(device); yb=yb.to(device); opt.zero_grad(set_to_none=True); loss=criterion(model(xb), yb); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),4); opt.step()
        if va is None: continue
        model.eval()
        with torch.no_grad():
            val=float(criterion(model(torch.tensor(X[va],dtype=torch.float32,device=device)), torch.tensor(Y[va],dtype=torch.float32,device=device)).cpu())
        if val < best_loss - 1e-4:
            best_loss=val; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale += 1
        if stale >= 5: break
    if best is not None: model.load_state_dict(best)
    return model, best_loss


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--include',nargs='*',default=['mitdb','incartdb','svdb']); ap.add_argument('--mitdb-raw-dir',type=Path,default=Path('/data1/jiahui/biosignal-agent/datasets/raw/mitdb')); ap.add_argument('--incart-raw-dir',type=Path,default=Path('/data1/jiahui/biosignal-agent/datasets/raw/incartdb')); ap.add_argument('--svdb-raw-dir',type=Path,default=Path('/data1/jiahui/biosignal-agent/datasets/raw/svdb')); ap.add_argument('--target-fs',type=float,default=250.0); ap.add_argument('--win-len',type=int,default=2048); ap.add_argument('--stride',type=int,default=1024); ap.add_argument('--radius-ms',type=float,default=32.0); ap.add_argument('--epochs',type=int,default=8); ap.add_argument('--seed',type=int,default=79); ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); ap.add_argument('--model-path',type=Path,default=MODEL_PATH); args=ap.parse_args(); seed_all(args.seed)
    specs=[]
    if 'mitdb' in args.include: specs.append(('mitdb',args.mitdb_raw_dir,MITDB_RECORDS))
    if 'incartdb' in args.include: specs.append(('incartdb',args.incart_raw_dir,wfdb.get_record_list('incartdb')))
    if 'svdb' in args.include: specs.append(('svdb',args.svdb_raw_dir,wfdb.get_record_list('svdb')))
    records=load_records(specs,args.target_fs)
    radius=max(1,int(round(args.radius_ms/1000.0*args.target_fs)))
    X,Y,groups=make_windows(records,args.win_len,args.stride,radius)
    print(json.dumps({'num_records':len(records),'num_windows':int(len(X)),'positive_fraction':float(Y.mean()),'include':args.include},indent=2),flush=True)
    splits=list(GroupKFold(n_splits=5).split(X, np.zeros(len(X)), groups=groups))
    folds=[]
    for fold,(tr,va) in enumerate(splits):
        model,vl=train_fold(X,Y,tr,va,args.epochs,args.seed+fold,args.device)
        folds.append({'fold':fold,'train_size':int(len(tr)),'val_size':int(len(va)),'val_loss':float(vl)})
        print('fold', fold, folds[-1], flush=True)
    allidx=np.arange(len(X)); final,_=train_fold(X,Y,allidx,None,max(args.epochs,10),args.seed+999,args.device)
    args.model_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':final.cpu().state_dict(),'architecture':'RPeakSegCNN','target_fs':args.target_fs,'win_len':args.win_len,'radius_ms':args.radius_ms,'threshold':0.5,'fold_reports':folds,'include':args.include},args.model_path)
    report={'model_path':str(args.model_path),'num_records':len(records),'num_windows':int(len(X)),'positive_fraction':float(Y.mean()),'fold_reports':folds}
    (OUT/'ecg_rpeak_segmentation_cnn_train_report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
