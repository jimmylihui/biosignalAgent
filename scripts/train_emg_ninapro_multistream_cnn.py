from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path('/data1/jiahui/biosignal-agent')
DATA = ROOT / 'datasets/raw/emg_ninapro_db1'
OUT = ROOT / 'outputs'


def build_cache(path: Path, window: int, step: int, max_per_class: int, seed: int = 17):
    rng = np.random.default_rng(seed)
    offsets = {1: 0, 2: 12, 3: 29}
    buckets = {}
    for mat_path in sorted(DATA.glob('s*/S*_A1_E*.mat')):
        mat = loadmat(mat_path)
        emg = np.asarray(mat['emg'], dtype=np.float32)
        stim = np.asarray(mat['restimulus']).ravel().astype(int)
        rep = np.asarray(mat['rerepetition']).ravel().astype(int)
        subject = f"s{int(np.asarray(mat['subject']).ravel()[0]):02d}"
        exercise = int(np.asarray(mat['exercise']).ravel()[0])
        off = offsets[exercise]
        emg = (emg - emg.mean(0, keepdims=True)) / (emg.std(0, keepdims=True) + 1e-6)
        for start in range(0, len(emg) - window + 1, step):
            local = int(np.bincount(stim[start:start + window], minlength=60).argmax())
            if local == 0 or np.mean(stim[start:start + window] == local) < 0.85:
                continue
            label = local + off
            repetition = int(np.bincount(rep[start:start + window]).argmax())
            key = f'{subject}_r{repetition}_g{label}'
            buckets.setdefault(label, []).append((emg[start:start + window].T.copy(), label, subject, repetition, key))
    rows = []
    for label, items in sorted(buckets.items()):
        if len(items) > max_per_class:
            idx = rng.choice(len(items), max_per_class, replace=False)
            items = [items[int(i)] for i in idx]
        rows.extend(items)
    rng.shuffle(rows)
    X = np.stack([r[0] for r in rows]).astype(np.float32)
    y = np.asarray([r[1] for r in rows])
    subjects = np.asarray([r[2] for r in rows])
    reps = np.asarray([r[3] for r in rows])
    trial_keys = np.asarray([r[4] for r in rows])
    np.savez_compressed(path, X=X, y=y, subjects=subjects, repetitions=reps, trial_keys=trial_keys)
    print(json.dumps({'cache': str(path), 'shape': X.shape, 'classes': len(set(y)), 'max_per_class': max_per_class}, indent=2), flush=True)


class SEBlock(nn.Module):
    def __init__(self, c: int, r: int = 8):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(c, max(4, c // r)), nn.ReLU(), nn.Linear(max(4, c // r), c), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1)

class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, d: int = 1):
        super().__init__()
        pad = (k // 2) * d
        self.net = nn.Sequential(nn.Conv1d(cin, cout, k, padding=pad, dilation=d), nn.BatchNorm1d(cout), nn.ReLU(), nn.Dropout(0.15), SEBlock(cout))
    def forward(self, x): return self.net(x)

class MultiStreamCNN(nn.Module):
    def __init__(self, cin: int, n_classes: int, width: int = 64):
        super().__init__()
        self.s3 = nn.Sequential(ConvBlock(cin, width, 3), ConvBlock(width, width, 3, 2))
        self.s5 = nn.Sequential(ConvBlock(cin, width, 5), ConvBlock(width, width, 5, 2))
        self.s9 = nn.Sequential(ConvBlock(cin, width, 9), ConvBlock(width, width, 9, 2))
        self.mix = nn.Sequential(nn.Conv1d(width * 3, width * 2, 1), nn.BatchNorm1d(width * 2), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.25), nn.Linear(width * 2, n_classes))
    def forward(self, x):
        return self.mix(torch.cat([self.s3(x), self.s5(x), self.s9(x)], dim=1))


def split(subjects, reps, protocol):
    if protocol == 'calibrated':
        return np.where(reps <= 7)[0], np.where(reps == 8)[0], np.where(reps >= 9)[0]
    order = sorted(set(subjects))
    val_subjects = set(order[-8:-5]); test_subjects = set(order[-5:])
    val = np.asarray([s in val_subjects for s in subjects]); test = np.asarray([s in test_subjects for s in subjects])
    return np.where(~(val | test))[0], np.where(val)[0], np.where(test)[0]


def metrics(y, proba):
    pred = proba.argmax(1)
    return {'accuracy': float(accuracy_score(y, pred)), 'balanced_accuracy': float(balanced_accuracy_score(y, pred)), 'macro_f1': float(f1_score(y, pred, average='macro')), 'weighted_f1': float(f1_score(y, pred, average='weighted')), 'top3_accuracy': float(top_k_accuracy_score(y, proba, k=3, labels=np.arange(proba.shape[1]))), 'top5_accuracy': float(top_k_accuracy_score(y, proba, k=5, labels=np.arange(proba.shape[1])))}

def predict(model, loader, device):
    model.eval(); ys=[]; probs=[]
    with torch.no_grad():
        for xb, yb in loader:
            probs.append(torch.softmax(model(xb.to(device)), 1).cpu().numpy()); ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(probs)

def trial_metrics(y, proba, keys):
    key_order = [] ; bucket = {}
    for i, k in enumerate(keys):
        if k not in bucket:
            bucket[k] = [] ; key_order.append(k)
        bucket[k].append(i)
    ty=[]; tp=[]
    for k in key_order:
        idx = bucket[k]
        ty.append(int(y[idx[0]]))
        tp.append(proba[idx].mean(0))
    return metrics(np.asarray(ty), np.vstack(tp)) | {'n_trials': len(ty)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', type=int, default=100)
    ap.add_argument('--step', type=int, default=25)
    ap.add_argument('--max-per-class', type=int, default=2200)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--protocol', choices=['calibrated','subject'], default='calibrated')
    ap.add_argument('--prepare', action='store_true')
    args = ap.parse_args()
    cache = OUT / f'emg_ninapro_db1_ms_window{args.window}_step{args.step}_mpc{args.max_per_class}.npz'
    if args.prepare or not cache.exists(): build_cache(cache, args.window, args.step, args.max_per_class)
    d = np.load(cache, allow_pickle=True)
    X = d['X'].astype(np.float32); raw_y=d['y']; subjects=d['subjects']; reps=d['repetitions']; keys=d['trial_keys']
    le=LabelEncoder(); y=le.fit_transform(raw_y)
    tr, va, te = split(subjects, reps, args.protocol)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=MultiStreamCNN(X.shape[1], len(le.classes_)).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn=nn.CrossEntropyLoss()
    train_loader=DataLoader(TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(y[tr]).long()), batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader=DataLoader(TensorDataset(torch.from_numpy(X[va]), torch.from_numpy(y[va]).long()), batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader=DataLoader(TensorDataset(torch.from_numpy(X[te]), torch.from_numpy(y[te]).long()), batch_size=args.batch_size, shuffle=False, num_workers=2)
    best=None; state=None; t0=time.time()
    for epoch in range(1,args.epochs+1):
        model.train(); total=0; n=0
        for xb,yb in train_loader:
            xb=xb.to(device); yb=yb.to(device); opt.zero_grad(set_to_none=True); loss=loss_fn(model(xb),yb); loss.backward(); opt.step(); total+=float(loss.item())*len(yb); n+=len(yb)
        vy, vp = predict(model, val_loader, device); vm = metrics(vy, vp)
        print(json.dumps({'epoch':epoch,'loss':total/max(1,n),'val':vm},indent=2), flush=True)
        if best is None or vm['macro_f1'] > best['macro_f1']:
            best=vm; state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    model.load_state_dict(state)
    ty,tp = predict(model, test_loader, device)
    report={'task':f'emg_ninapro_db1_52class_multistream_cnn_{args.protocol}','dataset':'NinaPro DB1 normalized raw windows','protocol':args.protocol,'window_samples':args.window,'step_samples':args.step,'n_total':int(len(y)),'n_train':int(len(tr)),'n_val':int(len(va)),'n_test':int(len(te)),'n_classes':int(len(le.classes_)),'best_val':best,'test_window':metrics(ty,tp),'test_trial':trial_metrics(ty,tp,keys[te]),'epochs':args.epochs,'elapsed_sec':round(time.time()-t0,2),'note':'Multi-stream CNN with channel attention and repetition/trial-level probability averaging.'}
    out=OUT/f'emg_ninapro_db1_52class_multistream_cnn_{args.protocol}_report.json'
    out.write_text(json.dumps(report,indent=2))
    torch.save({'state_dict':state,'labels':le.classes_.tolist(),'in_channels':int(X.shape[1]),'window_samples':args.window}, OUT/f'emg_ninapro_db1_52class_multistream_cnn_{args.protocol}.pt')
    print(json.dumps(report,indent=2), flush=True)
if __name__=='__main__': main()
