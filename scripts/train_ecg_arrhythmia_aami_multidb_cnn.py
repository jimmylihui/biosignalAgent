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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, classification_report
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

MITDB_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]
AAMI_CLASSES = ['N', 'S', 'V', 'F', 'Q']
LABEL_TO_IDX = {name: idx for idx, name in enumerate(AAMI_CLASSES)}
NORMAL = {'N', 'L', 'R', 'e', 'j'}
SUPRA = {'A', 'a', 'J', 'S'}
VENT = {'V', 'E'}
FUSION = {'F'}
UNKNOWN = {'/', 'f', 'Q'}
BEAT_SYMBOLS = NORMAL | SUPRA | VENT | FUSION | UNKNOWN
OUT = Path('/data1/jiahui/biosignal-agent/outputs')
MODEL_PATH = OUT / 'ecg_arrhythmia_aami_multidb_cnn_model.pt'


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def symbol_to_aami(symbol: str) -> str | None:
    if symbol in NORMAL: return 'N'
    if symbol in SUPRA: return 'S'
    if symbol in VENT: return 'V'
    if symbol in FUSION: return 'F'
    if symbol in UNKNOWN: return 'Q'
    return None


def robust_segment(values: np.ndarray, center: int, pre: int, post: int, beat_len: int) -> np.ndarray:
    start = center - pre; stop = center + post
    seg = np.zeros(pre + post, dtype=np.float32)
    src_start = max(start, 0); src_stop = min(stop, len(values))
    dst_start = src_start - start; dst_stop = dst_start + max(0, src_stop - src_start)
    if src_stop > src_start:
        seg[dst_start:dst_stop] = values[src_start:src_stop]
    med = float(np.median(seg)); q75, q25 = np.percentile(seg, [75, 25]); scale = float(q75 - q25)
    if scale < 1e-8: scale = float(np.std(seg)) + 1e-8
    seg = np.clip((seg - med) / scale, -8.0, 8.0)
    if len(seg) != beat_len:
        seg = scipy_signal.resample(seg, beat_len)
    return seg.astype(np.float32)


def complete(raw: Path, record: str) -> bool:
    return all((raw / f'{record}.{ext}').exists() for ext in ['hea', 'dat', 'atr'])


def load_beats(specs: list[tuple[str, Path, str, list[str]]], beat_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    X, F, y, groups = [], [], [], []
    for db_name, raw_dir, _slug, records in specs:
        for record in records:
            if not complete(raw_dir, record):
                continue
            try:
                header = wfdb.rdheader(str(raw_dir / record))
                rec = wfdb.rdrecord(str(raw_dir / record))
                lead = 'MLII' if 'MLII' in rec.sig_name else rec.sig_name[0]
                values = rec.p_signal[:, rec.sig_name.index(lead)].astype(np.float32)
                ann = wfdb.rdann(str(raw_dir / record), 'atr')
            except Exception as exc:
                print(f'skip {db_name}/{record}: {type(exc).__name__}: {exc}', flush=True)
                continue
            positions, labels = [], []
            for sample, symbol in zip(ann.sample, ann.symbol):
                cls = symbol_to_aami(symbol)
                if cls is None:
                    continue
                positions.append(int(sample)); labels.append(LABEL_TO_IDX[cls])
            if len(positions) < 3:
                continue
            positions = np.asarray(positions, dtype=int)
            rr_prev = np.r_[np.nan, np.diff(positions) / float(header.fs)]
            rr_next = np.r_[np.diff(positions) / float(header.fs), np.nan]
            med_rr = np.nanmedian(np.r_[rr_prev, rr_next])
            rr_prev = np.nan_to_num(rr_prev, nan=med_rr, posinf=med_rr, neginf=med_rr)
            rr_next = np.nan_to_num(rr_next, nan=med_rr, posinf=med_rr, neginf=med_rr)
            pre = int(0.25 * header.fs); post = int(0.45 * header.fs)
            for i, (sample, label) in enumerate(zip(positions, labels)):
                X.append(robust_segment(values, int(sample), pre, post, beat_len))
                local_hr = 60.0 / max(float((rr_prev[i] + rr_next[i]) / 2.0), 0.25)
                F.append([float(np.clip(rr_prev[i], 0.25, 3.0)), float(np.clip(rr_next[i], 0.25, 3.0)), float(np.clip(local_hr / 100.0, 0.2, 2.5))])
                y.append(int(label)); groups.append(f'{db_name}:{record}')
            print(f'loaded {db_name}/{record}: {len(positions)} beats', flush=True)
    return np.asarray(X, np.float32), np.asarray(F, np.float32), np.asarray(y, np.int64), groups


class AAMIBeatCNN(nn.Module):
    def __init__(self, feature_dim: int = 3, num_classes: int = 5, dropout: float = 0.30):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 9, padding=4, bias=False), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 7, padding=3, bias=False), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2, bias=False), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 192, 3, padding=1, bias=False), nn.BatchNorm1d(192), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(192 + feature_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2: x = x[:, None, :]
        return self.head(torch.cat([self.cnn(x), f], dim=1))


def make_loader(X, F, y, idx, batch_size, balanced):
    ds = TensorDataset(torch.tensor(X[idx], dtype=torch.float32), torch.tensor(F[idx], dtype=torch.float32), torch.tensor(y[idx], dtype=torch.long))
    if not balanced:
        return DataLoader(ds, batch_size=batch_size, shuffle=True)
    counts = Counter(map(int, y[idx]))
    weights = np.asarray([1.0 / counts[int(label)] for label in y[idx]], dtype=np.float32)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler)


def train_fold(X, F, y, tr, va, epochs, seed, device, balanced=False, class_weight_power=0.5):
    seed_all(seed)
    model = AAMIBeatCNN(feature_dim=F.shape[1]).to(device)
    counts = Counter(map(int, y[tr]))
    class_weights = torch.tensor([(len(tr) / max(counts.get(i, 1), 1)) ** class_weight_power for i in range(len(AAMI_CLASSES))], dtype=torch.float32, device=device)
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loader = make_loader(X, F, y, tr, 512, balanced)
    best_state = None; best_loss = math.inf; stale = 0
    for epoch in range(epochs):
        model.train()
        for xb, fb, yb in loader:
            xb=xb.to(device); fb=fb.to(device); yb=yb.to(device)
            opt.zero_grad(set_to_none=True); loss=criterion(model(xb, fb), yb); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 4.0); opt.step()
        if va is None: continue
        model.eval()
        with torch.no_grad():
            vv=torch.tensor(X[va], dtype=torch.float32, device=device); vf=torch.tensor(F[va], dtype=torch.float32, device=device); vy=torch.tensor(y[va], dtype=torch.long, device=device)
            val_loss=float(criterion(model(vv, vf), vy).cpu())
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss; best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale = 0
        else:
            stale += 1
        if stale >= 7: break
    if best_state is not None: model.load_state_dict(best_state)
    return model, best_loss


def predict(model, X, F, device):
    model.eval(); out=[]
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(F, dtype=torch.float32))
    with torch.no_grad():
        for xb, fb in DataLoader(ds, batch_size=1024):
            out.append(torch.softmax(model(xb.to(device), fb.to(device)), dim=1).cpu().numpy())
    return np.concatenate(out).astype(float)


def metrics(y, proba):
    pred = np.argmax(proba, axis=1)
    binary_y = (y != LABEL_TO_IDX['N']).astype(int)
    binary_score = 1.0 - proba[:, LABEL_TO_IDX['N']]
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t = float(max(((f1_score(binary_y, binary_score >= t, zero_division=0), t) for t in thresholds), key=lambda z:z[0])[1])
    binary_pred = (binary_score >= best_t).astype(int)
    out = {
        'accuracy': float(accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y, pred, average='weighted', zero_division=0)),
        'binary_threshold': best_t,
        'binary_accuracy': float(accuracy_score(binary_y, binary_pred)),
        'binary_precision': float(precision_score(binary_y, binary_pred, zero_division=0)),
        'binary_recall': float(recall_score(binary_y, binary_pred, zero_division=0)),
        'binary_f1': float(f1_score(binary_y, binary_pred, zero_division=0)),
        'binary_average_precision': float(average_precision_score(binary_y, binary_score)),
        'class_report': classification_report(y, pred, target_names=AAMI_CLASSES, zero_division=0, output_dict=True),
    }
    try: out['binary_roc_auc'] = float(roc_auc_score(binary_y, binary_score))
    except Exception: out['binary_roc_auc'] = 0.0
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mitdb-raw-dir', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/raw/mitdb'))
    ap.add_argument('--incart-raw-dir', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/raw/incartdb'))
    ap.add_argument('--svdb-raw-dir', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/raw/svdb'))
    ap.add_argument('--include', nargs='*', default=['mitdb','incartdb','svdb'])
    ap.add_argument('--beat-len', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--balanced-sampler', action='store_true')
    ap.add_argument('--class-weight-power', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=53)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--model-path', type=Path, default=MODEL_PATH)
    args=ap.parse_args(); seed_all(args.seed)
    specs=[]
    if 'mitdb' in args.include: specs.append(('mitdb', args.mitdb_raw_dir, 'mitdb', MITDB_RECORDS))
    if 'incartdb' in args.include: specs.append(('incartdb', args.incart_raw_dir, 'incartdb', wfdb.get_record_list('incartdb')))
    if 'svdb' in args.include: specs.append(('svdb', args.svdb_raw_dir, 'svdb', wfdb.get_record_list('svdb')))
    X,F,y,groups=load_beats(specs,args.beat_len)
    print(json.dumps({'num_beats':int(len(y)),'label_counts':{AAMI_CLASSES[k]:v for k,v in Counter(map(int,y)).items()},'records':len(set(groups)),'include':args.include, 'balanced_sampler': args.balanced_sampler, 'class_weight_power': args.class_weight_power}, indent=2), flush=True)
    proba=np.zeros((len(y), len(AAMI_CLASSES)), dtype=float); folds=[]
    for fold,(tr,va) in enumerate(GroupKFold(n_splits=5).split(X,y,groups=groups)):
        model,vl=train_fold(X,F,y,tr,va,args.epochs,args.seed+fold,args.device, balanced=args.balanced_sampler, class_weight_power=args.class_weight_power)
        proba[va]=predict(model,X[va],F[va],args.device)
        folds.append({'fold':fold,'train_size':int(len(tr)),'val_size':int(len(va)),'val_loss':float(vl),'val_label_counts':{AAMI_CLASSES[k]:v for k,v in Counter(map(int,y[va])).items()}})
        print('fold', fold, folds[-1], flush=True)
    cv_metrics=metrics(y, proba)
    allidx=np.arange(len(y)); final,_=train_fold(X,F,y,allidx,None,max(args.epochs,14),args.seed+999,args.device, balanced=args.balanced_sampler, class_weight_power=args.class_weight_power)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict':final.cpu().state_dict(),'architecture':'AAMIBeatCNN','beat_len':args.beat_len,'feature_dim':int(F.shape[1]),'classes':AAMI_CLASSES,'binary_threshold':cv_metrics['binary_threshold'],'cv_metrics':cv_metrics,'fold_reports':folds,'label_counts':{AAMI_CLASSES[k]:v for k,v in Counter(map(int,y)).items()}}, args.model_path)
    report={'model_path':str(args.model_path),'num_beats':int(len(y)),'label_counts':{AAMI_CLASSES[k]:v for k,v in Counter(map(int,y)).items()},'cv_metrics':cv_metrics,'fold_reports':folds}
    out=OUT/'ecg_arrhythmia_aami_multidb_cnn_train_report.json'; out.write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))

if __name__=='__main__': main()
