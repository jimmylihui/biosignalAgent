from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.pcg_tools import _load_pcg_signal_for_multisite, _resample_pcg_values, _center_crop_or_pad, _pcg_multisite_spec_image

LABELS = ['AS', 'AR', 'MR', 'MS', 'N']
OUT = Path('/data1/jiahui/biosignal-agent/outputs/pcg_bmdhs_valve_spectrogram_cnn.pt')
REPORT = Path('/data1/jiahui/biosignal-agent/outputs/pcg_bmdhs_valve_spectrogram_cnn_report.json')


class BMDHSSpecDataset(Dataset):
    def __init__(self, rows: list[dict], target_fs: int, seconds: float, freq_bins: int, time_bins: int, augment: bool = False):
        self.rows = rows
        self.target_fs = target_fs
        self.seconds = seconds
        self.freq_bins = freq_bins
        self.time_bins = time_bins
        self.augment = augment
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        fs, values = _load_pcg_signal_for_multisite(r['path'], r['sampling_rate'], None)
        values = _resample_pcg_values(values, fs, self.target_fs)
        length = int(round(self.target_fs * self.seconds))
        if self.augment and len(values) > length:
            start = np.random.randint(0, len(values) - length + 1)
            values = values[start:start + length]
        else:
            values = _center_crop_or_pad(values, length)
        if self.augment:
            values = values * np.float32(np.random.uniform(0.8, 1.2))
            if np.random.rand() < 0.5:
                values = values + np.random.normal(0, 0.01, size=len(values)).astype(np.float32)
        img = _pcg_multisite_spec_image(values, self.target_fs, self.freq_bins, self.time_bins)
        y = np.asarray([int(r[f'label_{lab.lower()}']) for lab in LABELS], dtype=np.float32)
        return torch.from_numpy(img[None, :, :].astype(np.float32)), torch.from_numpy(y)


class SpecCNN(nn.Module):
    def __init__(self, out_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 24, 5, padding=2, bias=False), nn.BatchNorm2d(24), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1, bias=False), nn.BatchNorm2d(48), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1, bias=False), nn.BatchNorm2d(96), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(96, 160, 3, padding=1, bias=False), nn.BatchNorm2d(160), nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(0.35), nn.Linear(160, out_dim),
        )
    def forward(self, x): return self.net(x)


def split_rows(rows: list[dict], val_fold: int) -> tuple[list[dict], list[dict]]:
    patients = sorted({r['patient_id'] for r in rows})
    val_patients = {p for i, p in enumerate(patients) if i % 5 == val_fold % 5}
    val = [r for r in rows if r['patient_id'] in val_patients]
    train = [r for r in rows if r['patient_id'] not in val_patients]
    return train, val


def choose_threshold(y, p):
    return float(max(((f1_score(y, p >= t, zero_division=0), t) for t in np.linspace(0.05, 0.95, 91)), key=lambda z: z[0])[1])


def metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    out = {}
    for i, lab in enumerate(LABELS):
        y = y_true[:, i].astype(int); p = probs[:, i]
        thr = choose_threshold(y, p)
        pred = (p >= thr).astype(int)
        out[lab] = {
            'threshold': thr,
            'accuracy': float(accuracy_score(y, pred)),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'f1': float(f1_score(y, pred, zero_division=0)),
            'average_precision': float(average_precision_score(y, p)) if int(y.sum()) else 0.0,
            'roc_auc': float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.0,
            'positive_count': int(y.sum()),
        }
    out['macro_f1'] = float(np.mean([out[lab]['f1'] for lab in LABELS]))
    out['macro_auroc'] = float(np.mean([out[lab]['roc_auc'] for lab in LABELS]))
    return out


def evaluate(model, loader, device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for xb, yb in loader:
            prob = torch.sigmoid(model(xb.to(device))).cpu().numpy()
            ps.append(prob); ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    ap=argparse.ArgumentParser(description='Train BMD-HS PCG valve multi-label spectrogram CNN.')
    ap.add_argument('--manifest',type=Path,default=Path('/data1/jiahui/biosignal-agent/datasets/processed/pcg_bmdhs_valve_manifest.json'))
    ap.add_argument('--model-path',type=Path,default=OUT)
    ap.add_argument('--report-path',type=Path,default=REPORT)
    ap.add_argument('--epochs',type=int,default=12)
    ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--val-fold',type=int,default=0)
    ap.add_argument('--target-fs',type=int,default=1000)
    ap.add_argument('--seconds',type=float,default=12.0)
    args=ap.parse_args()
    rows=json.load(open(args.manifest))['rows']
    train_rows,val_rows=split_rows(rows,args.val_fold)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    train_ds=BMDHSSpecDataset(train_rows,args.target_fs,args.seconds,80,128,augment=True)
    val_ds=BMDHSSpecDataset(val_rows,args.target_fs,args.seconds,80,128,augment=False)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=2)
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=2)
    y_train=np.asarray([[r[f'label_{lab.lower()}'] for lab in LABELS] for r in train_rows],dtype=float)
    pos_weight=(len(y_train)-y_train.sum(axis=0))/np.maximum(1,y_train.sum(axis=0))
    model=SpecCNN(len(LABELS)).to(device)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight,dtype=torch.float32,device=device))
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
    best=None; best_score=-1; history=[]
    for ep in range(1,args.epochs+1):
        model.train(); losses=[]
        for xb,yb in train_loader:
            xb=xb.to(device); yb=yb.to(device)
            opt.zero_grad(set_to_none=True); loss=loss_fn(model(xb),yb); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        yv,pv=evaluate(model,val_loader,device)
        m=metrics(yv,pv)
        row={'epoch':ep,'train_loss':float(np.mean(losses)),'val_macro_f1':m['macro_f1'],'val_macro_auroc':m['macro_auroc']}
        history.append(row); print(json.dumps(row),flush=True)
        score=m['macro_f1']
        if score>best_score:
            best_score=score; best={k:v.detach().cpu() for k,v in model.state_dict().items()}; best_metrics=m
    if best is not None:
        model.load_state_dict(best)
    bundle={'model_state_dict':model.cpu().state_dict(),'model':'BMDHS_SpecCNN','labels':LABELS,'target_fs':args.target_fs,'seconds':args.seconds,'freq_bins':80,'time_bins':128,'val_metrics':best_metrics,'history':history,'reference':'BMD-HS recording-level spectrogram CNN, patient-heldout fold'}
    args.model_path.parent.mkdir(parents=True,exist_ok=True); torch.save(bundle,args.model_path)
    report={'model_path':str(args.model_path),'num_train_records':len(train_rows),'num_val_records':len(val_rows),'val_fold':args.val_fold,'best_metrics':best_metrics,'history':history}
    args.report_path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
