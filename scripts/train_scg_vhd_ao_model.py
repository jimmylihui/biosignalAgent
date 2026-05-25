from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import signal as scipy_signal
from scipy.io import loadmat
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_scg_vhd_zenodo_benchmark import (  # noqa: E402
    MAT,
    SUMMARY,
    load_refs,
    load_vectors,
    match_peaks,
    hr_from_peaks,
)

OUT_DIR = Path('/data1/jiahui/biosignal-agent/outputs')


def set_seed(seed: int) -> None:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def safe_bandpass(x: np.ndarray, fs: float, low: float = 0.8, high: float = 35.0) -> np.ndarray:
    high = min(high, 0.45 * fs)
    x = np.asarray(x, dtype=np.float32)
    x = x - np.nanmedian(x)
    if len(x) < fs * 3 or high <= low:
        y = x
    else:
        sos = scipy_signal.butter(3, [low / (0.5 * fs), high / (0.5 * fs)], btype='bandpass', output='sos')
        y = scipy_signal.sosfiltfilt(sos, x).astype(np.float32)
    scale = np.nanpercentile(np.abs(y), 95) + 1e-6
    return np.clip(y / scale, -8, 8).astype(np.float32)


def make_label(n: int, center: int, sigma: float) -> np.ndarray:
    idx = np.arange(n, dtype=np.float32)
    return np.exp(-0.5 * ((idx - float(center)) / sigma) ** 2).astype(np.float32)


def pair_r_to_ao(r: np.ndarray, ao: np.ndarray, fs: float, start_s: float, end_s: float) -> list[tuple[int, int]]:
    pairs = []
    for rr in np.asarray(r, dtype=int):
        lo = rr + int(round(start_s * fs)); hi = rr + int(round(end_s * fs))
        cand = ao[(ao >= lo) & (ao <= hi)]
        if len(cand):
            pairs.append((int(rr), int(cand[0])))
    return pairs


def load_subjects() -> list[dict[str, Any]]:
    meta = pd.read_excel(SUMMARY)
    meta = meta[meta['Patient ID'].astype(str).str.startswith(('CP-', 'UP-'))]
    subjects = []
    for _, row in meta.iterrows():
        pid = str(row['Patient ID'])
        if not (MAT / f'{pid}-Vectors.mat').exists():
            continue
        fs = float(row['Sampling rate(Hz)'])
        try:
            vec = load_vectors(pid)
            refs = load_refs(pid, fs, len(vec['scg_z']))
            pairs = pair_r_to_ao(refs['r_lara'], refs['ao_z'], fs, 0.04, 0.32)
            if len(pairs) < 20:
                continue
            subjects.append({'pid': pid, 'fs': fs, 'scg_z': vec['scg_z'], 'r': refs['r_lara'], 'ao': refs['ao_z'], 'pairs': pairs})
        except Exception:
            continue
    return subjects


class BeatDataset(Dataset):
    def __init__(self, subjects: list[dict[str, Any]], target_fs: float, pre_s: float, post_s: float, train: bool, seed: int, mode: str):
        self.items = []
        self.target_fs = target_fs
        self.pre_s = pre_s
        self.post_s = post_s
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.train = train
        if mode == 'ecg_anchor':
            self.n = int(round((pre_s + post_s) * target_fs))
        else:
            self.n = int(round(post_s * target_fs))
        for subj in subjects:
            fs = subj['fs']
            x = safe_bandpass(subj['scg_z'], fs)
            if fs != target_fs:
                n_res = int(round(len(x) * target_fs / fs))
                x = scipy_signal.resample(x, n_res).astype(np.float32)
                ao = np.asarray(np.round(subj['ao'] * target_fs / fs), dtype=int)
                pairs = [(int(round(r * target_fs / fs)), int(round(a * target_fs / fs))) for r, a in subj['pairs']]
            else:
                ao = np.asarray(subj['ao'], dtype=int)
                pairs = subj['pairs']
            if mode == 'ecg_anchor':
                pre = int(round(pre_s * target_fs))
                for r, a in pairs:
                    start = r - pre
                    end = start + self.n
                    center = a - start
                    if start < 0 or end > len(x) or center < 0 or center >= self.n:
                        continue
                    self.items.append((subj['pid'], x[start:end].astype(np.float32), int(center), int(a)))
            else:
                step = max(1, self.n // 2)
                for start in range(0, max(1, len(x) - self.n + 1), step):
                    end = start + self.n
                    peaks = ao[(ao >= start) & (ao < end)]
                    if len(peaks) == 0 and train and self.rng.random() < 0.75:
                        continue
                    if len(peaks) == 0:
                        centers = []
                    else:
                        centers = [int(p - start) for p in peaks]
                    self.items.append((subj['pid'], x[start:end].astype(np.float32), centers, -1))
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        pid, x, centers, abs_a = self.items[i]
        if self.train:
            x = x * float(self.rng.uniform(0.85, 1.15)) + self.rng.normal(0, 0.03, len(x)).astype(np.float32)
        y = np.zeros(len(x), dtype=np.float32)
        if isinstance(centers, list):
            for c in centers:
                y = np.maximum(y, make_label(len(x), c, sigma=4.0))
        else:
            y = make_label(len(x), int(centers), sigma=4.0)
        return torch.from_numpy(x[None, :]), torch.from_numpy(y[None, :]), 0, pid


class BeatAOModel(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, base, 9, padding=4), nn.BatchNorm1d(base), nn.SiLU(),
            nn.Conv1d(base, base, 9, padding=4), nn.BatchNorm1d(base), nn.SiLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(base, base*2, 7, padding=3), nn.BatchNorm1d(base*2), nn.SiLU(),
            nn.Conv1d(base*2, base*2, 7, padding=3), nn.BatchNorm1d(base*2), nn.SiLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(base*2, base*4, 5, padding=2), nn.BatchNorm1d(base*4), nn.SiLU(),
            nn.Conv1d(base*4, base*4, 5, padding=2), nn.BatchNorm1d(base*4), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(base*4, base*2, 5, padding=2), nn.BatchNorm1d(base*2), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(base*2, base, 5, padding=2), nn.BatchNorm1d(base), nn.SiLU(),
            nn.Conv1d(base, 1, 1),
        )
    def forward(self, x):
        y = self.net(x)
        return y[..., :x.shape[-1]]


def predict_subject(model: BeatAOModel, subj: dict[str, Any], args: argparse.Namespace, dev: torch.device) -> np.ndarray:
    fs = subj['fs']; target_fs = args.target_fs
    x = safe_bandpass(subj['scg_z'], fs)
    if fs != target_fs:
        n_res = int(round(len(x) * target_fs / fs))
        x = scipy_signal.resample(x, n_res).astype(np.float32)
        r_peaks = np.asarray(np.round(subj['r'] * target_fs / fs), dtype=int)
    else:
        r_peaks = np.asarray(subj['r'], dtype=int)
    preds = []
    model.eval()
    with torch.no_grad():
        if args.mode == 'ecg_anchor':
            pre = int(round(args.pre_s * target_fs)); n = int(round((args.pre_s + args.post_s) * target_fs))
            for r in r_peaks:
                start = int(r) - pre; end = start + n
                if start < 0 or end > len(x):
                    continue
                seg = torch.from_numpy(x[start:end][None, None, :].astype(np.float32)).to(dev)
                prob = torch.sigmoid(model(seg)).cpu().numpy()[0, 0]
                lo = int(round((args.pre_s + 0.04) * target_fs)); hi = int(round((args.pre_s + 0.32) * target_fs))
                local = lo + int(np.argmax(prob[lo:hi]))
                pred = start + local
                if fs != target_fs:
                    pred = int(round(pred * fs / target_fs))
                preds.append(pred)
        else:
            n = int(round(args.post_s * target_fs))
            step = max(1, n // 2)
            prob_sum = np.zeros(len(x), dtype=np.float32)
            weight = np.zeros(len(x), dtype=np.float32)
            for start in range(0, len(x), step):
                seg = x[start:start+n]
                if len(seg) < n:
                    pad = np.zeros(n, dtype=np.float32); pad[:len(seg)] = seg; seg = pad
                pr = torch.sigmoid(model(torch.from_numpy(seg[None, None, :]).to(dev))).cpu().numpy()[0, 0]
                end = min(len(x), start+n)
                prob_sum[start:end] += pr[:end-start]
                weight[start:end] += 1.0
            prob = prob_sum / np.maximum(weight, 1.0)
            peaks, _ = scipy_signal.find_peaks(prob, height=0.35, distance=max(1, int(round(0.30 * target_fs))))
            if fs != target_fs:
                peaks = np.asarray(np.round(peaks * fs / target_fs), dtype=int)
            preds.extend(peaks.tolist())
    return np.asarray(preds, dtype=int)


def aggregate_subject_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    total = {'tp': 0, 'fp': 0, 'fn': 0}; maes=[]
    for row in rows:
        m=row[key]; total['tp']+=m['tp']; total['fp']+=m['fp']; total['fn']+=m['fn']
        if m.get('timing_mae_ms') is not None: maes.append(m['timing_mae_ms'])
    sens=total['tp']/(total['tp']+total['fn']) if total['tp']+total['fn'] else 0.0
    ppv=total['tp']/(total['tp']+total['fp']) if total['tp']+total['fp'] else 0.0
    f1=2*sens*ppv/(sens+ppv) if sens+ppv else 0.0
    return {**total,'sensitivity':sens,'ppv':ppv,'f1':f1,'timing_mae_ms':float(np.mean(maes)) if maes else None}


def eval_model(model, subjects, args, dev):
    rows=[]
    for subj in subjects:
        pred = predict_subject(model, subj, args, dev)
        ref = np.asarray(subj['ao'], dtype=int)
        ref_hr=hr_from_peaks(ref, subj['fs']); pred_hr=hr_from_peaks(pred, subj['fs'])
        rows.append({
            'pid': subj['pid'], 'ref_count': int(len(ref)), 'pred_count': int(len(pred)),
            'm100': match_peaks(ref, pred, subj['fs'], 0.10),
            'm50': match_peaks(ref, pred, subj['fs'], 0.05),
            'hr_abs_error_bpm': abs(pred_hr-ref_hr) if pred_hr is not None and ref_hr is not None else None,
        })
    hr=[r['hr_abs_error_bpm'] for r in rows if r['hr_abs_error_bpm'] is not None]
    return {'ao_100ms': aggregate_subject_metrics(rows,'m100'), 'ao_50ms': aggregate_subject_metrics(rows,'m50'), 'hr_mae_bpm': float(np.mean(hr)) if hr else None, 'per_subject': rows}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=113)
    ap.add_argument('--target-fs', type=float, default=256.0)
    ap.add_argument('--pre-s', type=float, default=0.05)
    ap.add_argument('--post-s', type=float, default=0.35)
    ap.add_argument('--mode', choices=['ecg_anchor', 'scg_free'], default='ecg_anchor')
    ap.add_argument('--out-model', default=None)
    ap.add_argument('--report', default=None)
    ap.add_argument('--cpu', action='store_true')
    args=ap.parse_args(); set_seed(args.seed)
    if args.out_model is None:
        args.out_model = str(OUT_DIR / f'scg_vhd_ao_{args.mode}_cnn.pt')
    if args.report is None:
        args.report = str(OUT_DIR / f'scg_vhd_ao_{args.mode}_cnn_report.json')
    subjects=load_subjects()
    # Deterministic subject split: CP-01..CP-50 train, CP-51..CP-70 val, UP cohort external test.
    train=[s for s in subjects if s['pid'].startswith('CP-') and int(s['pid'].split('-')[1]) <= 50]
    val=[s for s in subjects if s['pid'].startswith('CP-') and int(s['pid'].split('-')[1]) > 50]
    test=[s for s in subjects if s['pid'].startswith('UP-')]
    dev=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    trds=BeatDataset(train,args.target_fs,args.pre_s,args.post_s,True,args.seed,args.mode)
    vads=BeatDataset(val,args.target_fs,args.pre_s,args.post_s,False,args.seed+1,args.mode)
    trl=DataLoader(trds,batch_size=args.batch_size,shuffle=True,num_workers=0)
    model=BeatAOModel().to(dev)
    opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0], device=dev))
    best=None
    for ep in range(1,args.epochs+1):
        model.train(); loss_sum=0.0
        for x,y,center,pid in trl:
            x=x.to(dev); y=y.to(dev); opt.zero_grad(set_to_none=True); loss=loss_fn(model(x),y); loss.backward(); opt.step(); loss_sum += float(loss.item())*len(x)
        val_metrics=eval_model(model,val,args,dev)
        print(json.dumps({'epoch':ep,'loss':loss_sum/max(1,len(trds)),'val_100ms':{k:v for k,v in val_metrics['ao_100ms'].items() if k!='per_subject'},'val_hr_mae':val_metrics['hr_mae_bpm']}), flush=True)
        score=(val_metrics['ao_100ms']['f1'], -val_metrics['ao_100ms']['timing_mae_ms'])
        if best is None or score > best['score']:
            best={'epoch':ep,'score':score,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'val':val_metrics}
    model.load_state_dict(best['state_dict'])
    test_metrics=eval_model(model,test,args,dev)
    payload={'model_state_dict':best['state_dict'],'architecture':f'BeatAOModel_{args.mode}_CNN','target_fs':args.target_fs,'pre_s':args.pre_s,'post_s':args.post_s,'best_epoch':best['epoch'],'train_subjects':[s['pid'] for s in train],'val_subjects':[s['pid'] for s in val],'test_subjects':[s['pid'] for s in test]}
    torch.save(payload,args.out_model)
    report={'dataset':f'Zenodo 5279448 VHD; SCG_Z AO model mode={args.mode}','train_subjects':[s['pid'] for s in train],'val_subjects':[s['pid'] for s in val],'test_subjects':[s['pid'] for s in test],'n_train_beats':len(trds),'n_val_beats':len(vads),'best_epoch':best['epoch'],'val':best['val'],'test':test_metrics,'model_out':args.out_model}
    Path(args.report).write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['best_epoch','n_train_beats','n_val_beats','test']}, indent=2)[:4000])

if __name__=='__main__':
    main()
