
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wfdb
from scipy import signal as scipy_signal
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.utils.data import Dataset, DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biosignal_agent.tools.scg_tools import SCG_detect_j_peaks


def set_seed(seed:int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def complete_records(raw_dir: Path):
    out=[]
    for hea in sorted(raw_dir.glob('b*.hea')):
        root=hea.with_suffix('')
        if root.with_suffix('.dat').exists() and root.with_suffix('.atr').exists():
            out.append(root.name)
    return out


def derive_scg_j_peaks(x: np.ndarray, r_samples: np.ndarray, fs: float, start_s: float = 0.08, end_s: float = 0.28) -> np.ndarray:
    # ECG R-peaks precede the mechanical SCG J/AO complex. Use each R as an anchor and
    # choose the strongest local absolute SCG deflection in a physiological delay window.
    peaks=[]; a=int(round(start_s*fs)); b=int(round(end_s*fs))
    env=np.abs(x)
    for r in r_samples:
        lo=int(r)+a; hi=min(len(x), int(r)+b)
        if hi <= lo: continue
        peaks.append(lo + int(np.argmax(env[lo:hi])))
    if not peaks: return np.asarray([], dtype=int)
    out=[]
    min_dist=int(round(0.3*fs))
    for p in peaks:
        if not out or p-out[-1] >= min_dist:
            out.append(p)
        elif env[p] > env[out[-1]]:
            out[-1]=p
    return np.asarray(out, dtype=int)


def load_record(raw_dir: Path, rec: str, target_fs: float):
    record=wfdb.rdrecord(str(raw_dir/rec))
    ann=wfdb.rdann(str(raw_dir/rec), 'atr')
    fs=float(record.fs)
    names=list(record.sig_name)
    scg_idx=names.index('SCG') if 'SCG' in names else len(names)-1
    x=np.asarray(record.p_signal[:, scg_idx], dtype=np.float32)
    x=x[np.isfinite(x)]
    x=x-np.nanmedian(x)
    # SCG heartbeat energy is usually useful from roughly 0.8-35 Hz.
    high=min(35.0, 0.45*fs)
    sos=scipy_signal.butter(3, [0.8/(0.5*fs), high/(0.5*fs)], btype='bandpass', output='sos')
    x=scipy_signal.sosfiltfilt(sos, x).astype(np.float32)
    if fs != target_fs:
        n=max(16, int(round(len(x)*target_fs/fs)))
        x=scipy_signal.resample(x, n).astype(np.float32)
        samples=np.asarray(np.round(np.asarray(ann.sample)*target_fs/fs), dtype=int)
    else:
        samples=np.asarray(ann.sample, dtype=int)
    scale=np.nanpercentile(np.abs(x), 95)+1e-6
    x=np.clip(x/scale, -6, 6).astype(np.float32)
    samples=samples[(samples>=0)&(samples<len(x))]
    j_samples=derive_scg_j_peaks(x, samples, target_fs)
    return x, j_samples


def make_label(n:int, peaks:np.ndarray, fs:float, radius_s:float):
    y=np.zeros(n, dtype=np.float32)
    radius=max(1, int(round(radius_s*fs)))
    sigma=max(1.0, radius/2.5)
    for p in peaks:
        lo=max(0, p-radius); hi=min(n, p+radius+1)
        idx=np.arange(lo, hi)
        y[lo:hi]=np.maximum(y[lo:hi], np.exp(-0.5*((idx-p)/sigma)**2))
    return y


class SegmentDataset(Dataset):
    def __init__(self, raw_dir:Path, records:list[str], args, train:bool, seed:int):
        self.items=[]; self.args=args; self.train=train; self.rng=np.random.default_rng(seed); self.seg_len=int(args.seconds*args.target_fs)
        for rec in records:
            x, peaks=load_record(raw_dir, rec, args.target_fs)
            y=make_label(len(x), peaks, args.target_fs, args.label_radius_s)
            step=self.seg_len//2 if train else self.seg_len
            for start in range(0, max(1, len(x)-self.seg_len+1), step):
                end=start+self.seg_len
                if end <= len(x):
                    # Keep all validation segments; for training, skip rare empty tail windows.
                    if train and y[start:end].max() <= 0 and self.rng.random() < 0.8:
                        continue
                    self.items.append((rec, start, x[start:end].astype(np.float32), y[start:end].astype(np.float32)))
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        rec,start,x,y=self.items[i]
        if self.train:
            x=x*float(self.rng.uniform(0.85,1.15)) + self.rng.normal(0,0.025,size=x.shape).astype(np.float32)
            if self.rng.random()<0.2:
                x=np.roll(x, int(self.rng.integers(-int(0.15*self.args.target_fs), int(0.15*self.args.target_fs)+1)))
        return torch.from_numpy(x[None,:]), torch.from_numpy(y[None,:]), rec, start


class ConvBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__(); self.net=nn.Sequential(nn.Conv1d(ci,co,7,padding=3),nn.BatchNorm1d(co),nn.SiLU(),nn.Conv1d(co,co,5,padding=2),nn.BatchNorm1d(co),nn.SiLU())
    def forward(self,x): return self.net(x)


class UNet1D(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        self.e1=ConvBlock(1,base); self.e2=ConvBlock(base,base*2); self.e3=ConvBlock(base*2,base*4); self.e4=ConvBlock(base*4,base*8)
        self.pool=nn.MaxPool1d(2)
        self.b=ConvBlock(base*8,base*8)
        self.u4=nn.ConvTranspose1d(base*8,base*8,2,2); self.d4=ConvBlock(base*16,base*4)
        self.u3=nn.ConvTranspose1d(base*4,base*4,2,2); self.d3=ConvBlock(base*8,base*2)
        self.u2=nn.ConvTranspose1d(base*2,base*2,2,2); self.d2=ConvBlock(base*4,base)
        self.u1=nn.ConvTranspose1d(base,base,2,2); self.d1=ConvBlock(base*2,base)
        self.out=nn.Conv1d(base,1,1)
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2)); e4=self.e4(self.pool(e3)); b=self.b(self.pool(e4))
        x=self.u4(b); x=torch.cat([x[..., :e4.shape[-1]], e4],1); x=self.d4(x)
        x=self.u3(x); x=torch.cat([x[..., :e3.shape[-1]], e3],1); x=self.d3(x)
        x=self.u2(x); x=torch.cat([x[..., :e2.shape[-1]], e2],1); x=self.d2(x)
        x=self.u1(x); x=torch.cat([x[..., :e1.shape[-1]], e1],1); x=self.d1(x)
        return self.out(x)


def probs_to_peaks(prob, fs, threshold=0.35, distance_s=0.3):
    prob=np.asarray(prob, dtype=float)
    peaks,_=scipy_signal.find_peaks(prob, height=threshold, distance=max(1,int(distance_s*fs)))
    return peaks.astype(int)


def estimate_hr_from_peaks(peaks: np.ndarray, fs: float, n_samples: int) -> dict[str, float | None]:
    peaks=np.asarray(peaks, dtype=int)
    duration_s=float(n_samples)/float(fs) if fs else 0.0
    count_hr=float(len(peaks)*60.0/duration_s) if duration_s>0 else None
    if len(peaks) >= 3:
        rr=np.diff(peaks)/float(fs)
        rr=rr[(rr>=0.3)&(rr<=2.0)]
        interval_hr=float(60.0/np.median(rr)) if len(rr) else None
    else:
        interval_hr=None
    return {'count_hr_bpm':count_hr, 'interval_hr_bpm':interval_hr}


def add_hr_metrics(metrics: dict[str, Any], ref: np.ndarray, pred: np.ndarray, fs: float, n_samples: int) -> dict[str, Any]:
    ref_hr=estimate_hr_from_peaks(ref, fs, n_samples)
    pred_hr=estimate_hr_from_peaks(pred, fs, n_samples)
    metrics['ref_count_hr_bpm']=ref_hr['count_hr_bpm']
    metrics['ref_interval_hr_bpm']=ref_hr['interval_hr_bpm']
    metrics['pred_count_hr_bpm']=pred_hr['count_hr_bpm']
    metrics['pred_interval_hr_bpm']=pred_hr['interval_hr_bpm']
    metrics['count_hr_abs_error_bpm']=abs(pred_hr['count_hr_bpm']-ref_hr['count_hr_bpm']) if pred_hr['count_hr_bpm'] is not None and ref_hr['count_hr_bpm'] is not None else None
    metrics['interval_hr_abs_error_bpm']=abs(pred_hr['interval_hr_bpm']-ref_hr['interval_hr_bpm']) if pred_hr['interval_hr_bpm'] is not None and ref_hr['interval_hr_bpm'] is not None else None
    return metrics


def aggregate_hr_errors(per: list[dict[str, Any]]) -> dict[str, float | None]:
    count=[m['count_hr_abs_error_bpm'] for m in per if m.get('count_hr_abs_error_bpm') is not None]
    interval=[m['interval_hr_abs_error_bpm'] for m in per if m.get('interval_hr_abs_error_bpm') is not None]
    return {
        'count_hr_mae_bpm': float(np.mean(count)) if count else None,
        'interval_hr_mae_bpm': float(np.mean(interval)) if interval else None,
    }


def match_peaks(ref, pred, fs, tol_s=0.12):
    ref=np.asarray(ref,dtype=int); pred=np.asarray(pred,dtype=int); tol=int(round(tol_s*fs)); used=np.zeros(len(ref), dtype=bool); tp=0; errs=[]
    for p in pred:
        if len(ref)==0: continue
        j=int(np.argmin(np.abs(ref-p)))
        if not used[j] and abs(ref[j]-p)<=tol:
            used[j]=True; tp+=1; errs.append((p-ref[j])/fs)
    fp=len(pred)-tp; fn=len(ref)-tp
    sens=tp/(tp+fn) if tp+fn else 0.0; ppv=tp/(tp+fp) if tp+fp else 0.0; f1=2*sens*ppv/(sens+ppv) if sens+ppv else 0.0
    return {'tp':int(tp),'fp':int(fp),'fn':int(fn),'sensitivity':float(sens),'ppv':float(ppv),'f1':float(f1),'mae_ms':float(np.mean(np.abs(errs))*1000) if errs else None}


def predict_record(model, raw_dir, rec, args, dev):
    x, ref=load_record(raw_dir, rec, args.target_fs); seg_len=int(args.seconds*args.target_fs); step=seg_len
    out=np.zeros(len(x), dtype=np.float32); weight=np.zeros(len(x), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), step):
            seg=x[start:start+seg_len]
            if len(seg)<seg_len:
                pad=np.zeros(seg_len, np.float32); pad[:len(seg)]=seg; seg=pad
            pr=torch.sigmoid(model(torch.from_numpy(seg[None,None,:]).to(dev))).cpu().numpy()[0,0]
            end=min(len(x), start+seg_len); out[start:end]+=pr[:end-start]; weight[start:end]+=1
    prob=out/np.maximum(weight,1)
    pred=probs_to_peaks(prob, args.target_fs, args.threshold, args.min_distance_s)
    return ref, pred, prob, len(x)


def evaluate_model(model, raw_dir, records, args, dev):
    total={'tp':0,'fp':0,'fn':0}; maes=[]; per=[]
    for rec in records:
        ref,pred,prob,n_samples=predict_record(model, raw_dir, rec, args, dev)
        m=match_peaks(ref,pred,args.target_fs,args.tolerance_s); m=add_hr_metrics(m, ref, pred, args.target_fs, n_samples); m['record']=rec; m['ref_peaks']=int(len(ref)); m['pred_peaks']=int(len(pred)); per.append(m)
        total['tp']+=m['tp']; total['fp']+=m['fp']; total['fn']+=m['fn']
        if m['mae_ms'] is not None: maes.append(m['mae_ms'])
    sens=total['tp']/(total['tp']+total['fn']) if total['tp']+total['fn'] else 0.0; ppv=total['tp']/(total['tp']+total['fp']) if total['tp']+total['fp'] else 0.0; f1=2*sens*ppv/(sens+ppv) if sens+ppv else 0.0
    out={'tp':total['tp'],'fp':total['fp'],'fn':total['fn'],'sensitivity':sens,'ppv':ppv,'f1':f1,'mae_ms':float(np.mean(maes)) if maes else None,'per_record':per}
    out.update(aggregate_hr_errors(per))
    return out


def evaluate_baseline(raw_dir, records, args):
    import tempfile, pandas as pd
    per=[]; total={'tp':0,'fp':0,'fn':0}; maes=[]
    with tempfile.TemporaryDirectory() as td:
        for rec in records:
            x,ref=load_record(raw_dir, rec, args.target_fs)
            path=Path(td)/f'{rec}.csv'; pd.DataFrame({'signal':x}).to_csv(path,index=False)
            res=SCG_detect_j_peaks(str(path), args.target_fs)
            pred=np.asarray(res.get('j_peak_indices', []), dtype=int)
            m=match_peaks(ref,pred,args.target_fs,args.tolerance_s); m=add_hr_metrics(m, ref, pred, args.target_fs, len(x)); m['record']=rec; m['heart_rate_bpm']=res.get('heart_rate_bpm'); m['method']=res.get('method'); m['ref_peaks']=int(len(ref)); m['pred_peaks']=int(len(pred)); per.append(m)
            total['tp']+=m['tp']; total['fp']+=m['fp']; total['fn']+=m['fn']
            if m['mae_ms'] is not None: maes.append(m['mae_ms'])
    sens=total['tp']/(total['tp']+total['fn']) if total['tp']+total['fn'] else 0.0; ppv=total['tp']/(total['tp']+total['fp']) if total['tp']+total['fp'] else 0.0; f1=2*sens*ppv/(sens+ppv) if sens+ppv else 0.0
    out={'tp':total['tp'],'fp':total['fp'],'fn':total['fn'],'sensitivity':sens,'ppv':ppv,'f1':f1,'mae_ms':float(np.mean(maes)) if maes else None,'per_record':per}
    out.update(aggregate_hr_errors(per))
    return out


def evaluate_probability_sweep(model, raw_dir, records, args, dev):
    cached=[]
    for rec in records:
        ref, _, prob, n_samples=predict_record(model, raw_dir, rec, args, dev)
        cached.append((rec, ref, prob, n_samples))
    rows=[]
    for threshold in [0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70]:
        for distance_s in [0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70,0.80]:
            per=[]; total={'tp':0,'fp':0,'fn':0}; maes=[]
            for rec,ref,prob,n_samples in cached:
                pred=probs_to_peaks(prob, args.target_fs, threshold, distance_s)
                m=match_peaks(ref,pred,args.target_fs,args.tolerance_s); m=add_hr_metrics(m, ref, pred, args.target_fs, n_samples); m['record']=rec; m['ref_peaks']=int(len(ref)); m['pred_peaks']=int(len(pred)); per.append(m)
                total['tp']+=m['tp']; total['fp']+=m['fp']; total['fn']+=m['fn']
                if m['mae_ms'] is not None: maes.append(m['mae_ms'])
            sens=total['tp']/(total['tp']+total['fn']) if total['tp']+total['fn'] else 0.0; ppv=total['tp']/(total['tp']+total['fp']) if total['tp']+total['fp'] else 0.0; f1=2*sens*ppv/(sens+ppv) if sens+ppv else 0.0
            row={'threshold':threshold,'min_distance_s':distance_s,'tp':total['tp'],'fp':total['fp'],'fn':total['fn'],'sensitivity':sens,'ppv':ppv,'f1':f1,'mae_ms':float(np.mean(maes)) if maes else None,'per_record':per}
            row.update(aggregate_hr_errors(per)); rows.append(row)
    rows.sort(key=lambda r: (-(r['interval_hr_mae_bpm'] is not None), r['interval_hr_mae_bpm'] if r['interval_hr_mae_bpm'] is not None else 1e9, -r['f1']))
    return {'best_by_interval_hr_mae': rows[0] if rows else None, 'top5_by_interval_hr_mae': rows[:5]}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw-dir', default='/data1/jiahui/biosignal-agent/datasets/raw/cebsdb')
    ap.add_argument('--out-model', default='/data1/jiahui/biosignal-agent/outputs/scg_cebs_unet_heartbeat.pt')
    ap.add_argument('--report', default='/data1/jiahui/biosignal-agent/outputs/scg_cebs_unet_heartbeat_report.json')
    ap.add_argument('--target-fs', type=float, default=250.0)
    ap.add_argument('--seconds', type=float, default=8.0)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=8e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--label-radius-s', type=float, default=0.08)
    ap.add_argument('--threshold', type=float, default=0.35)
    ap.add_argument('--min-distance-s', type=float, default=0.30)
    ap.add_argument('--tolerance-s', type=float, default=0.12)
    ap.add_argument('--seed', type=int, default=59)
    ap.add_argument('--val-count', type=int, default=2)
    ap.add_argument('--test-count', type=int, default=2)
    ap.add_argument('--cpu', action='store_true')
    args=ap.parse_args(); set_seed(args.seed)
    raw_dir=Path(args.raw_dir); records=complete_records(raw_dir)
    if len(records)<3: raise SystemExit(f'Need at least 3 complete records, found {records}')
    holdout=args.val_count+args.test_count
    if len(records) <= holdout:
        raise SystemExit(f'Need more than {holdout} complete records, found {records}')
    train_records=records[:-holdout]; val_records=records[-holdout:-args.test_count] if args.test_count else records[-holdout:]; test_records=records[-args.test_count:] if args.test_count else []
    dev=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    trds=SegmentDataset(raw_dir, train_records, args, True, args.seed); vads=SegmentDataset(raw_dir, val_records, args, False, args.seed+1)
    trl=DataLoader(trds,batch_size=args.batch_size,shuffle=True,num_workers=0); val_loader=DataLoader(vads,batch_size=args.batch_size,shuffle=False,num_workers=0)
    model=UNet1D().to(dev)
    opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Positive samples are sparse; use a strong pos_weight for samplewise segmentation.
    pos_weight=torch.tensor([18.0], device=dev); loss_fn=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best=None
    for ep in range(1,args.epochs+1):
        model.train(); total=0.0
        for x,y,rec,start in trl:
            x=x.to(dev); y=y.to(dev); opt.zero_grad(set_to_none=True); loss=loss_fn(model(x), y); loss.backward(); opt.step(); total+=float(loss.item())*len(x)
        val=evaluate_model(model, raw_dir, val_records, args, dev)
        rep={'epoch':ep,'loss':total/max(1,len(trds)),'val':{k:v for k,v in val.items() if k!='per_record'}}
        print(json.dumps(rep), flush=True)
        score=(val['f1'], val['sensitivity'], val['ppv'])
        if best is None or score>best['score']:
            best={'epoch':ep,'score':score,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'val':val}
    model.load_state_dict(best['state_dict'])
    val_sweep=evaluate_probability_sweep(model, raw_dir, val_records, args, dev)
    original_threshold, original_distance = args.threshold, args.min_distance_s
    if val_sweep.get('best_by_interval_hr_mae'):
        args.threshold=val_sweep['best_by_interval_hr_mae']['threshold']; args.min_distance_s=val_sweep['best_by_interval_hr_mae']['min_distance_s']
    test=evaluate_model(model, raw_dir, test_records, args, dev)
    test_sweep=evaluate_probability_sweep(model, raw_dir, test_records, args, dev)
    baseline=evaluate_baseline(raw_dir, test_records, args)
    selected_postprocess={'selected_on':'validation_interval_hr_mae','threshold':args.threshold,'min_distance_s':args.min_distance_s,'original_threshold':original_threshold,'original_min_distance_s':original_distance}
    payload={'model_state_dict':best['state_dict'],'architecture':'UNet1D_SCGBPM_semantic_segmentation','target_fs':args.target_fs,'seconds':args.seconds,'threshold':args.threshold,'min_distance_s':args.min_distance_s,'tolerance_s':args.tolerance_s,'train_records':train_records,'val_records':val_records,'test_records':test_records,'best_epoch':best['epoch'],'selected_postprocess':selected_postprocess}
    Path(args.out_model).parent.mkdir(parents=True,exist_ok=True); torch.save(payload,args.out_model)
    report={'records':records,'train_records':train_records,'val_records':val_records,'test_records':test_records,'best_epoch':best['epoch'],'val':best['val'],'val_postprocess_sweep':val_sweep,'selected_postprocess':selected_postprocess,'test':test,'test_postprocess_oracle_sweep':test_sweep,'baseline_test':baseline,'model_out':args.out_model,'reference':'CEBSDB SCG heartbeat detection with ECG-anchored SCG J/AO labels: local SCG deflection searched 80-280 ms after ECG annotations; U-Net semantic segmentation inspired by SCG semantic-segmentation literature.'}
    Path(args.report).write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))

if __name__=='__main__': main()
