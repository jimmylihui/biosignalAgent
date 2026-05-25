from __future__ import annotations

import argparse, json, sys, math, random
from pathlib import Path
from typing import Any

import neurokit2 as nk
import numpy as np
import wfdb
from scipy import signal as scipy_signal
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.tools.common import bandpass_filter
from biosignal_agent.tools.peak_detectors import ppg_multiscale_systolic_peaks
from scripts.evaluate_ppg_peak_detectors import estimate_best_lag_match, match_ppg_to_ecg


def find_channel(sig_names, candidates):
    cleaned=[s.strip().strip(',').lower() for s in sig_names]
    for c in candidates:
        c=c.lower()
        if c in cleaned: return cleaned.index(c)
    for i,n in enumerate(cleaned):
        if any(c.lower() in n for c in candidates): return i
    return None


def weak_ppg_peak_labels(ppg, ecg, fs):
    _, info = nk.ecg_peaks(ecg, sampling_rate=fs, method='nabian2018', correct_artifacts=True)
    ecg_peaks=np.asarray(info.get('ECG_R_Peaks',[]), dtype=int)
    seed_peaks,_=ppg_multiscale_systolic_peaks(ppg, fs)
    lag=estimate_best_lag_match(ecg_peaks, seed_peaks, fs)['applied_lag_s'] if len(seed_peaks) else 0.0
    filt=bandpass_filter(ppg, fs, 0.4, min(8.0, fs*0.45))
    peaks=[]
    for r in ecg_peaks:
        lo=int(round(r + 0.08*fs - lag*fs)); hi=int(round(r + 0.60*fs - lag*fs))
        lo=max(0, lo); hi=min(len(ppg)-1, hi)
        if hi <= lo+2: continue
        loc=lo+int(np.argmax(filt[lo:hi+1]))
        peaks.append(loc)
    peaks=np.asarray(sorted(set(peaks)), dtype=int)
    return ecg_peaks, peaks, lag


def make_label(n, peaks, fs):
    y=np.zeros(n, dtype=np.float32)
    radius=max(1, int(round(0.04*fs)))
    sigma=max(1.0, 0.018*fs)
    for p in peaks:
        lo=max(0, int(p)-radius); hi=min(n, int(p)+radius+1)
        xs=np.arange(lo,hi)
        y[lo:hi]=np.maximum(y[lo:hi], np.exp(-0.5*((xs-p)/sigma)**2))
    return y


def load_records(raw_dir):
    rows=[]
    for hea in sorted(Path(raw_dir).glob('bidmc*.hea')):
        rec=wfdb.rdrecord(str(hea.with_suffix('')))
        ppg_idx=find_channel(rec.sig_name, ['PLETH','PPG'])
        ecg_idx=find_channel(rec.sig_name, ['II','V','ECG'])
        if ppg_idx is None or ecg_idx is None: continue
        ppg=rec.p_signal[:,ppg_idx].astype(float)
        ecg=rec.p_signal[:,ecg_idx].astype(float)
        fs=float(rec.fs)
        try:
            ecg_peaks, weak_peaks, lag=weak_ppg_peak_labels(ppg, ecg, fs)
        except Exception as exc:
            print('skip', hea.stem, exc, file=sys.stderr); continue
        if len(weak_peaks)<10: continue
        rows.append({'record':hea.stem,'fs':fs,'ppg':ppg,'ecg_peaks':ecg_peaks,'weak_peaks':weak_peaks,'lag':lag})
    return rows

class WindowDataset(Dataset):
    def __init__(self, records, window_s=8.192, stride_s=2.0, augment=False):
        self.items=[]; self.records=records; self.augment=augment
        for ri,r in enumerate(records):
            fs=r['fs']; w=int(round(window_s*fs)); stride=int(round(stride_s*fs))
            labels=make_label(len(r['ppg']), r['weak_peaks'], fs)
            r['label']=labels
            for start in range(0, max(1,len(r['ppg'])-w), stride):
                end=start+w
                if end>len(r['ppg']): continue
                if labels[start:end].max()<=0: continue
                self.items.append((ri,start,end))
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        ri,start,end=self.items[idx]; r=self.records[ri]; fs=r['fs']
        x=r['ppg'][start:end].astype(np.float32)
        x=bandpass_filter(x, fs, 0.4, min(8.0, fs*0.45)).astype(np.float32)
        x=x-np.median(x); scale=np.percentile(np.abs(x),95)+1e-6; x=x/scale
        y=r['label'][start:end].astype(np.float32)
        if self.augment:
            gain=np.random.uniform(0.8,1.2); x=x*gain
            if np.random.rand()<0.5: x=x+np.random.normal(0,0.03,size=x.shape).astype(np.float32)
            if np.random.rand()<0.25:
                t=np.linspace(0,1,len(x),dtype=np.float32); x=x+np.random.uniform(-0.1,0.1)*np.sin(2*np.pi*np.random.uniform(0.1,0.4)*t)
        return torch.tensor(x[None,:], dtype=torch.float32), torch.tensor(y[None,:], dtype=torch.float32)

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__(); self.net=nn.Sequential(nn.Conv1d(c_in,c_out,7,padding=3),nn.BatchNorm1d(c_out),nn.SiLU(),nn.Conv1d(c_out,c_out,5,padding=2),nn.BatchNorm1d(c_out),nn.SiLU())
    def forward(self,x): return self.net(x)
class PeakUNet(nn.Module):
    def __init__(self):
        super().__init__(); ch=[1,24,48,96,128]
        self.e1=ConvBlock(ch[0],ch[1]); self.e2=ConvBlock(ch[1],ch[2]); self.e3=ConvBlock(ch[2],ch[3]); self.b=ConvBlock(ch[3],ch[4])
        self.pool=nn.MaxPool1d(2)
        self.u3=nn.ConvTranspose1d(ch[4],ch[3],2,2); self.d3=ConvBlock(ch[3]*2,ch[3])
        self.u2=nn.ConvTranspose1d(ch[3],ch[2],2,2); self.d2=ConvBlock(ch[2]*2,ch[2])
        self.u1=nn.ConvTranspose1d(ch[2],ch[1],2,2); self.d1=ConvBlock(ch[1]*2,ch[1])
        self.out=nn.Conv1d(ch[1],1,1)
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2)); b=self.b(self.pool(e3))
        x=self.u3(b); x=torch.cat([x[...,:e3.shape[-1]],e3],1); x=self.d3(x)
        x=self.u2(x); x=torch.cat([x[...,:e2.shape[-1]],e2],1); x=self.d2(x)
        x=self.u1(x); x=torch.cat([x[...,:e1.shape[-1]],e1],1); x=self.d1(x)
        return self.out(x)

def prob_to_peaks(prob, fs, threshold=0.28):
    min_distance=max(1,int(60/220*fs))
    peaks,_=scipy_signal.find_peaks(prob, distance=min_distance, height=threshold, prominence=0.05)
    return peaks.astype(int)

def predict_record(model, ppg, fs, device, window=1024, hop=512):
    filt=bandpass_filter(ppg, fs, 0.4, min(8.0, fs*0.45)).astype(np.float32)
    filt=filt-np.median(filt); filt=filt/(np.percentile(np.abs(filt),95)+1e-6)
    acc=np.zeros(len(filt), dtype=np.float32); wt=np.zeros(len(filt), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        starts=list(range(0, max(1,len(filt)-window+1), hop))
        if starts[-1] != len(filt)-window: starts.append(max(0,len(filt)-window))
        win=np.hanning(window).astype(np.float32); win=np.maximum(win,0.05)
        for s in starts:
            seg=filt[s:s+window]
            if len(seg)<window: continue
            logits=model(torch.tensor(seg[None,None,:], device=device, dtype=torch.float32))
            pr=torch.sigmoid(logits).cpu().numpy()[0,0]
            acc[s:s+window]+=pr*win; wt[s:s+window]+=win
    prob=acc/np.maximum(wt,1e-6)
    return prob_to_peaks(prob, fs), prob

def evaluate_model(model, records, device):
    vals=[]
    for r in records:
        peaks,_=predict_record(model,r['ppg'],r['fs'],device)
        vals.append(estimate_best_lag_match(r['ecg_peaks'], peaks, r['fs']))
    return {'mean_sensitivity':float(np.mean([v['sensitivity'] for v in vals])), 'mean_ppv':float(np.mean([v['ppv'] for v in vals])), 'mean_f1':float(np.mean([v['f1'] for v in vals])), 'total_detected':int(sum(v['detected'] for v in vals)), 'total_reference':int(sum(v['reference'] for v in vals))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--raw-dir',default='/data1/jiahui/biosignal-agent/datasets/raw/bidmc'); ap.add_argument('--out',default='/data1/jiahui/biosignal-agent/outputs/ppg_peak_unet.pt'); ap.add_argument('--report',default='/data1/jiahui/biosignal-agent/outputs/ppg_peak_unet_report.json'); ap.add_argument('--epochs',type=int,default=10); args=ap.parse_args()
    torch.manual_seed(7); np.random.seed(7); random.seed(7)
    records=load_records(args.raw_dir); random.shuffle(records)
    n_val=max(8,int(len(records)*0.2)); val=records[:n_val]; train=records[n_val:]
    train_ds=WindowDataset(train, augment=True); val_ds=WindowDataset(val, augment=False)
    device='cuda' if torch.cuda.is_available() else 'cpu'; model=PeakUNet().to(device)
    pos_weight=torch.tensor([8.0], device=device); loss_fn=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt=torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader=DataLoader(train_ds,batch_size=64,shuffle=True,num_workers=0)
    history=[]; best=-1
    for ep in range(1,args.epochs+1):
        model.train(); losses=[]
        for x,y in train_loader:
            x=x.to(device); y=y.to(device); opt.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); opt.step(); losses.append(float(loss.item()))
        metrics=evaluate_model(model,val,device)
        history.append({'epoch':ep,'loss':float(np.mean(losses)),**metrics})
        print(json.dumps(history[-1]))
        if metrics['mean_f1']>best:
            best=metrics['mean_f1']; Path(args.out).parent.mkdir(parents=True,exist_ok=True); torch.save({'model_state_dict':model.state_dict(),'architecture':'shallow_1d_unet_peak_segmentation','fs_hint':125.0,'threshold':0.28,'window':1024,'hop':512,'cv_metrics':metrics}, args.out)
    ck=torch.load(args.out,map_location=device,weights_only=False); model.load_state_dict(ck['model_state_dict'])
    val_metrics=evaluate_model(model,val,device); train_metrics=evaluate_model(model,train[:min(12,len(train))],device)
    report={'model':'shallow_1d_unet_peak_segmentation','labeling':'weak labels from ECG R-peaks plus per-record PTT/channel-lag PPG local maxima','num_records':len(records),'train_records':len(train),'val_records':len(val),'train_windows':len(train_ds),'val_windows':len(val_ds),'device':device,'history':history,'val_metrics':val_metrics,'train_subset_metrics':train_metrics,'model_out':args.out,'caveat':'Weakly supervised BIDMC labels are ECG-aligned local PPG maxima, not manual PPG systolic annotations.'}
    Path(args.report).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
