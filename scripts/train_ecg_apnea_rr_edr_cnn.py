from __future__ import annotations
import argparse, json, math, random, sys, warnings
from collections import Counter
warnings.filterwarnings("ignore", category=RuntimeWarning)
from pathlib import Path
import numpy as np, torch
from scipy import interpolate, signal as scipy_signal
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.tools.common import load_csv_signal
from biosignal_agent.tools.peak_detectors import neurokit_nabian2018_peaks
OUT=Path('/data1/jiahui/biosignal-agent/outputs')
MODEL_PATH=OUT/'ecg_apnea_rr_edr_cnn_model.pt'

def seed_all(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def _robust_norm(seq):
    seq=np.asarray(seq,dtype=np.float32)
    seq=np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
    med=float(np.median(seq))
    iqr=float(np.percentile(seq,75)-np.percentile(seq,25))
    if iqr < 1e-8:
        iqr=float(np.std(seq)+1e-8)
    return np.nan_to_num(np.clip((seq-med)/iqr,-6,6), nan=0.0, posinf=6.0, neginf=-6.0).astype(np.float32)

def _smooth(seq, width):
    seq=np.asarray(seq,dtype=np.float32)
    width=max(3,int(width))
    if width % 2 == 0: width += 1
    if len(seq) < width: return seq.copy()
    return scipy_signal.savgol_filter(seq, width, 2).astype(np.float32)

def rr_edr_image(values, fs, seq_len=256, num_channels=6):
    values=np.asarray(values,dtype=float); values=values[np.isfinite(values)]
    peaks,_=neurokit_nabian2018_peaks(values,fs,low_hz=None,high_hz=None,fallback_threshold_scale=0.6)
    peaks=np.asarray(peaks,dtype=int); duration=len(values)/float(fs) if fs else 0.0
    grid=np.linspace(0,max(duration,1e-6),seq_len)
    if len(peaks)>=4:
        t=peaks/fs; rr=np.diff(t); rt=(t[:-1]+t[1:])/2
        keep=np.isfinite(rr)&(rr>=0.25)&(rr<=3.0)
        rr=rr[keep]; rt=rt[keep]
        rr_seq=np.interp(grid, rt, rr, left=float(np.median(rr)), right=float(np.median(rr))) if len(rr)>=2 else np.full(seq_len, float(np.median(rr)) if len(rr) else 0.8)
        amp=values[peaks]
        amp=(amp-np.median(amp))/(np.percentile(amp,75)-np.percentile(amp,25)+1e-8)
        edr=np.interp(grid,t,amp,left=float(amp[0]),right=float(amp[-1])) if len(amp)>=2 else np.zeros(seq_len)
    else:
        rr_seq=np.full(seq_len,0.8); edr=np.zeros(seq_len)
    rr_n=_robust_norm(rr_seq); edr_n=_robust_norm(edr)
    rr_delta=_robust_norm(np.gradient(rr_n)); edr_delta=_robust_norm(np.gradient(edr_n))
    slow_width=max(9, int(seq_len/10))
    rr_slow=_robust_norm(_smooth(rr_n, slow_width))
    edr_slow=_robust_norm(_smooth(edr_n, slow_width))
    chans=[rr_n, edr_n, rr_delta, edr_delta, rr_slow, edr_slow]
    return np.nan_to_num(np.stack(chans[:num_channels]), nan=0.0, posinf=6.0, neginf=-6.0).astype(np.float32)

def load_manifest(path, seq_len, num_channels):
    m=json.load(open(path)); X=[]; y=[]; groups=[]; sources=[]
    for r in m['records']:
        d=load_csv_signal(r['path'],float(r['sampling_rate']),None)
        X.append(rr_edr_image(d.values,d.sampling_rate,seq_len,num_channels)); y.append(1 if r['label']=='apnea' else 0); groups.append(str(r['record'])); sources.append(str(r.get('source') or r.get('dataset') or m.get('dataset', 'unknown')))
    return np.asarray(X,np.float32),np.asarray(y,np.int64),groups,sources

class RREdrCNN(nn.Module):
    def __init__(self, in_channels=6, dropout=0.30):
        super().__init__()
        self.cnn=nn.Sequential(
            nn.Conv1d(in_channels,32,9,padding=4,bias=False),nn.BatchNorm1d(32),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(32,64,7,padding=3,bias=False),nn.BatchNorm1d(64),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(64,96,5,padding=2,bias=False),nn.BatchNorm1d(96),nn.ReLU(),nn.MaxPool1d(2))
        self.rnn=nn.LSTM(96,64,batch_first=True,bidirectional=True)
        self.head=nn.Sequential(nn.LayerNorm(256),nn.Dropout(dropout),nn.Linear(256,96),nn.ReLU(),nn.Dropout(dropout),nn.Linear(96,1))
    def forward(self,x):
        z=self.cnn(x).transpose(1,2)
        z,_=self.rnn(z)
        pooled=torch.cat([z.mean(dim=1), z.amax(dim=1)], dim=1)
        return self.head(pooled).squeeze(-1)

def predict(model,X,device):
    model.eval(); out=[]
    with torch.no_grad():
        for xb in DataLoader(torch.tensor(X,dtype=torch.float32),batch_size=256): out.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    return np.concatenate(out).astype(float)

def train_fold(X,y,tr,va,epochs,seed,device):
    seed_all(seed); model=RREdrCNN(in_channels=X.shape[1]).to(device)
    pos=max(float((y[tr]==1).sum()),1); neg=max(float((y[tr]==0).sum()),1)
    crit=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos],dtype=torch.float32,device=device))
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
    ds=TensorDataset(torch.tensor(X[tr],dtype=torch.float32),torch.tensor(y[tr],dtype=torch.float32))
    loader=DataLoader(ds,batch_size=256,shuffle=True)
    best=None; best_loss=math.inf; stale=0
    for epoch in range(epochs):
        model.train()
        for xb,yb in loader:
            xb=xb.to(device); yb=yb.to(device); opt.zero_grad(set_to_none=True); loss=crit(model(xb),yb); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),4); opt.step()
        if va is None: continue
        model.eval()
        with torch.no_grad(): val=float(crit(model(torch.tensor(X[va],dtype=torch.float32,device=device)),torch.tensor(y[va],dtype=torch.float32,device=device)).cpu())
        if val<best_loss-1e-4: best_loss=val; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale+=1
        if stale>=8: break
    if best is not None: model.load_state_dict(best)
    return model,best_loss

def met(y,p,t):
    pred=(p>=t).astype(int); d={'threshold':float(t),'accuracy':float(accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'average_precision':float(average_precision_score(y,p))}
    try: d['roc_auc']=float(roc_auc_score(y,p))
    except Exception: d['roc_auc']=0.0
    return d

def source_metrics(y, p, sources, threshold):
    out={}
    sources=np.asarray(sources)
    for src in sorted(set(map(str, sources))):
        idx=np.where(sources==src)[0]
        if len(idx)==0:
            continue
        item=met(y[idx], p[idx], threshold)
        item['num']=int(len(idx)); item['positives']=int(np.sum(y[idx]))
        out[src]=item
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',default='/data1/jiahui/biosignal-agent/datasets/processed/apnea_ecg_large_manifest.json'); ap.add_argument('--seq-len',type=int,default=384); ap.add_argument('--num-channels',type=int,default=6); ap.add_argument('--epochs',type=int,default=50); ap.add_argument('--seed',type=int,default=41); ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); ap.add_argument('--model-path',type=Path,default=MODEL_PATH); ap.add_argument('--report-path',type=Path,default=OUT/'ecg_apnea_rr_edr_cnn_train_report.json'); args=ap.parse_args(); seed_all(args.seed)
    X,y,groups,sources=load_manifest(args.manifest,args.seq_len,args.num_channels); print(json.dumps({'num':len(y),'labels':dict(Counter(map(int,y))),'groups':len(set(groups)),'sources':dict(Counter(map(str,sources))),'shape':list(X.shape)},indent=2))
    proba=np.zeros(len(y)); folds=[]
    for fold,(tr,va) in enumerate(GroupKFold(n_splits=5).split(X,y,groups=groups)):
        model,vl=train_fold(X,y,tr,va,args.epochs,args.seed+fold,args.device); proba[va]=predict(model,X[va],args.device); folds.append({'fold':fold,'train_size':int(len(tr)),'val_size':int(len(va)),'val_loss':float(vl),'val_label_counts':dict(Counter(map(int,y[va])))}); print('fold',fold,folds[-1],flush=True)
    th=float(max(((f1_score(y,proba>=t,zero_division=0),t) for t in np.linspace(0.1,0.9,81)),key=lambda z:z[0])[1]); cv=met(y,proba,th); by_source=source_metrics(y,proba,sources,th)
    allidx=np.arange(len(y)); final,_=train_fold(X,y,allidx,None,max(args.epochs,45),args.seed+999,args.device)
    args.model_path.parent.mkdir(parents=True,exist_ok=True); torch.save({'state_dict':final.cpu().state_dict(),'architecture':'RREdrCNN_BiLSTM_multichannel','seq_len':args.seq_len,'num_channels':int(X.shape[1]),'threshold':th,'cv_metrics':cv,'source_metrics':by_source,'fold_reports':folds,'label_counts':dict(Counter(map(int,y)))},args.model_path)
    report={'model_path':str(args.model_path),'num_rows':int(len(y)),'label_counts':dict(Counter(map(int,y))),'threshold':th,'cv_metrics':cv,'source_metrics':by_source,'fold_reports':folds}; args.report_path.parent.mkdir(parents=True,exist_ok=True); args.report_path.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
