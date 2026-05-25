from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score, top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT=Path('/data1/jiahui/biosignal-agent')
DATA=ROOT/'datasets/raw/emg_geds'
OUT=ROOT/'outputs'
SPEED={'c':'comfortable','s':'slow','f':'fast'}
PHASE_COLS=['EMG_taR','EMG_taL','ACCx_taR','ACCy_taR','ACCz_taR','GYRx_taR','GYRy_taR','GYRz_taR','ACCx_taL','ACCy_taL','ACCz_taL','GYRx_taL','GYRy_taL','GYRz_taL','FSR_hsR','FSR_toR','FSR_hsL','FSR_toL']

class Block(nn.Module):
    def __init__(self,c,d=1):
        super().__init__(); self.net=nn.Sequential(nn.Conv1d(c,c,5,padding=2*d,dilation=d),nn.BatchNorm1d(c),nn.ReLU(),nn.Dropout(0.2),nn.Conv1d(c,c,3,padding=d,dilation=d),nn.BatchNorm1d(c)); self.act=nn.ReLU()
    def forward(self,x): return self.act(x+self.net(x))
class TCN(nn.Module):
    def __init__(self,cin,nc,width=96):
        super().__init__(); self.net=nn.Sequential(nn.Conv1d(cin,width,7,padding=3),nn.BatchNorm1d(width),nn.ReLU(),Block(width,1),Block(width,2),Block(width,4),nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Dropout(0.2),nn.Linear(width,nc))
    def forward(self,x): return self.net(x)

def meta(path):
    m=re.match(r's(\d+)([csf])(\d+)\.txt$', path.name)
    if not m: return None
    return f's{int(m.group(1)):02d}', SPEED[m.group(2)], int(m.group(3))

def labels_for_phase(ev):
    rhs=ev['RHS'].to_numpy(int); rto=ev['RTO'].to_numpy(int); seg=[]
    for i,(a,b) in enumerate(zip(rhs,rto)):
        seg.append((a,b,'right_stance'))
        if i+1<len(rhs): seg.append((b,rhs[i+1],'right_swing'))
    return seg

def label_at(segs, center):
    for a,b,l in segs:
        if a <= center < b: return l
    return None

def build_cache(path: Path, task: str, max_per_class: int=12000, seed: int=23):
    rng=np.random.default_rng(seed); buckets={}; rows=[]
    for txt in sorted(DATA.glob('S[0-9][0-9]/s*.txt')):
        m=meta(txt)
        if not m: continue
        subj,speed,trial=m
        if task=='speed':
            arr=np.loadtxt(txt,delimiter='\t',skiprows=1,dtype=np.float32)
            if arr.ndim==1: arr=arr.reshape(1,-1)
            sig=arr[:,1:]
            win=1000; step=500; label=speed
            for st in range(0,len(sig)-win+1,step):
                chunk=sig[st:st+win:5]  # 200 time steps
                chunk=(chunk-chunk.mean(0,keepdims=True))/(chunk.std(0,keepdims=True)+1e-6)
                buckets.setdefault(label,[]).append((chunk.T.copy(), label, subj))
            rows.append({'file':str(txt),'subject':subj,'label':label})
        else:
            ev=txt.with_name(txt.stem+'ev.txt')
            if not ev.exists(): continue
            with open(txt) as f: cols=f.readline().strip().split('\t')
            idx=[cols.index(c) for c in PHASE_COLS]
            arr=np.loadtxt(txt,delimiter='\t',skiprows=1,dtype=np.float32)
            if arr.ndim==1: arr=arr.reshape(1,-1)
            sig=arr[:,idx]
            segs=labels_for_phase(pd.read_csv(ev,sep='\t'))
            win=500; step=250
            for st in range(0,len(sig)-win+1,step):
                label=label_at(segs,st+win//2)
                if label is None: continue
                chunk=sig[st:st+win:2]  # 250 time steps
                chunk=(chunk-chunk.mean(0,keepdims=True))/(chunk.std(0,keepdims=True)+1e-6)
                buckets.setdefault(label,[]).append((chunk.T.copy(), label, subj))
            rows.append({'file':str(txt),'subject':subj,'event_file':str(ev)})
    packed=[]
    for label,items in sorted(buckets.items()):
        if len(items)>max_per_class:
            idx=rng.choice(len(items),max_per_class,replace=False); items=[items[int(i)] for i in idx]
        packed.extend(items)
    rng.shuffle(packed)
    X=np.stack([x for x,_,_ in packed]).astype(np.float32); y=np.asarray([y for _,y,_ in packed]); groups=np.asarray([g for *_,g in packed])
    np.savez_compressed(path,X=X,y=y,groups=groups)
    (OUT/f'emg_geds_{task}_tcn_manifest.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps({'cache':str(path),'task':task,'shape':X.shape,'counts':dict(Counter(y)),'subjects':len(set(groups))},indent=2),flush=True)

def split(groups):
    order=sorted(set(groups)); test_s=set(order[-5:]); val_s=set(order[-8:-5])
    test=np.asarray([g in test_s for g in groups]); val=np.asarray([g in val_s for g in groups]); train=~(test|val)
    return np.where(train)[0],np.where(val)[0],np.where(test)[0]

def predict(model,loader,device):
    model.eval(); ys=[]; probs=[]
    with torch.no_grad():
        for xb,yb in loader:
            probs.append(torch.softmax(model(xb.to(device)),1).cpu().numpy()); ys.append(yb.numpy())
    return np.concatenate(ys),np.vstack(probs)

def metrics(y,proba):
    pred=proba.argmax(1); out={'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),'weighted_f1':float(f1_score(y,pred,average='weighted'))}
    if proba.shape[1]==2:
        out['auroc']=float(roc_auc_score(y,proba[:,1])); out['auprc']=float(average_precision_score(y,proba[:,1]))
    else:
        out['top2_accuracy']=float(top_k_accuracy_score(y,proba,k=2,labels=np.arange(proba.shape[1])))
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--task',choices=['speed','phase'],required=True); ap.add_argument('--prepare',action='store_true'); ap.add_argument('--epochs',type=int,default=20); ap.add_argument('--batch-size',type=int,default=512); args=ap.parse_args()
    cache=OUT/f'emg_geds_{args.task}_raw_windows_tcn.npz'
    if args.prepare or not cache.exists(): build_cache(cache,args.task)
    d=np.load(cache,allow_pickle=True); X=d['X']; yr=d['y']; groups=d['groups']; le=LabelEncoder(); y=le.fit_transform(yr)
    tr,va,te=split(groups); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=TCN(X.shape[1],len(le.classes_)).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); loss_fn=nn.CrossEntropyLoss()
    loaders={
      'train':DataLoader(TensorDataset(torch.from_numpy(X[tr]),torch.from_numpy(y[tr]).long()),batch_size=args.batch_size,shuffle=True,num_workers=2),
      'val':DataLoader(TensorDataset(torch.from_numpy(X[va]),torch.from_numpy(y[va]).long()),batch_size=args.batch_size,shuffle=False,num_workers=2),
      'test':DataLoader(TensorDataset(torch.from_numpy(X[te]),torch.from_numpy(y[te]).long()),batch_size=args.batch_size,shuffle=False,num_workers=2)}
    best=None; state=None; t0=time.time()
    for ep in range(1,args.epochs+1):
        model.train(); total=0; n=0
        for xb,yb in loaders['train']:
            xb=xb.to(device); yb=yb.to(device); opt.zero_grad(set_to_none=True); loss=loss_fn(model(xb),yb); loss.backward(); opt.step(); total+=float(loss.item())*len(yb); n+=len(yb)
        vy,vp=predict(model,loaders['val'],device); vm=metrics(vy,vp); print(json.dumps({'epoch':ep,'loss':total/max(1,n),'val':vm},indent=2),flush=True)
        if best is None or vm['macro_f1']>best['macro_f1']:
            best=vm; state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    model.load_state_dict(state); ty,tp=predict(model,loaders['test'],device); tm=metrics(ty,tp)
    report={'task':f'emg_geds_{args.task}_tcn','dataset':'GEDS full S00-S22 raw downsampled windows','n_total':int(len(y)),'n_train':int(len(tr)),'n_val':int(len(va)),'n_test':int(len(te)),'labels':le.classes_.tolist(),'best_val':best,'test':tm,'epochs':args.epochs,'elapsed_sec':round(time.time()-t0,2),'method':'raw_sequence_tcn_subject_split'}
    (OUT/f'emg_geds_{args.task}_tcn_report.json').write_text(json.dumps(report,indent=2)); torch.save({'state_dict':state,'labels':le.classes_.tolist(),'in_channels':int(X.shape[1])},OUT/f'emg_geds_{args.task}_tcn.pt')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__': main()
