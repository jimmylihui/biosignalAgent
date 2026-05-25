from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT=Path('/data1/jiahui/biosignal-agent')
DATA=ROOT/'datasets/raw/emg_physical_action_uci/EMG Physical Action Data Set'
OUT=ROOT/'outputs'

class Block(nn.Module):
    def __init__(self,c,d=1):
        super().__init__(); self.net=nn.Sequential(nn.Conv1d(c,c,5,padding=2*d,dilation=d),nn.BatchNorm1d(c),nn.ReLU(),nn.Dropout(0.2),nn.Conv1d(c,c,3,padding=d,dilation=d),nn.BatchNorm1d(c)); self.act=nn.ReLU()
    def forward(self,x): return self.act(x+self.net(x))
class TCN(nn.Module):
    def __init__(self,cin,nc,width=64):
        super().__init__(); self.net=nn.Sequential(nn.Conv1d(cin,width,7,padding=3),nn.BatchNorm1d(width),nn.ReLU(),Block(width,1),Block(width,2),Block(width,4),nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(width,nc))
    def forward(self,x): return self.net(x)

def action_name(path: Path):
    return re.sub(r'(?<!^)([A-Z])', r'_\1', path.stem).lower().replace('frontkicking','front_kicking').replace('sidekicking','side_kicking')

def build_cache(path: Path):
    X=[]; y=[]; groups=[]; rows=[]; win=250; step=125
    for txt in sorted(DATA.glob('sub*/**/txt/*.txt')):
        arr=np.loadtxt(txt,dtype=np.float32)
        if arr.ndim==1: arr=arr.reshape(-1,1)
        if arr.shape[1] > 8: arr=arr[:,:8]
        if arr.shape[1] < 8: arr=np.pad(arr,((0,0),(0,8-arr.shape[1])))
        arr=(arr-arr.mean(0,keepdims=True))/(arr.std(0,keepdims=True)+1e-6)
        label=action_name(txt); subj=txt.parts[-4]; n=0
        for st in range(0,len(arr)-win+1,step):
            X.append(arr[st:st+win].T); y.append(label); groups.append(subj); n+=1
        rows.append({'file':str(txt),'label':label,'subject':subj,'windows':n})
    X=np.stack(X).astype(np.float32); y=np.asarray(y); groups=np.asarray(groups)
    np.savez_compressed(path,X=X,y=y,groups=groups)
    (OUT/'emg_physical_action_tcn_manifest.json').write_text(json.dumps(rows,indent=2))
    print('cache',X.shape,len(set(y)),sorted(set(groups)),flush=True)

def eval_model(model,loader,device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for xb,yb in loader:
            p=model(xb.to(device)).argmax(1).cpu().numpy(); ps.append(p); ys.append(yb.numpy())
    y=np.concatenate(ys); pred=np.concatenate(ps)
    return y,pred,{'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),'weighted_f1':float(f1_score(y,pred,average='weighted'))}

def main():
    cache=OUT/'emg_physical_action_raw_windows_tcn.npz'
    if not cache.exists(): build_cache(cache)
    d=np.load(cache,allow_pickle=True); X=d['X']; yr=d['y']; groups=d['groups']
    le=LabelEncoder(); y=le.fit_transform(yr); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    folds=[]; all_y=[]; all_p=[]; t0=time.time()
    subjects=sorted(set(groups))
    for fold,test_subj in enumerate(subjects,1):
        val_subj=subjects[fold % len(subjects)]
        test=groups==test_subj; val=groups==val_subj; train=~(test|val)
        model=TCN(X.shape[1],len(le.classes_)).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); loss_fn=nn.CrossEntropyLoss()
        train_loader=DataLoader(TensorDataset(torch.from_numpy(X[train]),torch.from_numpy(y[train]).long()),batch_size=256,shuffle=True)
        val_loader=DataLoader(TensorDataset(torch.from_numpy(X[val]),torch.from_numpy(y[val]).long()),batch_size=512)
        test_loader=DataLoader(TensorDataset(torch.from_numpy(X[test]),torch.from_numpy(y[test]).long()),batch_size=512)
        best=None; state=None
        for epoch in range(1,26):
            model.train(); total=0; n=0
            for xb,yb in train_loader:
                xb=xb.to(device); yb=yb.to(device); opt.zero_grad(set_to_none=True); loss=loss_fn(model(xb),yb); loss.backward(); opt.step(); total+=float(loss.item())*len(yb); n+=len(yb)
            _,_,vm=eval_model(model,val_loader,device)
            if best is None or vm['macro_f1']>best['macro_f1']:
                best=vm; state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        model.load_state_dict(state); yt,yp,tm=eval_model(model,test_loader,device); all_y.append(yt); all_p.append(yp)
        folds.append({'fold':fold,'test_subject':str(test_subj),'val_subject':str(val_subj),'best_val':best,'test':tm})
        print(json.dumps(folds[-1],indent=2),flush=True)
    yy=np.concatenate(all_y); pp=np.concatenate(all_p)
    report={'task':'emg_physical_action_20class_tcn','dataset':'UCI EMG Physical Action Data Set raw windows','n_windows':int(len(y)),'classes':le.classes_.tolist(),'folds':folds,'oof':{'accuracy':float(accuracy_score(yy,pp)),'balanced_accuracy':float(balanced_accuracy_score(yy,pp)),'macro_f1':float(f1_score(yy,pp,average='macro')),'weighted_f1':float(f1_score(yy,pp,average='weighted')),'confusion_matrix':confusion_matrix(yy,pp).tolist()},'elapsed_sec':round(time.time()-t0,2),'method':'raw_window_tcn_leave_one_subject_out'}
    (OUT/'emg_physical_action_20class_tcn_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['oof'],indent=2),flush=True)
if __name__=='__main__': main()
