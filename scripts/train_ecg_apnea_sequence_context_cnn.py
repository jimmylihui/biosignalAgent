from __future__ import annotations
import argparse, json, math, random, sys
from collections import Counter
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.tools.common import load_csv_signal
from scripts.train_ecg_apnea_context_rr_edr_cnn import rr_edr_image

OUT=Path('/data1/jiahui/biosignal-agent/outputs')
MODEL_PATH=OUT/'ecg_apnea_rr_edr_sequence_context_cnn_model.pt'

def seed_all(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def build_minute_features(manifest, per_minute_len, num_channels):
    minute_features=[]; y=[]; groups=[]; records=[]; minutes=[]
    for i,r in enumerate(manifest['records'], start=1):
        if i % 250 == 0: print(f'features {i}', flush=True)
        d=load_csv_signal(r['path'], float(r['sampling_rate']), None)
        minute_features.append(rr_edr_image(d.values, d.sampling_rate, per_minute_len, num_channels))
        y.append(1 if r['label']=='apnea' else 0); groups.append(str(r['record'])); records.append(str(r['record'])); minutes.append(int(r['minute']))
    return np.asarray(minute_features,np.float32), np.asarray(y,np.int64), groups, records, np.asarray(minutes,np.int64)

def load_manifest(path, context_radius=10, per_minute_len=64, num_channels=6, return_sources=False):
    m=json.load(open(path))
    F,y,groups,records,minutes=build_minute_features(m, per_minute_len, num_channels)
    index={(records[i], int(minutes[i])): i for i in range(len(records))}
    X=[]
    for i,(rec,minute) in enumerate(zip(records,minutes)):
        parts=[]
        for off in range(-context_radius, context_radius+1):
            j=index.get((rec, int(minute)+off), i)
            parts.append(F[j])
        X.append(np.concatenate(parts, axis=1))
    sources=[r.get('source', m.get('dataset','unknown')) for r in m['records']]
    if return_sources:
        return np.asarray(X,np.float32), y, groups, sources
    return np.asarray(X,np.float32), y, groups

class SeqContextCNN(nn.Module):
    def __init__(self, in_channels=6, dropout=0.35):
        super().__init__()
        self.cnn=nn.Sequential(
            nn.Conv1d(in_channels,32,11,padding=5,bias=False),nn.BatchNorm1d(32),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(32,64,9,padding=4,bias=False),nn.BatchNorm1d(64),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(64,96,7,padding=3,bias=False),nn.BatchNorm1d(96),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(96,128,5,padding=2,bias=False),nn.BatchNorm1d(128),nn.ReLU())
        self.rnn=nn.LSTM(128,64,batch_first=True,bidirectional=True)
        self.head=nn.Sequential(nn.LayerNorm(256),nn.Dropout(dropout),nn.Linear(256,96),nn.ReLU(),nn.Dropout(dropout),nn.Linear(96,1))
    def forward(self,x):
        z=self.cnn(x).transpose(1,2); z,_=self.rnn(z)
        pooled=torch.cat([z.mean(1), z.amax(1)], dim=1)
        return self.head(pooled).squeeze(-1)

def predict(model,X,device):
    model.eval(); out=[]
    with torch.no_grad():
        for xb in DataLoader(torch.tensor(X,dtype=torch.float32),batch_size=256):
            out.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    return np.concatenate(out).astype(float)

def train_fold(X,y,tr,va,epochs,seed,device):
    seed_all(seed); model=SeqContextCNN(in_channels=X.shape[1]).to(device)
    pos=max(float((y[tr]==1).sum()),1); neg=max(float((y[tr]==0).sum()),1)
    crit=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos],dtype=torch.float32,device=device))
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=2e-3)
    loader=DataLoader(TensorDataset(torch.tensor(X[tr],dtype=torch.float32),torch.tensor(y[tr],dtype=torch.float32)),batch_size=192,shuffle=True)
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
        if stale>=7: break
    if best is not None: model.load_state_dict(best)
    return model,best_loss

def met(y,p,t):
    pred=(p>=t).astype(int)
    d={'threshold':float(t),'accuracy':float(accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'average_precision':float(average_precision_score(y,p))}
    try: d['roc_auc']=float(roc_auc_score(y,p))
    except Exception: d['roc_auc']=0.0
    return d



def source_metrics(y, p, sources, threshold):
    out = {}
    sources = np.asarray(sources)
    for source in sorted(set(map(str, sources))):
        mask = sources == source
        if int(mask.sum()) == 0:
            continue
        yy = y[mask]
        pp = p[mask]
        item = met(yy, pp, threshold)
        item['num'] = int(mask.sum())
        item['positives'] = int(yy.sum())
        try:
            item['roc_auc'] = float(roc_auc_score(yy, pp))
        except Exception:
            item['roc_auc'] = 0.0
        out[source] = item
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',default='/data1/jiahui/biosignal-agent/datasets/processed/apnea_ecg_large_manifest.json')
    ap.add_argument('--context-radius',type=int,default=10)
    ap.add_argument('--per-minute-len',type=int,default=64)
    ap.add_argument('--num-channels',type=int,default=6)
    ap.add_argument('--epochs',type=int,default=40)
    ap.add_argument('--seed',type=int,default=43)
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--model-path',type=Path,default=MODEL_PATH)
    args=ap.parse_args(); seed_all(args.seed)
    X,y,groups,sources=load_manifest(args.manifest,args.context_radius,args.per_minute_len,args.num_channels,return_sources=True)
    print(json.dumps({'num':len(y),'labels':dict(Counter(map(int,y))),'groups':len(set(groups)),'shape':list(X.shape),'context_minutes':2*args.context_radius+1},indent=2), flush=True)
    proba=np.zeros(len(y)); folds=[]
    for fold,(tr,va) in enumerate(GroupKFold(n_splits=5).split(X,y,groups=groups)):
        model,vl=train_fold(X,y,tr,va,args.epochs,args.seed+fold,args.device)
        proba[va]=predict(model,X[va],args.device)
        rep={'fold':fold,'train_size':int(len(tr)),'val_size':int(len(va)),'val_loss':float(vl),'val_label_counts':dict(Counter(map(int,y[va])))}
        folds.append(rep); print('fold',fold,rep,flush=True)
    th=float(max(((f1_score(y,proba>=t,zero_division=0),t) for t in np.linspace(0.1,0.9,81)),key=lambda z:z[0])[1]); cv=met(y,proba,th); by_source=source_metrics(y,proba,sources,th)
    final,_=train_fold(X,y,np.arange(len(y)),None,max(args.epochs,45),args.seed+999,args.device)
    args.model_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':final.cpu().state_dict(),'architecture':'SeqContextCNN_BiLSTM','num_channels':int(X.shape[1]),'context_radius':args.context_radius,'per_minute_len':args.per_minute_len,'threshold':th,'cv_metrics':cv,'source_metrics':by_source,'fold_reports':folds,'label_counts':dict(Counter(map(int,y)))},args.model_path)
    report={'model_path':str(args.model_path),'num_rows':int(len(y)),'label_counts':dict(Counter(map(int,y))),'context_radius':args.context_radius,'per_minute_len':args.per_minute_len,'threshold':th,'cv_metrics':cv,'source_metrics':by_source,'fold_reports':folds}
    (OUT/'ecg_apnea_rr_edr_sequence_context_cnn_train_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__': main()
