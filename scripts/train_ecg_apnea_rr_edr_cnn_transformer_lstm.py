
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
from scripts.train_ecg_apnea_sequence_context_cnn import load_manifest

OUT = Path('/data1/jiahui/biosignal-agent/outputs')
MODEL_PATH = OUT / 'ecg_apnea_rr_edr_cnn_transformer_lstm_model.pt'

def seed_all(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=4096):
        super().__init__(); pe=torch.zeros(max_len,dim); pos=torch.arange(0,max_len,dtype=torch.float32).unsqueeze(1); div=torch.exp(torch.arange(0,dim,2).float()*(-math.log(10000.0)/dim)); pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div[:pe[:,1::2].shape[1]]); self.register_buffer('pe',pe.unsqueeze(0),persistent=False)
    def forward(self,x): return x+self.pe[:,:x.shape[1],:]

class RRTransformerLSTM(nn.Module):
    def __init__(self,in_channels=6,dropout=0.30,d_model=128):
        super().__init__()
        self.cnn=nn.Sequential(nn.Conv1d(in_channels,32,11,padding=5,bias=False),nn.BatchNorm1d(32),nn.GELU(),nn.MaxPool1d(2),nn.Conv1d(32,64,9,padding=4,bias=False),nn.BatchNorm1d(64),nn.GELU(),nn.MaxPool1d(2),nn.Conv1d(64,d_model,7,padding=3,bias=False),nn.BatchNorm1d(d_model),nn.GELU(),nn.MaxPool1d(2))
        self.pos=PositionalEncoding(d_model)
        enc=nn.TransformerEncoderLayer(d_model=d_model,nhead=4,dim_feedforward=256,dropout=dropout,batch_first=True,activation='gelu',norm_first=True)
        self.transformer=nn.TransformerEncoder(enc,num_layers=1)
        self.rnn=nn.LSTM(d_model,64,batch_first=True,bidirectional=True)
        self.head=nn.Sequential(nn.LayerNorm(256),nn.Dropout(dropout),nn.Linear(256,96),nn.GELU(),nn.Dropout(dropout),nn.Linear(96,1))
    def forward(self,x):
        z=self.cnn(x).transpose(1,2); z=self.transformer(self.pos(z)); z,_=self.rnn(z); return self.head(torch.cat([z.mean(1),z.amax(1)],1)).squeeze(-1)

def predict(model,X,device):
    model.eval(); out=[]
    with torch.no_grad():
        for xb in DataLoader(torch.tensor(X,dtype=torch.float32),batch_size=256): out.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    return np.concatenate(out).astype(float)

def train_fold(X,y,tr,va,epochs,seed,device):
    seed_all(seed); model=RRTransformerLSTM(in_channels=X.shape[1]).to(device); pos=max(float((y[tr]==1).sum()),1); neg=max(float((y[tr]==0).sum()),1); crit=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos],dtype=torch.float32,device=device)); opt=torch.optim.AdamW(model.parameters(),lr=7e-4,weight_decay=2e-3); loader=DataLoader(TensorDataset(torch.tensor(X[tr],dtype=torch.float32),torch.tensor(y[tr],dtype=torch.float32)),batch_size=192,shuffle=True); best=None; best_loss=math.inf; stale=0
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

def source_metrics(y,p,sources,t):
    out={}; sources=np.asarray(sources)
    for s in sorted(set(map(str,sources))):
        mask=sources==s; item=met(y[mask],p[mask],t); item['num']=int(mask.sum()); item['positives']=int(y[mask].sum()); out[s]=item
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',default='/data1/jiahui/biosignal-agent/datasets/processed/apnea_ecg_large_manifest.json'); ap.add_argument('--context-radius',type=int,default=10); ap.add_argument('--per-minute-len',type=int,default=64); ap.add_argument('--num-channels',type=int,default=6); ap.add_argument('--epochs',type=int,default=45); ap.add_argument('--seed',type=int,default=59); ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); ap.add_argument('--model-path',type=Path,default=MODEL_PATH); ap.add_argument('--report-path',type=Path,default=OUT/'ecg_apnea_rr_edr_cnn_transformer_lstm_report.json'); args=ap.parse_args(); seed_all(args.seed)
    X,y,groups,sources=load_manifest(args.manifest,args.context_radius,args.per_minute_len,args.num_channels,return_sources=True); print(json.dumps({'num':len(y),'labels':dict(Counter(map(int,y))),'groups':len(set(groups)),'shape':list(X.shape),'context_minutes':2*args.context_radius+1,'device':args.device},indent=2),flush=True)
    proba=np.zeros(len(y)); folds=[]
    for fold,(tr,va) in enumerate(GroupKFold(n_splits=5).split(X,y,groups=groups)):
        model,vl=train_fold(X,y,tr,va,args.epochs,args.seed+fold,args.device); proba[va]=predict(model,X[va],args.device); rep={'fold':fold,'train_size':int(len(tr)),'val_size':int(len(va)),'val_loss':float(vl),'val_label_counts':dict(Counter(map(int,y[va])))}; folds.append(rep); print('fold',fold,rep,flush=True)
    th=float(max(((f1_score(y,proba>=t,zero_division=0),t) for t in np.linspace(0.1,0.9,81)),key=lambda z:z[0])[1]); cv=met(y,proba,th); by_source=source_metrics(y,proba,sources,th); final,_=train_fold(X,y,np.arange(len(y)),None,max(args.epochs,50),args.seed+999,args.device); args.model_path.parent.mkdir(parents=True,exist_ok=True); torch.save({'state_dict':final.cpu().state_dict(),'architecture':'RR_EDR_CNNTransformerLSTM','num_channels':int(X.shape[1]),'context_radius':args.context_radius,'per_minute_len':args.per_minute_len,'threshold':th,'cv_metrics':cv,'source_metrics':by_source,'fold_reports':folds,'label_counts':dict(Counter(map(int,y)))},args.model_path); report={'model_path':str(args.model_path),'num_rows':int(len(y)),'label_counts':dict(Counter(map(int,y))),'context_radius':args.context_radius,'per_minute_len':args.per_minute_len,'threshold':th,'cv_metrics':cv,'source_metrics':by_source,'fold_reports':folds}; args.report_path.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__': main()
