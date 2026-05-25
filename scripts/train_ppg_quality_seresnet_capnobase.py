from __future__ import annotations
import argparse,json,random,sys
from pathlib import Path
import numpy as np, torch
from scipy import signal as scipy_signal
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from biosignal_agent.tools.common import bandpass_filter
from scripts.train_ppg_quality_capnobase import build_rows, label_natural_window, local_reference_peaks, augment_poor
from scripts.train_ppg_peak_unet_capnobase import load_capnobase

class QualityWindowDataset(Dataset):
    def __init__(self, rows, augment=False):
        self.rows=rows; self.augment=augment
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        row=self.rows[i]
        x=np.asarray(row['signal'], dtype=np.float32)
        fs=float(row['fs'])
        x=bandpass_filter(x, fs, 0.4, min(8.0, fs*0.45)).astype(np.float32)
        x=x-np.nanmedian(x); x=x/(np.nanpercentile(np.abs(x),95)+1e-6)
        if self.augment and row['y']=='good':
            if np.random.rand()<0.5: x=x+np.random.normal(0,0.03,size=x.shape).astype(np.float32)
            if np.random.rand()<0.25:
                t=np.arange(len(x),dtype=np.float32)/fs
                x=x+np.random.uniform(-0.08,0.08)*np.sin(2*np.pi*np.random.uniform(0.05,0.25)*t)
        d1=np.gradient(x).astype(np.float32)
        d2=np.gradient(d1).astype(np.float32)
        chans=np.stack([x,d1/(np.std(d1)+1e-6),d2/(np.std(d2)+1e-6)],axis=0)
        y=1 if row['y']=='good' else 0
        return torch.tensor(chans,dtype=torch.float32), torch.tensor(y,dtype=torch.float32)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__(); hidden=max(4,channels//reduction)
        self.fc=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(channels,hidden),nn.SiLU(),nn.Linear(hidden,channels),nn.Sigmoid())
    def forward(self,x):
        w=self.fc(x).unsqueeze(-1); return x*w
class SEResBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1, dilation=1, dropout=0.05):
        super().__init__()
        pad=dilation*3
        self.conv=nn.Sequential(nn.Conv1d(c_in,c_out,7,stride=stride,padding=pad,dilation=dilation),nn.BatchNorm1d(c_out),nn.SiLU(),nn.Dropout(dropout),nn.Conv1d(c_out,c_out,5,padding=2),nn.BatchNorm1d(c_out),SEBlock(c_out))
        self.skip=nn.Identity() if c_in==c_out and stride==1 else nn.Sequential(nn.Conv1d(c_in,c_out,1,stride=stride),nn.BatchNorm1d(c_out))
        self.act=nn.SiLU()
    def forward(self,x): return self.act(self.conv(x)+self.skip(x))
class PPGQualitySEResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(3,32,9,padding=4),nn.BatchNorm1d(32),nn.SiLU(),
            SEResBlock(32,48,stride=2,dilation=1),
            SEResBlock(48,64,stride=2,dilation=1),
            SEResBlock(64,96,stride=2,dilation=2),
            SEResBlock(96,128,stride=2,dilation=2),
            SEResBlock(128,160,stride=2,dilation=4),
        )
        self.head=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Dropout(0.25),nn.Linear(160,64),nn.SiLU(),nn.Dropout(0.15),nn.Linear(64,1))
    def forward(self,x): return self.head(self.net(x)).squeeze(-1)

def build_signal_rows(args):
    records=load_capnobase(args.raw_dir, target_fs=args.target_fs)
    rng=np.random.default_rng(args.seed)
    rows=[]; win=int(args.window_s*args.target_fs); step=int(args.step_s*args.target_fs)
    for rec in records:
        ppg=rec['ppg']; fs=rec['fs']; peaks=rec['peaks']
        for start in range(0,max(0,len(ppg)-win+1),step):
            end=start+win; sig=ppg[start:end]; ref=local_reference_peaks(peaks,start,end)
            label,details=label_natural_window(sig,ref,fs)
            if label is None: continue
            rows.append({'signal':sig.astype(np.float32),'fs':fs,'y':label,'group':rec['record'],'source':'natural','record':f"{rec['record']}_{start/fs:.1f}s"})
            if args.augment and label=='good':
                for name,aug in augment_poor(sig,fs,rng):
                    rows.append({'signal':aug.astype(np.float32),'fs':fs,'y':'poor','group':rec['record'],'source':name,'record':f"{rec['record']}_{start/fs:.1f}s_{name}"})
    return rows

def metrics(y, prob, threshold=0.5):
    pred=(np.asarray(prob)>=threshold).astype(int); y=np.asarray(y).astype(int)
    p,r,f,s=precision_recall_fscore_support(y,pred,labels=[0,1],zero_division=0)
    out={'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),'weighted_f1':float(f1_score(y,pred,average='weighted')),'auroc':float(roc_auc_score(y,prob)) if len(set(y))==2 else None,'labels':['poor','good'],'per_class':{'poor':{'precision':float(p[0]),'recall':float(r[0]),'f1':float(f[0]),'support':int(s[0])},'good':{'precision':float(p[1]),'recall':float(r[1]),'f1':float(f[1]),'support':int(s[1])}},'confusion_matrix_labels':['poor','good'],'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist()}
    return out

def train_fold(train_rows, val_rows, args, device):
    model=PPGQualitySEResNet().to(device)
    y_train=np.asarray([1 if r['y']=='good' else 0 for r in train_rows])
    pos=max(1,int(y_train.sum())); neg=max(1,int(len(y_train)-pos)); pos_weight=torch.tensor([neg/pos],device=device,dtype=torch.float32)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    train_loader=DataLoader(QualityWindowDataset(train_rows,augment=True),batch_size=args.batch_size,shuffle=True,num_workers=0)
    val_loader=DataLoader(QualityWindowDataset(val_rows),batch_size=args.batch_size,shuffle=False,num_workers=0)
    best_state=None; best=-1; history=[]
    for ep in range(1,args.epochs+1):
        model.train(); losses=[]
        for x,y in train_loader:
            x=x.to(device); y=y.to(device); opt.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); opt.step(); losses.append(float(loss.item()))
        yv=[]; pv=[]; model.eval()
        with torch.no_grad():
            for x,y in val_loader:
                prob=torch.sigmoid(model(x.to(device))).cpu().numpy(); pv.extend(prob.tolist()); yv.extend(y.numpy().tolist())
        m=metrics(yv,pv); item={'epoch':ep,'loss':float(np.mean(losses)),**m}; history.append(item)
        if m['macro_f1']>best:
            best=m['macro_f1']; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    yv=[]; pv=[]; model.eval()
    with torch.no_grad():
        for x,y in val_loader:
            prob=torch.sigmoid(model(x.to(device))).cpu().numpy(); pv.extend(prob.tolist()); yv.extend(y.numpy().tolist())
    return model, metrics(yv,pv), history, yv, pv

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--raw-dir',default='/data1/jiahui/biosignal-agent/datasets/raw/capnobase_benchmark'); ap.add_argument('--out',default='/data1/jiahui/biosignal-agent/outputs/ppg_quality_seresnet_capnobase.pt'); ap.add_argument('--report',default='/data1/jiahui/biosignal-agent/outputs/ppg_quality_seresnet_capnobase_report.json'); ap.add_argument('--window-s',type=float,default=30.0); ap.add_argument('--step-s',type=float,default=10.0); ap.add_argument('--target-fs',type=float,default=125.0); ap.add_argument('--folds',type=int,default=5); ap.add_argument('--epochs',type=int,default=8); ap.add_argument('--batch-size',type=int,default=96); ap.add_argument('--lr',type=float,default=8e-4); ap.add_argument('--seed',type=int,default=23); ap.add_argument('--augment',action='store_true',default=True); args=ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    rows=build_signal_rows(args); groups=np.asarray([r['group'] for r in rows]); y=np.asarray([1 if r['y']=='good' else 0 for r in rows])
    device='cuda' if torch.cuda.is_available() else 'cpu'; cv=GroupKFold(n_splits=min(args.folds,len(set(groups))))
    fold_reports=[]; all_y=[]; all_p=[]; best_model=None; best_macro=-1
    for fold,(tr,va) in enumerate(cv.split(np.zeros(len(rows)),y,groups),1):
        model,m,h,yv,pv=train_fold([rows[i] for i in tr],[rows[i] for i in va],args,device)
        fold_reports.append({'fold':fold,'metrics':m,'history':h[-3:]}); all_y.extend(yv); all_p.extend(pv)
        print(json.dumps({'fold':fold,'metrics':m}),flush=True)
        if m['macro_f1']>best_macro:
            best_macro=m['macro_f1']; best_model=model
    cv_metrics=metrics(all_y,all_p)
    torch.save({'model_state_dict':best_model.state_dict(),'architecture':'ppg_quality_se_resnet_1d','input_channels':['ppg','first_derivative','second_derivative'],'window_s':args.window_s,'target_fs':args.target_fs,'cv_metrics':cv_metrics,'labeling':'CapnoBase direct pleth_peak_x quality labels plus artifact augmentation'},args.out)
    report={'model':'ppg_quality_se_resnet_1d','training_windows':len(rows),'label_counts':{'poor':int((y==0).sum()),'good':int((y==1).sum())},'groups':int(len(set(groups))),'device':device,'cv_metrics':cv_metrics,'fold_reports':fold_reports,'model_out':args.out,'reference':'CapnoBase direct PPG peak labels; no ECG proxy; no lag correction.'}
    Path(args.report).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
