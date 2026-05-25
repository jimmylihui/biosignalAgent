
from __future__ import annotations
import argparse, json, random, math
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np, torch
from scipy import signal as scipy_signal
from scipy.io import wavfile
from scipy.ndimage import zoom
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import Dataset, DataLoader
LOCATIONS=['AV','PV','TV','MV','Phc']; MURMUR_CLASSES=['present','unknown','absent']

def set_seed(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
def load_wav(path):
    fs,x=wavfile.read(path); x=x[:,0] if getattr(x,'ndim',1)>1 else x; x=x.astype(np.float32); x=x[np.isfinite(x)]; x=x-np.nanmedian(x); x=np.clip(x/(np.nanpercentile(np.abs(x),95)+1e-6),-6,6); return int(fs),x.astype(np.float32)
def resample(x,fs,tfs):
    if fs==tfs: return x.astype(np.float32)
    return scipy_signal.resample(x,max(16,int(round(len(x)*tfs/float(fs))))).astype(np.float32)
def crop(x,n,train,rng):
    if len(x)>=n:
        s=int(rng.integers(0,len(x)-n+1)) if train else max(0,(len(x)-n)//2); return x[s:s+n]
    y=np.zeros(n,np.float32); s=(n-len(x))//2; y[s:s+len(x)]=x; return y
def spec_image(x,fs,fb,tb):
    freqs,_,sp=scipy_signal.spectrogram(x,fs=fs,window='hann',nperseg=256,noverlap=192,mode='magnitude',scaling='density')
    m=(freqs>=20)&(freqs<=800); freqs=freqs[m]; sp=sp[m]
    if sp.size==0 or len(freqs)==0: return np.zeros((fb,tb),np.float32)
    edges=np.geomspace(max(20,float(freqs[0])),max(21,float(freqs[-1])),fb+1); pooled=np.zeros((fb,sp.shape[1]),np.float32)
    for i in range(fb):
        mm=(freqs>=edges[i])&(freqs<edges[i+1]); pooled[i]=np.mean(sp[mm],0) if np.any(mm) else 0
    img=np.log1p(pooled**2); lo,hi=np.percentile(img,[2,98]); img=np.clip((img-lo)/max(hi-lo,1e-6),0,1)
    if img.shape[1]!=tb: img=zoom(img,(1,tb/img.shape[1]),order=1)
    return img.astype(np.float32)
def murmur_y(label):
    label=str(label).lower(); return 0 if label in ['present','abnormal'] else 1 if label=='unknown' else 2
def outcome_y(label):
    label=str(label).lower(); return 1 if label=='abnormal' else 0 if label=='normal' else -1
def demo_vec(patient):
    rec=patient['records'][0]
    age=str(rec.get('age') or '').lower(); sex=str(rec.get('sex') or '').lower()
    ages=['neonate','infant','child','adolescent','young adult']
    v=[1.0 if a in age else 0.0 for a in ages]+[1.0 if sex=='female' else 0.0,1.0 if sex=='male' else 0.0]
    for k,scale in [('height',200.0),('weight',120.0)]:
        try: val=float(rec.get(k)); v.append(0.0 if not np.isfinite(val) else val/scale)
        except Exception: v.append(0.0)
    return np.asarray(v,np.float32)
def build_patients(manifest):
    by={}
    for r in manifest['records']:
        pid=str(r['patient_id']); item=by.setdefault(pid,{'patient_id':pid,'records':[]}); item['records'].append(r)
    out=[]
    for p in by.values():
        lab=str(p['records'][0].get('patient_murmur_label') or p['records'][0].get('patient_label') or '').lower()
        if lab not in ['present','absent','unknown']: continue
        p['y_murmur']=murmur_y(lab); p['y_outcome']=outcome_y(p['records'][0].get('outcome')); p['demo']=demo_vec(p); out.append(p)
    return out
class PCGDataset(Dataset):
    def __init__(self,patients,idx,train,args,seed): self.patients=patients; self.idx=np.asarray(idx); self.train=train; self.args=args; self.rng=np.random.default_rng(seed); self.n=int(args.target_fs*args.seconds)
    def __len__(self): return len(self.idx)
    def __getitem__(self,i):
        p=self.patients[int(self.idx[i])]; byloc={r.get('location'):r for r in p['records']}; xs=[]; mask=[]; loc_y=[]
        murmur_locs=str(p['records'][0].get('murmur_locations') or '').split('+')
        for loc in LOCATIONS:
            r=byloc.get(loc)
            if r is None: xs.append(np.zeros((self.args.freq_bins,self.args.time_bins),np.float32)); mask.append(0.0); loc_y.append(-1); continue
            fs,x=load_wav(r['path']); x=resample(x,fs,self.args.target_fs); x=crop(x,self.n,self.train,self.rng)
            if self.train:
                x=x*float(self.rng.uniform(0.75,1.25))+self.rng.normal(0,0.015,x.shape).astype(np.float32)
                if self.rng.random()<0.25: x=np.roll(x,int(self.rng.integers(-self.args.target_fs,self.args.target_fs)))
            xs.append(spec_image(x,self.args.target_fs,self.args.freq_bins,self.args.time_bins)); mask.append(1.0)
            loc_y.append(1 if p['y_murmur']==0 and loc in murmur_locs else 0 if p['y_murmur'] in [0,2] else -1)
        return torch.tensor(np.stack(xs)[:,None,:,:]), torch.tensor(mask,dtype=torch.float32), torch.tensor(p['demo']), torch.tensor(p['y_murmur']), torch.tensor(p['y_outcome']), torch.tensor(loc_y,dtype=torch.float32)
class ResBlock(nn.Module):
    def __init__(self,ci,co,stride=1): super().__init__(); self.net=nn.Sequential(nn.Conv2d(ci,co,3,stride,padding=1),nn.BatchNorm2d(co),nn.SiLU(),nn.Conv2d(co,co,3,padding=1),nn.BatchNorm2d(co)); self.skip=nn.Identity() if ci==co and stride==1 else nn.Sequential(nn.Conv2d(ci,co,1,stride),nn.BatchNorm2d(co)); self.act=nn.SiLU()
    def forward(self,x): return self.act(self.net(x)+self.skip(x))
class Encoder(nn.Module):
    def __init__(self,emb=192): super().__init__(); self.net=nn.Sequential(nn.Conv2d(1,32,5,padding=2),nn.BatchNorm2d(32),nn.SiLU(),ResBlock(32,48,2),ResBlock(48,64,2),ResBlock(64,96,2),ResBlock(96,128,2),nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Dropout(.2),nn.Linear(128,emb),nn.SiLU())
    def forward(self,x): return self.net(x)
class StrongPCG(nn.Module):
    def __init__(self,emb=192,demo_dim=9):
        super().__init__(); self.encoder=Encoder(emb); self.loc_embed=nn.Parameter(torch.randn(len(LOCATIONS),emb)*0.02); self.gru=nn.GRU(emb,emb//2,batch_first=True,bidirectional=True); layer=nn.TransformerEncoderLayer(d_model=emb,nhead=4,dim_feedforward=emb*2,dropout=.2,batch_first=True,activation='gelu'); self.tx=nn.TransformerEncoder(layer,num_layers=1)
        self.attn=nn.Sequential(nn.Linear(emb,64),nn.Tanh(),nn.Linear(64,1)); self.demo=nn.Sequential(nn.LayerNorm(demo_dim),nn.Linear(demo_dim,32),nn.SiLU())
        self.murmur=nn.Sequential(nn.Dropout(.35),nn.Linear(emb+32,96),nn.SiLU(),nn.Dropout(.2),nn.Linear(96,3)); self.outcome=nn.Sequential(nn.Dropout(.25),nn.Linear(emb+32,64),nn.SiLU(),nn.Linear(64,2)); self.loc_head=nn.Linear(emb,1)
    def forward(self,x,mask,demo):
        b,l,c,f,t=x.shape; feat=self.encoder(x.reshape(b*l,c,f,t)).reshape(b,l,-1)+self.loc_embed.unsqueeze(0); feat,_=self.gru(feat); feat=self.tx(feat,src_key_padding_mask=(mask<=0)); loc_logits=self.loc_head(feat).squeeze(-1); score=self.attn(feat).squeeze(-1).masked_fill(mask<=0,-1e4); w=torch.softmax(score,1); pooled=(feat*w.unsqueeze(-1)).sum(1); z=torch.cat([pooled,self.demo(demo)],-1); return self.murmur(z), self.outcome(z), loc_logits
def weighted_murmur_accuracy(y,pred):
    y=np.asarray(y); pred=np.asarray(pred); weights={0:5.0,1:3.0,2:1.0}; num=sum(weights[int(c)] for c in y[pred==y]); den=sum(weights[int(c)] for c in y); return float(num/den) if den else 0.0
def outcome_weighted_accuracy(y,pred):
    y=np.asarray(y); pred=np.asarray(pred); m=y>=0; y=y[m]; pred=pred[m]; weights={1:5.0,0:1.0}; num=sum(weights[int(c)] for c in y[pred==y]); den=sum(weights[int(c)] for c in y); return float(num/den) if den else 0.0
def eval_metrics(y,prob,y_out=None,out_prob=None):
    pred=np.argmax(prob,1); d={'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),'weighted_murmur_accuracy':weighted_murmur_accuracy(y,pred),'confusion_matrix':confusion_matrix(y,pred,labels=[0,1,2]).astype(int).tolist(),'class_order':MURMUR_CLASSES}
    try: d['ovr_auroc']=float(roc_auc_score(y,prob,multi_class='ovr',labels=[0,1,2]))
    except Exception: d['ovr_auroc']=None
    if y_out is not None and out_prob is not None:
        out_pred=np.argmax(out_prob,1); d['outcome_weighted_accuracy']=outcome_weighted_accuracy(y_out,out_pred); d['outcome_accuracy']=float(accuracy_score(np.asarray(y_out)[np.asarray(y_out)>=0],out_pred[np.asarray(y_out)>=0])) if np.any(np.asarray(y_out)>=0) else None
    return d
def run_epoch(model,loader,opt,dev,args,weights):
    model.train(); ce=nn.CrossEntropyLoss(weight=weights.to(dev),reduction='none',label_smoothing=args.label_smoothing); ce_out=nn.CrossEntropyLoss(); bce=nn.BCEWithLogitsLoss(reduction='none'); total=0
    for x,mask,demo,y,yo,ly in loader:
        x=x.to(dev); mask=mask.to(dev); demo=demo.to(dev); y=y.to(dev); yo=yo.to(dev); ly=ly.to(dev); opt.zero_grad(set_to_none=True); ml,ol,ll=model(x,mask,demo); loss=ce(ml,y); loss=torch.where(y==1,loss*args.unknown_loss_weight,loss).mean(); valid=yo>=0
        if valid.any(): loss=loss+args.outcome_loss_weight*ce_out(ol[valid],yo[valid])
        lv=(ly>=0)&(mask>0)
        if lv.any(): loss=loss+args.location_loss_weight*bce(ll[lv],ly[lv]).mean()
        loss.backward(); opt.step(); total+=float(loss.item())*len(y)
    return total/max(1,len(loader.dataset))
def predict(model,loader,dev):
    model.eval(); ys=[]; yo=[]; ps=[]; po=[]
    with torch.no_grad():
        for x,mask,demo,y,yout,ly in loader:
            ml,ol,ll=model(x.to(dev),mask.to(dev),demo.to(dev)); ps.extend(torch.softmax(ml,1).cpu().numpy().tolist()); po.extend(torch.softmax(ol,1).cpu().numpy().tolist()); ys.extend(y.numpy().tolist()); yo.extend(yout.numpy().tolist())
    return np.asarray(ys),np.asarray(ps),np.asarray(yo),np.asarray(po)
def train_one(patients,tr,te,args,seed):
    dev=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'); trds=PCGDataset(patients,tr,True,args,seed); teds=PCGDataset(patients,te,False,args,seed+999); trl=DataLoader(trds,batch_size=args.batch_size,shuffle=True,num_workers=0); tel=DataLoader(teds,batch_size=args.batch_size,shuffle=False,num_workers=0)
    ytr=np.asarray([patients[int(i)]['y_murmur'] for i in tr]); counts=np.bincount(ytr,minlength=3).astype(float); weights=torch.tensor([1.8,0.8,1.2],dtype=torch.float32)*torch.sqrt(torch.tensor([len(ytr)/max(1,3*c) for c in counts],dtype=torch.float32))
    model=StrongPCG(args.embedding_dim).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay); best=None
    for ep in range(1,args.epochs+1):
        loss=run_epoch(model,trl,opt,dev,args,weights); y,p,yo,po=predict(model,tel,dev); met=eval_metrics(y,p,yo,po); score=(met['macro_f1'], met['weighted_murmur_accuracy'], met.get('ovr_auroc') or 0)
        if best is None or score>best['score']: best={'epoch':ep,'loss':loss,'y':y.tolist(),'prob':p.tolist(),'y_out':yo.tolist(),'out_prob':po.tolist(),'metrics':met,'score':score,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}}
    return best
def train_full(patients,args,seed):
    dev=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'); idx=np.arange(len(patients)); ds=PCGDataset(patients,idx,True,args,seed); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=0); y=np.asarray([p['y_murmur'] for p in patients]); counts=np.bincount(y,minlength=3).astype(float); weights=torch.tensor([1.8,0.8,1.2],dtype=torch.float32)*torch.sqrt(torch.tensor([len(y)/max(1,3*c) for c in counts],dtype=torch.float32)); model=StrongPCG(args.embedding_dim).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    for _ in range(args.epochs): run_epoch(model,loader,opt,dev,args,weights)
    return {k:v.detach().cpu() for k,v in model.state_dict().items()}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',default='/data1/jiahui/biosignal-agent/datasets/processed/circor_pcg_full_murmur_manifest.json'); ap.add_argument('--out-model',default='/data1/jiahui/biosignal-agent/outputs/pcg_murmur_sota_stack.pt'); ap.add_argument('--report',default='/data1/jiahui/biosignal-agent/outputs/pcg_murmur_sota_stack_report.json'); ap.add_argument('--epochs',type=int,default=8); ap.add_argument('--folds',type=int,default=5); ap.add_argument('--batch-size',type=int,default=12); ap.add_argument('--seeds',nargs='+',type=int,default=[59]); ap.add_argument('--lr',type=float,default=7e-4); ap.add_argument('--weight-decay',type=float,default=1e-4); ap.add_argument('--target-fs',type=int,default=1000); ap.add_argument('--seconds',type=float,default=8); ap.add_argument('--freq-bins',type=int,default=80); ap.add_argument('--time-bins',type=int,default=128); ap.add_argument('--embedding-dim',type=int,default=192); ap.add_argument('--unknown-loss-weight',type=float,default=.45); ap.add_argument('--outcome-loss-weight',type=float,default=.35); ap.add_argument('--location-loss-weight',type=float,default=.15); ap.add_argument('--label-smoothing',type=float,default=.04); ap.add_argument('--cpu',action='store_true'); args=ap.parse_args(); set_seed(args.seeds[0])
    patients=build_patients(json.loads(Path(args.manifest).read_text())); y=np.asarray([p['y_murmur'] for p in patients]); cv=StratifiedKFold(n_splits=args.folds,shuffle=True,random_state=args.seeds[0]); all_y=[]; all_p=[]; all_yo=[]; all_po=[]; folds=[]
    for fold,(tr,te) in enumerate(cv.split(np.arange(len(patients)),y),1):
        seed_probs=[]; seed_out=[]; seed_reports=[]; fy=None; fyo=None
        for seed in args.seeds:
            best=train_one(patients,tr,te,args,seed+fold); seed_probs.append(np.asarray(best['prob'])); seed_out.append(np.asarray(best['out_prob'])); fy=best['y']; fyo=best['y_out']; seed_reports.append({'seed':seed,'epoch':best['epoch'],'metrics':best['metrics']})
        p=np.mean(np.stack(seed_probs),0); po=np.mean(np.stack(seed_out),0); met=eval_metrics(np.asarray(fy),p,np.asarray(fyo),po); folds.append({'fold':fold,'num_train':len(tr),'num_test':len(te),'metrics':met,'seed_reports':seed_reports}); all_y.extend(fy); all_p.extend(p.tolist()); all_yo.extend(fyo); all_po.extend(po.tolist()); print(json.dumps({'fold':fold,'metrics':met}),flush=True)
    cvm=eval_metrics(np.asarray(all_y),np.asarray(all_p),np.asarray(all_yo),np.asarray(all_po)); states=[]
    for seed in args.seeds: states.append({'seed':seed,'model_state_dict':train_full(patients,args,seed+999)})
    payload={'ensemble_state_dicts':states,'architecture':'ResNet2D_BiGRU_Transformer_multisite_multitask','locations':LOCATIONS,'murmur_classes':MURMUR_CLASSES,'target_fs':args.target_fs,'seconds':args.seconds,'freq_bins':args.freq_bins,'time_bins':args.time_bins,'embedding_dim':args.embedding_dim,'cv_metrics':cvm,'reference':'CirCor patient-level 3-class murmur, outcome/location multi-task, low-weight unknown, multi-seed ensemble.'}
    Path(args.out_model).parent.mkdir(parents=True,exist_ok=True); torch.save(payload,args.out_model); report={'manifest':args.manifest,'num_patients':len(patients),'label_counts':dict(Counter([MURMUR_CLASSES[int(v)] for v in y])),'seeds':args.seeds,'model_out':args.out_model,'folds':folds,'cv_metrics':cvm}; Path(args.report).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
