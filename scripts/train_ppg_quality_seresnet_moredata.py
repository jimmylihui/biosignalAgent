from __future__ import annotations
import argparse,json,random,sys
from pathlib import Path
import numpy as np, torch
from scipy import signal as scipy_signal
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from biosignal_agent.tools.common import bandpass_filter
from scripts.train_ppg_quality_capnobase import label_natural_window, local_reference_peaks, augment_poor
from scripts.train_ppg_peak_unet_capnobase import load_capnobase
from scripts.train_ppg_quality_seresnet_capnobase import PPGQualitySEResNet, metrics

TARGET_FS=64.0
TARGET_LEN=1920

def resample_to_64(sig, fs):
    sig=np.asarray(sig,dtype=np.float32)
    if abs(fs-TARGET_FS)>1e-3:
        sig=scipy_signal.resample(sig,TARGET_LEN).astype(np.float32)
    elif len(sig)!=TARGET_LEN:
        sig=scipy_signal.resample(sig,TARGET_LEN).astype(np.float32)
    return sig

def load_segade(root, good_thr=0.10, poor_thr=0.30):
    root=Path(root); rows=[]
    specs=[('TROIKA_channel_1/processed_dataset','segade_troika','train'),('WESAD_all/processed_dataset','segade_wesad','train'),('new_PPG_DaLiA_train/processed_dataset','segade_dalia','train'),('new_PPG_DaLiA_test/processed_dataset','segade_dalia','test')]
    skipped=0
    for rel,src,split in specs:
        x=np.load(root/rel/'scaled_ppgs.npy')
        y=np.load(root/rel/'seg_labels.npy')
        for i in range(len(x)):
            frac=float(np.mean(y[i]))
            if frac<=good_thr: lab='good'
            elif frac>=poor_thr: lab='poor'
            else:
                skipped+=1; continue
            rows.append({'signal':x[i].astype(np.float32),'fs':TARGET_FS,'y':lab,'group':f'{src}_{i}','source':src,'split':split,'artifact_fraction':frac})
    return rows, skipped

def load_capnobase_rows(raw_dir, seed=23, max_aug_per_good=3):
    rng=np.random.default_rng(seed); rows=[]; records=load_capnobase(raw_dir,target_fs=125.0)
    win=int(30*125); step=int(10*125)
    for rec in records:
        ppg=rec['ppg']; fs=rec['fs']; peaks=rec['peaks']
        for start in range(0,max(0,len(ppg)-win+1),step):
            end=start+win; sig=ppg[start:end]; ref=local_reference_peaks(peaks,start,end)
            label,_=label_natural_window(sig,ref,fs)
            if label is None: continue
            rows.append({'signal':resample_to_64(sig,fs),'fs':TARGET_FS,'y':label,'group':rec['record'],'source':'capnobase_direct','split':'train','artifact_fraction':0.0 if label=='good' else 1.0})
            if label=='good':
                for name,aug in augment_poor(sig,fs,rng)[:max_aug_per_good]:
                    rows.append({'signal':resample_to_64(aug,fs),'fs':TARGET_FS,'y':'poor','group':rec['record'],'source':f'capnobase_aug_{name}','split':'train','artifact_fraction':1.0})
    return rows

class RowsDataset(Dataset):
    def __init__(self, rows, augment=False): self.rows=rows; self.augment=augment
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        row=self.rows[i]; x=np.asarray(row['signal'],dtype=np.float32)
        x=x-np.nanmedian(x); x=x/(np.nanpercentile(np.abs(x),95)+1e-6)
        if self.augment and row['y']=='good':
            if np.random.rand()<0.4: x=x+np.random.normal(0,0.025,size=x.shape).astype(np.float32)
        d1=np.gradient(x).astype(np.float32); d2=np.gradient(d1).astype(np.float32)
        chans=np.stack([x,d1/(np.std(d1)+1e-6),d2/(np.std(d2)+1e-6)],axis=0)
        y=1 if row['y']=='good' else 0
        return torch.tensor(chans,dtype=torch.float32), torch.tensor(y,dtype=torch.float32)

def evaluate(model, rows, device, batch=256):
    loader=DataLoader(RowsDataset(rows),batch_size=batch,shuffle=False,num_workers=0)
    yy=[]; pp=[]; model.eval()
    with torch.no_grad():
        for x,y in loader:
            prob=torch.sigmoid(model(x.to(device))).cpu().numpy(); pp.extend(prob.tolist()); yy.extend(y.numpy().tolist())
    return metrics(yy,pp), yy, pp

def train(args):
    segade, skipped=load_segade(args.segade_dir,args.good_thr,args.poor_thr)
    cap=load_capnobase_rows(args.capnobase_dir,args.seed,args.max_capno_aug) if args.use_capnobase else []
    train_rows=[r for r in segade if r['split']=='train']+cap
    test_rows=[r for r in segade if r['split']=='test']
    rng=random.Random(args.seed); rng.shuffle(train_rows)
    n_val=max(1,int(len(train_rows)*0.15)); val_rows=train_rows[:n_val]; tr_rows=train_rows[n_val:]
    device='cuda' if torch.cuda.is_available() else 'cpu'; model=PPGQualitySEResNet().to(device)
    y_train=np.asarray([1 if r['y']=='good' else 0 for r in tr_rows]); pos=max(1,int(y_train.sum())); neg=max(1,int(len(y_train)-pos))
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos],device=device,dtype=torch.float32))
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    loader=DataLoader(RowsDataset(tr_rows,augment=True),batch_size=args.batch_size,shuffle=True,num_workers=0)
    best=-1; best_state=None; hist=[]
    for ep in range(1,args.epochs+1):
        model.train(); losses=[]
        for x,y in loader:
            x=x.to(device); y=y.to(device); opt.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); opt.step(); losses.append(float(loss.item()))
        val_m,_,_=evaluate(model,val_rows,device,args.batch_size)
        test_m,_,_=evaluate(model,test_rows,device,args.batch_size)
        item={'epoch':ep,'loss':float(np.mean(losses)),'val_macro_f1':val_m['macro_f1'],'val_good_f1':val_m['per_class']['good']['f1'],'test_macro_f1':test_m['macro_f1'],'test_good_f1':test_m['per_class']['good']['f1'],'test_auroc':test_m['auroc']}
        hist.append(item); print(json.dumps(item),flush=True)
        if val_m['macro_f1']>best:
            best=val_m['macro_f1']; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    val_m,_,_=evaluate(model,val_rows,device,args.batch_size); test_m,_,_=evaluate(model,test_rows,device,args.batch_size)
    out={'model_state_dict':model.state_dict(),'architecture':'ppg_quality_se_resnet_1d','input_channels':['ppg','first_derivative','second_derivative'],'target_fs':TARGET_FS,'window_samples':TARGET_LEN,'cv_metrics':test_m,'labeling':'Segade DaLiA/TROIKA/WESAD per-sample artifact labels converted to window quality plus optional CapnoBase direct-label augmentation'}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); torch.save(out,args.out)
    report={'model':'ppg_quality_se_resnet_1d_moredata','segade_rows':len(segade),'segade_skipped_ambiguous':skipped,'capnobase_rows':len(cap),'train_rows':len(tr_rows),'val_rows':len(val_rows),'test_rows':len(test_rows),'label_counts_train':{lab:sum(r['y']==lab for r in tr_rows) for lab in ['poor','good']},'label_counts_test':{lab:sum(r['y']==lab for r in test_rows) for lab in ['poor','good']},'device':device,'history':hist,'val_metrics':val_m,'test_metrics_segade_dalia':test_m,'model_out':args.out,'decision_note':'Compare against CapnoBase direct-label RF before integrating.'}
    Path(args.report).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--segade-dir',default='/data1/jiahui/biosignal-agent/datasets/raw/segade_ppg_quality'); ap.add_argument('--capnobase-dir',default='/data1/jiahui/biosignal-agent/datasets/raw/capnobase_benchmark'); ap.add_argument('--out',default='/data1/jiahui/biosignal-agent/outputs/ppg_quality_seresnet_moredata.pt'); ap.add_argument('--report',default='/data1/jiahui/biosignal-agent/outputs/ppg_quality_seresnet_moredata_report.json'); ap.add_argument('--epochs',type=int,default=10); ap.add_argument('--batch-size',type=int,default=256); ap.add_argument('--lr',type=float,default=8e-4); ap.add_argument('--seed',type=int,default=29); ap.add_argument('--good-thr',type=float,default=0.10); ap.add_argument('--poor-thr',type=float,default=0.30); ap.add_argument('--use-capnobase',action='store_true',default=True); ap.add_argument('--max-capno-aug',type=int,default=3); args=ap.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed); train(args)
if __name__=='__main__': main()
