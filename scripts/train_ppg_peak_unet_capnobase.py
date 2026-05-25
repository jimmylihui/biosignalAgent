from __future__ import annotations
import argparse,json,random,sys,re
from pathlib import Path
import numpy as np,pandas as pd,torch
from scipy import signal as scipy_signal
from torch import nn
from torch.utils.data import Dataset,DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from biosignal_agent.tools.common import bandpass_filter
from biosignal_agent.tools.peak_detectors import ppg_multiscale_systolic_peaks, neurokit_nabian2018_peaks
from scripts.train_ppg_peak_unet import PeakUNet, make_label, predict_record

TARGET_FS=125.0

def _parse_sample_list(value):
    if value is None or (isinstance(value,float) and np.isnan(value)): return np.asarray([],dtype=int)
    nums=re.findall(r'-?\d+', str(value))
    return np.asarray([int(x) for x in nums if int(x)>0], dtype=int)

def load_capnobase(raw_dir, target_fs=TARGET_FS):
    raw=Path(raw_dir); rows=[]
    for sig_path in sorted(raw.glob('*_8min_signal.tab')):
        rec=sig_path.name.replace('_signal.tab','')
        label_path=raw/f'{rec}_labels.tab'; param_path=raw/f'{rec}_param.tab'
        if not label_path.exists() or not param_path.exists(): continue
        sig=pd.read_csv(sig_path,sep='\t')
        lab=pd.read_csv(label_path,sep='\t')
        par=pd.read_csv(param_path,sep='\t')
        fs=float(par['samplingrate_pleth'].iloc[0])
        ppg=sig['pleth_y'].to_numpy(float)
        peaks=_parse_sample_list(lab['pleth_peak_x'].iloc[0])
        if len(peaks)<10: continue
        # CapnoBase annotations are sample indices; map by time to target fs.
        ppg_rs=scipy_signal.resample_poly(ppg, int(target_fs), int(fs)).astype(float)
        peaks_rs=np.asarray(sorted(set(int(round((p-1)/fs*target_fs)) for p in peaks)), dtype=int)
        peaks_rs=peaks_rs[(peaks_rs>=0)&(peaks_rs<len(ppg_rs))]
        rows.append({'record':rec,'fs':target_fs,'ppg':ppg_rs,'peaks':peaks_rs,'native_fs':fs,'native_num_peaks':int(len(peaks))})
    return rows

class DirectPeakDataset(Dataset):
    def __init__(self,records,window=1024,stride=256,augment=False):
        self.records=records; self.items=[]; self.augment=augment; self.window=window
        for ri,r in enumerate(records):
            r['label']=make_label(len(r['ppg']),r['peaks'],r['fs'])
            for start in range(0,max(1,len(r['ppg'])-window),stride):
                end=start+window
                if end>len(r['ppg']): continue
                if r['label'][start:end].max()<=0: continue
                self.items.append((ri,start,end))
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        ri,s,e=self.items[i]; r=self.records[ri]; fs=r['fs']
        x=bandpass_filter(r['ppg'][s:e].astype(np.float32),fs,0.4,min(8.0,fs*0.45)).astype(np.float32)
        x=x-np.median(x); x=x/(np.percentile(np.abs(x),95)+1e-6)
        y=r['label'][s:e].astype(np.float32)
        if self.augment:
            x=x*np.random.uniform(0.8,1.2)
            if np.random.rand()<0.6: x=x+np.random.normal(0,0.04,size=x.shape).astype(np.float32)
            if np.random.rand()<0.3:
                t=np.arange(len(x),dtype=np.float32)/fs; x=x+np.random.uniform(-0.12,0.12)*np.sin(2*np.pi*np.random.uniform(0.08,0.35)*t)
        return torch.tensor(x[None,:],dtype=torch.float32),torch.tensor(y[None,:],dtype=torch.float32)

def match_direct(ref,det,fs,tol_s=0.10):
    ref=np.asarray(ref,dtype=int); det=np.asarray(det,dtype=int); tol=int(round(tol_s*fs))
    used=np.zeros(len(det),dtype=bool); matched=0; errs=[]
    for r in ref:
        cand=np.where((np.abs(det-r)<=tol)&(~used))[0]
        if len(cand)==0: continue
        j=cand[np.argmin(np.abs(det[cand]-r))]; used[j]=True; matched+=1; errs.append((det[j]-r)/fs)
    sens=matched/len(ref) if len(ref) else 0.0; ppv=matched/len(det) if len(det) else 0.0
    f1=2*sens*ppv/(sens+ppv) if sens+ppv else 0.0
    return {'matched':int(matched),'reference':int(len(ref)),'detected':int(len(det)),'sensitivity':float(sens),'ppv':float(ppv),'f1':float(f1),'median_abs_timing_error_ms':float(np.median(np.abs(errs))*1000) if errs else None}

def summarize(vals):
    return {'mean_sensitivity':float(np.mean([v['sensitivity'] for v in vals])),'mean_ppv':float(np.mean([v['ppv'] for v in vals])),'mean_f1':float(np.mean([v['f1'] for v in vals])),'total_matched':int(sum(v['matched'] for v in vals)),'total_reference':int(sum(v['reference'] for v in vals)),'total_detected':int(sum(v['detected'] for v in vals)),'median_abs_timing_error_ms':float(np.nanmedian([v['median_abs_timing_error_ms'] for v in vals if v['median_abs_timing_error_ms'] is not None]))}

def eval_model(model,records,device,threshold=0.28,prominence=0.05):
    vals=[]
    for r in records:
        peaks,prob=predict_record(model,r['ppg'],r['fs'],device)
        if threshold!=0.28 or prominence!=0.05:
            peaks,_=scipy_signal.find_peaks(prob,distance=max(1,int(60/220*r['fs'])),height=threshold,prominence=prominence)
        vals.append(match_direct(r['peaks'],peaks,r['fs']))
    return summarize(vals),vals

def eval_baselines(records):
    out={}
    for name,fn in {
        'ppg_multiscale': lambda x,fs: ppg_multiscale_systolic_peaks(x,fs)[0],
        'nabian_on_ppg': lambda x,fs: neurokit_nabian2018_peaks(x,fs,low_hz=0.4,high_hz=min(8.0,fs*0.45),fallback_threshold_scale=0.35)[0],
    }.items():
        vals=[match_direct(r['peaks'],fn(r['ppg'],r['fs']),r['fs']) for r in records]
        out[name]=summarize(vals)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--raw-dir',default='/data1/jiahui/biosignal-agent/datasets/raw/capnobase_benchmark'); ap.add_argument('--epochs',type=int,default=8); ap.add_argument('--out',default='/data1/jiahui/biosignal-agent/outputs/ppg_peak_unet_capnobase.pt'); ap.add_argument('--report',default='/data1/jiahui/biosignal-agent/outputs/ppg_peak_unet_capnobase_report.json'); args=ap.parse_args()
    torch.manual_seed(11); np.random.seed(11); random.seed(11)
    records=load_capnobase(args.raw_dir); random.shuffle(records); n_val=max(8,int(0.2*len(records))); val=records[:n_val]; train=records[n_val:]
    train_ds=DirectPeakDataset(train,augment=True); val_ds=DirectPeakDataset(val,augment=False)
    device='cuda' if torch.cuda.is_available() else 'cpu'; model=PeakUNet().to(device)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.0],device=device)); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    loader=DataLoader(train_ds,batch_size=64,shuffle=True,num_workers=0); history=[]; best=-1
    for ep in range(1,args.epochs+1):
        model.train(); losses=[]
        for x,y in loader:
            x=x.to(device); y=y.to(device); opt.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); opt.step(); losses.append(float(loss.item()))
        metrics,_=eval_model(model,val,device); item={'epoch':ep,'loss':float(np.mean(losses)),**metrics}; history.append(item); print(json.dumps(item),flush=True)
        if metrics['mean_f1']>best:
            best=metrics['mean_f1']; Path(args.out).parent.mkdir(parents=True,exist_ok=True); torch.save({'model_state_dict':model.state_dict(),'architecture':'shallow_1d_unet_peak_segmentation','training_dataset':'capnobase_direct_pleth_peak_labels_resampled_125hz','threshold':0.28,'prominence':0.05,'window':1024,'hop':512,'cv_metrics':metrics},args.out)
    ck=torch.load(args.out,map_location=device,weights_only=False); model.load_state_dict(ck['model_state_dict'])
    val_metrics,val_rows=eval_model(model,val,device); train_metrics,_=eval_model(model,train[:min(12,len(train))],device); baselines=eval_baselines(val)
    report={'model':'shallow_1d_unet_peak_segmentation','labeling':'direct CapnoBase pleth_peak_x PPG peak annotations; no ECG-derived labels; no lag correction','num_records':len(records),'train_records':len(train),'val_records':len(val),'train_windows':len(train_ds),'val_windows':len(val_ds),'target_fs':TARGET_FS,'device':device,'history':history,'val_metrics':val_metrics,'train_subset_metrics':train_metrics,'val_baselines':baselines,'model_out':args.out,'match_tolerance_s':0.10}
    Path(args.report).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
