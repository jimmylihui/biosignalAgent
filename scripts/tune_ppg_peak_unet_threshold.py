from pathlib import Path
import json, sys
import numpy as np, pandas as pd, torch, neurokit2 as nk
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scipy import signal as scipy_signal
from scripts.train_ppg_peak_unet import PeakUNet, predict_record
from biosignal_agent.tools.common import bandpass_filter
from scripts.evaluate_ppg_peak_detectors import estimate_best_lag_match, match_ppg_to_ecg, summarize

def peaks_from_prob(prob, fs, thr, prom):
    peaks,_=scipy_signal.find_peaks(prob, distance=max(1,int(60/220*fs)), height=thr, prominence=prom)
    return peaks.astype(int)

def predict_prob(model, ppg, fs, device, window=1024, hop=512):
    filt=bandpass_filter(ppg, fs, 0.4, min(8.0, fs*0.45)).astype(np.float32)
    filt=filt-np.median(filt); filt=filt/(np.percentile(np.abs(filt),95)+1e-6)
    acc=np.zeros(len(filt),dtype=np.float32); wt=np.zeros(len(filt),dtype=np.float32); win=np.maximum(np.hanning(window).astype(np.float32),0.05)
    starts=list(range(0,max(1,len(filt)-window+1),hop))
    if starts and starts[-1]!=len(filt)-window: starts.append(max(0,len(filt)-window))
    model.eval()
    with torch.no_grad():
        for s in starts:
            seg=filt[s:s+window]
            if len(seg)<window: continue
            pr=torch.sigmoid(model(torch.tensor(seg[None,None,:],device=device,dtype=torch.float32))).cpu().numpy()[0,0]
            acc[s:s+window]+=pr*win; wt[s:s+window]+=win
    return acc/np.maximum(wt,1e-6)

def load_mimic(raw='/data1/jiahui/biosignal-agent/datasets/raw/mimic_perform_af', seconds=60):
    rows=[]
    for path in sorted(Path(raw).rglob('*_data.csv')):
        f=pd.read_csv(path)
        if 'PPG' not in f or 'ECG' not in f: continue
        fs=1.0/float(np.median(np.diff(f['Time'].to_numpy(float)))) if 'Time' in f else 125.0
        n=int(seconds*fs); ppg=f['PPG'].to_numpy(float)[:n]; ecg=f['ECG'].to_numpy(float)[:n]
        try:
            _,info=nk.ecg_peaks(ecg,sampling_rate=fs,method='nabian2018',correct_artifacts=True)
            ecg_peaks=np.asarray(info.get('ECG_R_Peaks',[]),dtype=int)
        except Exception: continue
        rows.append((path.stem.replace('_data',''),ppg,ecg_peaks,fs))
    return rows

def main():
    device='cuda' if torch.cuda.is_available() else 'cpu'; ck=torch.load('/data1/jiahui/biosignal-agent/outputs/ppg_peak_unet.pt',map_location=device,weights_only=False); model=PeakUNet().to(device); model.load_state_dict(ck['model_state_dict'])
    rows=load_mimic(); probs=[]
    for rec,ppg,ecg_peaks,fs in rows:
        probs.append((rec,predict_prob(model,ppg,fs,device),ecg_peaks,fs))
    best=None
    for thr in [0.08,0.12,0.16,0.20,0.24,0.28,0.32,0.36,0.42,0.50]:
      for prom in [0.01,0.03,0.05,0.08,0.12]:
        vals=[]
        for rec,prob,ecg_peaks,fs in probs:
            peaks=peaks_from_prob(prob,fs,thr,prom); vals.append(estimate_best_lag_match(ecg_peaks,peaks,fs))
        f1=float(np.mean([v['f1'] for v in vals])); ppv=float(np.mean([v['ppv'] for v in vals])); sens=float(np.mean([v['sensitivity'] for v in vals])); det=sum(v['detected'] for v in vals)
        item={'thr':thr,'prom':prom,'f1':f1,'ppv':ppv,'sens':sens,'detected':int(det)}
        if best is None or item['f1']>best['f1']: best=item
    print(json.dumps(best,indent=2))
if __name__=='__main__': main()
