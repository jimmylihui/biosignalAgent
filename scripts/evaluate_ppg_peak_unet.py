from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np, pandas as pd, torch, neurokit2 as nk
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.tools.peak_detectors import neurokit_nabian2018_peaks, ppg_multiscale_systolic_peaks
from scripts.evaluate_ppg_peak_detectors import match_ppg_to_ecg, estimate_best_lag_match, summarize
from scripts.train_ppg_peak_unet import PeakUNet, predict_record

def record_paths(raw_dir: Path): return sorted(raw_dir.rglob('*_data.csv'))

def evaluate_record(path: Path, seconds: float, model, device):
    frame=pd.read_csv(path)
    if 'PPG' not in frame.columns or 'ECG' not in frame.columns: return None
    fs=1.0/float(np.median(np.diff(frame['Time'].to_numpy(float)))) if 'Time' in frame.columns else 125.0
    n=int(seconds*fs); ppg=frame['PPG'].to_numpy(float)[:n]; ecg=frame['ECG'].to_numpy(float)[:n]
    if len(ppg)<fs*20 or len(ecg)<fs*20: return None
    try:
        _,info=nk.ecg_peaks(ecg,sampling_rate=fs,method='nabian2018',correct_artifacts=True)
        ecg_peaks=np.asarray(info.get('ECG_R_Peaks',[]),dtype=int)
    except Exception: return None
    rows={'record':path.stem.replace('_data',''),'sampling_rate':fs,'ecg_peaks':int(len(ecg_peaks))}
    detectors={
      'nabian_on_ppg': lambda x: neurokit_nabian2018_peaks(x, fs, low_hz=0.4, high_hz=min(8.0, fs*0.45), fallback_threshold_scale=0.35)[0],
      'ppg_multiscale': lambda x: ppg_multiscale_systolic_peaks(x, fs)[0],
      'ppg_peak_unet': lambda x: predict_record(model, x, fs, device)[0],
    }
    for name,fn in detectors.items():
        peaks=fn(ppg); fixed=match_ppg_to_ecg(ecg_peaks,peaks,fs); lag=estimate_best_lag_match(ecg_peaks,peaks,fs)
        rows[name]={'fixed_ptt_window':fixed,'lag_corrected':lag,'estimated_channel_lag_s':lag['applied_lag_s']}
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--raw-dir',type=Path,default=Path('/data1/jiahui/biosignal-agent/datasets/raw/mimic_perform_af')); ap.add_argument('--model',default='/data1/jiahui/biosignal-agent/outputs/ppg_peak_unet.pt'); ap.add_argument('--seconds',type=float,default=60.0); ap.add_argument('--out-json',type=Path,default=Path('/data1/jiahui/biosignal-agent/outputs/ppg_peak_unet_mimic_eval.json')); args=ap.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'; ck=torch.load(args.model,map_location=device,weights_only=False); model=PeakUNet().to(device); model.load_state_dict(ck['model_state_dict']); model.eval()
    rows=[row for p in record_paths(args.raw_dir) if (row:=evaluate_record(p,args.seconds,model,device)) is not None]
    report=summarize(rows); report['rows']=rows; report['model']=args.model; report['source']='mimic_perform_af_60s'; args.out_json.parent.mkdir(parents=True,exist_ok=True); args.out_json.write_text(json.dumps(report,indent=2))
    print(json.dumps({'num_records':report['num_records'],'detectors':report['detectors']},indent=2))
if __name__=='__main__': main()
