from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from scipy import signal as scipy_signal
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_scg_vhd_ao_model import BeatAOModel, load_subjects, safe_bandpass, aggregate_subject_metrics
from scripts.evaluate_scg_vhd_zenodo_benchmark import match_peaks, hr_from_peaks


def predict_free(model, subj, payload, threshold, distance_s, dev):
    fs=subj['fs']; target_fs=float(payload['target_fs']); n=int(round(float(payload['post_s'])*target_fs)); step=max(1,n//2)
    x=safe_bandpass(subj['scg_z'], fs)
    if fs != target_fs:
        x=scipy_signal.resample(x, int(round(len(x)*target_fs/fs))).astype(np.float32)
    prob_sum=np.zeros(len(x),np.float32); weight=np.zeros(len(x),np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0,len(x),step):
            seg=x[start:start+n]
            if len(seg)<n:
                pad=np.zeros(n,np.float32); pad[:len(seg)]=seg; seg=pad
            pr=torch.sigmoid(model(torch.from_numpy(seg[None,None,:]).to(dev))).cpu().numpy()[0,0]
            end=min(len(x),start+n); prob_sum[start:end]+=pr[:end-start]; weight[start:end]+=1
    prob=prob_sum/np.maximum(weight,1)
    peaks,_=scipy_signal.find_peaks(prob, height=threshold, distance=max(1,int(round(distance_s*target_fs))))
    if fs != target_fs:
        peaks=np.asarray(np.round(peaks*fs/target_fs),dtype=int)
    return peaks.astype(int)


def eval_subjects(model, subjects, payload, threshold, distance_s, dev):
    rows=[]
    for subj in subjects:
        pred=predict_free(model,subj,payload,threshold,distance_s,dev); ref=np.asarray(subj['ao'],dtype=int)
        ref_hr=hr_from_peaks(ref,subj['fs']); pred_hr=hr_from_peaks(pred,subj['fs'])
        rows.append({'pid':subj['pid'],'ref_count':len(ref),'pred_count':len(pred),'m100':match_peaks(ref,pred,subj['fs'],0.10),'m50':match_peaks(ref,pred,subj['fs'],0.05),'hr_abs_error_bpm':abs(pred_hr-ref_hr) if pred_hr is not None and ref_hr is not None else None})
    hr=[r['hr_abs_error_bpm'] for r in rows if r['hr_abs_error_bpm'] is not None]
    return {'ao_100ms':aggregate_subject_metrics(rows,'m100'),'ao_50ms':aggregate_subject_metrics(rows,'m50'),'hr_mae_bpm':float(np.mean(hr)) if hr else None,'per_subject':rows}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='/data1/jiahui/biosignal-agent/outputs/scg_vhd_ao_scg_free_cnn.pt'); ap.add_argument('--report',default='/data1/jiahui/biosignal-agent/outputs/scg_vhd_ao_scg_free_cnn_postprocess_report.json'); ap.add_argument('--cpu',action='store_true')
    args=ap.parse_args(); dev=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    payload=torch.load(args.model,map_location='cpu'); model=BeatAOModel(); model.load_state_dict(payload['model_state_dict']); model.to(dev)
    subjects=load_subjects(); val=[s for s in subjects if s['pid'].startswith('CP-') and int(s['pid'].split('-')[1])>50]; test=[s for s in subjects if s['pid'].startswith('UP-')]
    rows=[]
    for th in [0.35,0.45,0.55,0.65,0.75,0.85,0.90,0.95]:
        for dist in [0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80]:
            r=eval_subjects(model,val,payload,th,dist,dev)
            rows.append({'threshold':th,'distance_s':dist,'val':r})
            print(json.dumps({'th':th,'dist':dist,'f1':r['ao_100ms']['f1'],'hr':r['hr_mae_bpm']}),flush=True)
    # prioritize AO F1, then avoid unusable HR if tied
    rows.sort(key=lambda x:(x['val']['ao_100ms']['f1'], -(x['val']['hr_mae_bpm'] or 1e9)), reverse=True)
    best=rows[0]; test_r=eval_subjects(model,test,payload,best['threshold'],best['distance_s'],dev)
    report={'selected':{'threshold':best['threshold'],'distance_s':best['distance_s'],'val_ao_100ms':best['val']['ao_100ms'],'val_ao_50ms':best['val']['ao_50ms'],'val_hr_mae_bpm':best['val']['hr_mae_bpm']},'test':test_r,'top5':[{'threshold':x['threshold'],'distance_s':x['distance_s'],'f1':x['val']['ao_100ms']['f1'],'hr':x['val']['hr_mae_bpm']} for x in rows[:5]]}
    Path(args.report).write_text(json.dumps(report,indent=2)); print(json.dumps({'selected':report['selected'],'test':{k:test_r[k] for k in ['ao_100ms','ao_50ms','hr_mae_bpm']}},indent=2))
if __name__=='__main__': main()
