from __future__ import annotations
import argparse,json,random,sys,re
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from biosignal_agent.tools.ppg_tools import _ppg_artifact_metrics, _ppg_quality_feature_vector
from biosignal_agent.tools.peak_detectors import ppg_multiscale_systolic_peaks
from scripts.train_ppg_peak_unet_capnobase import load_capnobase, match_direct

FEATURE_NAMES = [
    "flatline_fraction", "saturation_fraction", "baseline_wander_ratio", "high_frequency_noise_ratio", "artifact_score",
    "skewness", "kurtosis", "zero_crossing_rate", "normalized_dynamic_range", "num_peaks", "peak_rate_per_min",
    "pulse_interval_cv", "robust_pulse_interval_cv", "normalized_rmssd", "successive_change_fraction",
    "turning_point_ratio", "short_interval_fraction", "long_interval_fraction",
]

def local_reference_peaks(peaks, start, end):
    peaks=np.asarray(peaks,dtype=int)
    return peaks[(peaks>=start)&(peaks<end)]-start

def label_natural_window(ppg, ref_peaks, fs):
    detected,_=ppg_multiscale_systolic_peaks(ppg,fs)
    match=match_direct(ref_peaks,detected,fs,tol_s=0.10)
    artifact=_ppg_artifact_metrics(ppg,fs)
    ratio=len(detected)/max(1,len(ref_peaks))
    label=None
    if len(ref_peaks)>=5 and match['f1']>=0.96 and match['sensitivity']>=0.94 and match['ppv']>=0.94 and 0.80<=ratio<=1.25 and artifact['artifact_score']<0.45:
        label='good'
    elif len(ref_peaks)<3 or match['f1']<0.82 or match['sensitivity']<0.80 or match['ppv']<0.80 or ratio<0.65 or ratio>1.50 or artifact['artifact_score']>=0.60:
        label='poor'
    return label,{**match,'peak_count_ratio':float(ratio),**artifact}

def augment_poor(ppg, fs, rng):
    finite=ppg[np.isfinite(ppg)]
    if len(finite)==0: return []
    scale=float(np.nanstd(finite))+1e-8; t=np.arange(len(ppg))/fs; out=[]
    out.append(('hf_noise', ppg + rng.normal(0,0.45*scale,size=len(ppg))))
    out.append(('baseline_drift', ppg + np.sin(2*np.pi*rng.uniform(0.04,0.16)*t)*rng.uniform(0.8,1.8)*scale))
    out.append(('clipping', np.clip(ppg, np.nanpercentile(ppg,12), np.nanpercentile(ppg,88))))
    dropout=ppg.copy()
    for _ in range(2):
        width=int(rng.uniform(0.8,4.0)*fs)
        if width>=len(dropout): continue
        start=int(rng.integers(0,len(dropout)-width)); dropout[start:start+width]=np.nanmedian(dropout)
    out.append(('dropout', dropout))
    inverted=-ppg + 2*np.nanmedian(ppg)
    out.append(('polarity_inversion', inverted))
    return out

def add(rows, signal, fs, label, record, group, source, details):
    rows.append({'x':_ppg_quality_feature_vector(signal,fs),'y':label,'record':record,'group':group,'source':source,'details':details})

def build_rows(args):
    rng=np.random.default_rng(args.seed); rows=[]; records=load_capnobase(args.raw_dir, target_fs=args.target_fs)
    win=int(args.window_s*args.target_fs); step=int(args.step_s*args.target_fs)
    for rec in records:
        ppg=rec['ppg']; fs=rec['fs']; peaks=rec['peaks']
        for start in range(0,max(0,len(ppg)-win+1),step):
            end=start+win; sig=ppg[start:end]; ref=local_reference_peaks(peaks,start,end)
            label,details=label_natural_window(sig,ref,fs)
            if label is None: continue
            wrec=f"{rec['record']}_{start/fs:.1f}s"
            add(rows,sig,fs,label,wrec,rec['record'],'capnobase_natural_direct_label',details)
            if args.augment and label=='good':
                for name,aug in augment_poor(sig,fs,rng):
                    det={'augmentation':name,'parent_record':wrec,**_ppg_artifact_metrics(aug,fs)}
                    add(rows,aug,fs,'poor',f'{wrec}_{name}',rec['record'],f'augmented_{name}',det)
    return rows

def metrics(y_true,y_pred):
    labels=['poor','good']; p,r,f,s=precision_recall_fscore_support(y_true,y_pred,labels=labels,zero_division=0)
    return {'accuracy':float(accuracy_score(y_true,y_pred)),'macro_f1':float(f1_score(y_true,y_pred,average='macro')),'weighted_f1':float(f1_score(y_true,y_pred,average='weighted')),'labels':labels,'per_class':{lab:{'precision':float(p[i]),'recall':float(r[i]),'f1':float(f[i]),'support':int(s[i])} for i,lab in enumerate(labels)},'confusion_matrix_labels':labels,'confusion_matrix':confusion_matrix(y_true,y_pred,labels=labels).tolist()}

def models(seed):
    return {
        'logistic_l2_balanced': make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000,class_weight='balanced',C=0.7,random_state=seed)),
        'random_forest_balanced': RandomForestClassifier(n_estimators=600,max_depth=10,min_samples_leaf=6,class_weight='balanced',random_state=seed,n_jobs=-1),
        'extra_trees_balanced': ExtraTreesClassifier(n_estimators=600,max_depth=12,min_samples_leaf=4,class_weight='balanced',random_state=seed,n_jobs=-1),
    }

def train(args):
    rows=build_rows(args)
    if len(rows)<50 or len(set(r['y'] for r in rows))<2: raise RuntimeError('not enough labeled rows')
    x=np.asarray([r['x'] for r in rows],dtype=float); y=np.asarray([r['y'] for r in rows]); groups=np.asarray([r['group'] for r in rows])
    cv=GroupKFold(n_splits=min(args.folds,len(set(groups))))
    reports={}; best_name=None; best_score=-1; best_model=None
    for name,model in models(args.seed).items():
        pred=cross_val_predict(model,x,y,groups=groups,cv=cv)
        rep=metrics(list(y),list(pred)); reports[name]=rep
        score=rep['macro_f1'] + 0.25*rep['per_class']['good']['f1']
        if score>best_score: best_name=name; best_score=score; best_model=model
    best_model.fit(x,y)
    payload={'model':best_model,'model_name':best_name,'feature_names':FEATURE_NAMES,'cv_metrics':reports[best_name],'all_cv_metrics':reports,'training_windows':int(len(y)),'label_counts':dict(Counter(y)),'source_counts':dict(Counter(r['source'] for r in rows)),'reference':'CapnoBase direct pleth_peak_x labels: natural windows labeled by direct PPG peak match; poor class augmented with common PPG artifacts. No ECG proxy and no lag-corrected labels.','window_s':args.window_s,'step_s':args.step_s,'target_fs':args.target_fs}
    Path(args.model_out).parent.mkdir(parents=True,exist_ok=True); joblib.dump(payload,args.model_out)
    report={k:v for k,v in payload.items() if k!='model'}; report['sample_preview']=[{k:r[k] for k in ['record','y','source','details']} for r in rows[:80]]
    Path(args.report_out).write_text(json.dumps(report,indent=2)); return report

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--raw-dir',default='/data1/jiahui/biosignal-agent/datasets/raw/capnobase_benchmark'); ap.add_argument('--model-out',default='/data1/jiahui/biosignal-agent/outputs/ppg_signal_quality_capnobase_classifier.joblib'); ap.add_argument('--report-out',default='/data1/jiahui/biosignal-agent/outputs/ppg_signal_quality_capnobase_classifier_report.json'); ap.add_argument('--window-s',type=float,default=30.0); ap.add_argument('--step-s',type=float,default=10.0); ap.add_argument('--target-fs',type=float,default=125.0); ap.add_argument('--folds',type=int,default=5); ap.add_argument('--seed',type=int,default=13); ap.add_argument('--augment',action='store_true',default=True); args=ap.parse_args(); print(json.dumps(train(args),indent=2))
if __name__=='__main__': main()
