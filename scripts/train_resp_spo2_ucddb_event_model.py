from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load(path):
    df=pd.read_csv(path); col=df.columns[0]
    return pd.to_numeric(df[col],errors='coerce').to_numpy(dtype=float)


def clean(x):
    x=np.asarray(x,dtype=float).ravel(); finite=np.isfinite(x)
    if not finite.any(): return np.zeros_like(x)
    x=np.where(finite,x,float(np.nanmedian(x[finite])))
    lo,hi=np.percentile(x,[0.5,99.5])
    if hi>lo: x=np.clip(x,lo,hi)
    return x


def bandpower(x,fs,lo,hi):
    if len(x)<8: return 0.0
    f,p=scipy_signal.welch(x-np.mean(x),fs=fs,nperseg=min(len(x),int(fs*8)))
    m=(f>=lo)&(f<hi)
    return float(np.trapezoid(p[m],f[m])) if m.any() else 0.0


def resp_features(x,fs):
    x=clean(x); n=len(x)
    if n==0: return [0.0]*45
    try:
        filt=scipy_signal.sosfiltfilt(scipy_signal.butter(3,[0.05,0.7],btype='bandpass',fs=fs,output='sos'),x)
    except Exception:
        filt=x-np.mean(x)
    win=max(1,int(2*fs)); ker=np.ones(win)/win
    env=np.sqrt(np.convolve(filt*filt,ker,mode='same'))
    base=float(np.percentile(env,75)+1e-12)
    low=env<base*0.25; reduced=(env<base*0.7)&(env>=base*0.25)
    peaks,_=scipy_signal.find_peaks(filt,distance=max(1,int(1.5*fs)),prominence=max(np.std(filt)*0.2,1e-8))
    if len(peaks)<2:
        peaks,_=scipy_signal.find_peaks(-filt,distance=max(1,int(1.5*fs)),prominence=max(np.std(filt)*0.2,1e-8))
    intervals=np.diff(peaks)/fs if len(peaks)>1 else np.array([])
    intervals=intervals[(intervals>=1.0)&(intervals<=15.0)]
    rate=float(60/np.median(intervals)) if len(intervals) else 0.0
    cv=float(np.std(intervals)/(np.mean(intervals)+1e-12)) if len(intervals) else 0.0
    q=np.percentile(filt,[1,5,25,50,75,95,99]); eq=np.percentile(env,[1,5,25,50,75,95,99])
    bp=[bandpower(filt,fs,0.05,0.15),bandpower(filt,fs,0.15,0.35),bandpower(filt,fs,0.35,0.7)]
    total=sum(bp)+1e-12
    feats=[float(np.mean(filt)),float(np.std(filt)),float(np.ptp(filt)),*[float(v) for v in q],float(skew(filt)),float(kurtosis(filt)),float(np.mean(env)),float(np.std(env)),float(np.min(env)),float(np.max(env)),*[float(v) for v in eq],float(np.mean(low)),float(np.mean(reduced)),float(np.mean(env<base*0.5)),float(np.mean(env<base*0.8)),rate,cv,float(len(peaks)),float(len(peaks)/(n/fs/60+1e-12)),*bp,*[b/total for b in bp]]
    return [0.0 if not np.isfinite(v) else float(v) for v in feats]


def spo2_features(x,fs):
    x=clean(x); plausible=x[(x>=50)&(x<=100)]
    if len(plausible)==0: plausible=x
    if len(plausible)==0: return [0.0]*30
    target=plausible
    baseline=float(np.percentile(target,90)); desat=target<=baseline-3
    below90=target<90; below88=target<88
    dx=np.diff(target,prepend=target[0])*fs
    q=np.percentile(target,[1,5,25,50,75,95,99])
    # crude sustained desaturation segments.
    min_len=max(1,int(10*fs)); events=0; start=None
    for i,flag in enumerate(desat):
        if flag and start is None: start=i
        elif not flag and start is not None:
            if i-start>=min_len: events+=1
            start=None
    if start is not None and len(desat)-start>=min_len: events+=1
    duration_h=len(target)/fs/3600 if fs else 0
    odi=events/duration_h if duration_h>0 else 0.0
    feats=[float(np.mean(target)),float(np.std(target)),float(np.min(target)),float(np.max(target)),float(np.ptp(target)),*[float(v) for v in q],baseline,float(np.mean(below90)),float(np.mean(below88)),float(np.mean(desat)),float(events),float(odi),float(np.mean(dx)),float(np.std(dx)),float(np.percentile(np.abs(dx),95)),float(np.mean(np.abs(dx)>1)),float(len(target)/max(1,len(x)))]
    return [0.0 if not np.isfinite(v) else float(v) for v in feats]


def metrics(y_true,y_pred,y_prob):
    labels=['normal','respiratory_event']
    out={'accuracy':float(accuracy_score(y_true,y_pred)),'balanced_accuracy':float(balanced_accuracy_score(y_true,y_pred)),'macro_f1':float(f1_score(y_true,y_pred,average='macro')),'weighted_f1':float(f1_score(y_true,y_pred,average='weighted')),'confusion_matrix':confusion_matrix(y_true,y_pred,labels=labels).tolist(),'labels':labels}
    try:
        pos=labels.index('respiratory_event'); out['auroc']=float(roc_auc_score((y_true=='respiratory_event').astype(int),y_prob[:,pos]))
    except Exception as exc: out['auroc_error']=str(exc)
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,default=Path('/data1/jiahui/biosignal-agent/datasets/processed/psg_sleep_manifest.json')); ap.add_argument('--out-dir',type=Path,default=Path('/data1/jiahui/biosignal-agent/outputs/resp_spo2'))
    args=ap.parse_args(); d=json.loads(args.manifest.read_text())
    Xf=[]; Xr=[]; Xs=[]; y=[]
    for rec in d['records']:
        rf=resp_features(load(rec['resp_path']),float(rec['resp_sampling_rate']))
        sf=spo2_features(load(rec['spo2_path']),float(rec['spo2_sampling_rate']))
        Xr.append(rf); Xs.append(sf); Xf.append(rf+sf); y.append(rec['respiratory_event_label'])
    Xf=np.asarray(Xf,dtype=np.float32); Xr=np.asarray(Xr,dtype=np.float32); y=np.asarray(y)
    def fit_eval(X,name):
        clf=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('ensemble',VotingClassifier([('extra',ExtraTreesClassifier(n_estimators=700,min_samples_leaf=2,class_weight='balanced',random_state=71,n_jobs=-1)),('rf',RandomForestClassifier(n_estimators=500,min_samples_leaf=2,class_weight='balanced',random_state=73,n_jobs=-1))],voting='soft'))])
        skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=79); all_t=[]; all_p=[]; all_prob=[]; folds=[]
        from sklearn.base import clone
        for fold,(tr,te) in enumerate(skf.split(X,y),1):
            m=clone(clf); m.fit(X[tr],y[tr]); pred=m.predict(X[te]); prob=m.predict_proba(X[te])
            all_t.extend(y[te]); all_p.extend(pred); all_prob.extend(prob.tolist()); folds.append({'fold':fold,**metrics(y[te],pred,prob)})
        clf.fit(X,y)
        return clf, {'name':name,'overall':metrics(np.asarray(all_t),np.asarray(all_p),np.asarray(all_prob)),'folds':folds}
    fusion, fusion_report=fit_eval(Xf,'resp_flow_plus_spo2')
    resp, resp_report=fit_eval(Xr,'resp_flow_only')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    fusion_path=args.out_dir/'resp_spo2_ucddb_event_fusion_ensemble.joblib'; resp_path=args.out_dir/'resp_ucddb_event_flow_ensemble.joblib'
    joblib.dump({'model':fusion,'features':'resp_features_plus_spo2_features','labels':['normal','respiratory_event']},fusion_path)
    joblib.dump({'model':resp,'features':'resp_features_only','labels':['normal','respiratory_event']},resp_path)
    report={'dataset':'UCDDB ucddb002 PSG Flow+SpO2 30s windows','task':'respiratory_event vs normal','validation':'5-fold stratified window CV on one UCDDB record; not subject-independent','num_windows':int(len(y)),'label_counts':dict(Counter(y.tolist())),'fusion':fusion_report,'resp_only':resp_report,'fusion_model_path':str(fusion_path),'resp_model_path':str(resp_path)}
    (args.out_dir/'resp_spo2_ucddb_event_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'num_windows':report['num_windows'],'label_counts':report['label_counts'],'fusion':fusion_report['overall'],'resp_only':resp_report['overall'],'fusion_model_path':str(fusion_path),'resp_model_path':str(resp_path)},indent=2))

if __name__=='__main__': main()
