from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_SPEC = importlib.util.spec_from_file_location('prepare_chbmit', Path(__file__).with_name('prepare_chbmit_seizure_dataset.py'))
_mod = importlib.util.module_from_spec(_SPEC); assert _SPEC.loader is not None; _SPEC.loader.exec_module(_mod)
parse_summary = _mod.parse_summary
read_edf_channel = _mod.read_edf_channel
parse_edf_header = _mod.parse_edf_header

RAW = Path('/data1/jiahui/biosignal-agent/datasets/raw/chbmit/chb01')
OUT = Path('/data1/jiahui/biosignal-agent/outputs/eeg_seizure')
CHANNEL = 'FP1-F7'
WINDOW_S = 10.0
STEP_S = 5.0


def clean(x):
    x=np.asarray(x,dtype=float).ravel(); finite=np.isfinite(x)
    if not finite.any(): return np.zeros_like(x)
    med=float(np.nanmedian(x[finite])); x=np.where(finite,x,med)
    lo,hi=np.percentile(x,[0.5,99.5])
    if hi>lo: x=np.clip(x,lo,hi)
    return x-np.median(x)


def feats(x, fs):
    x=clean(x); n=len(x)
    if n==0: return [0.0]*60
    freqs, psd=scipy_signal.welch(x, fs=fs, nperseg=min(n,int(fs*2)))
    def band(lo,hi):
        m=(freqs>=lo)&(freqs<hi)
        return float(np.trapezoid(psd[m],freqs[m])) if m.any() else 0.0
    bands={'delta':(0.5,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,min(80,fs*0.45)),'hfo_proxy':(80,min(120,fs*0.45))}
    bp={k:band(*v) for k,v in bands.items()}; total=sum(bp.values())+1e-12
    rel={k:v/total for k,v in bp.items()}
    dx=np.diff(x,prepend=x[0])*fs
    env=np.abs(scipy_signal.hilbert(x)) if len(x)>8 else np.abs(x)
    robust=float(np.median(np.abs(x))*1.4826+1e-8)
    spike=np.abs(x)>6*robust
    spike_edges=np.flatnonzero(np.diff(spike.astype(int),prepend=0)==1)
    line_len=float(np.sum(np.abs(np.diff(x))))/(n+1e-12)
    hj_act=float(np.var(x)); hj_mob=float(np.sqrt(np.var(dx)/(np.var(x)+1e-12)))
    ddx=np.diff(dx,prepend=dx[0])*fs
    hj_comp=float(np.sqrt(np.var(ddx)/(np.var(dx)+1e-12))/(hj_mob+1e-12))
    p=psd[(freqs>=0.5)&(freqs<=min(80,fs*0.45))]; p=p/(p.sum()+1e-12)
    sent=float(-np.sum(p*np.log2(p+1e-12))/np.log2(len(p)+1e-12)) if len(p) else 0.0
    q=np.percentile(x,[1,5,25,50,75,95,99]); eq=np.percentile(env,[50,75,90,95,99])
    out=[float(np.mean(x)),float(np.std(x)),float(np.min(x)),float(np.max(x)),float(np.ptp(x)),*[float(v) for v in q],float(skew(x)),float(kurtosis(x)),float(np.mean(np.abs(x))),float(np.sqrt(np.mean(x*x))),line_len,hj_act,hj_mob,hj_comp,sent,float(np.mean(env)),float(np.std(env)),*[float(v) for v in eq],float(len(spike_edges)),float(len(spike_edges)/(n/fs/60.0)),float(np.mean(dx)),float(np.std(dx)),float(np.percentile(np.abs(dx),95)),*[float(bp[k]) for k in sorted(bp)],*[float(rel[k]) for k in sorted(rel)],rel['gamma']/(rel['alpha']+rel['theta']+1e-12),rel['beta']/(rel['delta']+1e-12)]
    return [0.0 if not np.isfinite(v) else float(v) for v in out]


def file_duration(path):
    h=parse_edf_header(path)
    return float(h['num_records']*h['record_duration'])


def overlaps(start, stop, intervals, margin=60.0):
    return any(start < e+margin and stop > s-margin for s,e in intervals)


def build():
    summary=parse_summary(RAW/'chb01-summary.txt')
    X=[]; y=[]; groups=[]; meta=[]
    for info in summary:
        path=RAW/info['file']
        if not path.exists() or path.suffix!='.edf' or not info.get('seizures'):
            continue
        try:
            dur=file_duration(path)
        except Exception:
            continue
        intervals=[(float(z['start_s']),float(z.get('end_s',z['start_s']+30))) for z in info['seizures']]
        starts=[]
        for s,e in intervals:
            t=max(0.0,s-5.0)
            while t+WINDOW_S<=min(dur,e+5.0):
                starts.append((t,'seizure'))
                t+=STEP_S
        non=[]
        for t in np.arange(60.0, max(61.0,dur-WINDOW_S), 30.0):
            if not overlaps(t,t+WINDOW_S,intervals,margin=120.0):
                non.append((float(t),'non_seizure'))
            if len(non)>=max(30,len(starts)*3):
                break
        for t,label in starts+non:
            try:
                vals,fs,ch=read_edf_channel(path,CHANNEL,t,WINDOW_S)
            except Exception:
                continue
            if len(vals)<int(fs*WINDOW_S*0.8): continue
            X.append(feats(vals,fs)); y.append(label); groups.append(path.stem); meta.append({'file':path.name,'start_s':t,'label':label,'fs':fs,'channel':ch})
    return np.asarray(X,dtype=np.float32),np.asarray(y),np.asarray(groups),meta


def metrics(y_true,y_pred,y_prob,labels):
    out={'accuracy':float(accuracy_score(y_true,y_pred)),'balanced_accuracy':float(balanced_accuracy_score(y_true,y_pred)),'macro_f1':float(f1_score(y_true,y_pred,average='macro')),'weighted_f1':float(f1_score(y_true,y_pred,average='weighted')),'confusion_matrix':confusion_matrix(y_true,y_pred,labels=labels).tolist(),'labels':labels}
    try:
        pos=labels.index('seizure')
        out['auroc']=float(roc_auc_score((y_true=='seizure').astype(int),y_prob[:,pos]))
    except Exception as exc: out['auroc_error']=str(exc)
    return out


def main():
    X,y,groups,meta=build(); labels=sorted(set(y.tolist()))
    clf=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('ensemble',VotingClassifier([('extra',ExtraTreesClassifier(n_estimators=600,min_samples_leaf=2,class_weight='balanced',random_state=41,n_jobs=-1)),('rf',RandomForestClassifier(n_estimators=500,min_samples_leaf=2,class_weight='balanced',random_state=43,n_jobs=-1))],voting='soft'))])
    n_splits=min(3,len(set(groups.tolist())))
    all_t=[]; all_p=[]; all_prob=[]; folds=[]
    from sklearn.base import clone
    if n_splits>=2:
        for fold,(tr,te) in enumerate(GroupKFold(n_splits=n_splits).split(X,y,groups),1):
            m=clone(clf); m.fit(X[tr],y[tr]); pred=m.predict(X[te]); prob=m.predict_proba(X[te])
            all_t.extend(y[te]); all_p.extend(pred); all_prob.extend(prob.tolist()); folds.append({'fold':fold,'test_files':sorted(set(groups[te])),**metrics(y[te],pred,prob,labels)})
    clf.fit(X,y); OUT.mkdir(parents=True,exist_ok=True)
    model_path=OUT/'eeg_chbmit_chb01_seizure_feature_ensemble.joblib'
    joblib.dump({'model':clf,'labels':labels,'channel':CHANNEL,'window_seconds':WINDOW_S,'feature':'single_channel_eeg_seizure_features'},model_path)
    report={'dataset':'CHB-MIT chb01 available seizure EDFs','task':'seizure vs non_seizure windows','model':'single-channel EEG feature ensemble ExtraTrees+RF','validation':f'{n_splits}-fold EDF-file grouped CV; small chb01 subset only','num_windows':int(len(y)),'label_counts':dict(Counter(y.tolist())),'file_counts':dict(Counter(groups.tolist())),'overall':metrics(np.asarray(all_t),np.asarray(all_p),np.asarray(all_prob),labels) if all_t else None,'folds':folds,'model_path':str(model_path),'examples':meta[:5]}
    (OUT/'eeg_chbmit_chb01_seizure_feature_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'num_windows':report['num_windows'],'label_counts':report['label_counts'],'file_counts':report['file_counts'],'overall':report['overall'],'model_path':str(model_path)},indent=2))

if __name__=='__main__': main()
