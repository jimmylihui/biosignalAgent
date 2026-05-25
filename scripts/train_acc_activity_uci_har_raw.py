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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_SPEC=importlib.util.spec_from_file_location('prepare_acc', Path(__file__).with_name('prepare_acc_activity_dataset.py'))
_mod=importlib.util.module_from_spec(_SPEC); assert _SPEC.loader is not None; _SPEC.loader.exec_module(_mod)
ensure_dataset=_mod.ensure_dataset; load_split=_mod.load_split; ACTIVITY_NAMES=_mod.ACTIVITY_NAMES; ACTIVE=_mod.ACTIVE
RAW=Path('/data1/jiahui/biosignal-agent/datasets/raw/uci_har')
OUT=Path('/data1/jiahui/biosignal-agent/outputs/acc_activity')
FS=50.0


def feat(mag, fs=FS):
    x=np.asarray(mag,dtype=float).ravel(); finite=np.isfinite(x)
    if not finite.any(): return [0.0]*64
    med=float(np.nanmedian(x[finite])); x=np.where(finite,x,med)
    x=x-np.mean(x)
    n=len(x)
    freqs,psd=scipy_signal.welch(x,fs=fs,nperseg=min(n,int(fs*2)))
    def band(lo,hi):
        m=(freqs>=lo)&(freqs<hi)
        return float(np.trapezoid(psd[m],freqs[m])) if m.any() else 0.0
    bands={'very_low':(0.0,0.3),'low':(0.3,1.0),'step':(1.0,3.0),'mid':(3.0,8.0),'high':(8.0,min(20,fs*0.45))}
    bp={k:band(*v) for k,v in bands.items()}; total=sum(bp.values())+1e-12; rel={k:v/total for k,v in bp.items()}
    dx=np.diff(x,prepend=x[0])*fs
    q=np.percentile(x,[1,5,25,50,75,95,99])
    peaks,_=scipy_signal.find_peaks(np.abs(x),distance=max(1,int(fs*0.25)),prominence=max(np.std(x)*0.25,1e-8))
    dominant=float(freqs[np.argmax(psd)]) if len(psd) else 0.0
    p=psd/(psd.sum()+1e-12)
    sent=float(-np.sum(p*np.log2(p+1e-12))/np.log2(len(p)+1e-12)) if len(p) else 0.0
    out=[float(np.mean(x)),float(np.std(x)),float(np.min(x)),float(np.max(x)),float(np.ptp(x)),*[float(v) for v in q],float(skew(x)),float(kurtosis(x)),float(np.mean(np.abs(x))),float(np.sqrt(np.mean(x*x))),float(np.mean(dx)),float(np.std(dx)),float(np.percentile(np.abs(dx),95)),float(len(peaks)),float(len(peaks)/(n/fs/60.0+1e-12)),dominant,sent,*[float(bp[k]) for k in sorted(bp)],*[float(rel[k]) for k in sorted(rel)],rel['step']/(rel['low']+1e-12),rel['high']/(rel['step']+1e-12)]
    return [0.0 if not np.isfinite(v) else float(v) for v in out]


def split(dataset_dir, name):
    labels, subjects, features, acc = load_split(dataset_dir, name)
    X=[]; y=[]; coarse=[]
    for lab, arr in zip(labels, acc):
        label=ACTIVITY_NAMES[int(lab)]
        mag=np.linalg.norm(arr,axis=1)
        X.append(feat(mag)); y.append(label); coarse.append('active' if label in ACTIVE else 'rest')
    return np.asarray(X,dtype=np.float32), np.asarray(y), np.asarray(coarse), subjects


def metrics(y_true,y_pred,y_prob,labels):
    out={'accuracy':float(accuracy_score(y_true,y_pred)),'balanced_accuracy':float(balanced_accuracy_score(y_true,y_pred)),'macro_f1':float(f1_score(y_true,y_pred,average='macro')),'weighted_f1':float(f1_score(y_true,y_pred,average='weighted')),'confusion_matrix':confusion_matrix(y_true,y_pred,labels=labels).tolist(),'labels':labels}
    try: out['macro_auroc_ovr']=float(roc_auc_score(y_true,y_prob,labels=labels,multi_class='ovr',average='macro'))
    except Exception as exc: out['macro_auroc_error']=str(exc)
    return out


def binary(y_true,y_pred):
    labels=['active','rest']
    return {'accuracy':float(accuracy_score(y_true,y_pred)),'balanced_accuracy':float(balanced_accuracy_score(y_true,y_pred)),'macro_f1':float(f1_score(y_true,y_pred,average='macro')),'confusion_matrix':confusion_matrix(y_true,y_pred,labels=labels).tolist(),'labels':labels}


def main():
    ds=ensure_dataset(RAW, False)
    Xtr,ytr,ctr,subtr=split(ds,'train'); Xte,yte,cte,subte=split(ds,'test')
    clf=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('ensemble',VotingClassifier([('extra',ExtraTreesClassifier(n_estimators=700,min_samples_leaf=2,class_weight='balanced',random_state=51,n_jobs=-1)),('rf',RandomForestClassifier(n_estimators=500,min_samples_leaf=2,class_weight='balanced',random_state=53,n_jobs=-1))],voting='soft'))])
    clf.fit(Xtr,ytr); pred=clf.predict(Xte); prob=clf.predict_proba(Xte); labels=sorted(set(ytr.tolist()))
    coarse_pred=np.asarray(['active' if p in ACTIVE else 'rest' for p in pred])
    OUT.mkdir(parents=True,exist_ok=True)
    model_path=OUT/'acc_uci_har_raw_magnitude_activity_ensemble.joblib'
    joblib.dump({'model':clf,'labels':labels,'sampling_rate':FS,'feature':'raw_acc_magnitude_stats_spectrum'},model_path)
    report={'dataset':'UCI HAR full train/test raw total_acc magnitude','task':'6-class activity and active/rest','model':'raw magnitude feature ensemble ExtraTrees+RF','validation':'official train/test subject split','train_windows':int(len(ytr)),'test_windows':int(len(yte)),'train_label_counts':dict(Counter(ytr.tolist())),'test_label_counts':dict(Counter(yte.tolist())),'overall':metrics(yte,pred,prob,labels),'coarse_active_rest':binary(cte,coarse_pred),'model_path':str(model_path)}
    (OUT/'acc_uci_har_raw_magnitude_activity_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'train_windows':report['train_windows'],'test_windows':report['test_windows'],'overall':report['overall'],'coarse_active_rest':report['coarse_active_rest'],'model_path':str(model_path)},indent=2))

if __name__=='__main__': main()
