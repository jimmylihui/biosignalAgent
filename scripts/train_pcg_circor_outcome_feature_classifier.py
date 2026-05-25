from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import signal as scipy_signal
from scipy.io import wavfile
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    'duration_s','rms','zcr','envelope_cv','envelope_p95_p50','envelope_p99_p50',
    'spectral_centroid','spectral_entropy','rolloff85','band_20_60','band_60_150','band_150_400','band_400_800',
    'temporal_energy_cv','heart_sound_rate','peak_interval_cv',
    'loc_AV','loc_PV','loc_TV','loc_MV','loc_Phc',
]


def load_signal(path: str) -> tuple[int, np.ndarray]:
    fs, values = wavfile.read(path)
    if values.ndim > 1:
        values = values[:, 0]
    values = values.astype(float)
    values = values[np.isfinite(values)]
    values = values - np.nanmedian(values)
    scale = np.nanpercentile(np.abs(values), 95) + 1e-8
    return int(fs), np.clip(values / scale, -8, 8)


def entropy(power: np.ndarray) -> float:
    p = power / (np.sum(power) + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)) / np.log2(len(power))) if len(power) > 1 else 0.0


def extract_features(record: dict[str, Any]) -> dict[str, float]:
    fs, values = load_signal(record['path'])
    duration = len(values) / float(fs) if fs else 0.0
    high = min(800.0, fs * 0.45)
    if len(values) >= fs and high > 30:
        sos = scipy_signal.butter(3, [20.0/(0.5*fs), high/(0.5*fs)], btype='bandpass', output='sos')
        filt = scipy_signal.sosfiltfilt(sos, values)
    else:
        filt = values
    env = np.abs(scipy_signal.hilbert(filt)) if len(filt) else np.asarray([])
    freqs, psd = scipy_signal.welch(filt, fs=fs, nperseg=min(len(filt), int(fs*2))) if len(filt) > 16 else (np.asarray([]), np.asarray([]))
    total = float(np.trapezoid(psd, freqs) + 1e-12) if len(freqs) else 1e-12
    def band(low, hi):
        mask = (freqs >= low) & (freqs < min(hi, high))
        return float(np.trapezoid(psd[mask], freqs[mask]) / total) if np.any(mask) else 0.0
    centroid = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-12)) if len(freqs) else 0.0
    cdf = np.cumsum(psd) if len(psd) else np.asarray([])
    rolloff = float(freqs[np.searchsorted(cdf, 0.85*cdf[-1])]) if len(freqs) and cdf[-1] > 0 else 0.0
    temporal_cv = 0.0
    if len(filt) > fs:
        nper = max(128, min(len(filt), int(0.5*fs)))
        _, _, spec = scipy_signal.spectrogram(filt, fs=fs, nperseg=nper, noverlap=nper//2, mode='magnitude')
        energy = np.sum(spec**2, axis=0)
        temporal_cv = float(np.std(energy) / (np.mean(energy) + 1e-12)) if len(energy) else 0.0
    peaks = np.asarray([], dtype=int)
    if len(env):
        peaks, _ = scipy_signal.find_peaks(env, distance=max(1, int(0.18*fs)), prominence=max(float(np.std(env))*0.25, 1e-8))
    intervals = np.diff(peaks) / float(fs) if len(peaks) > 1 else np.asarray([])
    loc = str(record.get('location') or '')
    out = {
        'duration_s': float(duration), 'rms': float(np.sqrt(np.mean(filt**2))) if len(filt) else 0.0,
        'zcr': float(np.mean(np.diff(np.signbit(filt)) != 0)) if len(filt) > 1 else 0.0,
        'envelope_cv': float(np.std(env)/(np.mean(env)+1e-12)) if len(env) else 0.0,
        'envelope_p95_p50': float(np.percentile(env,95)/(np.percentile(env,50)+1e-12)) if len(env) else 0.0,
        'envelope_p99_p50': float(np.percentile(env,99)/(np.percentile(env,50)+1e-12)) if len(env) else 0.0,
        'spectral_centroid': centroid, 'spectral_entropy': entropy(psd) if len(psd) else 0.0, 'rolloff85': rolloff,
        'band_20_60': band(20,60), 'band_60_150': band(60,150), 'band_150_400': band(150,400), 'band_400_800': band(400,800),
        'temporal_energy_cv': temporal_cv,
        'heart_sound_rate': float(len(peaks)/duration*60.0) if duration > 0 else 0.0,
        'peak_interval_cv': float(np.std(intervals)/(np.mean(intervals)+1e-12)) if len(intervals) else 0.0,
        'loc_AV': float(loc == 'AV'), 'loc_PV': float(loc == 'PV'), 'loc_TV': float(loc == 'TV'), 'loc_MV': float(loc == 'MV'), 'loc_Phc': float(loc == 'Phc'),
    }
    return out


def metrics(y_true, prob, threshold=0.5):
    y=np.asarray(y_true,dtype=int); p=np.asarray(prob,dtype=float); pred=(p>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'true_positive':int(tp),'true_negative':int(tn),'false_positive':int(fp),'false_negative':int(fn),'accuracy':float(accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),'recall_sensitivity':float(recall_score(y,pred,zero_division=0)),'specificity':float(tn/(tn+fp)) if tn+fp else 0.0,'f1':float(f1_score(y,pred,zero_division=0)),'auroc':float(roc_auc_score(y,p)) if len(set(y.tolist()))>1 else None,'threshold':float(threshold)}


def best_threshold(y, prob):
    best=(0.5,metrics(y,prob,0.5))
    for t in np.linspace(0.1,0.9,81):
        m=metrics(y,prob,float(t))
        if (m['f1'],m['accuracy'],m['specificity'])>(best[1]['f1'],best[1]['accuracy'],best[1]['specificity']): best=(float(t),m)
    return best


def models(seed):
    return {
        'logistic_balanced': Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('classifier',LogisticRegression(max_iter=2000,class_weight='balanced',solver='liblinear',C=0.8,random_state=seed))]),
        'random_forest_balanced': Pipeline([('imputer',SimpleImputer(strategy='median')),('classifier',RandomForestClassifier(n_estimators=600,max_depth=10,min_samples_leaf=3,class_weight='balanced',random_state=seed))]),
        'extra_trees_balanced': Pipeline([('imputer',SimpleImputer(strategy='median')),('classifier',ExtraTreesClassifier(n_estimators=800,max_depth=10,min_samples_leaf=2,class_weight='balanced',random_state=seed))]),
        'gradient_boosting': Pipeline([('imputer',SimpleImputer(strategy='median')),('classifier',GradientBoostingClassifier(n_estimators=220,max_depth=2,learning_rate=0.035,random_state=seed))]),
    }


def patient_metrics(records, prob, threshold=0.5):
    bins={}; truth={}
    for rec,p in zip(records,prob):
        pid=rec['patient_id']; bins.setdefault(pid,[]).append(float(p)); truth[pid]=1 if str(rec.get('outcome','')).lower()=='abnormal' else 0
    ids=sorted(bins); y=[truth[i] for i in ids]; pp=[max(bins[i]) for i in ids]
    return metrics(y,pp,threshold), y, pp


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',default='/data1/jiahui/biosignal-agent/datasets/processed/circor_pcg_murmur_manifest.json')
    ap.add_argument('--out-model',default='/data1/jiahui/biosignal-agent/outputs/pcg_circor_outcome_feature_classifier.joblib')
    ap.add_argument('--report',default='/data1/jiahui/biosignal-agent/outputs/pcg_circor_outcome_feature_classifier_report.json')
    ap.add_argument('--seed',type=int,default=41)
    args=ap.parse_args()
    manifest=json.loads(Path(args.manifest).read_text())
    records=[r for r in manifest['records'] if str(r.get('outcome','')).lower() in {'normal','abnormal'}]
    rows=[]
    for i,rec in enumerate(records,1):
        row=extract_features(rec); rows.append(row)
        if i%200==0: print('features',i,flush=True)
    x=np.asarray([[row[k] for k in FEATURES] for row in rows],dtype=float)
    y=np.asarray([1 if str(r.get('outcome','')).lower()=='abnormal' else 0 for r in records],dtype=int)
    groups=np.asarray([r['patient_id'] for r in records])
    cv=GroupKFold(n_splits=5)
    summaries={}; best_name=None; best_f1=(-1.0,-1.0,-1.0); best_model=None; best_prob=None
    for name,model in models(args.seed).items():
        prob=cross_val_predict(model,x,y,groups=groups,cv=cv,method='predict_proba')[:,1]
        rec_m=metrics(y,prob,0.5); bt,bm=best_threshold(y,prob); pat_m,py,pp=patient_metrics(records,prob,0.5); pbt,pbm=best_threshold(py,pp)
        summaries[name]={'record_metrics':rec_m,'record_best_threshold':bt,'record_best_threshold_metrics':bm,'patient_metrics':pat_m,'patient_best_threshold':pbt,'patient_best_threshold_metrics':pbm}
        print(name,summaries[name],flush=True)
        score=(pat_m.get('auroc') or 0.0, pat_m.get('specificity') or 0.0, pat_m.get('f1') or 0.0)
        if best_name is None or score>best_f1:
            best_f1=score; best_name=name; best_model=model; best_prob=prob
    best_model.fit(x,y)
    payload={'model':best_model,'model_name':best_name,'feature_names':FEATURES,'cv_metrics':summaries[best_name],'reference':'CirCor 1.0.3 outcome Abnormal/Normal labels, location-aware handcrafted PCG features, patient-group CV, patient max aggregation.'}
    Path(args.out_model).parent.mkdir(parents=True,exist_ok=True); joblib.dump(payload,args.out_model)
    report={'manifest':args.manifest,'num_records':len(records),'num_patients':len(set(groups.tolist())),'outcome_label_counts':dict(Counter(['abnormal' if v else 'normal' for v in y.tolist()])),'best_model':best_name,'best_cv_metrics':summaries[best_name],'all_cv_summaries':summaries,'model_out':args.out_model}
    Path(args.report).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
