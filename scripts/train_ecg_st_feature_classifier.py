from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
from scipy import signal as scipy_signal
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.tools.common import load_csv_signal
from biosignal_agent.tools.peak_detectors import neurokit_nabian2018_peaks

OUT=Path('/data1/jiahui/biosignal-agent/outputs')
MODEL_PATH=OUT/'ecg_st_feature_classifier.joblib'

def safe(v, default=0.0):
    try: v=float(v)
    except Exception: return default
    return v if np.isfinite(v) else default

def st_features(path, fs):
    data=load_csv_signal(path,float(fs),None); x=np.asarray(data.values,dtype=float); fs=float(data.sampling_rate)
    x=x[np.isfinite(x)]
    if len(x)<max(32,int(fs*3)):
        return {}
    # mild baseline removal only; ST level should remain relative to local PR baseline.
    try:
        b,a=scipy_signal.butter(2,0.5/(fs/2),btype='highpass')
        xf=scipy_signal.filtfilt(b,a,x).astype(float)
    except Exception:
        xf=x-np.nanmedian(x)
    peaks,_=neurokit_nabian2018_peaks(xf,fs,low_hz=None,high_hz=None,fallback_threshold_scale=0.6)
    peaks=np.asarray(peaks,dtype=int)
    peaks=peaks[(peaks>int(0.25*fs))&(peaks<len(xf)-int(0.35*fs))]
    rr=np.diff(peaks)/fs if len(peaks)>1 else np.asarray([])
    st_vals=[]; st60=[]; st80=[]; st_slopes=[]; t_vals=[]; qrs_amp=[]
    for p in peaks:
        base0=max(0,p-int(0.08*fs)); base1=max(base0+1,p-int(0.02*fs))
        baseline=float(np.median(xf[base0:base1])) if base1>base0 else float(xf[p])
        j60=p+int(0.06*fs); j80=p+int(0.08*fs); j120=p+int(0.12*fs); t0=p+int(0.12*fs); t1=min(len(xf),p+int(0.36*fs))
        if j120 < len(xf):
            v60=float(xf[j60]-baseline); v80=float(xf[j80]-baseline); v120=float(xf[j120]-baseline)
            st60.append(v60); st80.append(v80); st_vals.append(v80); st_slopes.append((v120-v60)/0.06)
        if t1>t0:
            seg=xf[t0:t1]-baseline; t_vals.append(float(seg[np.argmax(np.abs(seg))]))
        q0=max(0,p-int(0.04*fs)); q1=min(len(xf),p+int(0.04*fs))
        if q1>q0: qrs_amp.append(float(np.max(xf[q0:q1])-np.min(xf[q0:q1])))
    st=np.asarray(st_vals,dtype=float); st60=np.asarray(st60,dtype=float); st80=np.asarray(st80,dtype=float); slope=np.asarray(st_slopes,dtype=float); tv=np.asarray(t_vals,dtype=float); qa=np.asarray(qrs_amp,dtype=float)
    dyn=float(np.percentile(xf,95)-np.percentile(xf,5)) if len(xf) else 0.0
    qrs_med=float(np.median(qa)) if len(qa) else dyn
    scale=max(qrs_med,dyn,1e-8)
    def stat(prefix, arr):
        arr=np.asarray(arr,dtype=float); arr=arr[np.isfinite(arr)]
        if len(arr)==0:
            return {prefix+'_median':0,prefix+'_mean_abs':0,prefix+'_p95_abs':0,prefix+'_std':0,prefix+'_iqr':0,prefix+'_pos_frac':0,prefix+'_neg_frac':0}
        return {prefix+'_median':float(np.median(arr)),prefix+'_mean_abs':float(np.mean(np.abs(arr))),prefix+'_p95_abs':float(np.percentile(np.abs(arr),95)),prefix+'_std':float(np.std(arr)),prefix+'_iqr':float(np.percentile(arr,75)-np.percentile(arr,25)),prefix+'_pos_frac':float(np.mean(arr>0)),prefix+'_neg_frac':float(np.mean(arr<0))}
    feats={
        'duration_s':len(xf)/fs,'num_beats':len(peaks),'hr_bpm':float(60/np.median(rr)) if len(rr) else 0.0,'rr_cv':float(np.std(rr)/np.mean(rr)) if len(rr) and np.mean(rr)>0 else 0.0,
        'signal_dynamic_range':dyn,'qrs_amp_median':qrs_med,'st_abs_over_qrs':float(np.median(np.abs(st))/scale) if len(st) else 0.0,'st_p95_abs_over_qrs':float(np.percentile(np.abs(st),95)/scale) if len(st) else 0.0,
        'st_elevation_fraction_0p1mv':float(np.mean(st>0.1)) if len(st) else 0.0,'st_depression_fraction_0p1mv':float(np.mean(st<-0.1)) if len(st) else 0.0,
    }
    for k,v in stat('st80',st).items(): feats[k]=v
    for k,v in stat('st60',st60).items(): feats[k]=v
    for k,v in stat('st_slope',slope).items(): feats[k]=v
    for k,v in stat('t_amp',tv).items(): feats[k]=v
    return {k:safe(v) for k,v in feats.items()}

def metrics(y, proba, threshold):
    pred=(proba>=threshold).astype(int)
    return {'threshold':float(threshold),'accuracy':float(accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'average_precision':float(average_precision_score(y,proba)),'roc_auc':float(roc_auc_score(y,proba)) if len(set(y))>1 else 0.0,'class_report':classification_report(y,pred,labels=[0,1],target_names=['normal','st_abnormal'],zero_division=0,output_dict=True)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',default='/data1/jiahui/biosignal-agent/datasets/processed/edb_st_windows_6rec_manifest.json'); ap.add_argument('--model-path',type=Path,default=MODEL_PATH); ap.add_argument('--report-path',type=Path,default=OUT/'ecg_st_feature_classifier_train_report.json'); args=ap.parse_args()
    m=json.load(open(args.manifest)); rows=[]; y=[]; groups=[]
    for i,r in enumerate(m['records'],1):
        if i%250==0: print('features',i,flush=True)
        rows.append(st_features(r['path'],r['sampling_rate'])); y.append(1 if r['label']=='st_abnormal' else 0); groups.append(str(r.get('group') or r['record']))
    names=sorted({k for row in rows for k in row})
    X=np.asarray([[row.get(k,0.0) for k in names] for row in rows],dtype=float); y=np.asarray(y,dtype=int); groups=np.asarray(groups)
    candidates={
      'logreg': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000,class_weight='balanced')),
      'extra_trees': ExtraTreesClassifier(n_estimators=500,max_features='sqrt',min_samples_leaf=2,class_weight='balanced',random_state=17,n_jobs=-1),
      'random_forest': RandomForestClassifier(n_estimators=400,max_features='sqrt',min_samples_leaf=2,class_weight='balanced',random_state=23,n_jobs=-1),
      'hgb': HistGradientBoostingClassifier(max_iter=250,learning_rate=0.04,l2_regularization=0.05,random_state=29),
    }
    splits=list(GroupKFold(n_splits=min(5,len(set(groups)))).split(X,y,groups=groups))
    reports={}; best_name=None; best_score=-1; best_thr=0.5
    for name,model in candidates.items():
        proba=np.zeros(len(y),dtype=float)
        for tr,va in splits:
            clf=clone(model); clf.fit(X[tr],y[tr]); proba[va]=clf.predict_proba(X[va])[:,1]
        th=float(max(((f1_score(y,proba>=t,zero_division=0),t) for t in np.linspace(0.05,0.95,91)),key=lambda z:z[0])[1])
        rep=metrics(y,proba,th); reports[name]=rep
        print(name,json.dumps({k:rep[k] for k in ['threshold','accuracy','precision','recall','f1','average_precision','roc_auc']},indent=2),flush=True)
        score=rep['f1']+0.1*rep['average_precision']
        if score>best_score: best_score=score; best_name=name; best_thr=th
    final=clone(candidates[best_name]); final.fit(X,y)
    bundle={'model':final,'feature_names':names,'threshold':best_thr,'best_model':best_name,'cv_metrics':reports[best_name],'all_model_reports':reports,'label_counts':dict(Counter(map(int,y))),'manifest':args.manifest}
    args.model_path.parent.mkdir(parents=True,exist_ok=True); joblib.dump(bundle,args.model_path)
    report={'model_path':str(args.model_path),**{k:bundle[k] for k in ['best_model','threshold','cv_metrics','all_model_reports','label_counts','manifest']}}
    args.report_path.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
