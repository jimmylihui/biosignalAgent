from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from scipy import signal as scipy_signal
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_image_scale_prior import DURATIONS
from biosignal_agent.tools.image_modality_tools import extract_image_modality_features, IMAGE_FEATURE_NAMES

OUT = Path('/data1/jiahui/biosignal-agent/outputs/image_scale_prior')
ROWS = OUT / 'scale_prior_dataset.json'


def trace_from_image(path: str) -> np.ndarray:
    img = Image.open(path).convert('L').resize((256, 96), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    dark = arr < 0.75
    ys=[]
    for x in range(dark.shape[1]):
        idx=np.flatnonzero(dark[:,x])
        if len(idx): ys.append(float(np.median(idx)/max(1,dark.shape[0]-1)))
    if len(ys)<4: return np.zeros(4,dtype=float)
    y=np.asarray(ys,dtype=float)
    y=np.interp(np.linspace(0,len(y)-1,256), np.arange(len(y)), y)
    return y


def extra_features(path: str) -> dict[str,float]:
    y=trace_from_image(path)
    centered=y-np.mean(y)
    diff=np.diff(centered)
    peaks,_=scipy_signal.find_peaks(-centered, distance=3, prominence=max(np.std(centered)*0.2,1e-8))
    fft=np.abs(np.fft.rfft(centered))**2
    freqs=np.fft.rfftfreq(len(centered),d=1.0)
    power=float(np.sum(fft))+1e-12
    centroid=float(np.sum(freqs*fft)/power)
    entropy=float(-np.sum((fft/power)*np.log2(fft/power+1e-12))/np.log2(len(fft))) if len(fft)>1 else 0.0
    return {
        'trace_peak_count_norm': float(len(peaks)/len(y)),
        'trace_zero_crossing_rate_resampled': float(np.mean(np.diff(np.signbit(centered))!=0)),
        'trace_abs_slope_mean_resampled': float(np.mean(np.abs(diff))) if len(diff) else 0.0,
        'trace_slope_std_resampled': float(np.std(diff)) if len(diff) else 0.0,
        'trace_fft_centroid_resampled': centroid,
        'trace_fft_entropy_resampled': entropy,
        'trace_fft_low': float(np.sum(fft[freqs<0.03])/power),
        'trace_fft_mid': float(np.sum(fft[(freqs>=0.03)&(freqs<0.12)])/power),
        'trace_fft_high': float(np.sum(fft[freqs>=0.12])/power),
    }


def build_features(rows):
    names=None; X=[]; y=[]
    modality_labels = ['ecg','ppg','resp','bcg','scg','abp','pcg','eeg','emg','eda','acc','spo2']
    for row in rows:
        try:
            # Use only fixed-size trace/spectral features plus modality context. The
            # generic image features include raw width and full-column FFT terms that
            # leak rendering resolution and did not generalize to high-res plots.
            f=extra_features(row['image_path'])
            base=extract_image_modality_features(row['image_path'])
            for key in ['trace_y_mean','trace_y_std','trace_y_range','trace_slope_std','trace_slope_abs_mean',
                        'trace_zero_crossing_rate','trace_fft_centroid','trace_fft_entropy',
                        'trace_low_ratio','trace_mid_ratio','trace_high_ratio','trace_skew','trace_kurtosis',
                        'row_dark_entropy','col_dark_entropy','row_projection_peaks','col_projection_peaks',
                        'dark_fraction','edge_density']:
                f[key]=float(base.get(key,0.0))
            for mod in modality_labels:
                f[f'modality_{mod}']=1.0 if str(row.get('modality')).lower()==mod else 0.0
        except Exception:
            continue
        if names is None: names=sorted(f)
        X.append([float(f.get(n,0.0)) for n in names])
        y.append(int(row['label']))
    return np.asarray(X,dtype=float), np.asarray(y,dtype=int), names


def main():
    rows=json.loads(ROWS.read_text())
    X,y,names=build_features(rows)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,random_state=19,stratify=y)
    models={
        'extra_trees': ExtraTreesClassifier(n_estimators=700, random_state=19, class_weight='balanced', min_samples_leaf=2, max_features='sqrt'),
        'random_forest': RandomForestClassifier(n_estimators=500, random_state=19, class_weight='balanced', min_samples_leaf=2, max_features='sqrt'),
        'logistic': Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(max_iter=3000,class_weight='balanced',C=1.0))]),
    }
    results={}
    best_name=None; best_acc=-1
    for name,model in models.items():
        model.fit(Xtr,ytr)
        pred=model.predict(Xte)
        proba=model.predict_proba(Xte)
        acc=accuracy_score(yte,pred)
        top2=top_k_accuracy_score(yte, proba, k=2, labels=list(range(len(DURATIONS))))
        top3=top_k_accuracy_score(yte, proba, k=3, labels=list(range(len(DURATIONS))))
        results[name]={'accuracy':float(acc),'top2_accuracy':float(top2),'top3_accuracy':float(top3),'confusion_matrix':confusion_matrix(yte,pred,labels=list(range(len(DURATIONS)))).tolist()}
        if acc>best_acc:
            best_acc=acc; best_name=name
    best=models[best_name]
    model_path=OUT/'image_scale_prior_feature_model.joblib'
    joblib.dump({'model':best,'feature_names':names,'durations':DURATIONS,'results':results,'best_model':best_name}, model_path)
    report={'num_rows':int(len(y)),'num_train':int(len(ytr)),'num_test':int(len(yte)),'durations':DURATIONS,'results':results,'best_model':best_name,'model_path':str(model_path)}
    (OUT/'image_scale_prior_feature_train_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
