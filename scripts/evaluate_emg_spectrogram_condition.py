#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.tools.spectrogram_tools import Signal_extract_spectrogram_features

FEATURE_NAMES = [
    'spectrogram_log_power_mean', 'spectrogram_log_power_std', 'spectral_centroid_mean_hz',
    'spectral_centroid_std_hz', 'spectral_rolloff85_mean_hz', 'spectral_rolloff85_std_hz',
    'spectral_entropy', 'temporal_energy_cv', 'temporal_energy_p95_p50_ratio',
    'band_20_60_ratio', 'band_60_150_ratio', 'band_150_300_ratio', 'band_300_450_ratio'
]


def load_values(path):
    frame=pd.read_csv(path)
    col='signal' if 'signal' in frame.columns else frame.select_dtypes('number').columns[-1]
    return frame[col].to_numpy(dtype=float)


def label_from_record(record):
    name=str(record).lower()
    if 'healthy' in name: return 'healthy'
    if 'myopathy' in name: return 'myopathy'
    if 'neuropathy' in name: return 'neuropathy'
    return 'unknown'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/datasets/processed/dedicated_common_manifest.json')
    ap.add_argument('--window-seconds', type=float, default=1.0)
    ap.add_argument('--out-json', default='/data1/jiahui/biosignal-agent/outputs/emg_spectrogram_condition_eval.json')
    ap.add_argument('--out-csv', default='/data1/jiahui/biosignal-agent/outputs/emg_spectrogram_condition_eval.csv')
    ap.add_argument('--window-dir', default='/data1/jiahui/biosignal-agent/datasets/processed/emg_spectrogram_windows')
    args=ap.parse_args()
    manifest=json.loads(Path(args.manifest).read_text())
    out_dir=Path(args.window_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows=[]; x=[]; labels=[]
    for rec in manifest.get('records', []):
        if rec.get('modality')!='emg' or rec.get('dataset')!='emgdb':
            continue
        label=label_from_record(rec.get('record'))
        values=load_values(rec['path'])
        fs=float(rec['sampling_rate'])
        win=max(1,int(args.window_seconds*fs))
        n_windows=len(values)//win
        for i in range(n_windows):
            chunk=values[i*win:(i+1)*win]
            chunk_path=out_dir/f"{rec['record']}_w{i:02d}.csv"
            pd.DataFrame({'signal':chunk}).to_csv(chunk_path,index=False)
            feats=Signal_extract_spectrogram_features(str(chunk_path), fs, modality='emg', window_seconds=0.25, overlap=0.5, max_frequency_hz=450.0)
            vector=[np.nan if feats.get(k) is None else feats.get(k) for k in FEATURE_NAMES]
            x.append(vector); labels.append(label)
            rows.append({'record':rec.get('record'), 'window':i, 'truth':label, 'path':str(chunk_path), **{k:feats.get(k) for k in FEATURE_NAMES}, 'feature_error':feats.get('error')})
    x=np.asarray(x,dtype=float)
    y=np.asarray(labels)
    classes=sorted(set(labels))
    if len(classes)<2:
        raise SystemExit('need at least two EMG classes')
    min_class=min(Counter(labels).values())
    n_splits=min(5,min_class)
    cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=13)
    model=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('classifier',LogisticRegression(class_weight='balanced',solver='liblinear',random_state=13))])
    pred=cross_val_predict(model,x,y,cv=cv)
    for row,p in zip(rows,pred.tolist()):
        row['prediction']=p
    report={
        'manifest':args.manifest,
        'note':'window-level proof-of-concept; windows from the same source record appear across folds, so this is not subject-independent validation.',
        'num_windows':len(rows),
        'num_source_records':len({r['record'] for r in rows}),
        'feature_names':FEATURE_NAMES,
        'model':'spectrogram_features_multiclass_logistic_regression',
        'cv':f'stratified_{n_splits}_fold_window_level',
        'truth_counts':dict(Counter(labels)),
        'prediction_counts':dict(Counter(pred.tolist())),
        'metrics':{'accuracy':float(accuracy_score(y,pred)), 'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0))},
        'confusion_matrix':{'labels':classes,'matrix':confusion_matrix(y,pred,labels=classes).tolist()},
        'rows':rows,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report,indent=2))
    with Path(args.out_csv).open('w',newline='') as f:
        keys=sorted({k for r in rows for k in r})
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(json.dumps({k:report[k] for k in ['num_windows','num_source_records','cv','truth_counts','prediction_counts','metrics','confusion_matrix']},indent=2))

if __name__=='__main__':
    main()
