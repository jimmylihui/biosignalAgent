#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.tools.spectrogram_tools import Signal_extract_spectrogram_features

FEATURE_NAMES = [
    'spectrogram_log_power_mean', 'spectrogram_log_power_std', 'spectral_centroid_mean_hz',
    'spectral_centroid_std_hz', 'spectral_rolloff85_mean_hz', 'spectral_rolloff85_std_hz',
    'spectral_entropy', 'temporal_energy_cv', 'temporal_energy_p95_p50_ratio',
    'band_20_60_ratio', 'band_60_150_ratio', 'band_150_400_ratio'
]


def binary_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'true_positive': int(tp), 'true_negative': int(tn), 'false_positive': int(fp), 'false_negative': int(fn),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall_sensitivity': float(recall_score(y_true, y_pred, zero_division=0)),
        'specificity': float(tn / (tn + fp)) if tn + fp else 0.0,
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }


def choose_cv(y):
    counts = Counter(y.tolist())
    min_class = min(counts.values()) if counts else 0
    if min_class >= 3:
        n = min(5, min_class)
        return StratifiedKFold(n_splits=n, shuffle=True, random_state=13), f'stratified_{n}_fold'
    return LeaveOneOut(), 'leave_one_out'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/datasets/processed/pcg_murmur_manifest.json')
    ap.add_argument('--out-json', default='/data1/jiahui/biosignal-agent/outputs/pcg_spectrogram_murmur_eval.json')
    ap.add_argument('--out-csv', default='/data1/jiahui/biosignal-agent/outputs/pcg_spectrogram_murmur_eval.csv')
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    rows=[]; x=[]; y=[]
    for rec in manifest.get('records', []):
        if rec.get('label') not in {'normal','abnormal'}:
            continue
        feats = Signal_extract_spectrogram_features(rec['path'], float(rec['sampling_rate']), modality='pcg', window_seconds=1.0, overlap=0.5, max_frequency_hz=500.0)
        vector=[np.nan if feats.get(k) is None else feats.get(k) for k in FEATURE_NAMES]
        x.append(vector)
        truth=1 if rec['label']=='abnormal' else 0
        y.append(truth)
        rows.append({'record':rec.get('record'), 'truth':rec.get('label'), **{k:feats.get(k) for k in FEATURE_NAMES}, 'feature_error':feats.get('error')})
    x=np.asarray(x,dtype=float); y_arr=np.asarray(y,dtype=int)
    cv, cv_name = choose_cv(y_arr)
    model=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('classifier',LogisticRegression(class_weight='balanced',solver='liblinear',random_state=13))])
    proba=cross_val_predict(model,x,y_arr,cv=cv,method='predict_proba')[:,1]
    pred=(proba>=0.5).astype(int)
    for row,pred_i,prob in zip(rows,pred.tolist(),proba.tolist()):
        row['prediction']='abnormal' if pred_i else 'normal'
        row['prediction_probability_abnormal']=float(prob)
    report={'manifest':args.manifest,'num_records':len(rows),'feature_names':FEATURE_NAMES,'model':'spectrogram_features_logistic_regression','cv':cv_name,'truth_counts':dict(Counter(['abnormal' if v else 'normal' for v in y])),'prediction_counts':dict(Counter(r['prediction'] for r in rows)),'metrics':binary_metrics(y,pred.tolist()),'rows':rows}
    Path(args.out_json).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report,indent=2))
    with Path(args.out_csv).open('w',newline='') as f:
        keys=sorted({k for r in rows for k in r})
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(json.dumps({k:report[k] for k in ['num_records','cv','truth_counts','prediction_counts','metrics']},indent=2))

if __name__=='__main__':
    main()
