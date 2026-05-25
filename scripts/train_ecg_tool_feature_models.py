from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.tools.common import load_csv_signal
from biosignal_agent.tools.ecg_tools import ECG_APNEA_MODEL_PATH, ECG_ARRHYTHMIA_MODEL_PATH, ECG_detect_r_peaks, _ecg_feature_dict

OUT = Path('/data1/jiahui/biosignal-agent/outputs')


def records_to_xy(manifest_path: str, label_fn) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    manifest = json.load(open(manifest_path))
    rows=[]; y=[]; groups=[]
    names=None
    for rec in manifest.get('records', []):
        try:
            data = load_csv_signal(rec['path'], float(rec['sampling_rate']), column=None)
            peaks = np.asarray(ECG_detect_r_peaks(rec['path'], float(rec['sampling_rate']), column=None).get('r_peak_indices', []), dtype=int)
            feats = _ecg_feature_dict(data.values, data.sampling_rate, peaks)
        except Exception:
            continue
        if names is None:
            names = sorted(feats)
        rows.append([float(feats.get(n,0.0)) for n in names])
        y.append(int(label_fn(rec)))
        groups.append(str(rec.get('record') or rec.get('path')))
    return np.asarray(rows, dtype=float), np.asarray(y, dtype=int), names or [], groups


def evaluate_models(X, y, groups, seed=17) -> tuple[Any, str, dict[str, Any]]:
    models = {
        'extra_trees': ExtraTreesClassifier(n_estimators=600, random_state=seed, class_weight='balanced', min_samples_leaf=2, max_features='sqrt', n_jobs=-1),
        'random_forest': RandomForestClassifier(n_estimators=400, random_state=seed, class_weight='balanced', min_samples_leaf=2, max_features='sqrt', n_jobs=-1),
        'logistic': Pipeline([('scale', StandardScaler()), ('clf', LogisticRegression(max_iter=3000, class_weight='balanced', C=0.8))]),
    }
    uniq_groups = len(set(groups))
    if uniq_groups >= 3:
        cv = GroupKFold(n_splits=min(5, uniq_groups))
        splits = cv.split(X, y, groups=groups)
    else:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        splits = cv.split(X, y)
    results={}; best_name=None; best_score=-1
    # Need fresh split generator per model.
    for name, model in models.items():
        if uniq_groups >= 3:
            cv = GroupKFold(n_splits=min(5, uniq_groups))
            split_iter = cv.split(X, y, groups=groups)
        else:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            split_iter = cv.split(X, y)
        proba = cross_val_predict(model, X, y, cv=split_iter, method='predict_proba')[:, 1]
        # choose threshold maximizing F1 on CV predictions; store it with model
        thresholds=np.linspace(0.2,0.8,61)
        f1s=[f1_score(y, proba>=t, zero_division=0) for t in thresholds]
        best_t=float(thresholds[int(np.argmax(f1s))])
        pred=(proba>=best_t).astype(int)
        metrics={
            'threshold': best_t,
            'accuracy': float(accuracy_score(y,pred)),
            'precision': float(precision_score(y,pred,zero_division=0)),
            'recall': float(recall_score(y,pred,zero_division=0)),
            'f1': float(f1_score(y,pred,zero_division=0)),
            'average_precision': float(average_precision_score(y,proba)),
        }
        try: metrics['roc_auc']=float(roc_auc_score(y,proba))
        except Exception: metrics['roc_auc']=0.0
        results[name]=metrics
        score=metrics['f1'] + 0.05*metrics['average_precision']
        if score>best_score:
            best_score=score; best_name=name
    best=models[best_name]
    best.fit(X,y)
    return best, best_name, results


def train_one(name: str, manifest: str, model_path: Path, label_fn) -> dict[str, Any]:
    X,y,names,groups=records_to_xy(manifest, label_fn)
    model,best_name,results=evaluate_models(X,y,groups)
    threshold=results[best_name]['threshold']
    joblib.dump({'model':model,'model_name':best_name,'feature_names':names,'threshold':threshold,'cv_results':results}, model_path)
    return {'task':name,'manifest':manifest,'num_rows':int(len(y)),'label_counts':dict(Counter(map(int,y))),'best_model':best_name,'model_path':str(model_path),'cv_results':results}


def main():
    report={}
    report['arrhythmia']=train_one(
        'arrhythmia',
        '/data1/jiahui/biosignal-agent/datasets/processed/labeled_arrhythmia_manifest.json',
        ECG_ARRHYTHMIA_MODEL_PATH,
        lambda rec: 1 if rec.get('binary_label') == 'abnormal' else 0,
    )
    report['apnea']=train_one(
        'apnea',
        '/data1/jiahui/biosignal-agent/datasets/processed/apnea_ecg_manifest.json',
        ECG_APNEA_MODEL_PATH,
        lambda rec: 1 if rec.get('label') == 'apnea' else 0,
    )
    out=OUT/'ecg_tool_feature_model_train_report.json'
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
