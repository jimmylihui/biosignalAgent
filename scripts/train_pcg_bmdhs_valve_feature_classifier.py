from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.pcg_tools import PCG_extract_murmur_features

LABELS = ['AS', 'AR', 'MR', 'MS', 'N']
OUT = Path('/data1/jiahui/biosignal-agent/outputs/pcg_bmdhs_valve_feature_classifier.joblib')
REPORT = Path('/data1/jiahui/biosignal-agent/outputs/pcg_bmdhs_valve_feature_classifier_report.json')
FEATURE_CACHE = Path('/data1/jiahui/biosignal-agent/datasets/processed/pcg_bmdhs_valve_features.csv')


def numeric_feature_row(tool_out: dict) -> dict:
    out = {}
    for k, v in tool_out.items():
        if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(float(v)):
            out[k] = float(v)
    return out


def build_features(rows: list[dict], cache: Path, refresh: bool = False) -> pd.DataFrame:
    if cache.exists() and not refresh:
        return pd.read_csv(cache)
    feats = []
    for i, row in enumerate(rows, 1):
        tool = PCG_extract_murmur_features(row['path'], float(row['sampling_rate']), None)
        item = {
            'recording': row['recording'],
            'patient_id': row['patient_id'],
            'site': row.get('site') or 'unknown',
            'posture': row.get('posture') or 'unknown',
            'duration_s': row.get('duration_s'),
        }
        item.update(numeric_feature_row(tool))
        for lab in LABELS:
            item[f'label_{lab.lower()}'] = int(row[f'label_{lab.lower()}'])
        feats.append(item)
        if i % 50 == 0:
            print('features', i, '/', len(rows), flush=True)
    df = pd.DataFrame(feats)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def choose_threshold(y: np.ndarray, p: np.ndarray) -> float:
    return float(max(((f1_score(y, p >= t, zero_division=0), t) for t in np.linspace(0.05, 0.95, 91)), key=lambda z: z[0])[1])


def metric_dict(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    return {
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'average_precision': float(average_precision_score(y, p)) if int(y.sum()) else 0.0,
        'roc_auc': float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.0,
        'positive_count': int(y.sum()),
        'num_records': int(len(y)),
    }


def make_pipeline(numeric_cols: list[str], cat_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric_cols),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols),
    ])
    return Pipeline([('preprocess', pre), ('model', LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear'))])


def main() -> None:
    ap = argparse.ArgumentParser(description='Train BMD-HS PCG valve subtype feature classifiers with patient-grouped CV.')
    ap.add_argument('--manifest', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/processed/pcg_bmdhs_valve_manifest.json'))
    ap.add_argument('--feature-cache', type=Path, default=FEATURE_CACHE)
    ap.add_argument('--model-path', type=Path, default=OUT)
    ap.add_argument('--report-path', type=Path, default=REPORT)
    ap.add_argument('--refresh-features', action='store_true')
    args = ap.parse_args()
    manifest = json.load(open(args.manifest))
    rows = manifest['rows']
    df = build_features(rows, args.feature_cache, args.refresh_features)
    cat_cols = ['site', 'posture']
    label_cols = [f'label_{lab.lower()}' for lab in LABELS]
    ignore = {'recording', 'patient_id', *cat_cols, *label_cols}
    numeric_cols = [c for c in df.columns if c not in ignore and pd.api.types.is_numeric_dtype(df[c])]
    X = df[numeric_cols + cat_cols]
    groups = df['patient_id'].to_numpy()
    cv = GroupKFold(n_splits=5)
    models = {}
    report = {'dataset': manifest['dataset'], 'num_records': len(df), 'num_patients': int(df['patient_id'].nunique()), 'labels': LABELS, 'numeric_features': numeric_cols, 'categorical_features': cat_cols, 'targets': {}}
    for lab, ycol in zip(LABELS, label_cols):
        y = df[ycol].to_numpy(dtype=int)
        oof = np.zeros(len(df), dtype=float)
        fold_rows = []
        for fold, (tr, va) in enumerate(cv.split(X, y, groups), 1):
            pipe = make_pipeline(numeric_cols, cat_cols)
            pipe.fit(X.iloc[tr], y[tr])
            p = pipe.predict_proba(X.iloc[va])[:, 1]
            oof[va] = p
            thr = choose_threshold(y[va], p)
            fold_rows.append({'fold': fold, **metric_dict(y[va], p, thr)})
        threshold = choose_threshold(y, oof)
        cv_metrics = metric_dict(y, oof, threshold)
        final = make_pipeline(numeric_cols, cat_cols)
        final.fit(X, y)
        models[lab] = final
        report['targets'][lab] = {'cv_metrics': cv_metrics, 'fold_metrics': fold_rows}
        print(lab, json.dumps(cv_metrics), flush=True)
    bundle = {'models': models, 'labels': LABELS, 'numeric_features': numeric_cols, 'categorical_features': cat_cols, 'report': report, 'feature_cache': str(args.feature_cache), 'reference': 'BMD-HS patient-grouped recording-level feature baseline'}
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_path)
    args.report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'model_path': str(args.model_path), 'targets': {k: v['cv_metrics'] for k, v in report['targets'].items()}}, indent=2))


if __name__ == '__main__':
    main()
