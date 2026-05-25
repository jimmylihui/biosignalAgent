from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_scg_vhd_zenodo_benchmark import MAT, SUMMARY, load_vectors
from biosignal_agent.tools.scg_tools import _scg_mechanical_feature_map

OUT_MODEL = Path('/data1/jiahui/biosignal-agent/outputs/scg_vhd_mechanical_subtype_classifier.joblib')
OUT_REPORT = Path('/data1/jiahui/biosignal-agent/outputs/scg_vhd_mechanical_subtype_classifier_report.json')
TMP = Path('/data1/jiahui/biosignal-agent/outputs/scg_vhd_mechanical_csv_cache')
LABEL_COLUMNS = {
    'as': 'Moderate or greater AS',
    'ms': 'Moderate or greater MS',
    'mr': 'Moderate or greater MR',
    'ar': 'Moderate or greater AR',
    'tr': 'moderate or greater TR',
}


def safe_label(v) -> int:
    try:
        return int(float(v) > 0)
    except Exception:
        return 0


def build_table() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    TMP.mkdir(parents=True, exist_ok=True)
    meta = pd.read_excel(SUMMARY)
    meta = meta[meta['Patient ID'].astype(str).str.startswith(('CP-', 'UP-'))]
    rows = []
    labels = {k: [] for k in LABEL_COLUMNS}
    for _, m in meta.iterrows():
        pid = str(m['Patient ID'])
        if not (MAT / f'{pid}-Vectors.mat').exists():
            continue
        try:
            fs = float(m['Sampling rate(Hz)'])
            vec = load_vectors(pid)
        except Exception as exc:
            print('skip load', pid, repr(exc), flush=True)
            continue
        scg_csv = TMP / f'{pid}_scg_z.csv'
        ecg_csv = TMP / f'{pid}_ecg_lara.csv'
        if not scg_csv.exists():
            pd.DataFrame({'signal': vec['scg_z']}).to_csv(scg_csv, index=False)
        if not ecg_csv.exists():
            pd.DataFrame({'signal': vec['ecg_lara']}).to_csv(ecg_csv, index=False)
        try:
            feat = _scg_mechanical_feature_map(str(scg_csv), fs, ecg_path=str(ecg_csv), ecg_sampling_rate=fs)
        except Exception as exc:
            print('skip features', pid, repr(exc), flush=True)
            continue
        feat['patient_id'] = pid
        feat['cohort_cp'] = 1.0 if pid.startswith('CP-') else 0.0
        rows.append(feat)
        for key, col in LABEL_COLUMNS.items():
            labels[key].append(safe_label(m.get(col, 0)))
        print(pid, {k: labels[k][-1] for k in LABEL_COLUMNS}, flush=True)
    df = pd.DataFrame(rows)
    y = {k: np.asarray(v, dtype=int) for k, v in labels.items()}
    return df, y


def candidate_models(seed: int):
    return {
        'logreg_balanced': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=2000, C=0.5, random_state=seed)),
        'extra_trees_balanced': make_pipeline(SimpleImputer(strategy='median'), ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3, class_weight='balanced', random_state=seed)),
        'random_forest_balanced': make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(n_estimators=500, min_samples_leaf=3, class_weight='balanced', random_state=seed)),
    }


def eval_model_cv(x: np.ndarray, y: np.ndarray, model_name: str, seed: int = 113) -> dict:
    if len(np.unique(y)) < 2:
        return {'model': model_name, 'n': int(len(y)), 'positive_rate': float(np.mean(y)), 'auroc': None, 'auprc': None, 'f1_at_0p5': None}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = np.zeros(len(y), dtype=float)
    for tr, te in cv.split(x, y):
        model = candidate_models(seed)[model_name]
        model.fit(x[tr], y[tr])
        scores[te] = model.predict_proba(x[te])[:, 1]
    pred = (scores >= 0.5).astype(int)
    return {
        'model': model_name,
        'n': int(len(y)),
        'positive_rate': float(np.mean(y)),
        'auroc': float(roc_auc_score(y, scores)),
        'auprc': float(average_precision_score(y, scores)),
        'f1_at_0p5': float(f1_score(y, pred, zero_division=0)),
    }


def main() -> None:
    df, labels = build_table()
    feature_names = [c for c in df.columns if c != 'patient_id']
    x = df[feature_names].to_numpy(dtype=float)
    label_reports = {}
    selected_models = []
    selected_names = []
    for label, y in labels.items():
        reports = [eval_model_cv(x, y, name) for name in candidate_models(113)]
        usable = [r for r in reports if r['auroc'] is not None]
        best = max(usable, key=lambda r: (r['auroc'], r['auprc'])) if usable else reports[0]
        model = candidate_models(113)[best['model']]
        model.fit(x, y)
        selected_models.append(model)
        selected_names.append(label)
        label_reports[label] = {'candidates': reports, 'selected': best}
        print(label, best, flush=True)
    macro_auc = float(np.mean([r['selected']['auroc'] for r in label_reports.values() if r['selected']['auroc'] is not None]))
    bundle = {
        'models': selected_models,
        'labels': selected_names,
        'feature_names': feature_names,
        'label_columns': LABEL_COLUMNS,
        'training_subjects': df['patient_id'].tolist(),
        'cv_report': label_reports,
        'macro_auroc': macro_auc,
        'note': 'VHD Zenodo SCG_Z + ECG timing feature classifier for moderate-or-greater valve subtype screening; small dataset, screening only.',
    }
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUT_MODEL)
    OUT_REPORT.write_text(json.dumps({k: v for k, v in bundle.items() if k != 'models'}, indent=2))
    print(json.dumps({'model': str(OUT_MODEL), 'report': str(OUT_REPORT), 'macro_auroc': macro_auc, 'labels': label_reports}, indent=2)[:5000])


if __name__ == '__main__':
    main()
