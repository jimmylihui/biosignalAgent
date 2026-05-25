from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.tools.pcg_tools import PCG_extract_murmur_features
from biosignal_agent.tools.spectrogram_tools import Signal_extract_spectrogram_features

PCG_FEATURES = [
    'low_band_power', 'mid_band_power', 'high_band_power', 'very_high_band_power',
    'mid_band_ratio', 'high_band_ratio', 'spectral_centroid_hz', 'spectral_entropy',
    'zero_crossing_rate', 'envelope_std', 'envelope_p90_median_ratio',
    'envelope_p95_median_ratio', 'envelope_p99_median_ratio', 'continuous_fraction_60',
    'continuous_fraction_75', 'num_sounds', 'heart_rate_bpm', 'sound_interval_cv',
]
SPECTROGRAM_FEATURES = [
    'spectrogram_log_power_mean', 'spectrogram_log_power_std', 'spectral_centroid_mean_hz',
    'spectral_centroid_std_hz', 'spectral_rolloff85_mean_hz', 'spectral_rolloff85_std_hz',
    'spectral_entropy', 'temporal_energy_cv', 'temporal_energy_p95_p50_ratio',
    'band_20_60_ratio', 'band_60_150_ratio', 'band_150_400_ratio',
]
FEATURES = [f'pcg_{name}' for name in PCG_FEATURES] + [f'spec_{name}' for name in SPECTROGRAM_FEATURES]


def metrics(y_true: np.ndarray, y_pred: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        'true_positive': int(tp), 'true_negative': int(tn), 'false_positive': int(fp), 'false_negative': int(fn),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall_sensitivity': float(recall_score(y_true, y_pred, zero_division=0)),
        'specificity': float(tn / (tn + fp)) if tn + fp else 0.0,
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out['auroc'] = float(roc_auc_score(y_true, prob))
    except ValueError:
        out['auroc'] = None
    return out


def collect_row(record: dict[str, Any]) -> dict[str, Any]:
    pcg = PCG_extract_murmur_features(record['path'], float(record['sampling_rate']))
    spec = Signal_extract_spectrogram_features(record['path'], float(record['sampling_rate']), modality='pcg', window_seconds=1.0, overlap=0.5, max_frequency_hz=500.0)
    row = {'record': record.get('record'), 'truth': record.get('label')}
    for name in PCG_FEATURES:
        row[f'pcg_{name}'] = pcg.get(name)
    for name in SPECTROGRAM_FEATURES:
        row[f'spec_{name}'] = spec.get(name)
    row['pcg_error'] = pcg.get('error')
    row['spec_error'] = spec.get('error')
    return row


def models(seed: int):
    return {
        'logistic_balanced': Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('classifier', LogisticRegression(class_weight='balanced', solver='liblinear', C=0.7, random_state=seed))]),
        'random_forest_balanced': Pipeline([('imputer', SimpleImputer(strategy='median')), ('classifier', RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=3, class_weight='balanced', random_state=seed))]),
        'extra_trees_balanced': Pipeline([('imputer', SimpleImputer(strategy='median')), ('classifier', ExtraTreesClassifier(n_estimators=700, max_depth=8, min_samples_leaf=2, class_weight='balanced', random_state=seed))]),
        'gradient_boosting': Pipeline([('imputer', SimpleImputer(strategy='median')), ('classifier', GradientBoostingClassifier(n_estimators=180, max_depth=2, learning_rate=0.04, random_state=seed))]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/datasets/processed/pcg_murmur_manifest.json')
    parser.add_argument('--out-model', default='/data1/jiahui/biosignal-agent/outputs/pcg_murmur_feature_classifier.joblib')
    parser.add_argument('--report', default='/data1/jiahui/biosignal-agent/outputs/pcg_murmur_feature_classifier_report.json')
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    records = [r for r in manifest.get('records', []) if r.get('label') in {'normal', 'abnormal'}]
    rows = [collect_row(r) for r in records]
    x = np.asarray([[np.nan if row.get(name) is None else row.get(name) for name in FEATURES] for row in rows], dtype=float)
    y = np.asarray([1 if row['truth'] == 'abnormal' else 0 for row in rows], dtype=int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    summaries = {}
    best_name = None
    best_f1 = -1.0
    best_model = None
    for name, model in models(args.seed).items():
        prob = cross_val_predict(model, x, y, cv=cv, method='predict_proba')[:, 1]
        pred = (prob >= 0.5).astype(int)
        summary = metrics(y, pred, prob)
        summaries[name] = summary
        print(name, summary, flush=True)
        if summary['f1'] > best_f1:
            best_f1 = summary['f1']
            best_name = name
            best_model = model
    assert best_model is not None and best_name is not None
    best_model.fit(x, y)
    payload = {'model': best_model, 'model_name': best_name, 'feature_names': FEATURES, 'cv_metrics': summaries[best_name], 'label_mapping': {'normal': 0, 'abnormal': 1}, 'reference': 'PhysioNet/CinC 2016 training-a normal/abnormal, first 30s windows, stratified 5-fold CV.'}
    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.out_model)
    report = {'manifest': args.manifest, 'num_records': len(rows), 'truth_counts': dict(Counter(row['truth'] for row in rows)), 'feature_names': FEATURES, 'best_model': best_name, 'best_cv_metrics': summaries[best_name], 'all_cv_summaries': summaries, 'model_out': args.out_model}
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
