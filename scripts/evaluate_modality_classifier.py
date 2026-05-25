from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.modality_tools import FEATURE_NAMES, Signal_classify_modality, extract_modality_features
from scripts.benchmark_metrics import multiclass_metrics


def choose_cv(labels: list[str]):
    min_class = min(Counter(labels).values()) if labels else 0
    if min_class < 2:
        raise ValueError('Need at least two records per class for cross-validation.')
    n_splits = min(5, min_class)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=13), f'stratified_{n_splits}_fold'


def evaluate(manifest_path: str | Path, model_path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text())
    feature_rows = []
    records = []
    for record in manifest.get('records', []):
        try:
            features = extract_modality_features(record['path'], float(record['sampling_rate']), column=None)
        except Exception as exc:
            records.append({'record': record.get('record'), 'truth': record.get('modality'), 'prediction': 'error', 'error': str(exc)})
            continue
        feature_rows.append([features[name] for name in FEATURE_NAMES])
        records.append({'record': record.get('record'), 'dataset': record.get('dataset'), 'truth': record['modality'], 'path': record['path'], 'sampling_rate': float(record['sampling_rate']), 'features': features})
    valid = [row for row in records if 'features' in row]
    x = np.asarray(feature_rows, dtype=float)
    y = [row['truth'] for row in valid]
    cv, cv_name = choose_cv(y)
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=13, class_weight='balanced')),
    ])
    pred = cross_val_predict(model, x, y, cv=cv)
    model.fit(x, y)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'feature_names': FEATURE_NAMES}, model_path)
    rows = []
    for record, prediction in zip(valid, pred.tolist()):
        tool_result = Signal_classify_modality(record['path'], float(record['sampling_rate']), column=None)
        rows.append({
            'record': record['record'],
            'dataset': record.get('dataset'),
            'truth': record['truth'],
            'prediction': prediction,
            'tool_prediction_after_fit': tool_result.get('predicted_modality'),
            'tool_confidence_after_fit': tool_result.get('confidence'),
            'model_source': tool_result.get('model_source'),
        })
    return {
        'manifest': str(manifest_path),
        'num_records': len(rows),
        'model_path': str(model_path),
        'feature_names': FEATURE_NAMES,
        'cv': cv_name,
        'truth_counts': dict(Counter(y)),
        'prediction_counts': dict(Counter(pred.tolist())),
        'metrics': {'accuracy': float(accuracy_score(y, pred)), 'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0))},
        'multiclass_metrics': multiclass_metrics(rows),
        'rows': rows,
    }


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='Train/evaluate feature-based signal modality classifier baseline.')
    parser.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/datasets/processed/modality_classifier_manifest.json')
    parser.add_argument('--model-path', default='/data1/jiahui/biosignal-agent/outputs/modality_classifier_model.joblib')
    parser.add_argument('--out-json', default='/data1/jiahui/biosignal-agent/outputs/modality_classifier_eval.json')
    parser.add_argument('--out-csv', default='/data1/jiahui/biosignal-agent/outputs/modality_classifier_eval.csv')
    args = parser.parse_args()
    report = evaluate(args.manifest, args.model_path)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    write_csv(report['rows'], args.out_csv)
    print(json.dumps({key: report[key] for key in ['num_records', 'cv', 'truth_counts', 'prediction_counts', 'metrics', 'model_path']}, indent=2))


if __name__ == '__main__':
    main()
