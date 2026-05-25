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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.image_modality_tools import IMAGE_FEATURE_NAMES, Signal_classify_modality_from_image, extract_image_modality_features
from scripts.benchmark_metrics import multiclass_metrics


def choose_cv(labels: list[str]):
    min_class = min(Counter(labels).values()) if labels else 0
    if min_class < 2:
        raise ValueError('Need at least two records per class for cross-validation.')
    n_splits = min(5, min_class)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=13), f'stratified_{n_splits}_fold'


def load_records(manifest_paths: list[str]) -> list[dict[str, Any]]:
    records = []
    seen = set()
    for manifest_path in manifest_paths:
        manifest = json.loads(Path(manifest_path).read_text())
        for record in manifest.get('records', []):
            image_path = record.get('image_path')
            modality = record.get('modality')
            if not image_path or not modality:
                continue
            key = (str(image_path), str(modality))
            if key in seen:
                continue
            seen.add(key)
            enriched = dict(record)
            enriched['manifest'] = manifest_path
            records.append(enriched)
    return records


def build_model(model_name: str):
    if model_name == 'random_forest':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=500, random_state=13, class_weight='balanced', min_samples_leaf=1)),
        ])
    if model_name == 'extra_trees':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', ExtraTreesClassifier(n_estimators=700, random_state=13, class_weight='balanced', min_samples_leaf=1)),
        ])
    if model_name == 'linear_svm':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='linear', C=0.5, class_weight='balanced', probability=True, random_state=13)),
        ])
    if model_name == 'rbf_svm':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='rbf', C=3.0, gamma='scale', class_weight='balanced', probability=True, random_state=13)),
        ])
    if model_name == 'logistic':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=3000, class_weight='balanced', C=1.0)),
        ])
    if model_name == 'voting':
        return VotingClassifier(
            estimators=[
                ('rf', build_model('random_forest')),
                ('et', build_model('extra_trees')),
                ('svm', build_model('rbf_svm')),
            ],
            voting='soft',
        )
    raise ValueError(f'unknown model: {model_name}')


def evaluate(manifest_paths: list[str], model_path: str | Path, model_name: str = 'extra_trees') -> dict[str, Any]:
    records = load_records(manifest_paths)
    feature_rows = []
    valid = []
    errors = []
    for record in records:
        try:
            features = extract_image_modality_features(
                record['image_path'],
                crop_left=int(record.get('crop_left') or 0),
                crop_right=int(record.get('crop_right') or 0),
                crop_top=int(record.get('crop_top') or 0),
                crop_bottom=int(record.get('crop_bottom') or 0),
            )
        except Exception as exc:
            errors.append({'record': record.get('record'), 'truth': record.get('modality'), 'image_path': record.get('image_path'), 'error': str(exc)})
            continue
        feature_rows.append([features[name] for name in IMAGE_FEATURE_NAMES])
        valid.append({**record, 'features': features})
    x = np.asarray(feature_rows, dtype=float)
    y = [str(row['modality']).lower() for row in valid]
    cv, cv_name = choose_cv(y)
    model = build_model(model_name)
    pred = cross_val_predict(model, x, y, cv=cv)
    model.fit(x, y)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'feature_names': IMAGE_FEATURE_NAMES, 'model_name': model_name}, model_path)
    rows = []
    for record, prediction in zip(valid, pred.tolist()):
        tool_result = Signal_classify_modality_from_image(
            record['image_path'],
            crop_left=int(record.get('crop_left') or 0),
            crop_right=int(record.get('crop_right') or 0),
            crop_top=int(record.get('crop_top') or 0),
            crop_bottom=int(record.get('crop_bottom') or 0),
            model_path=str(model_path),
        )
        rows.append({
            'record': record.get('record'),
            'variant': record.get('variant'),
            'truth': str(record['modality']).lower(),
            'prediction': prediction,
            'tool_prediction_after_fit': tool_result.get('predicted_modality'),
            'tool_confidence_after_fit': tool_result.get('confidence'),
            'image_path': record.get('image_path'),
            'manifest': record.get('manifest'),
        })
    return {
        'manifests': manifest_paths,
        'num_records': len(rows),
        'num_errors': len(errors),
        'model_path': str(model_path),
        'model_name': model_name,
        'feature_names': IMAGE_FEATURE_NAMES,
        'cv': cv_name,
        'truth_counts': dict(Counter(y)),
        'prediction_counts': dict(Counter(pred.tolist())),
        'metrics': {'accuracy': float(accuracy_score(y, pred)), 'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0))},
        'multiclass_metrics': multiclass_metrics(rows),
        'rows': rows,
        'errors': errors,
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
    parser = argparse.ArgumentParser(description='Train/evaluate feature-based image modality classifier baseline.')
    parser.add_argument('--manifest', action='append', default=None, help='Image digitization manifest. Repeat to combine manifests.')
    parser.add_argument('--model-path', default='/data1/jiahui/biosignal-agent/outputs/image_modality_classifier_model.joblib')
    parser.add_argument('--model-name', default='linear_svm', choices=['random_forest', 'extra_trees', 'linear_svm', 'rbf_svm', 'logistic', 'voting'])
    parser.add_argument('--out-json', default='/data1/jiahui/biosignal-agent/outputs/image_modality_classifier_eval.json')
    parser.add_argument('--out-csv', default='/data1/jiahui/biosignal-agent/outputs/image_modality_classifier_eval.csv')
    args = parser.parse_args()
    manifests = args.manifest or ['/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark_highres_aligned_manifest.json']
    report = evaluate(manifests, args.model_path, model_name=args.model_name)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    write_csv(report['rows'], args.out_csv)
    print(json.dumps({key: report[key] for key in ['num_records', 'num_errors', 'cv', 'truth_counts', 'prediction_counts', 'metrics', 'model_name', 'model_path']}, indent=2))


if __name__ == '__main__':
    main()
