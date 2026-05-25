from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biosignal_agent.tools.spo2_tools import SpO2_extract_oximetry_features

DEFAULT_MANIFEST = Path('/data1/jiahui/biosignal-agent/datasets/processed/psg_sleep_manifest.json')
DEFAULT_MODEL = Path('/data1/jiahui/biosignal-agent/outputs/spo2_ucddb_event_feature_model.joblib')
DEFAULT_REPORT = Path('/data1/jiahui/biosignal-agent/outputs/spo2_ucddb_event_feature_model_report.json')


def numeric_features(result: dict) -> dict[str, float]:
    return {k: float(v) for k, v in result.items() if k != 'tool' and isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(float(v))}


def load_examples(manifest_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[dict]]:
    manifest = json.loads(manifest_path.read_text())
    rows = []
    feature_names = set()
    for idx, rec in enumerate(manifest['records']):
        feats = SpO2_extract_oximetry_features(rec['spo2_path'], float(rec['spo2_sampling_rate']))
        if 'error' in feats:
            continue
        row = numeric_features(feats)
        if not row:
            continue
        row.update({
            'record': rec['record'],
            'window_start_s': rec['window_start_s'],
            'label': rec['respiratory_event_label'],
            'event_count': rec.get('event_count', 0),
        })
        feature_names.update(k for k in row if k not in {'record', 'window_start_s', 'label', 'event_count'})
        rows.append(row)
    cols = sorted(feature_names)
    x = np.asarray([[row.get(col, 0.0) for col in cols] for row in rows], dtype=np.float32)
    y = np.asarray([row['label'] for row in rows])
    groups = np.asarray([row['record'] for row in rows])
    return x, y, groups, cols, rows


def metric_block(y_true, y_pred, y_prob) -> dict:
    labels = ['normal', 'respiratory_event']
    out = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted')),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        'labels': labels,
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        pos = labels.index('respiratory_event')
        binary = (np.asarray(y_true) == 'respiratory_event').astype(int)
        out['auroc'] = float(roc_auc_score(binary, y_prob[:, pos]))
        out['average_precision'] = float(average_precision_score(binary, y_prob[:, pos]))
    return out


def heuristic_eval(rows: list[dict]) -> dict:
    truth = []
    pred = []
    prob = []
    for row in rows:
        truth.append(row['label'])
        odi3 = float(row.get('odi3_per_hour', 0.0) or 0.0)
        odi4 = float(row.get('odi4_per_hour', 0.0) or 0.0)
        t90 = float(row.get('time_below_90_fraction', 0.0) or 0.0)
        score = min(1.0, 0.55 * min(odi3 / 30.0, 1.0) + 0.25 * min(odi4 / 20.0, 1.0) + 0.20 * min(t90 / 0.15, 1.0))
        prob.append([1.0 - score, score])
        pred.append('respiratory_event' if score >= 0.35 else 'normal')
    return metric_block(np.asarray(truth), np.asarray(pred), np.asarray(prob, dtype=float))


next_rec_paths = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument('--model-out', type=Path, default=DEFAULT_MODEL)
    ap.add_argument('--report-out', type=Path, default=DEFAULT_REPORT)
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    global next_rec_paths
    next_rec_paths = {(rec['record'], rec['window_start_s']): (rec['spo2_path'], float(rec['spo2_sampling_rate'])) for rec in manifest['records']}
    x, y, groups, cols, rows = load_examples(args.manifest)
    labels = ['normal', 'respiratory_event']
    clf = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('ensemble', VotingClassifier([
            ('extra', ExtraTreesClassifier(n_estimators=700, min_samples_leaf=2, class_weight='balanced', random_state=101, n_jobs=-1)),
            ('rf', RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight='balanced_subsample', random_state=103, n_jobs=-1)),
            ('hgb', HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, l2_regularization=0.05, random_state=107)),
        ], voting='soft')),
    ])
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    all_true, all_pred, all_prob, folds = [], [], [], []
    from sklearn.base import clone
    for fold, (tr, te) in enumerate(gkf.split(x, y, groups), 1):
        model = clone(clf)
        model.fit(x[tr], y[tr])
        pred = model.predict(x[te])
        prob = model.predict_proba(x[te])
        all_true.extend(y[te].tolist()); all_pred.extend(pred.tolist()); all_prob.extend(prob.tolist())
        folds.append({'fold': fold, 'test_records': sorted(set(groups[te].tolist())), **metric_block(y[te], pred, prob)})
    clf.fit(x, y)
    overall = metric_block(np.asarray(all_true), np.asarray(all_pred), np.asarray(all_prob, dtype=float))
    heuristic = heuristic_eval(rows)
    report = {
        'dataset': 'UCDDB PSG SpO2 30s windows',
        'task': 'respiratory_event vs normal from SpO2 only',
        'validation': 'record-level GroupKFold across UCDDB records',
        'num_windows': int(len(y)),
        'num_records': int(len(np.unique(groups))),
        'label_counts': dict(Counter(y.tolist())),
        'model_metrics': overall,
        'heuristic_oximetry_metrics': heuristic,
        'folds': folds,
        'feature_columns': cols,
        'model_path': str(args.model_out),
        'disclaimer': 'UCDDB respiratory-event windows are not identical to full-night AHI scoring; SpO2-only models miss arousal-only hypopneas and respiratory morphology.',
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': clf, 'feature_columns': cols, 'labels': labels, 'metrics': overall, 'validation': report['validation']}, args.model_out)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ['num_windows','num_records','label_counts','model_metrics','heuristic_oximetry_metrics','model_path']}, indent=2))

if __name__ == '__main__':
    main()
