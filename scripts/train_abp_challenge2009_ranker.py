from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_abp_challenge2009 import RAW_DIR, evaluate_event, parse_answers

OUT_DIR = Path('/data1/jiahui/biosignal-agent/outputs/abp_challenge2009')
CACHE_DIR = OUT_DIR / 'segment_cache'
FEATURE_NAMES = [
    'score', 'num_pulses', 'heart_rate_bpm', 'median_systolic', 'median_diastolic', 'median_map', 'map_p05',
    'median_pulse_pressure', 'hypotensive_beat_fraction', 'official_low_map_beat_fraction', 'low_map_minute_fraction',
    'minute_map_slope_mmHg_per_min', 'valid_minute_count', 'severe_hypotensive_beat_fraction',
    'narrow_pulse_pressure_fraction', 'artifact_rejected_fraction'
]
AGGS = ['max', 'mean', 'min']


def ensure_smoke_cache() -> None:
    # These calls are cheap after segment_cache is populated and fill missing JSONs otherwise.
    evaluate_event(1, 1.0, top_h=5, suffixes=['a', 'b', 'c'])
    evaluate_event(2, 1.0, top_h=10, suffixes=['a', 'b', 'c'])


def answers_for_event(event: int) -> dict[str, str]:
    return parse_answers(RAW_DIR / ('event-1-answers.txt' if event == 1 else 'event-2-answers.txt'))


def segment_json(event: int, record: str, suffix: str, tail_seconds: int = 60) -> dict[str, Any] | None:
    path = CACHE_DIR / f'event{event}_{record}{suffix}_tail{tail_seconds}s.json'
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def record_features(event: int, record: str, suffixes: list[str]) -> tuple[list[float], dict[str, Any]]:
    segs = [segment_json(event, record, suffix) for suffix in suffixes]
    segs = [s for s in segs if s and s.get('score') is not None]
    details = {'valid_segment_count': len(segs), 'segments': [s.get('segment') for s in segs]}
    feats: list[float] = []
    for name in FEATURE_NAMES:
        vals = np.asarray([float(s.get(name)) if s.get(name) is not None else np.nan for s in segs], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            feats.extend([np.nan, np.nan, np.nan])
        else:
            feats.extend([float(np.nanmax(vals)), float(np.nanmean(vals)), float(np.nanmin(vals))])
    best = max(segs, key=lambda s: float(s.get('score', -np.inf))) if segs else {}
    details['best_segment'] = best.get('segment')
    details['heuristic_score'] = float(best.get('score', np.nan)) if best else np.nan
    return feats, details


def make_model(kind: str) -> Pipeline:
    if kind == 'logistic':
        clf = LogisticRegression(class_weight='balanced', C=0.5, max_iter=2000, random_state=13)
    else:
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features='sqrt', class_weight='balanced_subsample', random_state=13, n_jobs=-1)
        et = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=1, max_features='sqrt', class_weight='balanced', random_state=17, n_jobs=-1)
        clf = VotingClassifier([('rf', rf), ('et', et)], voting='soft', weights=[1, 2])
    return Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('classifier', clf)])


def topk_score(records: list[str], truth: list[int], scores: np.ndarray, event: int, top_h: int | None = None) -> dict[str, Any]:
    if top_h is None:
        top_h = 5 if event == 1 else 10
    order = np.argsort(-scores)
    pred_h_idx = set(order[:top_h].tolist())
    pred = np.asarray([1 if i in pred_h_idx else 0 for i in range(len(records))], dtype=int)
    return {
        'top_h': int(top_h),
        'score': int(np.sum(pred == np.asarray(truth))),
        'accuracy': float(np.mean(pred == np.asarray(truth))),
        'tp': int(np.sum((pred == 1) & (np.asarray(truth) == 1))),
        'tn': int(np.sum((pred == 0) & (np.asarray(truth) == 0))),
        'fp': int(np.sum((pred == 1) & (np.asarray(truth) == 0))),
        'fn': int(np.sum((pred == 0) & (np.asarray(truth) == 1))),
        'predicted_h_records': [records[i] for i in order[:top_h]],
    }


def build_dataset(events: list[int], suffixes: list[str]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rows=[]; X=[]; y=[]
    for event in events:
        answers=answers_for_event(event)
        for record, label in sorted(answers.items()):
            feats, details = record_features(event, record, suffixes)
            X.append(feats); y.append(1 if label == 'H' else 0)
            rows.append({'event': event, 'record': record, 'truth': label, **details})
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), rows


def evaluate(kind: str, suffixes: list[str]) -> dict[str, Any]:
    ensure_smoke_cache()
    X, y, rows = build_dataset([1, 2], suffixes)
    model = make_model(kind)
    loo = LeaveOneOut()
    proba = cross_val_predict(model, X, y, cv=loo, method='predict_proba')[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        'loo_accuracy_threshold_0p5': float(accuracy_score(y, pred)),
        'loo_balanced_accuracy_threshold_0p5': float(balanced_accuracy_score(y, pred)),
        'loo_f1_threshold_0p5': float(f1_score(y, pred)),
        'loo_auroc': float(roc_auc_score(y, proba)),
    }
    for row, p, pr in zip(rows, proba, pred):
        row['cv_probability_h'] = float(p)
        row['cv_prediction_threshold_0p5'] = 'H' if pr else 'C'
    event_scores = {}
    for event in [1, 2]:
        idx = [i for i, row in enumerate(rows) if row['event'] == event]
        records = [rows[i]['record'] for i in idx]
        truth = [int(y[i]) for i in idx]
        scores = proba[idx]
        if event == 2:
            sweep = {str(k): topk_score(records, truth, scores, event, top_h=k) for k in range(10, 17)}
            best_k = max(sweep, key=lambda k: sweep[k]['score'])
            event_scores['event2_best_valid_toph'] = sweep[best_k]
            event_scores['event2_toph_sweep'] = sweep
        event_scores[f'event{event}_official_style'] = topk_score(records, truth, scores, event)
    final = make_model(kind)
    final.fit(X, y)
    out_model = OUT_DIR / f'abp_challenge2009_{kind}_ranker.joblib'
    joblib.dump({'model': final, 'feature_names': [f'{name}_{agg}' for name in FEATURE_NAMES for agg in AGGS], 'suffixes': suffixes, 'metrics': metrics, 'event_scores': event_scores}, out_model)
    return {'kind': kind, 'num_records': len(rows), 'positive_records': int(np.sum(y)), 'metrics': metrics, 'event_scores': event_scores, 'model_path': str(out_model), 'rows': rows}


def main() -> None:
    parser = argparse.ArgumentParser(description='Train/evaluate a Challenge 2009 ABP AHE ranker from cached segment features.')
    parser.add_argument('--kind', choices=['logistic','ensemble'], default='ensemble')
    parser.add_argument('--suffixes', default='a,b,c')
    parser.add_argument('--out-json', type=Path, default=None)
    args = parser.parse_args()
    report = evaluate(args.kind, [x.strip() for x in args.suffixes.split(',') if x.strip()])
    out = args.out_json or OUT_DIR / f'abp_challenge2009_{args.kind}_ranker_report.json'
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ['kind','num_records','positive_records','metrics','event_scores','model_path']}, indent=2))


if __name__ == '__main__':
    main()
