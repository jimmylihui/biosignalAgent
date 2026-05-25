from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_edb_st_dataset import st_intervals  # noqa: E402
from scripts.train_ecg_st_feature_classifier import st_features  # noqa: E402


def _metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    return {
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'average_precision': float(average_precision_score(y, p)) if int(y.sum()) else 0.0,
        'roc_auc': float(roc_auc_score(y, p)) if len(set(map(int, y))) > 1 else 0.0,
    }


def _merge_intervals(intervals: list[tuple[float, float]], max_gap_s: float) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [list(intervals[0])]
    for start, stop in intervals[1:]:
        if start <= out[-1][1] + max_gap_s:
            out[-1][1] = max(out[-1][1], stop)
        else:
            out.append([start, stop])
    return [(float(a), float(b)) for a, b in out]


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _event_metrics(true_by_key: dict, pred_by_key: dict, min_overlap_s: float) -> dict:
    tp = fp = fn = 0
    per_key = {}
    for key in sorted(set(true_by_key) | set(pred_by_key)):
        true_eps = list(true_by_key.get(key, []))
        pred_eps = list(pred_by_key.get(key, []))
        matched_true = set()
        key_tp = key_fp = 0
        for pe in pred_eps:
            best_i = None
            best_ov = 0.0
            for i, te in enumerate(true_eps):
                if i in matched_true:
                    continue
                ov = _overlap(pe, te)
                if ov > best_ov:
                    best_i = i
                    best_ov = ov
            if best_i is not None and best_ov >= min_overlap_s:
                matched_true.add(best_i)
                key_tp += 1
            else:
                key_fp += 1
        key_fn = len(true_eps) - len(matched_true)
        tp += key_tp; fp += key_fp; fn += key_fn
        per_key[f'{key[0]}:ch{key[1]}'] = {'true': len(true_eps), 'pred': len(pred_eps), 'tp': key_tp, 'fp': key_fp, 'fn': key_fn}
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn, 'per_record_channel': per_key}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/processed/edb_st_windows_12rec_manifest.json'))
    ap.add_argument('--model-path', type=Path, default=Path('/data1/jiahui/biosignal-agent/outputs/ecg_st_feature_classifier_edb12.joblib'))
    ap.add_argument('--raw-dir', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/raw/edb'))
    ap.add_argument('--report-path', type=Path, default=Path('/data1/jiahui/biosignal-agent/outputs/ecg_st_episode_scoring_edb12_report.json'))
    ap.add_argument('--min-event-overlap-s', type=float, default=10.0)
    ap.add_argument('--merge-gap-s', type=float, default=45.0)
    ap.add_argument('--cross-val', action='store_true', help='Use record-heldout out-of-fold probabilities instead of in-sample final-model probabilities.')
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    bundle = joblib.load(args.model_path)
    names = list(bundle.get('feature_names', []))
    rows = manifest['records']
    X = []
    y = []
    groups = []
    for i, row in enumerate(rows, 1):
        if i % 250 == 0:
            print('features', i, flush=True)
        feats = st_features(row['path'], row['sampling_rate'])
        X.append([float(feats.get(name, 0.0)) for name in names])
        y.append(1 if row['label'] == 'st_abnormal' else 0)
        groups.append(str(row.get('group') or row.get('record')))
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    if args.cross_val:
        candidates = {
            'logreg': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced')),
            'extra_trees': ExtraTreesClassifier(n_estimators=500, max_features='sqrt', min_samples_leaf=2, class_weight='balanced', random_state=17, n_jobs=-1),
            'random_forest': RandomForestClassifier(n_estimators=400, max_features='sqrt', min_samples_leaf=2, class_weight='balanced', random_state=23, n_jobs=-1),
            'hgb': HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, l2_regularization=0.05, random_state=29),
        }
        model_name = str(bundle.get('best_model') or 'hgb')
        base_model = candidates.get(model_name, candidates['hgb'])
        proba = np.zeros(len(y), dtype=float)
        splits = list(GroupKFold(n_splits=min(5, len(set(groups)))).split(X, y, groups=groups))
        for fold, (tr, va) in enumerate(splits, 1):
            print('cv_fold', fold, 'train', len(tr), 'valid', len(va), flush=True)
            clf = clone(base_model)
            clf.fit(X[tr], y[tr])
            proba[va] = clf.predict_proba(X[va])[:, list(clf.classes_).index(1)]
    else:
        proba = bundle['model'].predict_proba(X)[:, list(bundle['model'].classes_).index(1)]

    true_by_key = defaultdict(list)
    by_key_windows = defaultdict(list)
    for idx, row in enumerate(rows):
        key = (row['record'], int(row['channel']))
        start = float(row['window_start_s']); stop = start + float(row['duration_s'])
        by_key_windows[key].append((start, stop, idx))
    for (record, channel), wins in by_key_windows.items():
        fs = float(next(row['sampling_rate'] for row in rows if row['record'] == record and int(row['channel']) == channel))
        coverage_start = min(w[0] for w in wins); coverage_stop = max(w[1] for w in wins)
        header_len = int(round(coverage_stop * fs))
        for ep in st_intervals(args.raw_dir, record, header_len):
            if int(ep['channel']) != int(channel):
                continue
            start = max(float(ep['start_sample']) / fs, coverage_start)
            stop = min(float(ep['end_sample']) / fs, coverage_stop)
            if stop - start >= args.min_event_overlap_s:
                true_by_key[(record, channel)].append((start, stop))

    thresholds = np.linspace(0.05, 0.95, 91)
    reports = []
    for threshold in thresholds:
        pred_by_key = defaultdict(list)
        pred = proba >= threshold
        for (record, channel), wins in by_key_windows.items():
            ints = [(start, stop) for start, stop, idx in wins if pred[idx]]
            pred_by_key[(record, channel)] = _merge_intervals(ints, args.merge_gap_s)
        ev = _event_metrics(true_by_key, pred_by_key, args.min_event_overlap_s)
        wm = _metrics(y, proba, float(threshold))
        reports.append({'threshold': float(threshold), 'window': wm, 'episode': {k: ev[k] for k in ['precision', 'recall', 'f1', 'tp', 'fp', 'fn']}})
    default_threshold = float(bundle.get('threshold', 0.5))
    nearest = min(reports, key=lambda r: abs(r['threshold'] - default_threshold))
    best_episode = max(reports, key=lambda r: (r['episode']['f1'], r['episode']['recall'], r['window']['f1']))
    best_window = max(reports, key=lambda r: (r['window']['f1'], r['episode']['f1']))
    detail_threshold = best_episode['threshold']
    pred_by_key = defaultdict(list)
    pred = proba >= detail_threshold
    for (record, channel), wins in by_key_windows.items():
        ints = [(start, stop) for start, stop, idx in wins if pred[idx]]
        pred_by_key[(record, channel)] = _merge_intervals(ints, args.merge_gap_s)
    detail = _event_metrics(true_by_key, pred_by_key, args.min_event_overlap_s)
    report = {
        'model_path': str(args.model_path),
        'manifest': str(args.manifest),
        'num_windows': int(len(y)),
        'positive_windows': int(y.sum()),
        'probability_source': 'record_heldout_oof' if args.cross_val else 'final_model_in_sample',
        'default_threshold': default_threshold,
        'default_threshold_metrics': nearest,
        'best_episode_threshold_metrics': best_episode,
        'best_window_threshold_metrics': best_window,
        'best_episode_per_record_channel': detail['per_record_channel'],
        'scoring': {'min_event_overlap_s': args.min_event_overlap_s, 'merge_gap_s': args.merge_gap_s},
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ['model_path','num_windows','positive_windows','default_threshold_metrics','best_episode_threshold_metrics','best_window_threshold_metrics','scoring']}, indent=2))


if __name__ == '__main__':
    main()
