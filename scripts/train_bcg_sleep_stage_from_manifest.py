from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, cohen_kappa_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biosignal_agent.tools.bcg_tools import _bcg_feature_summary, _detect_bcg_peaks
from biosignal_agent.tools.common import load_csv_signal

OUT = Path('/data1/jiahui/biosignal-agent/outputs/bcg_sleep_stage_feature_model.joblib')
METRICS = Path('/data1/jiahui/biosignal-agent/outputs/bcg_sleep_stage_feature_model_metrics.json')

STAGE_MAP = {
    'w': 'W', 'wake': 'W', '0': 'W', 0: 'W',
    'n1': 'N1', '1': 'N1', 1: 'N1',
    'n2': 'N2', '2': 'N2', 2: 'N2',
    'n3': 'N3', '3': 'N3', 3: 'N3', 'n4': 'N3', '4': 'N3', 4: 'N3',
    'r': 'REM', 'rem': 'REM', '5': 'REM', 5: 'REM',
}

REQUIRED_MANIFEST_COLUMNS = ['record', 'signal_path', 'sampling_rate', 'stage_path']
STAGE_FILE_COLUMNS = 'stage CSV/TSV must contain either (epoch, stage) or (start_sec, stage). Epoch duration defaults to --epoch-seconds.'


def normalize_stage(value) -> str | None:
    key = str(value).strip().lower()
    return STAGE_MAP.get(key)


def read_stage_file(path: Path, epoch_seconds: float) -> pd.DataFrame:
    sep = '\t' if path.suffix.lower() in {'.tsv', '.txt'} else ','
    df = pd.read_csv(path, sep=sep)
    lower = {c.lower(): c for c in df.columns}
    if 'stage' not in lower:
        raise ValueError(f'{path} missing stage column')
    stage_col = lower['stage']
    if 'start_sec' in lower:
        start = pd.to_numeric(df[lower['start_sec']], errors='coerce').to_numpy(dtype=float)
    elif 'epoch' in lower:
        start = pd.to_numeric(df[lower['epoch']], errors='coerce').to_numpy(dtype=float) * float(epoch_seconds)
    else:
        raise ValueError(f'{path} missing epoch or start_sec column')
    stages = [normalize_stage(v) for v in df[stage_col]]
    out = pd.DataFrame({'start_sec': start, 'stage': stages})
    return out[np.isfinite(out['start_sec']) & out['stage'].notna()].reset_index(drop=True)


def epoch_features(values: np.ndarray, fs: float, start_sec: float, epoch_seconds: float, record: str) -> dict | None:
    start = int(round(start_sec * fs))
    stop = int(round((start_sec + epoch_seconds) * fs))
    if start < 0 or stop > len(values) or stop - start < int(0.75 * epoch_seconds * fs):
        return None
    segment = np.asarray(values[start:stop], dtype=float)
    segment = segment[np.isfinite(segment)]
    if len(segment) < int(10 * fs):
        return None
    peaks, details = _detect_bcg_peaks(segment, fs)
    features = _bcg_feature_summary(segment, fs, peaks)
    centered = segment - float(np.nanmedian(segment))
    diff = np.diff(centered)
    extra = {
        'record': record,
        'start_sec': float(start_sec),
        'spectral_hr_bpm': details.get('spectral_hr_bpm') or 0.0,
        'harmonic_ambiguity': float(bool(details.get('harmonic_ambiguity'))),
        'candidate_hr_spread': float((details.get('candidate_hr_max_bpm') or 0.0) - (details.get('candidate_hr_min_bpm') or 0.0)),
        'signal_std': float(np.nanstd(segment)),
        'signal_iqr': float(np.nanpercentile(segment, 75) - np.nanpercentile(segment, 25)),
        'diff_std': float(np.nanstd(diff)) if len(diff) else 0.0,
        'motion_p95': float(np.nanpercentile(np.abs(diff), 95)) if len(diff) else 0.0,
    }
    features.update(extra)
    return features


def build_table(manifest: Path, epoch_seconds: float) -> pd.DataFrame:
    mf = pd.read_csv(manifest)
    missing = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in mf.columns]
    if missing:
        raise ValueError(f'manifest missing columns {missing}; required={REQUIRED_MANIFEST_COLUMNS}; {STAGE_FILE_COLUMNS}')
    rows = []
    for _, row in mf.iterrows():
        record = str(row['record'])
        fs = float(row['sampling_rate'])
        signal = load_csv_signal(str(row['signal_path']), fs, row.get('column') if pd.notna(row.get('column', None)) else None)
        stages = read_stage_file(Path(row['stage_path']), epoch_seconds)
        for _, st in stages.iterrows():
            feat = epoch_features(signal.values, signal.sampling_rate, float(st['start_sec']), epoch_seconds, record)
            if feat is None:
                continue
            feat['stage'] = st['stage']
            rows.append(feat)
    if not rows:
        raise RuntimeError('no valid labeled epochs extracted')
    return pd.DataFrame(rows)


def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {'record', 'start_sec', 'stage', 'quality'}
    cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return sorted(cols)


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a BCG sleep-stage classifier from a user-supplied PSG-labeled BCG manifest.')
    parser.add_argument('--manifest', type=Path, required=False, help='CSV with record, signal_path, sampling_rate, optional column, stage_path')
    parser.add_argument('--epoch-seconds', type=float, default=30.0)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--metrics-output', type=Path, default=METRICS)
    args = parser.parse_args()
    if args.manifest is None:
        print(json.dumps({
            'status': 'needs_manifest',
            'required_manifest_columns': REQUIRED_MANIFEST_COLUMNS,
            'optional_manifest_columns': ['column'],
            'stage_file_contract': STAGE_FILE_COLUMNS,
            'reason': 'No public PSG-labeled BCG benchmark was confirmed; train this model when synchronized BCG+PSG hypnogram labels are available.',
        }, indent=2))
        return
    df = build_table(args.manifest, args.epoch_seconds)
    cols = feature_columns(df)
    x = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    y = df['stage'].to_numpy()
    groups = df['record'].to_numpy()
    base = VotingClassifier([
        ('rf', RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight='balanced_subsample', random_state=7, n_jobs=-1)),
        ('hgb', HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, l2_regularization=0.05, random_state=7)),
    ], voting='soft')
    model = Pipeline([('scale', StandardScaler()), ('clf', base)])
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 3:
        splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
        splits = splitter.split(x, y, groups)
        split_name = 'group_kfold_by_record'
    else:
        splitter = StratifiedKFold(n_splits=min(5, np.min(np.bincount(pd.factorize(y)[0]))), shuffle=True, random_state=7)
        splits = splitter.split(x, y)
        split_name = 'stratified_kfold_epoch_level_fallback'
    preds = np.empty_like(y, dtype=object)
    for train_idx, test_idx in splits:
        fold_model = Pipeline([('scale', StandardScaler()), ('clf', base)])
        fold_model.fit(x[train_idx], y[train_idx])
        preds[test_idx] = fold_model.predict(x[test_idx])
    metrics = {
        'num_epochs': int(len(df)),
        'num_records': int(len(unique_groups)),
        'classes': sorted(set(y.tolist())),
        'split': split_name,
        'accuracy': float(accuracy_score(y, preds)),
        'balanced_accuracy': float(balanced_accuracy_score(y, preds)),
        'macro_f1': float(f1_score(y, preds, average='macro')),
        'cohen_kappa': float(cohen_kappa_score(y, preds)),
        'classification_report': classification_report(y, preds, output_dict=True, zero_division=0),
        'feature_columns': cols,
        'disclaimer': 'BCG sleep staging must be validated by subject-held-out PSG labels; do not use epoch-level fallback as final evidence.',
    }
    model.fit(x, y)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'feature_columns': cols, 'classes': sorted(set(y.tolist())), 'epoch_seconds': args.epoch_seconds, 'metrics': metrics}, args.output)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k not in {'classification_report', 'feature_columns'}}, indent=2))
    print('wrote', args.output)
    print('wrote', args.metrics_output)


if __name__ == '__main__':
    main()
