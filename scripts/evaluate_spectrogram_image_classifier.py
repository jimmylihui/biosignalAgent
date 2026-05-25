#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.tools.spectrogram_tools import Signal_render_spectrogram_image


def load_values(path: str) -> np.ndarray:
    frame = pd.read_csv(path)
    col = 'signal' if 'signal' in frame.columns else frame.select_dtypes('number').columns[-1]
    return frame[col].to_numpy(dtype=float)


def write_window(values: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'signal': values}).to_csv(out_path, index=False)


def image_vector(path: str, size: int) -> np.ndarray:
    img = Image.open(path).convert('L').resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.ravel()


def label_from_emg_record(record: str) -> str:
    name = str(record).lower()
    if 'healthy' in name: return 'healthy'
    if 'myopathy' in name: return 'myopathy'
    if 'neuropathy' in name: return 'neuropathy'
    return 'unknown'


def collect_pcg(manifest: dict[str, Any], out_dir: Path, image_size: int) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    rows=[]; x=[]; y=[]
    for rec in manifest.get('records', []):
        if rec.get('label') not in {'normal','abnormal'}:
            continue
        label = rec['label']
        out_png = out_dir / f"pcg_{rec['record']}.png"
        render = Signal_render_spectrogram_image(rec['path'], float(rec['sampling_rate']), out_png=str(out_png), modality='pcg', window_seconds=1.0, overlap=0.5, max_frequency_hz=500, width=256, height=256)
        if render.get('error'):
            rows.append({'record': rec.get('record'), 'truth': label, 'error': render.get('error')})
            continue
        rows.append({'record': rec.get('record'), 'truth': label, 'spectrogram_image': render['image_path'], 'source_record': rec.get('record')})
        x.append(image_vector(render['image_path'], image_size)); y.append(label)
    return rows, np.asarray(x), np.asarray(y)


def collect_emg(manifest: dict[str, Any], out_dir: Path, image_size: int, window_seconds: float) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    rows=[]; x=[]; y=[]
    window_dir = out_dir / 'windows'
    for rec in manifest.get('records', []):
        if rec.get('modality') != 'emg' or rec.get('dataset') != 'emgdb':
            continue
        label = label_from_emg_record(rec.get('record'))
        values = load_values(rec['path'])
        fs = float(rec['sampling_rate'])
        win = max(1, int(window_seconds * fs))
        for i in range(len(values)//win):
            chunk_path = window_dir / f"{rec['record']}_w{i:02d}.csv"
            write_window(values[i*win:(i+1)*win], chunk_path)
            out_png = out_dir / f"emg_{rec['record']}_w{i:02d}.png"
            render = Signal_render_spectrogram_image(str(chunk_path), fs, out_png=str(out_png), modality='emg', window_seconds=0.25, overlap=0.5, max_frequency_hz=450, width=256, height=256)
            if render.get('error'):
                rows.append({'record': rec.get('record'), 'window': i, 'truth': label, 'error': render.get('error')})
                continue
            rows.append({'record': rec.get('record'), 'window': i, 'truth': label, 'spectrogram_image': render['image_path'], 'source_record': rec.get('record')})
            x.append(image_vector(render['image_path'], image_size)); y.append(label)
    return rows, np.asarray(x), np.asarray(y)


def choose_cv(y: np.ndarray):
    counts = Counter(y.tolist())
    min_class = min(counts.values()) if counts else 0
    if min_class >= 3:
        n = min(5, min_class)
        return StratifiedKFold(n_splits=n, shuffle=True, random_state=13), f'stratified_{n}_fold'
    return LeaveOneOut(), 'leave_one_out'


def metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    labels = sorted(set(y_true.tolist()) | set(pred.tolist()))
    out = {'accuracy': float(accuracy_score(y_true, pred)), 'macro_f1': float(f1_score(y_true, pred, average='macro', zero_division=0))}
    if len(labels) == 2:
        positive = 'abnormal' if 'abnormal' in labels else labels[-1]
        y_bin = [1 if item == positive else 0 for item in y_true]
        p_bin = [1 if item == positive else 0 for item in pred]
        tn, fp, fn, tp = confusion_matrix(y_bin, p_bin, labels=[0,1]).ravel()
        out.update({
            'positive_label': positive,
            'precision': float(precision_score(y_bin, p_bin, zero_division=0)),
            'recall_sensitivity': float(recall_score(y_bin, p_bin, zero_division=0)),
            'specificity': float(tn/(tn+fp)) if tn+fp else 0.0,
            'f1': float(f1_score(y_bin, p_bin, zero_division=0)),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['pcg_murmur','emg_condition'], required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--image-size', type=int, default=64)
    ap.add_argument('--window-seconds', type=float, default=1.0)
    ap.add_argument('--out-dir', default='/data1/jiahui/biosignal-agent/datasets/processed/spectrogram_image_benchmark')
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-csv', required=True)
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    out_dir = Path(args.out_dir) / args.task
    if args.task == 'pcg_murmur':
        rows, x, y = collect_pcg(manifest, out_dir, args.image_size)
        note = 'record-level PCG spectrogram image classifier baseline.'
    else:
        rows, x, y = collect_emg(manifest, out_dir, args.image_size, args.window_seconds)
        note = 'window-level EMG spectrogram image classifier smoke test; not subject-independent.'
    ok_rows = [r for r in rows if not r.get('error')]
    cv, cv_name = choose_cv(y)
    n_components = max(1, min(4, x.shape[0]-2, x.shape[1]))
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components, random_state=13)),
        ('classifier', LogisticRegression(class_weight='balanced', solver='liblinear', random_state=13)),
    ])
    pred = cross_val_predict(model, x, y, cv=cv)
    for row, p in zip(ok_rows, pred.tolist()):
        row['prediction'] = p
    labels = sorted(set(y.tolist()) | set(pred.tolist()))
    report = {
        'task': args.task,
        'manifest': args.manifest,
        'note': note,
        'num_records': len(ok_rows),
        'image_size': args.image_size,
        'model': f'spectrogram_image_pixels_pca{n_components}_logistic_regression',
        'cv': cv_name,
        'truth_counts': dict(Counter(y.tolist())),
        'prediction_counts': dict(Counter(pred.tolist())),
        'metrics': metrics(y, pred),
        'confusion_matrix': {'labels': labels, 'matrix': confusion_matrix(y, pred, labels=labels).tolist()},
        'rows': rows,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    with Path(args.out_csv).open('w', newline='') as f:
        keys = sorted({k for r in rows for k in r})
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({k: report[k] for k in ['task','num_records','cv','truth_counts','prediction_counts','metrics','confusion_matrix']}, indent=2))

if __name__ == '__main__':
    main()
