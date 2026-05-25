
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from train_pcg_murmur_patient_multiloc_cnn import (
    LOCATIONS,
    build_patients,
    set_seed,
    train_fold,
    train_full,
)


def metrics(y_true, prob, threshold=0.5):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else None,
        "threshold": float(threshold),
    }


def best_threshold(y, prob):
    best = (0.5, metrics(y, prob, 0.5))
    for t in np.linspace(0.1, 0.9, 81):
        m = metrics(y, prob, float(t))
        if (m["f1"], m["accuracy"], m["specificity"]) > (best[1]["f1"], best[1]["accuracy"], best[1]["specificity"]):
            best = (float(t), m)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/datasets/processed/circor_pcg_murmur_manifest.json')
    ap.add_argument('--out-model', default='/data1/jiahui/biosignal-agent/outputs/pcg_murmur_patient_multiloc_ensemble.pt')
    ap.add_argument('--report', default='/data1/jiahui/biosignal-agent/outputs/pcg_murmur_patient_multiloc_ensemble_report.json')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=8e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--target-fs', type=int, default=1000)
    ap.add_argument('--seconds', type=float, default=8.0)
    ap.add_argument('--freq-bins', type=int, default=80)
    ap.add_argument('--time-bins', type=int, default=128)
    ap.add_argument('--embedding-dim', type=int, default=128)
    ap.add_argument('--seeds', nargs='+', type=int, default=[59, 101, 211])
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    patients = build_patients(manifest, include_unknown=False)
    patients = [p for p in patients if p['y'] in {0, 1}]
    y = np.asarray([p['y'] for p in patients], dtype=int)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seeds[0])
    all_y = []
    all_prob = []
    folds = []
    for fold, (tr, te) in enumerate(cv.split(np.arange(len(patients)), y), 1):
        seed_probs = []
        fold_y = None
        seed_reports = []
        for seed in args.seeds:
            set_seed(seed + fold)
            best = train_fold(patients, tr, te, args, seed + fold)
            seed_probs.append(np.asarray(best['prob'], dtype=float))
            fold_y = best['y']
            seed_reports.append({
                'seed': seed,
                'epoch': best['epoch'],
                'metrics': best['metrics'],
                'best_threshold': best['best_threshold'],
                'best_threshold_metrics': best['best_threshold_metrics'],
            })
        prob = np.mean(np.stack(seed_probs, axis=0), axis=0).tolist()
        m = metrics(fold_y, prob, 0.5)
        bt, bm = best_threshold(fold_y, prob)
        all_y.extend(fold_y)
        all_prob.extend(prob)
        rep = {
            'fold': fold,
            'num_train': len(tr),
            'num_test': len(te),
            'metrics': m,
            'best_threshold': bt,
            'best_threshold_metrics': bm,
            'seed_reports': seed_reports,
        }
        folds.append(rep)
        print(json.dumps({k: rep[k] for k in ['fold', 'num_train', 'num_test', 'metrics', 'best_threshold', 'best_threshold_metrics']}), flush=True)
    cv_metrics = metrics(all_y, all_prob, 0.5)
    bt, bm = best_threshold(all_y, all_prob)
    states = []
    for seed in args.seeds:
        set_seed(seed + 999)
        states.append({'seed': seed, 'model_state_dict': train_full(patients, args)})
    payload = {
        'ensemble_state_dicts': states,
        'architecture': 'PatientMultiLocCNN_logfreq_spectrogram_attention_ensemble',
        'locations': LOCATIONS,
        'target_fs': args.target_fs,
        'seconds': args.seconds,
        'freq_bins': args.freq_bins,
        'time_bins': args.time_bins,
        'embedding_dim': args.embedding_dim,
        'seeds': args.seeds,
        'cv_metrics': cv_metrics,
        'best_threshold': bt,
        'best_threshold_metrics': bm,
        'reference': 'CirCor 1.0.3 patient-level murmur present/absent, multi-location attention CNN, multi-seed probability ensemble, stratified patient CV.',
    }
    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out_model)
    report = {
        'manifest': args.manifest,
        'num_patients': len(patients),
        'label_counts': dict(Counter(['present' if v else 'absent' for v in y.tolist()])),
        'model_out': args.out_model,
        'seeds': args.seeds,
        'folds': folds,
        'cv_metrics': cv_metrics,
        'best_threshold': bt,
        'best_threshold_metrics': bm,
    }
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
