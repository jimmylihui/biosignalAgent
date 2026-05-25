from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from scipy import signal as scipy_signal
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.tools.common import load_csv_signal
from biosignal_agent.tools.peak_detectors import neurokit_nabian2018_peaks
from scripts.train_ecg_st_feature_classifier import st_features

OUT = Path('/data1/jiahui/biosignal-agent/outputs')


def safe(v: float, default: float = 0.0) -> float:
    try:
        v = float(v)
    except Exception:
        return default
    return v if np.isfinite(v) else default


def width_features(values: np.ndarray, fs: float) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < int(3 * fs):
        return {}
    try:
        b, a = scipy_signal.butter(2, [0.5 / (fs / 2), min(35.0 / (fs / 2), 0.99)], btype='bandpass')
        xf = scipy_signal.filtfilt(b, a, x)
    except Exception:
        xf = x - np.nanmedian(x)
    peaks, _ = neurokit_nabian2018_peaks(xf, fs, low_hz=None, high_hz=None, fallback_threshold_scale=0.6)
    peaks = np.asarray(peaks, dtype=int)
    peaks = peaks[(peaks > int(0.18 * fs)) & (peaks < len(xf) - int(0.22 * fs))]
    widths = []
    slopes = []
    amps = []
    for p in peaks:
        lo = p - int(0.16 * fs); hi = p + int(0.16 * fs)
        seg = xf[lo:hi]
        if len(seg) < 8:
            continue
        base = float(np.median(np.r_[seg[:max(1,int(0.04*fs))], seg[-max(1,int(0.04*fs)):]]))
        centered = seg - base
        peak_local = int(np.argmax(np.abs(centered)))
        amp = float(np.max(centered) - np.min(centered))
        if amp <= 1e-8:
            continue
        thresh = 0.35 * np.max(np.abs(centered))
        above = np.flatnonzero(np.abs(centered) >= thresh)
        if len(above):
            widths.append(float((above[-1] - above[0] + 1) * 1000.0 / fs))
        diff = np.diff(seg)
        slopes.append(float(np.percentile(np.abs(diff), 95)))
        amps.append(amp)
    rr = np.diff(peaks) / fs if len(peaks) > 1 else np.asarray([])
    def stat(prefix, arr):
        arr = np.asarray(arr, dtype=float); arr = arr[np.isfinite(arr)]
        if not len(arr):
            return {prefix + '_median': 0.0, prefix + '_iqr': 0.0, prefix + '_p90': 0.0}
        return {prefix + '_median': float(np.median(arr)), prefix + '_iqr': float(np.percentile(arr,75)-np.percentile(arr,25)), prefix + '_p90': float(np.percentile(arr,90))}
    out = {'num_beats': float(len(peaks)), 'rr_cv': float(np.std(rr)/np.mean(rr)) if len(rr) and np.mean(rr)>0 else 0.0}
    out.update(stat('qrs_width35_ms', widths)); out.update(stat('qrs_slope', slopes)); out.update(stat('qrs_amp', amps))
    return {k: safe(v) for k, v in out.items()}


def features(path: str, fs: float) -> dict[str, float]:
    data = load_csv_signal(path, fs, None)
    out = {}
    out.update(st_features(path, fs))
    out.update(width_features(data.values, data.sampling_rate))
    return {k: safe(v) for k, v in out.items()}


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return {'threshold': float(threshold), 'accuracy': float(accuracy_score(y,pred)), 'precision': float(precision_score(y,pred,zero_division=0)), 'recall': float(recall_score(y,pred,zero_division=0)), 'f1': float(f1_score(y,pred,zero_division=0)), 'average_precision': float(average_precision_score(y,p)) if int(np.sum(y)) else 0.0, 'roc_auc': float(roc_auc_score(y,p)) if len(set(map(int,y)))>1 else 0.0, 'class_report': classification_report(y,pred,labels=[0,1],zero_division=0,output_dict=True)}


def train_target(rows, target: str, model_path: Path):
    y = np.asarray([int(r[f'label_{target}']) for r in rows], dtype=int)
    groups = np.asarray([str(r.get('strat_fold') or r.get('group')) for r in rows])
    feats=[]
    for i,r in enumerate(rows,1):
        if i % 200 == 0: print(target, 'features', i, flush=True)
        feats.append(features(r['path'], r['sampling_rate']))
    names = sorted({k for row in feats for k in row})
    X = np.asarray([[row.get(k,0.0) for k in names] for row in feats], dtype=float)
    candidates = {
        'logreg': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced')),
        'extra_trees': ExtraTreesClassifier(n_estimators=500, max_features='sqrt', min_samples_leaf=2, class_weight='balanced', random_state=7, n_jobs=-1),
        'random_forest': RandomForestClassifier(n_estimators=500, max_features='sqrt', min_samples_leaf=2, class_weight='balanced', random_state=11, n_jobs=-1),
        'hgb': HistGradientBoostingClassifier(max_iter=220, learning_rate=0.04, l2_regularization=0.05, random_state=13),
    }
    splits = list(GroupKFold(n_splits=min(5, len(set(groups)))).split(X, y, groups=groups))
    reports={}; best_name=None; best_score=-1; best_thr=0.5
    for name, model in candidates.items():
        proba=np.zeros(len(y),dtype=float)
        for tr,va in splits:
            clf=clone(model); clf.fit(X[tr],y[tr]); proba[va]=clf.predict_proba(X[va])[:,1]
        th=float(max(((f1_score(y,proba>=t,zero_division=0),t) for t in np.linspace(0.05,0.95,91)), key=lambda z:z[0])[1])
        rep=metrics(y,proba,th); reports[name]=rep
        print(target, name, json.dumps({k:rep[k] for k in ['threshold','precision','recall','f1','average_precision','roc_auc']},indent=2), flush=True)
        score=rep['f1'] + 0.1*rep['average_precision']
        if score > best_score:
            best_score=score; best_name=name; best_thr=th
    final=clone(candidates[best_name]); final.fit(X,y)
    bundle={'model': final, 'feature_names': names, 'target': target, 'threshold': best_thr, 'best_model': best_name, 'cv_metrics': reports[best_name], 'all_model_reports': reports, 'label_counts': dict(Counter(map(int,y)))}
    model_path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(bundle, model_path)
    return {'model_path': str(model_path), **{k: bundle[k] for k in ['target','threshold','best_model','cv_metrics','all_model_reports','label_counts']}}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/processed/ptbxl_superclass_lead2_manifest.json')); ap.add_argument('--out-dir', type=Path, default=OUT); ap.add_argument('--report-path', type=Path, default=OUT/'ecg_ptbxl_superclass_feature_train_report.json'); args=ap.parse_args()
    rows=json.load(open(args.manifest))['rows']
    report={'manifest': str(args.manifest), 'num_records': len(rows), 'targets': {}}
    for target in ['cd','sttc']:
        report['targets'][target]=train_target(rows,target,args.out_dir/f'ecg_ptbxl_{target}_lead2_feature_classifier.joblib')
    args.report_path.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
