
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.tools.common import load_csv_signal
from biosignal_agent.tools.ecg_tools import ECG_detect_r_peaks, _clean_rr_intervals, _ecg_feature_dict, _ecg_artifact_metrics, _predict_arrhythmia_beat_model, ECG_ARRHYTHMIA_BEAT_MODEL_PATH

CLASSES = ['normal', 'af', 'other_rhythm']
OUT = Path('/data1/jiahui/biosignal-agent/outputs')
MODEL_PATH = OUT / 'ecg_rhythm_feature_classifier.joblib'

def safe(v, default=0.0):
    try: v=float(v)
    except Exception: return default
    return v if np.isfinite(v) else default

def feature_row(record):
    data = load_csv_signal(record['path'], float(record['sampling_rate']), None)
    peak = ECG_detect_r_peaks(record['path'], float(record['sampling_rate']), None)
    peaks = np.asarray(peak.get('r_peak_indices', []), dtype=int)
    features = _ecg_feature_dict(data.values, data.sampling_rate, peaks if len(peaks) else None)
    rr = _clean_rr_intervals(peaks, data.sampling_rate) if len(peaks) > 1 else np.asarray([])
    diff = np.diff(rr) if len(rr) > 1 else np.asarray([])
    artifacts = _ecg_artifact_metrics(data.values, data.sampling_rate)
    beat_score, beat_details, *_ = _predict_arrhythmia_beat_model(ECG_ARRHYTHMIA_BEAT_MODEL_PATH, data.values, data.sampling_rate, peaks)
    beat_details = beat_details or {}
    subtype = beat_details.get('subtype_details') or {}
    subtype_counts = subtype.get('subtype_counts') or {}
    total_sub = max(1, sum(int(v) for v in subtype_counts.values()))
    nbeats = max(1, int(len(peaks)))
    vals = {
        'duration_s': safe(features.get('duration_s')),
        'heart_rate_bpm': safe(features.get('heart_rate_bpm')),
        'peaks_per_minute': safe(features.get('peaks_per_minute')),
        'rr_count': safe(features.get('rr_count')),
        'rr_mean_s': safe(features.get('rr_mean_s')),
        'rr_median_s': safe(features.get('rr_median_s')),
        'rr_std_s': safe(features.get('rr_std_s')),
        'rr_cv': safe(features.get('rr_cv')),
        'rr_iqr_s': safe(features.get('rr_iqr_s')),
        'rr_range_s': safe(features.get('rr_range_s')),
        'rmssd_s': safe(features.get('rmssd_s')),
        'pnn50': safe(features.get('pnn50')),
        'pnn120': safe(features.get('pnn120')),
        'successive_change_fraction': safe(features.get('successive_change_fraction')),
        'pause_fraction': safe(features.get('pause_fraction')),
        'short_rr_fraction': safe(features.get('short_rr_fraction')),
        'long_rr_fraction': safe(features.get('long_rr_fraction')),
        'rr_lf_hf_ratio': safe(features.get('rr_lf_hf_ratio')),
        'rr_dominant_freq_hz': safe(features.get('rr_dominant_freq_hz')),
        'rr_diff_mean_abs': float(np.mean(np.abs(diff))) if len(diff) else 0.0,
        'rr_diff_std': float(np.std(diff)) if len(diff) else 0.0,
        'rr_diff_sign_changes': float(np.mean(np.diff(np.sign(diff)) != 0)) if len(diff) > 2 else 0.0,
        'signal_std': safe(features.get('signal_std')),
        'signal_mad': safe(features.get('signal_mad')),
        'flat_fraction': safe(features.get('flat_fraction')),
        'baseline_wander_ratio': safe(artifacts.get('baseline_wander_ratio')),
        'high_frequency_noise_ratio': safe(artifacts.get('high_frequency_noise_ratio')),
        'powerline_noise_ratio': safe(artifacts.get('powerline_noise_ratio')),
        'beat_max_abnormal_probability': safe(beat_details.get('max_beat_abnormal_probability')),
        'beat_top5_abnormal_probability': safe(beat_details.get('mean_top5_beat_abnormal_probability')),
        'beat_abnormal_fraction': safe(beat_details.get('beat_abnormal_fraction_at_threshold')),
        'beat_abnormal_count_per_beat': safe(beat_details.get('num_abnormal_beats_at_threshold')) / nbeats,
        'subtype_s_fraction': safe(subtype_counts.get('S')) / total_sub,
        'subtype_v_fraction': safe(subtype_counts.get('V')) / total_sub,
        'subtype_f_fraction': safe(subtype_counts.get('F')) / total_sub,
        'subtype_q_fraction': safe(subtype_counts.get('Q')) / total_sub,
    }
    return vals

def af_metrics(y_true, y_pred):
    yt=np.asarray([1 if y=='af' else 0 for y in y_true]); yp=np.asarray([1 if y=='af' else 0 for y in y_pred])
    return {'precision':float(precision_score(yt,yp,zero_division=0)),'recall':float(recall_score(yt,yp,zero_division=0)),'f1':float(f1_score(yt,yp,zero_division=0))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',default='/data1/jiahui/biosignal-agent/datasets/processed/ecg_rhythm_beat_full_manifest.json'); ap.add_argument('--model-path',type=Path,default=MODEL_PATH); args=ap.parse_args()
    manifest=json.load(open(args.manifest)); records=manifest['records']
    rows=[]; y=[]; groups=[]
    for i,r in enumerate(records,1):
        if i%100==0: print('features',i,flush=True)
        rows.append(feature_row(r)); y.append(r['coarse_rhythm_label']); groups.append(str(r['record']))
    feature_names=list(rows[0].keys()); X=np.asarray([[row[k] for k in feature_names] for row in rows],dtype=float); y=np.asarray(y)
    candidates={
      'extra_trees': ExtraTreesClassifier(n_estimators=450, max_features='sqrt', min_samples_leaf=2, class_weight='balanced', random_state=17, n_jobs=-1),
      'random_forest': RandomForestClassifier(n_estimators=350, max_features='sqrt', min_samples_leaf=2, class_weight='balanced', random_state=23, n_jobs=-1),
      'logreg': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced', multi_class='auto')),
    }
    reports={}; best_name=None; best_score=-1; best_pred=None
    splits=list(GroupKFold(n_splits=5).split(X,y,groups=groups))
    for name,model in candidates.items():
        pred=np.empty(len(y),dtype=object); pred[:]='normal'
        for fold,(tr,va) in enumerate(splits):
            m=model
            # fresh clone without pulling sklearn clone dependency explicitly
            import sklearn.base
            m=sklearn.base.clone(model)
            m.fit(X[tr],y[tr]); pred[va]=m.predict(X[va])
        rep={'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),'weighted_f1':float(f1_score(y,pred,average='weighted',zero_division=0)),'af_metrics':af_metrics(y,pred),'class_report':classification_report(y,pred,labels=CLASSES,zero_division=0,output_dict=True)}
        reports[name]=rep; score=rep['af_metrics']['f1'] + 0.25*rep['macro_f1']
        print(name,json.dumps({k:rep[k] for k in ['accuracy','macro_f1','weighted_f1','af_metrics']},indent=2),flush=True)
        if score>best_score: best_score=score; best_name=name; best_pred=pred
    final=candidates[best_name]; final.fit(X,y)
    bundle={'model':final,'feature_names':feature_names,'classes':CLASSES,'best_model':best_name,'cv_metrics':reports[best_name],'all_model_reports':reports,'label_counts':dict(Counter(map(str,y))),'manifest':args.manifest}
    args.model_path.parent.mkdir(parents=True,exist_ok=True); joblib.dump(bundle,args.model_path)
    report={'model_path':str(args.model_path),**{k:bundle[k] for k in ['best_model','cv_metrics','all_model_reports','label_counts','manifest']}}
    (OUT/'ecg_rhythm_feature_classifier_train_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__': main()
