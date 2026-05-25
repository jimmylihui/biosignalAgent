from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from collections import Counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from biosignal_agent.tools.ppg_tools import PPG_estimate_respiration_modulation
from scripts.evaluate_ppg_respiration_bidmc import iter_wfdb_records, reference_resp_rate
from scripts.train_ppg_respiration_candidate_selector import collect_candidates, FEATURES, evaluate_rows

AERATION_FS_OUT = 125.0


def reference_rate_from_waveform(values: np.ndarray, fs: float) -> float | None:
    values = values[np.isfinite(values)]
    if len(values) < fs * 20:
        return None
    values = values - np.nanmedian(values)
    high = min(0.7, fs * 0.45)
    if high <= 0.08:
        return None
    sos = scipy_signal.butter(3, [0.08/(0.5*fs), high/(0.5*fs)], btype='bandpass', output='sos')
    filt = scipy_signal.sosfiltfilt(sos, values)
    freqs, psd = scipy_signal.welch(filt, fs=fs, nperseg=min(len(filt), int(fs*32)))
    mask=(freqs>=0.08)&(freqs<=high)
    if not np.any(mask): return None
    return float(freqs[mask][int(np.argmax(psd[mask]))]*60.0)


def ensure_csv(path: Path, signal: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pd.DataFrame({'signal': signal}).to_csv(path, index=False)
    return path


def build_bidmc(raw_dir: Path, processed_dir: Path) -> list[dict[str, Any]]:
    items=[]
    for item in iter_wfdb_records(raw_dir, processed_dir):
        truth = reference_resp_rate(item['resp'], item['fs']).get('respiratory_rate_bpm')
        if truth is None: continue
        items.append({'record': f"bidmc_{item['record']}", 'group': f"bidmc_{item['record']}", 'db': 'BIDMC', 'ppg_path': item['ppg_path'], 'fs': float(item['fs']), 'truth': float(truth)})
    return items


def build_capnobase(raw_dir: Path, processed_dir: Path) -> list[dict[str, Any]]:
    items=[]
    for sig_path in sorted(raw_dir.glob('*_8min_signal.tab')):
        rec=sig_path.name.replace('_signal.tab','')
        param=raw_dir/f'{rec}_param.tab'
        if not param.exists(): continue
        sig=pd.read_csv(sig_path, sep='\t')
        par=pd.read_csv(param, sep='\t')
        fs=float(par['samplingrate_pleth'].iloc[0])
        ppg=sig['pleth_y'].to_numpy(float)
        co2=sig['co2_y'].to_numpy(float)
        truth=reference_rate_from_waveform(co2, fs)
        if truth is None: continue
        ppg_path=ensure_csv(processed_dir/f'{rec}_ppg_full.csv', ppg)
        items.append({'record': f"capnobase_{rec}", 'group': f"capnobase_{rec}", 'db': 'CapnoBase', 'ppg_path': ppg_path, 'fs': fs, 'truth': float(truth)})
    return items



def _unique_positive_time_series(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(times) & np.isfinite(values) & (times > 0)
    times = times[mask]
    values = values[mask]
    if len(times) < 4:
        return times, values
    order = np.argsort(times)
    times = times[order]
    values = values[order]
    keep = np.r_[True, np.diff(times) > 1e-6]
    return times[keep], values[keep]


def build_aeration(raw_dir: Path, processed_dir: Path) -> list[dict[str, Any]]:
    items = []
    processed_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in sorted(raw_dir.glob('ProcessedData_Subject*_PEEP.csv')):
        record = csv_path.stem.replace('ProcessedData_', 'aeration_')
        df = pd.read_csv(csv_path)
        flow_t, flow = _unique_positive_time_series(df['PSD Time [s]'].to_numpy(float), df['PSD Flow [L/s]'].to_numpy(float))
        if len(flow_t) < 200:
            continue
        flow_fs = float(1.0 / np.nanmedian(np.diff(flow_t)))
        truth = reference_rate_from_waveform(flow, flow_fs)
        if truth is None:
            continue
        for channel in ('PPG0', 'PPG1', 'PPG2'):
            if channel not in df.columns:
                continue
            ppg_t = df['PPG Time [s]'].to_numpy(float)
            ppg = df[channel].to_numpy(float)
            mask = np.isfinite(ppg_t) & np.isfinite(ppg)
            ppg_t = ppg_t[mask]
            ppg = ppg[mask]
            if len(ppg_t) < AERATION_FS_OUT * 20:
                continue
            order = np.argsort(ppg_t)
            ppg_t = ppg_t[order]
            ppg = ppg[order]
            keep = np.r_[True, np.diff(ppg_t) > 1e-6]
            ppg_t = ppg_t[keep]
            ppg = ppg[keep]
            duration = float(ppg_t[-1] - ppg_t[0])
            if duration < 20.0:
                continue
            out_n = int(duration * AERATION_FS_OUT)
            out_t = np.arange(out_n, dtype=float) / AERATION_FS_OUT + float(ppg_t[0])
            ppg_resampled = np.interp(out_t, ppg_t, ppg)
            channel_record = f'{record}_{channel}'
            ppg_path = ensure_csv(processed_dir / f'{channel_record}_125hz.csv', ppg_resampled)
            items.append({'record': channel_record, 'group': record, 'db': 'Aeration', 'ppg_path': ppg_path, 'fs': AERATION_FS_OUT, 'truth': float(truth)})
    return items

def build_candidate_rows(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows=[]; current=[]
    for item in items:
        out=PPG_estimate_respiration_modulation(str(item['ppg_path']), item['fs'])
        estimate=out.get('respiratory_rate_bpm')
        current.append({'record': item['record'], 'group': item.get('group', item['record']), 'db': item['db'], 'truth': item['truth'], 'estimate': estimate, 'error': abs(float(estimate)-item['truth']) if estimate is not None else None})
        cands=collect_candidates(out)
        if not cands: continue
        distances=[abs(c['rate_bpm']-item['truth']) for c in cands]
        best_i=int(np.argmin(distances))
        for i,c in enumerate(cands):
            rows.append({**c, 'record': item['record'], 'group': item.get('group', item['record']), 'db': item['db'], 'truth': item['truth'], 'label': 1 if i==best_i else 0, 'abs_error_if_selected': float(abs(c['rate_bpm']-item['truth']))})
    return rows,current


def models(seed:int):
    return {
        'logistic_balanced': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000,class_weight='balanced',C=0.8,random_state=seed)),
        'random_forest_balanced': RandomForestClassifier(n_estimators=500,max_depth=6,min_samples_leaf=3,class_weight='balanced',random_state=seed),
        'extra_trees_balanced': ExtraTreesClassifier(n_estimators=500,max_depth=6,min_samples_leaf=3,class_weight='balanced',random_state=seed),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=160,max_depth=2,learning_rate=0.04,random_state=seed),
    }


def eval_by_database(rec_rows):
    out={}
    for db in sorted(set(r['record'].split('_',1)[0] for r in rec_rows)):
        vals=[r['abs_error_bpm'] for r in rec_rows if r['record'].startswith(db+'_')]
        out[db]={'num_records':len(vals),'mae_bpm':float(np.mean(vals)) if vals else None,'median_abs_error_bpm':float(np.median(vals)) if vals else None}
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bidmc-raw', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/raw/bidmc'))
    ap.add_argument('--bidmc-processed', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/processed/bidmc_full'))
    ap.add_argument('--capnobase-raw', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/raw/capnobase_benchmark'))
    ap.add_argument('--capnobase-processed', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/processed/capnobase_resp'))
    ap.add_argument('--aeration-raw', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/raw/respiratory_heartrate_aeration/Processed_Dataset'))
    ap.add_argument('--aeration-processed', type=Path, default=Path('/data1/jiahui/biosignal-agent/datasets/processed/aeration_ppg_resp'))
    ap.add_argument('--out-model', default='/data1/jiahui/biosignal-agent/outputs/ppg_respiration_candidate_selector_multidb.joblib')
    ap.add_argument('--report', default='/data1/jiahui/biosignal-agent/outputs/ppg_respiration_candidate_selector_multidb_report.json')
    ap.add_argument('--seed', type=int, default=37)
    args=ap.parse_args()

    items=(build_bidmc(args.bidmc_raw,args.bidmc_processed)
           +build_capnobase(args.capnobase_raw,args.capnobase_processed)
           +build_aeration(args.aeration_raw,args.aeration_processed))
    rows,current=build_candidate_rows(items)
    x=np.asarray([[r[f] for f in FEATURES] for r in rows], dtype=float)
    y=np.asarray([r['label'] for r in rows])
    groups=np.asarray([r.get('group', r['record']) for r in rows])
    dbs=np.asarray([r['db'] for r in rows])

    logo=LeaveOneGroupOut(); reports={}; best_name=None; best_score=999; best_model=None
    for name,model in models(args.seed).items():
        prob=cross_val_predict(model,x,y,groups=groups,cv=logo,method='predict_proba')[:,1]
        rep=evaluate_rows(rows,prob)
        rep['by_database']=eval_by_database(rep['rows'])
        reports[name]=rep
        print(name, rep['mae_bpm'], rep['median_abs_error_bpm'], rep['by_database'], flush=True)
        if rep['mae_bpm']<best_score:
            best_score=rep['mae_bpm']; best_name=name; best_model=model

    best_model.fit(x,y)
    payload={'model':best_model,'model_name':best_name,'feature_names':FEATURES,'cv_metrics':{k:v for k,v in reports[best_name].items() if k!='rows'},'reference':'BIDMC RESP waveform + CapnoBase CO2 waveform + Aeration flow waveform spectral RR; PPG-only multi-source candidate selector; leave-one-record-out.'}
    Path(args.out_model).parent.mkdir(parents=True,exist_ok=True); joblib.dump(payload,args.out_model)
    current_errors=[r['error'] for r in current if r['error'] is not None]
    current_by_db={}
    for db in sorted(set(r['db'] for r in current)):
        vals=[r['error'] for r in current if r['db']==db and r['error'] is not None]
        current_by_db[db]={'num_records':len(vals),'mae_bpm':float(np.mean(vals)) if vals else None,'median_abs_error_bpm':float(np.median(vals)) if vals else None}
    report={'num_records':len(items),'record_counts':dict(Counter(i['db'] for i in items)),'num_candidate_rows':len(rows),'label_counts':{str(k):int(v) for k,v in Counter(y).items()},'features':FEATURES,'current_tool_mae_bpm':float(np.mean(current_errors)),'current_tool_median_abs_error_bpm':float(np.median(current_errors)),'current_by_database':current_by_db,'best_model':best_name,'best_cv_metrics':reports[best_name],'all_cv_summaries':{k:{kk:vv for kk,vv in v.items() if kk!='rows'} for k,v in reports.items()},'model_out':args.out_model,'decision_note':'Multidatabase candidate selector; integrate if it improves LOOCV and does not degrade BIDMC/CapnoBase subsets.'}
    Path(args.report).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
