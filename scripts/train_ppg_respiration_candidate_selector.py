from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from typing import Any
from collections import Counter
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from biosignal_agent.tools.ppg_tools import PPG_estimate_respiration_modulation
from scripts.evaluate_ppg_respiration_bidmc import iter_wfdb_records, reference_resp_rate

SOURCES=['baseline_wander','hilbert_envelope','pulse_amplitude','pulse_interval']

def source_index(src):
    try: return SOURCES.index(src)+1
    except ValueError: return 0

def collect_candidates(tool_out:dict[str,Any]) -> list[dict[str,Any]]:
    cands=tool_out.get('respiration_candidate_rates') or []
    bins:dict[float,dict[str,Any]]={}
    for c in cands:
        src=c.get('source')
        selected=c.get('respiratory_rate_bpm')
        if selected is not None:
            key=round(float(selected)/1.875)*1.875
            item=bins.setdefault(key, {'rate_bpm':key,'source_scores':{s:0.0 for s in SOURCES},'source_best_rank':{s:99 for s in SOURCES},'selected_sources':set(),'top_sources':set(),'max_power':0.0,'sum_power':0.0,'num_sources':0})
            item['selected_sources'].add(src)
        for rank,p in enumerate(c.get('top_respiration_peaks_bpm') or []):
            rate=p.get('rate_bpm'); power=float(p.get('power_ratio') or 0.0)
            if rate is None or not (5.0 <= rate <= 35.0): continue
            key=round(float(rate)/1.875)*1.875
            item=bins.setdefault(key, {'rate_bpm':key,'source_scores':{s:0.0 for s in SOURCES},'source_best_rank':{s:99 for s in SOURCES},'selected_sources':set(),'top_sources':set(),'max_power':0.0,'sum_power':0.0,'num_sources':0})
            if src in SOURCES:
                item['source_scores'][src]+=power*(0.85**rank)
                item['source_best_rank'][src]=min(item['source_best_rank'][src],rank)
                if power>=1.3: item['top_sources'].add(src)
            item['max_power']=max(item['max_power'],power)
            item['sum_power']+=power*(0.85**rank)
    rows=[]
    selected_rate=tool_out.get('respiratory_rate_bpm')
    for rate,item in bins.items():
        scores=item['source_scores']; ranks=item['source_best_rank']
        low_neighbor=sum(v['sum_power'] for k,v in bins.items() if 5<=k<12 and abs(k-rate)>1e-6)
        double_support=bins.get(round((rate/2)/1.875)*1.875,{}).get('sum_power',0.0) if rate>=12 else 0.0
        half_support=bins.get(round((rate*2)/1.875)*1.875,{}).get('sum_power',0.0) if rate<18 else 0.0
        rows.append({
            'rate_bpm':float(rate),
            'rate_norm':float(rate/40.0),
            'is_low_rate':float(rate<10.0),
            'is_adult_plausible':float(12.0<=rate<=24.0),
            'distance_to_18':float(abs(rate-18.0)/18.0),
            'sum_power':float(item['sum_power']),
            'max_power':float(item['max_power']),
            'num_strong_sources':float(len(item['top_sources'])),
            'num_selected_sources':float(len(item['selected_sources'])),
            'chosen_by_current_rule':float(selected_rate is not None and abs(rate-float(selected_rate))<=1.0),
            'low_neighbor_power':float(low_neighbor),
            'double_rate_support':float(double_support),
            'half_rate_support':float(half_support),
            **{f'{s}_score':float(scores[s]) for s in SOURCES},
            **{f'{s}_rank':float(8 if ranks[s]==99 else ranks[s]) for s in SOURCES},
            'supporting_sources':','.join(sorted(str(x) for x in item['top_sources'] if x)),
        })
    return rows

FEATURES=['rate_norm','is_low_rate','is_adult_plausible','distance_to_18','sum_power','max_power','num_strong_sources','num_selected_sources','chosen_by_current_rule','low_neighbor_power','double_rate_support','half_rate_support'] + [f'{s}_score' for s in SOURCES] + [f'{s}_rank' for s in SOURCES]

def build_dataset(raw_dir:Path, processed_out:Path):
    records=iter_wfdb_records(raw_dir, processed_out)
    rows=[]; current=[]
    for item in records:
        ref=reference_resp_rate(item['resp'], item['fs'])
        truth=ref.get('respiratory_rate_bpm')
        if truth is None: continue
        out=PPG_estimate_respiration_modulation(str(item['ppg_path']), item['fs'])
        estimate=out.get('respiratory_rate_bpm')
        current.append({'record':item['record'],'truth':truth,'estimate':estimate,'error':abs(estimate-truth) if estimate is not None else None})
        cands=collect_candidates(out)
        if not cands: continue
        distances=[abs(c['rate_bpm']-truth) for c in cands]
        best_i=int(np.argmin(distances))
        for i,c in enumerate(cands):
            rows.append({**c,'record':item['record'],'truth':float(truth),'label':1 if i==best_i else 0,'abs_error_if_selected':float(abs(c['rate_bpm']-truth))})
    return rows,current

def models(seed:int):
    return {
        'logistic_balanced': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000,class_weight='balanced',C=0.8,random_state=seed)),
        'random_forest_balanced': RandomForestClassifier(n_estimators=300,max_depth=5,min_samples_leaf=3,class_weight='balanced',random_state=seed),
        'extra_trees_balanced': ExtraTreesClassifier(n_estimators=300,max_depth=5,min_samples_leaf=3,class_weight='balanced',random_state=seed),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=120,max_depth=2,learning_rate=0.05,random_state=seed),
    }

def evaluate_rows(rows,pred_prob):
    by_record={}
    for row,prob in zip(rows,pred_prob):
        item={**row,'prob':float(prob)}
        by_record.setdefault(row['record'],[]).append(item)
    rec_rows=[]
    for rec,items in by_record.items():
        chosen=max(items,key=lambda x:x['prob'])
        current=next((x for x in items if x['chosen_by_current_rule']>0.5), None)
        rec_rows.append({'record':rec,'truth':chosen['truth'],'pred':chosen['rate_bpm'],'abs_error_bpm':abs(chosen['rate_bpm']-chosen['truth']),'selected_supporting_sources':chosen.get('supporting_sources'),'current_pred':current['rate_bpm'] if current else None,'current_abs_error_bpm':abs(current['rate_bpm']-chosen['truth']) if current else None})
    return {'num_records':len(rec_rows),'mae_bpm':float(np.mean([r['abs_error_bpm'] for r in rec_rows])),'median_abs_error_bpm':float(np.median([r['abs_error_bpm'] for r in rec_rows])),'current_mae_on_candidate_grid':float(np.mean([r['current_abs_error_bpm'] for r in rec_rows if r['current_abs_error_bpm'] is not None])),'rows':sorted(rec_rows,key=lambda x:x['abs_error_bpm'],reverse=True)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--raw-wfdb-dir',type=Path,default=Path('/data1/jiahui/biosignal-agent/datasets/raw/bidmc')); ap.add_argument('--processed-out-dir',type=Path,default=Path('/data1/jiahui/biosignal-agent/datasets/processed/bidmc_full')); ap.add_argument('--out-model',default='/data1/jiahui/biosignal-agent/outputs/ppg_respiration_candidate_selector.joblib'); ap.add_argument('--report',default='/data1/jiahui/biosignal-agent/outputs/ppg_respiration_candidate_selector_report.json'); ap.add_argument('--seed',type=int,default=31); args=ap.parse_args()
    rows,current=build_dataset(args.raw_wfdb_dir,args.processed_out_dir)
    x=np.asarray([[r[f] for f in FEATURES] for r in rows],dtype=float); y=np.asarray([r['label'] for r in rows]); groups=np.asarray([r['record'] for r in rows])
    logo=LeaveOneGroupOut(); reports={}; best_name=None; best_score=999; best_model=None
    for name,model in models(args.seed).items():
        if hasattr(model,'predict_proba'):
            prob=cross_val_predict(model,x,y,groups=groups,cv=logo,method='predict_proba')[:,1]
        else:
            pred=cross_val_predict(model,x,y,groups=groups,cv=logo); prob=pred.astype(float)
        rep=evaluate_rows(rows,prob); reports[name]=rep
        score=rep['mae_bpm']
        print(name, score, rep['median_abs_error_bpm'], flush=True)
        if score<best_score: best_name=name; best_score=score; best_model=model
    best_model.fit(x,y); joblib.dump({'model':best_model,'model_name':best_name,'feature_names':FEATURES,'cv_metrics':{k:v for k,v in reports[best_name].items() if k!='rows'},'reference':'BIDMC RESP waveform spectral RR; selector chooses among PPG-only multi-source candidate rates. Leave-one-record-out evaluation.'},args.out_model)
    current_errors=[r['error'] for r in current if r['error'] is not None]
    report={'num_candidate_rows':len(rows),'num_records':len(set(groups)),'label_counts': {str(k): int(v) for k, v in Counter(y).items()},'features':FEATURES,'current_tool_mae_bpm':float(np.mean(current_errors)),'current_tool_median_abs_error_bpm':float(np.median(current_errors)),'best_model':best_name,'best_cv_metrics':reports[best_name],'all_cv_summaries':{k:{kk:vv for kk,vv in v.items() if kk!='rows'} for k,v in reports.items()},'model_out':args.out_model,'decision_note':'Only integrate if leave-one-record-out materially improves current rule without obvious overfit.'}
    Path(args.report).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
