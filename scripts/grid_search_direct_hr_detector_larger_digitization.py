from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_fixed_hr_recipes_larger_digitization import digitize, load_signal, peaks, hr_from_peaks, reference_hr
from biosignal_agent.tools.digitize_tools import ML_MODEL_PATH

BANDS=[None,(0.3,8),(0.5,12),(0.8,25),(1,35),(2,40),(5,80),(20,150),(20,250)]
DISTANCES=[0.18,0.2,0.25,0.3,0.35,0.4,0.5]
PROMINENCES=[0.15,0.2,0.35,0.5,0.75,1.0,1.5]

def mean(xs):
    xs=[float(x) for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark_more_10s_manifest.json')
    ap.add_argument('--out-dir', default='/data1/jiahui/biosignal-agent/outputs/direct_hr_grid_more_10s')
    ap.add_argument('--out-json', default='/data1/jiahui/biosignal-agent/outputs/direct_hr_grid_more_10s_eval.json')
    ap.add_argument('--model-path', default=str(ML_MODEL_PATH))
    ap.add_argument('--probability-threshold', type=float, default=0.5)
    ap.add_argument('--include-modality', action='append', default=['pcg','eeg','eda','bcg','emg','scg'])
    args=ap.parse_args()
    wanted={m.lower() for m in args.include_modality}
    records=[r for r in json.loads(Path(args.manifest).read_text()).get('records',[]) if str(r.get('modality','')).lower() in wanted]
    rows=[]
    out_dir=Path(args.out_dir)
    for rec in records:
        csv, fs, err=digitize(rec,out_dir,'lowres_direct',args.model_path,args.probability_threshold,4)
        ref_hr=reference_hr(rec)
        rows.append({'record':rec['record'],'modality':rec['modality'],'variant':rec.get('variant'),'csv':csv,'fs':fs,'digitizer_error':err,'reference_hr_bpm':ref_hr})
    all_results=[]; summary=[]
    for modality in sorted(wanted):
        subset=[r for r in rows if r['modality']==modality and not r.get('digitizer_error') and r.get('reference_hr_bpm') is not None]
        baseline_errs=[]
        for r in subset:
            pred=hr_from_peaks(peaks(load_signal(r['csv']),r['fs'],0.25,0.5,None),r['fs'])
            baseline_errs.append(abs(r['reference_hr_bpm']-pred) if pred is not None else None)
        for dist in DISTANCES:
            for prom in PROMINENCES:
                for band in BANDS:
                    errs=[]; details=[]
                    for r in subset:
                        pred=hr_from_peaks(peaks(load_signal(r['csv']),r['fs'],dist,prom,band),r['fs'])
                        if pred is None: continue
                        err=abs(r['reference_hr_bpm']-pred)
                        errs.append(err)
                        details.append({'record':r['record'],'variant':r.get('variant'),'reference_hr_bpm':r['reference_hr_bpm'],'predicted_hr_bpm':pred,'hr_abs_error_bpm':err,'csv':r['csv']})
                    if errs:
                        all_results.append({'modality':modality,'candidate':'lowres_direct','distance_s':dist,'prominence_scale':prom,'band':band,'num_records':len(errs),'mean_hr_abs_error_bpm':float(np.mean(errs)),'median_hr_abs_error_bpm':float(np.median(errs)),'max_hr_abs_error_bpm':float(np.max(errs)),'details':details})
        best=sorted([r for r in all_results if r['modality']==modality], key=lambda x:(x['mean_hr_abs_error_bpm'],x['max_hr_abs_error_bpm']))[0]
        summary.append({'modality':modality,'num_records':len(subset),'baseline_hr_mae_bpm':mean(baseline_errs),'best_direct_grid_hr_mae_bpm':best['mean_hr_abs_error_bpm'],'improvement_bpm':mean(baseline_errs)-best['mean_hr_abs_error_bpm'] if mean(baseline_errs) is not None else None,'best_distance_s':best['distance_s'],'best_prominence_scale':best['prominence_scale'],'best_band':best['band'],'best_max_error_bpm':best['max_hr_abs_error_bpm']})
    report={'manifest':args.manifest,'summary':summary,'results':all_results,'rows':rows}
    Path(args.out_json).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report,indent=2))
    print(json.dumps({'out_json':args.out_json,'summary':summary},indent=2))
if __name__=='__main__': main()
