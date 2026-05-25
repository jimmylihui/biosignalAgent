from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_fixed_hr_recipes_larger_digitization import digitize, load_signal, peaks, hr_from_peaks, reference_hr
from biosignal_agent.tools.digitize_tools import ML_MODEL_PATH

CANDIDATES = ["lowres_direct", "x4_median", "x4_path", "x4_momentum"]
BANDS = [None, (0.3, 8), (0.5, 12), (0.8, 25), (1, 35), (2, 40), (5, 80), (20, 150), (20, 250)]
DISTANCES = [0.18, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
PROMINENCES = [0.15, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5]


def mean(vals):
    vals=[float(v) for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark_more_10s_manifest.json')
    ap.add_argument('--out-dir', default='/data1/jiahui/biosignal-agent/outputs/hr_grid_more_10s')
    ap.add_argument('--out-json', default='/data1/jiahui/biosignal-agent/outputs/hr_grid_more_10s_eval.json')
    ap.add_argument('--model-path', default=str(ML_MODEL_PATH))
    ap.add_argument('--probability-threshold', type=float, default=0.5)
    ap.add_argument('--scale', type=int, default=4)
    ap.add_argument('--include-modality', action='append', default=['pcg','eeg','eda','bcg','emg','scg'])
    args=ap.parse_args()
    wanted={m.lower() for m in args.include_modality}
    records=[r for r in json.loads(Path(args.manifest).read_text()).get('records',[]) if str(r.get('modality','')).lower() in wanted]
    out_dir=Path(args.out_dir)
    generated=[]
    for rec in records:
        ref_hr=reference_hr(rec)
        for cand in CANDIDATES:
            csv, fs, err = digitize(rec, out_dir, cand, args.model_path, args.probability_threshold, args.scale)
            generated.append({'record':rec['record'],'modality':rec['modality'],'variant':rec.get('variant'),'candidate':cand,'csv':csv,'fs':fs,'digitizer_error':err,'reference_hr_bpm':ref_hr})
    grids=[]
    for modality in sorted(wanted):
        for cand in CANDIDATES:
            subset=[r for r in generated if r['modality']==modality and r['candidate']==cand and not r.get('digitizer_error') and r.get('reference_hr_bpm') is not None]
            if not subset:
                continue
            for dist in DISTANCES:
                for prom in PROMINENCES:
                    for band in BANDS:
                        errs=[]; details=[]
                        for row in subset:
                            vals=load_signal(row['csv'])
                            pred_hr=hr_from_peaks(peaks(vals, float(row['fs']), dist, prom, band), float(row['fs']))
                            if pred_hr is None:
                                continue
                            err=abs(float(row['reference_hr_bpm'])-pred_hr)
                            errs.append(err)
                            details.append({'record':row['record'],'variant':row.get('variant'),'reference_hr_bpm':row['reference_hr_bpm'],'predicted_hr_bpm':pred_hr,'hr_abs_error_bpm':err,'csv':row['csv']})
                        if errs:
                            grids.append({'modality':modality,'candidate':cand,'distance_s':dist,'prominence_scale':prom,'band':band,'num_records':len(errs),'mean_hr_abs_error_bpm':float(np.mean(errs)),'median_hr_abs_error_bpm':float(np.median(errs)),'max_hr_abs_error_bpm':float(np.max(errs)),'details':details})
    grids.sort(key=lambda r:(r['modality'], r['mean_hr_abs_error_bpm'], r['max_hr_abs_error_bpm']))
    by_modality={}
    for modality in sorted(wanted):
        rows=[r for r in grids if r['modality']==modality]
        by_modality[modality]=rows[:10]
    baseline=[]
    for rec in records:
        row=next(r for r in generated if r['record']==rec['record'] and r['candidate']=='lowres_direct')
        pred=hr_from_peaks(peaks(load_signal(row['csv']), float(row['fs']), 0.25, 0.5, None), float(row['fs'])) if row['csv'] else None
        err=abs(row['reference_hr_bpm']-pred) if row['reference_hr_bpm'] is not None and pred is not None else None
        baseline.append({'record':rec['record'],'modality':rec['modality'],'err':err})
    summary=[]
    for modality in sorted(wanted):
        base_mae=mean([r['err'] for r in baseline if r['modality']==modality])
        best=by_modality.get(modality,[None])[0]
        summary.append({'modality':modality,'num_records':sum(1 for r in records if r['modality']==modality),'baseline_hr_mae_bpm':base_mae,'best_grid_hr_mae_bpm':best['mean_hr_abs_error_bpm'] if best else None,'best_candidate':best['candidate'] if best else None,'best_distance_s':best['distance_s'] if best else None,'best_prominence_scale':best['prominence_scale'] if best else None,'best_band':best['band'] if best else None,'best_max_error_bpm':best['max_hr_abs_error_bpm'] if best else None})
    report={'manifest':args.manifest,'num_generated':len(generated),'generated_failure_counts':dict(Counter(r.get('digitizer_error') for r in generated if r.get('digitizer_error'))),'summary':summary,'by_modality_top10':by_modality,'generated':generated}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    print(json.dumps({'out_json':args.out_json,'summary':summary,'failure_counts':report['generated_failure_counts']}, indent=2))

if __name__=='__main__':
    main()
