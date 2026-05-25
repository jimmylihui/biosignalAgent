from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.common import bpm_from_peaks
from scripts.evaluate_pcg_springer_segmentation import event_metrics, state_centers
from scripts.train_pcg_springer_segmentation_tcn import PCGStateTCN, predict_labels


def merged_state_centers(labels: np.ndarray, target: int, fs: float, min_ms: float, merge_gap_ms: float) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    mask = labels == int(target)
    if not np.any(mask):
        return np.asarray([], dtype=int)
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts = list(np.where(edges == 1)[0])
    ends = list(np.where(edges == -1)[0])
    min_len = max(1, int(round(min_ms * fs / 1000.0)))
    max_gap = max(0, int(round(merge_gap_ms * fs / 1000.0)))
    merged: list[tuple[int, int]] = []
    for a, b in zip(starts, ends):
        if not merged:
            merged.append((int(a), int(b)))
        else:
            pa, pb = merged[-1]
            if int(a) - pb <= max_gap:
                merged[-1] = (pa, int(b))
            else:
                merged.append((int(a), int(b)))
    filtered = [(a, b) for a, b in merged if b - a >= min_len]
    return np.asarray([int(round((a + b - 1) / 2.0)) for a, b in filtered], dtype=int)


def evaluate(model, rows, device, chunk_len, tolerance_ms, min_ms, merge_gap_ms):
    s1_items=[]; s2_items=[]; hr_errors=[]
    for row in rows:
        df=pd.read_csv(row['path'])
        values=df['signal'].to_numpy(dtype=np.float32)
        labels=df['state_label'].to_numpy(dtype=int)
        fs=float(row['sampling_rate'])
        pred=predict_labels(model, values, device, chunk_len)
        true_s1=state_centers(labels,1); true_s2=state_centers(labels,3)
        pred_s1=merged_state_centers(pred,1,fs,min_ms,merge_gap_ms)
        pred_s2=merged_state_centers(pred,3,fs,min_ms,merge_gap_ms)
        s1_items.append(event_metrics(true_s1,pred_s1,fs,tolerance_ms))
        s2_items.append(event_metrics(true_s2,pred_s2,fs,tolerance_ms))
        true_hr=bpm_from_peaks(true_s1,fs) if len(true_s1)>=2 else None
        pred_hr=bpm_from_peaks(pred_s1,fs) if len(pred_s1)>=2 else None
        if true_hr is not None and pred_hr is not None:
            hr_errors.append(abs(float(pred_hr)-float(true_hr)))
    def summary(items):
        tp=sum(x['tp'] for x in items); fp=sum(x['fp'] for x in items); fn=sum(x['fn'] for x in items)
        p=tp/max(1,tp+fp); r=tp/max(1,tp+fn)
        maes=[x['mae_ms'] for x in items if x.get('mae_ms') is not None]
        return {'micro_precision':float(p),'micro_recall':float(r),'micro_f1':float(2*p*r/max(1e-12,p+r)),'mean_record_f1':float(np.mean([x['f1'] for x in items])),'mae_ms_mean_record':float(np.mean(maes)) if maes else None,'tp':int(tp),'fp':int(fp),'fn':int(fn)}
    return {'min_ms':min_ms,'merge_gap_ms':merge_gap_ms,'s1_summary':summary(s1_items),'s2_summary':summary(s2_items),'heart_rate_mae_bpm':float(np.mean(hr_errors)) if hr_errors else None,'heart_rate_num_records':len(hr_errors)}


def main():
    ap=argparse.ArgumentParser(description='Tune/evaluate TCN state-to-event postprocessing on Springer PCG segmentation.')
    ap.add_argument('--manifest',type=Path,default=Path('/data1/jiahui/biosignal-agent/datasets/processed/pcg_springer_segmentation_manifest.json'))
    ap.add_argument('--model-path',type=Path,default=Path('/data1/jiahui/biosignal-agent/outputs/pcg_springer_segmentation_tcn.pt'))
    ap.add_argument('--report-path',type=Path,default=Path('/data1/jiahui/biosignal-agent/outputs/pcg_springer_segmentation_tcn_postprocess_report.json'))
    ap.add_argument('--val-fold',type=int,default=0)
    ap.add_argument('--tolerance-ms',type=float,default=80.0)
    args=ap.parse_args()
    rows=json.load(open(args.manifest))['rows']
    val=[r for r in rows if int(r.get('record_id','0')) % 5 == args.val_fold % 5]
    device='cuda' if torch.cuda.is_available() else 'cpu'
    bundle=torch.load(args.model_path,map_location=device)
    model=PCGStateTCN().to(device)
    model.load_state_dict(bundle['model_state_dict'])
    chunk_len=int(bundle.get('chunk_len',4096))
    results=[]
    for min_ms in [30,50,70,90,120,150]:
        for gap_ms in [0,30,60,100,150,220]:
            res=evaluate(model,val,device,chunk_len,args.tolerance_ms,min_ms,gap_ms)
            score=(res['s1_summary']['micro_f1']+res['s2_summary']['micro_f1'])/2.0
            hr=res['heart_rate_mae_bpm'] if res['heart_rate_mae_bpm'] is not None else 9999.0
            res['selection_score']=float(score - 0.01*min(hr,100.0))
            results.append(res)
            print(json.dumps({'min_ms':min_ms,'gap_ms':gap_ms,'s1_f1':res['s1_summary']['micro_f1'],'s2_f1':res['s2_summary']['micro_f1'],'hr_mae':res['heart_rate_mae_bpm'],'score':res['selection_score']}),flush=True)
    best=max(results,key=lambda r:r['selection_score'])
    report={'num_val_records':len(val),'tolerance_ms':args.tolerance_ms,'best':best,'all_results':results,'model_path':str(args.model_path)}
    args.report_path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'best':best},indent=2))

if __name__=='__main__':
    main()
