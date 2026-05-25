from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.tools.ecg_tools import ECG_detect_afib

def metrics(rows):
    tp=sum(1 for r in rows if r['truth']=='af' and r['prediction']=='af')
    tn=sum(1 for r in rows if r['truth']!='af' and r['prediction']!='af')
    fp=sum(1 for r in rows if r['truth']!='af' and r['prediction']=='af')
    fn=sum(1 for r in rows if r['truth']=='af' and r['prediction']!='af')
    p=tp/(tp+fp) if tp+fp else 0.0; rec=tp/(tp+fn) if tp+fn else 0.0; sp=tn/(tn+fp) if tn+fp else 0.0
    f1=2*p*rec/(p+rec) if p+rec else 0.0
    return {'true_positive':tp,'true_negative':tn,'false_positive':fp,'false_negative':fn,'precision':p,'recall_sensitivity':rec,'specificity':sp,'f1':f1,'accuracy':(tp+tn)/len(rows) if rows else 0.0}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',default='/data1/jiahui/biosignal-agent/datasets/processed/ecg_rhythm_beat_manifest.json')
    ap.add_argument('--out-json',default='/data1/jiahui/biosignal-agent/outputs/ecg_af_detector_eval.json')
    args=ap.parse_args()
    d=json.load(open(args.manifest)); rows=[]
    for i,r in enumerate(d.get('records',[]),1):
        if i%250==0: print('processed',i,flush=True)
        out=ECG_detect_afib(r['path'],float(r['sampling_rate']),None)
        pred='af' if out.get('afib_risk')=='afib_likely' else 'non_af'
        rows.append({'record':r['record'],'window_start_s':r.get('window_start_s'),'truth':r['coarse_rhythm_label'],'prediction':pred,'afib_probability':out.get('afib_probability'),'threshold':out.get('decision_threshold'),'method':out.get('method')})
    report={'manifest':args.manifest,'num_windows':len(rows),'truth_counts':dict(Counter(x['truth'] for x in rows)),'prediction_counts':dict(Counter(x['prediction'] for x in rows)),'metrics':metrics(rows),'rows':rows}
    Path(args.out_json).parent.mkdir(parents=True,exist_ok=True); Path(args.out_json).write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['num_windows','truth_counts','prediction_counts','metrics']},indent=2))
if __name__=='__main__': main()
