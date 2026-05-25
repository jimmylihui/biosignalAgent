from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_scg_heartbeat_unet_cebs import complete_records, load_record
from biosignal_agent.tools.scg_tools import SCG_assess_quality

RAW_DIR = Path('/data1/jiahui/biosignal-agent/datasets/raw/cebsdb')
OUT = Path('/data1/jiahui/biosignal-agent/outputs/scg_quality_synthetic_cebs_report.json')


def norm(x):
    x=np.asarray(x,dtype=float)
    x=x-np.nanmedian(x)
    scale=np.nanpercentile(np.abs(x),95)+1e-8
    return np.clip(x/scale,-8,8)


def artifacts(x, fs, rng):
    n=len(x); t=np.arange(n)/fs; sig=np.std(x)+1e-8
    out=[]
    out.append(('good', x.copy(), 'good'))
    y=x.copy(); start=int(rng.integers(0, max(1,n-int(fs*5)))); y[start:start+int(fs*5)]=0; out.append(('dropout', y, 'bad'))
    y=np.clip(x, np.percentile(x,20), np.percentile(x,80)); out.append(('clipping', y, 'bad'))
    y=x + rng.normal(0, 2.5*sig, n); out.append(('white_noise', y, 'bad'))
    y=x + 4.0*sig*np.sin(2*np.pi*0.25*t); out.append(('motion_baseline', y, 'bad'))
    y=x*0.03 + rng.normal(0,0.02*sig,n); out.append(('low_amplitude', y, 'bad'))
    return out


def pred_label(out):
    if out.get('quality') == 'bad' or out.get('confidence', 0) < 0.5:
        return 'bad'
    return 'good'


def main():
    rng=np.random.default_rng(7)
    rows=[]
    with tempfile.TemporaryDirectory() as td:
        for rec in complete_records(RAW_DIR):
            x,_=load_record(RAW_DIR, rec, 250.0)
            x=norm(x)
            fs=250.0; seg_len=int(fs*30)
            starts=np.linspace(0, max(0,len(x)-seg_len), 3, dtype=int)
            for idx,start in enumerate(starts):
                seg=x[start:start+seg_len]
                if len(seg)<seg_len: continue
                for artifact, values, label in artifacts(seg, fs, rng):
                    path=Path(td)/f'{rec}_{idx}_{artifact}.csv'
                    pd.DataFrame({'signal': values}).to_csv(path,index=False)
                    res=SCG_assess_quality(str(path), fs)
                    rows.append({'record':rec,'segment':idx,'artifact':artifact,'label':label,'pred':pred_label(res),'tool_quality':res.get('quality'),'confidence':res.get('confidence'),'result':res})
    y=[r['label'] for r in rows]; p=[r['pred'] for r in rows]
    summary={
        'n_examples':len(rows),
        'accuracy':float(accuracy_score(y,p)),
        'macro_f1':float(f1_score(y,p,average='macro')),
        'bad_f1':float(f1_score([1 if a=='bad' else 0 for a in y],[1 if a=='bad' else 0 for a in p])),
        'confusion_matrix_labels':['bad','good'],
        'confusion_matrix':confusion_matrix(y,p,labels=['bad','good']).tolist(),
        'classification_report':classification_report(y,p,labels=['bad','good'],output_dict=True,zero_division=0),
    }
    report={'summary':summary,'per_example':rows}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
