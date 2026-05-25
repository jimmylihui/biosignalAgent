from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import wfdb
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.abp_tools import ABP_detect_pulses, ABP_compute_hemodynamics, ABP_screen_pressure_events, ABP_classify_pressure_events

RAW=Path('/data1/jiahui/biosignal-agent/datasets/raw/bidmc')
OUT=Path('/data1/jiahui/biosignal-agent/outputs/abp_bidmc')
CSV=OUT/'csv'


def has_abp(path: Path) -> bool:
    return 'ABP' in path.read_text(errors='replace')


def main():
    rows=[]
    CSV.mkdir(parents=True,exist_ok=True)
    for hea in sorted(RAW.glob('*.hea')):
        if not has_abp(hea):
            continue
        name=hea.stem
        rec=wfdb.rdrecord(str(RAW/name))
        names=[s.strip(',') for s in rec.sig_name]
        if 'ABP' not in names:
            continue
        idx=names.index('ABP')
        x=rec.p_signal[:,idx]
        out=CSV/f'{name}_abp.csv'
        pd.DataFrame({'abp':x}).to_csv(out,index=False)
        pulses=ABP_detect_pulses(str(out),float(rec.fs),'abp')
        hemo=ABP_compute_hemodynamics(str(out),float(rec.fs),'abp')
        pressure=ABP_screen_pressure_events(str(out),float(rec.fs),'abp')
        events=ABP_classify_pressure_events(str(out),float(rec.fs),'abp')
        plausible=not pulses.get('error') and pulses.get('heart_rate_bpm') is not None and 35<=pulses['heart_rate_bpm']<=180 and 30<=pulses.get('approx_diastolic_value',0)<=140 and 50<=pulses.get('median_systolic_value',0)<=220 and 5<=pulses.get('median_pulse_pressure',0)<=120
        rows.append({'record':name,'fs':rec.fs,'path':str(out),'num_pulses':pulses.get('num_pulses'),'heart_rate_bpm':pulses.get('heart_rate_bpm'),'systolic':pulses.get('median_systolic_value'),'diastolic':pulses.get('approx_diastolic_value'),'map':hemo.get('mean_arterial_pressure_proxy'),'pulse_pressure':pulses.get('median_pulse_pressure'),'pressure_risk':pressure.get('pressure_risk'),'pressure_event_risk':events.get('pressure_event_risk'),'hypotensive_beat_fraction':events.get('hypotensive_beat_fraction'),'severe_hypotensive_beat_fraction':events.get('severe_hypotensive_beat_fraction'),'confidence':pulses.get('confidence'),'artifact_rejected_fraction':pulses.get('artifact_rejected_fraction'),'plausible_summary':plausible,'error':pulses.get('error')})
    vals=[r for r in rows if r['plausible_summary']]
    report={'dataset':'BIDMC waveform records with ABP channel','num_records':len(rows),'plausible_records':len(vals),'plausible_fraction':len(vals)/len(rows) if rows else 0.0,'median_hr':float(np.nanmedian([r['heart_rate_bpm'] for r in vals])) if vals else None,'median_systolic':float(np.nanmedian([r['systolic'] for r in vals])) if vals else None,'median_diastolic':float(np.nanmedian([r['diastolic'] for r in vals])) if vals else None,'median_map':float(np.nanmedian([r['map'] for r in vals])) if vals else None,'pressure_event_counts':dict(Counter(r.get('pressure_event_risk') for r in vals)),'rows':rows}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'abp_bidmc_eval.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['num_records','plausible_records','plausible_fraction','median_hr','median_systolic','median_diastolic','median_map','pressure_event_counts']},indent=2))
    print(json.dumps(rows,indent=2))

if __name__=='__main__': main()
