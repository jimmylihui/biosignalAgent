from __future__ import annotations
import argparse, json, re, sys
from collections import Counter
from pathlib import Path
from typing import Any
import pandas as pd
import wfdb

EDB_DEFAULT_RECORDS = ['e0103','e0104','e0105','e0106','e0107','e0108','e0110','e0111','e0112','e0113','e0114','e0115']
ST_START_RE = re.compile(r'^\(ST(\d)([+-])')
ST_END_RE = re.compile(r'^ST(\d)([+-])\)')
AST_RE = re.compile(r'^AST(\d)([+-])(\d+)')

def clean(note: str) -> str:
    return (note or '').replace('\x00','').strip()

def maybe_download(raw_dir: Path, records: list[str]) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    missing=[r for r in records if not (raw_dir/f'{r}.hea').exists() or not (raw_dir/f'{r}.dat').exists() or not (raw_dir/f'{r}.atr').exists()]
    for r in missing:
        print('download', r, flush=True)
        wfdb.dl_database('edb', dl_dir=str(raw_dir), records=[r])

def st_intervals(raw_dir: Path, record: str, total_samples: int) -> list[dict[str, Any]]:
    ann=wfdb.rdann(str(raw_dir/record),'atr')
    open_by_key: dict[tuple[int,str], int] = {}
    intervals=[]
    peaks=[]
    for sample, note in zip(ann.sample, ann.aux_note):
        n=clean(note)
        m=ST_START_RE.match(n)
        if m:
            key=(int(m.group(1)), m.group(2)); open_by_key[key]=int(sample); continue
        m=ST_END_RE.match(n)
        if m:
            key=(int(m.group(1)), m.group(2)); start=open_by_key.pop(key, None)
            if start is not None and int(sample)>start:
                intervals.append({'start_sample':start,'end_sample':int(sample),'channel':key[0],'direction':key[1]})
            continue
        m=AST_RE.match(n)
        if m:
            peaks.append({'sample':int(sample),'channel':int(m.group(1)),'direction':m.group(2),'magnitude_uv':int(m.group(3))})
    for (ch, direction), start in open_by_key.items():
        intervals.append({'start_sample':start,'end_sample':int(total_samples),'channel':ch,'direction':direction})
    # Attach nearest peak magnitude inside each interval where available.
    for ep in intervals:
        mags=[p['magnitude_uv'] for p in peaks if p['channel']==ep['channel'] and ep['start_sample']<=p['sample']<=ep['end_sample']]
        ep['peak_magnitude_uv']=max(mags) if mags else None
    return intervals

def overlap_seconds(intervals, start, stop, channel, fs):
    ov=0; dirs=Counter(); mags=[]
    for ep in intervals:
        if int(ep['channel']) != int(channel):
            continue
        n=max(0, min(stop, ep['end_sample']) - max(start, ep['start_sample']))
        if n>0:
            ov += n; dirs[ep['direction']] += n
            if ep.get('peak_magnitude_uv') is not None: mags.append(ep['peak_magnitude_uv'])
    return ov/float(fs), dict(dirs), max(mags) if mags else None

def export_record(raw_dir: Path, out_dir: Path, record: str, seconds: int, stride_seconds: int, min_overlap_s: float, max_windows_per_record_channel: int | None):
    h=wfdb.rdheader(str(raw_dir/record)); intervals=st_intervals(raw_dir, record, h.sig_len)
    rows=[]
    for ch, lead in enumerate(h.sig_name):
        kept=0
        for start_s in range(0, max(0, int(h.sig_len/h.fs)-seconds+1), stride_seconds):
            if max_windows_per_record_channel is not None and kept >= max_windows_per_record_channel:
                break
            start=int(round(start_s*h.fs)); stop=int(round((start_s+seconds)*h.fs))
            ov, dirs, mag=overlap_seconds(intervals,start,stop,ch,h.fs)
            label='st_abnormal' if ov >= min_overlap_s else 'normal'
            rec=wfdb.rdrecord(str(raw_dir/record), sampfrom=start, sampto=stop)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_csv=out_dir/f'edb_{record}_ch{ch}_{start_s:05d}_{seconds}s_ecg.csv'
            pd.DataFrame({'signal': rec.p_signal[:,ch].astype(float)}).to_csv(out_csv,index=False)
            rows.append({'dataset':'edb_st_windows','source_database':'edb','record':record,'group':record,'channel':ch,'lead':lead,'window_start_s':start_s,'duration_s':seconds,'path':str(out_csv),'sampling_rate':float(h.fs),'label':label,'st_overlap_s':ov,'st_direction_overlap_samples':dirs,'st_peak_magnitude_uv':mag,'modality':'ecg'})
            kept += 1
    return rows, intervals

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--raw-dir',type=Path,default=Path('/data1/jiahui/biosignal-agent/datasets/raw/edb')); ap.add_argument('--out-dir',type=Path,default=Path('/data1/jiahui/biosignal-agent/datasets/processed/edb_st_windows')); ap.add_argument('--manifest',type=Path,default=Path('/data1/jiahui/biosignal-agent/datasets/processed/edb_st_windows_manifest.json')); ap.add_argument('--records',nargs='*',default=EDB_DEFAULT_RECORDS); ap.add_argument('--seconds',type=int,default=60); ap.add_argument('--stride-seconds',type=int,default=30); ap.add_argument('--min-overlap-s',type=float,default=10.0); ap.add_argument('--max-windows-per-record-channel',type=int,default=180); ap.add_argument('--no-download',action='store_true'); args=ap.parse_args()
    if not args.no_download: maybe_download(args.raw_dir,args.records)
    rows=[]; episode_counts={}; skipped={}
    for r in args.records:
        try:
            part, eps=export_record(args.raw_dir,args.out_dir,r,args.seconds,args.stride_seconds,args.min_overlap_s,args.max_windows_per_record_channel); rows.extend(part); episode_counts[r]=len(eps); print(r,len(part),Counter(x['label'] for x in part), 'episodes', len(eps), flush=True)
        except Exception as exc:
            skipped[r]=f'{type(exc).__name__}:{str(exc)[:160]}'; print('skip',r,skipped[r],flush=True)
    manifest={'dataset':'edb_st_windows','records':rows,'num_windows':len(rows),'label_counts':dict(Counter(x['label'] for x in rows)),'record_episode_counts':episode_counts,'seconds':args.seconds,'stride_seconds':args.stride_seconds,'min_overlap_s':args.min_overlap_s,'skipped_records':skipped}
    args.manifest.parent.mkdir(parents=True,exist_ok=True); args.manifest.write_text(json.dumps(manifest,indent=2)); print(json.dumps({k:manifest[k] for k in ['num_windows','label_counts','record_episode_counts','skipped_records']},indent=2))
if __name__=='__main__': main()
