
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
from typing import Any
import pandas as pd, requests
from scipy.io import wavfile
BASE_URL='https://physionet.org/files/circor-heart-sound/1.0.3'
LOCATIONS=('AV','PV','TV','MV','Phc')

def download(url: str, path: Path, enabled: bool):
    if path.exists() and path.stat().st_size > 256: return path
    if not enabled: return None
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.part')
    with requests.get(url, stream=True, timeout=90) as r:
        r.raise_for_status()
        with tmp.open('wb') as f:
            for chunk in r.iter_content(1024*1024):
                if chunk: f.write(chunk)
    tmp.replace(path); return path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--raw-dir',type=Path,default=Path('/data1/jiahui/biosignal-agent/datasets/raw/circor-heart-sound/1.0.3')); ap.add_argument('--manifest',type=Path,default=Path('/data1/jiahui/biosignal-agent/datasets/processed/circor_pcg_full_murmur_manifest.json')); ap.add_argument('--download',action='store_true'); ap.add_argument('--min-seconds',type=float,default=3.0); args=ap.parse_args()
    df=pd.read_csv(download(f'{BASE_URL}/training_data.csv', args.raw_dir/'training_data.csv', args.download))
    records_path=download(f'{BASE_URL}/RECORDS', args.raw_dir/'RECORDS', args.download)
    stems={Path(x.strip()).name for x in records_path.read_text(errors='replace').splitlines() if x.strip()}
    rows=[]
    for _,patient in df.iterrows():
        pid=str(int(patient['Patient ID'])); murmur=str(patient.get('Murmur') or '').strip(); outcome=str(patient.get('Outcome') or '').strip()
        loc_text=str(patient.get('Recording locations:') or '')
        locs=[loc for loc in LOCATIONS if loc in loc_text.split('+')]
        murmur_locs=str(patient.get('Murmur locations') or '')
        for loc in locs:
            stem=f'{pid}_{loc}'
            if stem not in stems: continue
            wav=download(f'{BASE_URL}/training_data/{stem}.wav', args.raw_dir/'training_data'/f'{stem}.wav', args.download)
            if wav is None:
                continue
            fs,x=wavfile.read(wav)
            dur=float(len(x)/fs)
            if dur < args.min_seconds: continue
            record_present = murmur == 'Present' and loc in murmur_locs.split('+')
            rows.append({'dataset':'circor_heart_sound_1.0.3','patient_id':pid,'record':stem,'location':loc,'modality':'pcg','path':str(wav),'sampling_rate':float(fs),'duration_s':dur,'patient_murmur_label':murmur.lower(),'record_murmur_label':'present' if record_present else 'unknown' if murmur=='Unknown' else 'absent','label':'abnormal' if record_present else 'unknown' if murmur=='Unknown' else 'normal','patient_label':'abnormal' if murmur=='Present' else 'unknown' if murmur=='Unknown' else 'normal','outcome':outcome,'age':patient.get('Age'),'sex':patient.get('Sex'),'height':patient.get('Height'),'weight':patient.get('Weight'),'murmur_locations':patient.get('Murmur locations'),'most_audible_location':patient.get('Most audible location')})
    out={'dataset':'circor_heart_sound_1.0.3_full','num_records':len(rows),'num_patients':len(set(r['patient_id'] for r in rows)),'patient_murmur_counts':dict(Counter(r['patient_murmur_label'] for r in {r['patient_id']:r for r in rows}.values())),'record_label_counts':dict(Counter(r['label'] for r in rows)),'records':rows}
    args.manifest.parent.mkdir(parents=True,exist_ok=True); args.manifest.write_text(json.dumps(out,indent=2)); print(json.dumps({k:out[k] for k in ['num_records','num_patients','patient_murmur_counts','record_label_counts']},indent=2))
if __name__=='__main__': main()
