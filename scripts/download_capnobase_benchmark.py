from __future__ import annotations
import json, urllib.request
from pathlib import Path

DOI='10.5683/SP2/NLB8IT'
API=f'https://borealisdata.ca/api/datasets/:persistentId/?persistentId=doi:{DOI}'
OUT=Path('/data1/jiahui/biosignal-agent/datasets/raw/capnobase_benchmark')

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data=json.load(urllib.request.urlopen(API, timeout=60))
    files=data['data']['latestVersion']['files']
    wanted=[]
    for f in files:
        name=f['dataFile']['filename']
        if name.endswith('_8min_signal.tab') or name.endswith('_8min_labels.tab') or name.endswith('_8min_param.tab'):
            wanted.append((f['dataFile']['id'], name))
    for fid,name in wanted:
        path=OUT/name
        if path.exists() and path.stat().st_size>0:
            continue
        print('download', fid, name, flush=True)
        path.write_bytes(urllib.request.urlopen(f'https://borealisdata.ca/api/access/datafile/{fid}', timeout=120).read())
    print(json.dumps({'out':str(OUT),'files':len(wanted),'downloaded':len(list(OUT.glob('*_8min_*.tab')))}, indent=2))
if __name__=='__main__': main()
