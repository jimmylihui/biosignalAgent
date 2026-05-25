from pathlib import Path
import urllib.request,json
ROOT=Path('/data1/jiahui/biosignal-agent/datasets/raw/segade_ppg_quality')
DIRS=['TROIKA_channel_1/processed_dataset','WESAD_all/processed_dataset','new_PPG_DaLiA_train/processed_dataset','new_PPG_DaLiA_test/processed_dataset']
for d in DIRS:
    out=(ROOT/d); out.mkdir(parents=True,exist_ok=True)
    api=f'https://api.github.com/repos/chengstark/Segade/contents/data/{d}'
    items=json.load(urllib.request.urlopen(api,timeout=30))
    for item in items:
        path=out/item['name']
        if path.exists() and path.stat().st_size==item['size']:
            continue
        print('download',d,item['name'],item['size'],flush=True)
        path.write_bytes(urllib.request.urlopen(item['download_url'],timeout=180).read())
print('done', ROOT)
