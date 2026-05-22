from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

ARTICLE_API = 'https://api.figshare.com/v2/articles/28643153'
DATASET_DOI = '10.6084/m9.figshare.28643153'
DEFAULT_FS = 125.0


def figshare_files() -> list[dict]:
    with urlopen(ARTICLE_API, timeout=60) as response:
        payload = json.load(response)
    return payload.get('files', [])


def select_bcg_files(limit: int) -> list[dict]:
    files = [item for item in figshare_files() if item.get('name', '').lower().endswith('_bcg.csv')]
    files.sort(key=lambda item: item['name'])
    return files[:limit]


def stream_first_seconds(file_info: dict, seconds: float, out_path: Path, sampling_rate: float = DEFAULT_FS) -> int:
    max_rows = int(round(seconds * sampling_rate))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with urlopen(file_info['download_url'], timeout=120) as response:
        text_stream = (line.decode('utf-8', errors='replace') for line in response)
        reader = csv.DictReader(text_stream)
        for row in reader:
            value = row.get('value')
            if value is None:
                continue
            rows.append(float(value))
            if len(rows) >= max_rows:
                break
    pd.DataFrame({'signal': rows}).to_csv(out_path, index=False)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare a small dedicated BCG subset from the Figshare bedside BCG dataset.')
    parser.add_argument('--out-dir', default='/data1/jiahui/biosignal-agent/datasets/processed/dedicated_bcg')
    parser.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/datasets/processed/dedicated_bcg_manifest.json')
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--seconds', type=float, default=60.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    records = []
    for file_info in select_bcg_files(args.limit):
        subject = file_info['name'].replace('_bcg.csv', '')
        out_path = out_dir / f'figshare_bed_bcg_{subject.lower()}_{int(args.seconds)}s.csv'
        num_samples = stream_first_seconds(file_info, args.seconds, out_path)
        records.append({
            'dataset': 'figshare_bed_bcg_2025',
            'record': subject,
            'modality': 'bcg',
            'path': str(out_path),
            'sampling_rate': DEFAULT_FS,
            'source_sampling_rate': DEFAULT_FS,
            'seconds': args.seconds,
            'source_channel': 'value',
            'num_samples': num_samples,
            'source_file_id': file_info.get('id'),
            'source_file_name': file_info.get('name'),
            'source_download_url': file_info.get('download_url'),
            'doi': DATASET_DOI,
            'license': 'CC BY 4.0',
        })
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({'records': records}, indent=2))
    pd.DataFrame(records).to_csv(manifest_path.with_suffix('.csv'), index=False)
    print(json.dumps({'manifest': str(manifest_path), 'csv_manifest': str(manifest_path.with_suffix('.csv')), 'num_records': len(records)}, indent=2))


if __name__ == '__main__':
    main()
