from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

RAW = Path('/data1/jiahui/biosignal-agent/datasets/raw/scg_rhc_physionet')
BASE_URL = 'https://physionet.org/files/scg-rhc-wearable-database/1.0.0'


def run(cmd: list[str]) -> None:
    print(' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def record_to_wfdb_name(record: str) -> str:
    return record.strip().replace('.RHC', '-RHC')


def main() -> None:
    ap = argparse.ArgumentParser(description='Download SCG-RHC metadata and selected WFDB processed records from PhysioNet.')
    ap.add_argument('--max-records', type=int, default=0, help='0 means metadata only; positive N downloads first N exported records; -1 downloads all exported records.')
    ap.add_argument('--records-file', default=None, help='Optional file with Study IDs such as TRM155.RHC1 or WFDB names such as TRM155-RHC1.')
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    meta = RAW / 'meta_information'
    processed = RAW / 'processed_data'
    meta.mkdir(exist_ok=True)
    processed.mkdir(exist_ok=True)
    for rel in ['RECORDS', 'LICENSE.txt']:
        run(['wget', '-N', '-c', f'{BASE_URL}/{rel}', '-P', str(RAW)])
    run(['wget', '-r', '-N', '-c', '-np', '-nH', '--cut-dirs=3', f'{BASE_URL}/meta_information/', '-P', str(RAW)])
    if args.max_records == 0 and not args.records_file:
        return
    if args.records_file:
        records = [x.strip() for x in Path(args.records_file).read_text().splitlines() if x.strip()]
    else:
        exported = meta / 'list_exported_recs.txt'
        records = [x.strip() for x in exported.read_text().splitlines() if x.strip()]
        if args.max_records > 0:
            records = records[:args.max_records]
    for record in records:
        wfdb_name = record_to_wfdb_name(record)
        for ext in ['hea', 'json', 'dat']:
            run(['wget', '-N', '-c', f'{BASE_URL}/processed_data/{wfdb_name}.{ext}', '-P', str(processed)])


if __name__ == '__main__':
    main()
