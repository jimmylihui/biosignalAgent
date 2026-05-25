from __future__ import annotations
import argparse
from pathlib import Path
import wfdb


def complete(raw: Path, record: str) -> bool:
    return all((raw / f'{record}.{ext}').exists() for ext in ['hea', 'dat', 'atr'])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--db', required=True)
    p.add_argument('--raw-dir', type=Path, required=True)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=3)
    args = p.parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    records = wfdb.get_record_list(args.db)
    missing = [r for r in records if not complete(args.raw_dir, r)]
    if args.limit is not None:
        missing = missing[:args.limit]
    print({'db': args.db, 'total_records': len(records), 'missing_to_download': len(missing), 'raw_dir': str(args.raw_dir)}, flush=True)
    downloaded = 0
    failed = []
    for i in range(0, len(missing), args.batch_size):
        batch = missing[i:i + args.batch_size]
        files = []
        for record in batch:
            files.extend([f'{record}.hea', f'{record}.dat', f'{record}.atr'])
        try:
            print('download batch', batch, flush=True)
            wfdb.dl_files(args.db, str(args.raw_dir), files=files, keep_subdirs=False, overwrite=False)
            downloaded += sum(1 for record in batch if complete(args.raw_dir, record))
        except Exception as exc:
            print('batch failed', batch, type(exc).__name__, exc, flush=True)
            failed.extend(batch)
    complete_now = [r for r in records if complete(args.raw_dir, r)]
    print({'downloaded_complete_in_run': downloaded, 'complete_total': len(complete_now), 'failed': failed[:20], 'num_failed': len(failed)}, flush=True)

if __name__ == '__main__':
    main()
