from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import wfdb

APNEA_ECG_RECORDS = [
    "a01", "a02", "a03", "a04", "a05", "a06", "a07", "a08", "a09", "a10",
    "a11", "a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19", "a20",
    "b01", "b02", "b03", "b04", "b05",
    "c01", "c02", "c03", "c04", "c05", "c06", "c07", "c08", "c09", "c10",
]


def ensure_record(record: str, raw_dir: Path, download: bool) -> None:
    required = [raw_dir / f"{record}.hea", raw_dir / f"{record}.dat", raw_dir / f"{record}.apn"]
    if all(path.exists() for path in required):
        return
    if not download:
        missing = [str(path) for path in required if not path.exists()]
        raise FileNotFoundError(f"Missing Apnea-ECG files for {record}: {missing}. Re-run with --download.")
    raw_dir.mkdir(parents=True, exist_ok=True)
    wfdb.dl_files("apnea-ecg", str(raw_dir), files=[f"{record}.hea", f"{record}.dat", f"{record}.apn"], keep_subdirs=False, overwrite=False)


def export_minute(record: str, raw_dir: Path, out_dir: Path, minute: int) -> dict[str, Any]:
    header = wfdb.rdheader(str(raw_dir / record))
    start = int(minute * 60 * header.fs)
    stop = int((minute + 1) * 60 * header.fs)
    signal = wfdb.rdrecord(str(raw_dir / record), sampfrom=start, sampto=stop)
    lead_idx = 0
    ann = wfdb.rdann(str(raw_dir / record), "apn")
    symbol = ann.symbol[minute]
    label = "apnea" if symbol == "A" else "normal"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"apneaecg_{record}_{minute:04d}_60s.csv"
    pd.DataFrame({"signal": signal.p_signal[:, lead_idx]}).to_csv(out_csv, index=False)
    return {
        "dataset": "apnea_ecg_minutes",
        "record": record,
        "minute": minute,
        "duration_s": 60,
        "modality": "ecg",
        "path": str(out_csv),
        "sampling_rate": float(header.fs),
        "source_channel": signal.sig_name[lead_idx] if signal.sig_name else "ECG",
        "label": label,
        "annotation_symbol": symbol,
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    for record in args.records:
        ensure_record(record, raw_dir, args.download)
        ann = wfdb.rdann(str(raw_dir / record), "apn")
        max_minutes = len(ann.symbol)
        kept = 0
        for minute in range(args.start_minute, max_minutes):
            if args.max_minutes_per_record is not None and kept >= args.max_minutes_per_record:
                break
            rows.append(export_minute(record, raw_dir, out_dir, minute))
            kept += 1
    counts = Counter(row["label"] for row in rows)
    return {
        "dataset": "apnea_ecg_minutes",
        "records": rows,
        "num_windows": len(rows),
        "label_counts": dict(sorted(counts.items())),
        "records_requested": list(args.records),
        "max_minutes_per_record": args.max_minutes_per_record,
        "start_minute": args.start_minute,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Apnea-ECG one-minute ECG windows with apnea/normal labels.")
    parser.add_argument("--raw-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/raw/apnea-ecg"))
    parser.add_argument("--out-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/apnea_ecg"))
    parser.add_argument("--manifest", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/apnea_ecg_manifest.json"))
    parser.add_argument("--records", nargs="*", default=["a01", "b01", "c01"])
    parser.add_argument("--start-minute", type=int, default=0)
    parser.add_argument("--max-minutes-per-record", type=int, default=20)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest(args)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: manifest[key] for key in ["num_windows", "label_counts", "records_requested"]}, indent=2))


if __name__ == "__main__":
    main()
