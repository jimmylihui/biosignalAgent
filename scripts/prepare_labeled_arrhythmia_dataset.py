from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import wfdb

MITDB_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]
BEAT_SYMBOLS = {"N", "L", "R", "A", "a", "J", "S", "V", "F", "e", "j", "E", "/", "f", "Q"}
NORMAL_RHYTHM_SYMBOLS = {"N", "L", "R", "e", "j"}
SUPRAVENTRICULAR_SYMBOLS = {"A", "a", "J", "S"}
VENTRICULAR_SYMBOLS = {"V", "E"}
FUSION_OR_PACED_SYMBOLS = {"F", "/", "f", "Q"}


def classify_symbols(symbols: list[str]) -> dict[str, Any]:
    beat_symbols = [symbol for symbol in symbols if symbol in BEAT_SYMBOLS]
    counts = Counter(beat_symbols)
    abnormal = [symbol for symbol in beat_symbols if symbol not in NORMAL_RHYTHM_SYMBOLS]
    if any(symbol in VENTRICULAR_SYMBOLS for symbol in beat_symbols):
        label = "ventricular_ectopy"
    elif any(symbol in SUPRAVENTRICULAR_SYMBOLS for symbol in beat_symbols):
        label = "supraventricular_ectopy"
    elif any(symbol in FUSION_OR_PACED_SYMBOLS for symbol in beat_symbols):
        label = "paced_fusion_or_unknown"
    else:
        label = "normal_or_bundle_branch"
    return {
        "label": label,
        "binary_label": "abnormal" if abnormal else "normal",
        "num_beats": len(beat_symbols),
        "num_abnormal_beats": len(abnormal),
        "abnormal_fraction": len(abnormal) / len(beat_symbols) if beat_symbols else 0.0,
        "annotation_counts": dict(sorted(counts.items())),
    }


def export_window(record: str, raw_dir: Path, out_dir: Path, start_s: int, seconds: int) -> dict[str, Any] | None:
    header = wfdb.rdheader(str(raw_dir / record))
    start = int(start_s * header.fs)
    stop = int((start_s + seconds) * header.fs)
    wfdb_record = wfdb.rdrecord(str(raw_dir / record), sampfrom=start, sampto=stop)
    lead = "MLII" if "MLII" in wfdb_record.sig_name else wfdb_record.sig_name[0]
    lead_idx = wfdb_record.sig_name.index(lead)
    ann = wfdb.rdann(str(raw_dir / record), "atr", sampfrom=start, sampto=stop)
    labels = classify_symbols(list(ann.symbol))
    if labels["num_beats"] < 5:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"mitdb_{record}_{start_s:05d}_{seconds}s.csv"
    pd.DataFrame({"signal": wfdb_record.p_signal[:, lead_idx]}).to_csv(out_csv, index=False)
    return {
        "dataset": "mitdb_arrhythmia_windows",
        "record": record,
        "window_start_s": start_s,
        "duration_s": seconds,
        "modality": "ecg",
        "path": str(out_csv),
        "sampling_rate": float(header.fs),
        "source_channel": lead,
        **labels,
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    out_dir = Path(args.out_dir)
    raw_dir = Path(args.raw_dir)
    for record in args.records:
        header = wfdb.rdheader(str(raw_dir / record))
        total_seconds = int(header.sig_len / header.fs)
        starts = range(0, max(0, total_seconds - args.seconds + 1), args.stride_seconds)
        kept_for_record = 0
        for start_s in starts:
            if args.max_windows_per_record is not None and kept_for_record >= args.max_windows_per_record:
                break
            row = export_window(record, raw_dir, out_dir, start_s, args.seconds)
            if row is None:
                continue
            rows.append(row)
            kept_for_record += 1
    label_counts = Counter(row["binary_label"] for row in rows)
    detailed_counts = Counter(row["label"] for row in rows)
    return {
        "dataset": "mitdb_arrhythmia_windows",
        "records": rows,
        "num_windows": len(rows),
        "label_counts": dict(sorted(label_counts.items())),
        "detailed_label_counts": dict(sorted(detailed_counts.items())),
        "window_seconds": args.seconds,
        "stride_seconds": args.stride_seconds,
        "max_windows_per_record": args.max_windows_per_record,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MIT-BIH ECG windows with annotation-derived arrhythmia labels.")
    parser.add_argument("--raw-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/raw/mitdb"))
    parser.add_argument("--out-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/labeled_arrhythmia"))
    parser.add_argument("--manifest", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/labeled_arrhythmia_manifest.json"))
    parser.add_argument("--records", nargs="*", default=MITDB_RECORDS)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--stride-seconds", type=int, default=60)
    parser.add_argument("--max-windows-per-record", type=int, default=5)
    args = parser.parse_args()
    manifest = build_manifest(args)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: manifest[key] for key in ["num_windows", "label_counts", "detailed_label_counts"]}, indent=2))


if __name__ == "__main__":
    main()
