from __future__ import annotations

import argparse
import json
import sys
import re
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typing import Any

import pandas as pd
import wfdb

from scripts.prepare_labeled_arrhythmia_dataset import MITDB_RECORDS, BEAT_SYMBOLS

AF_RHYTHMS = {"AFIB", "AFL"}
NORMAL_RHYTHMS = {"N"}
NORMAL_BEATS = {"N", "L", "R", "e", "j"}
SUPRAVENTRICULAR_BEATS = {"A", "a", "J", "S"}
VENTRICULAR_BEATS = {"V", "E"}
FUSION_PACED_BEATS = {"F", "/", "f", "Q"}


def clean_rhythm(note: str) -> str | None:
    note = note.replace("\x00", "").strip()
    if not note.startswith("("):
        return None
    token = re.sub(r"[^A-Za-z0-9_+-]", "", note[1:])
    return token or None


def rhythm_intervals(raw_dir: Path, record: str, total_samples: int) -> list[dict[str, Any]]:
    ann = wfdb.rdann(str(raw_dir / record), "atr")
    starts = []
    for sample, note in zip(ann.sample, ann.aux_note):
        rhythm = clean_rhythm(note)
        if rhythm:
            starts.append((int(sample), rhythm))
    if not starts:
        starts = [(0, "UNKNOWN")]
    if starts[0][0] > 0:
        starts.insert(0, (0, starts[0][1]))
    intervals = []
    for idx, (start, rhythm) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else total_samples
        intervals.append({"start_sample": start, "end_sample": int(end), "rhythm": rhythm})
    return intervals


def rhythm_label_for_window(intervals: list[dict[str, Any]], start: int, stop: int) -> dict[str, Any]:
    overlap_by_rhythm: Counter[str] = Counter()
    for item in intervals:
        overlap = max(0, min(stop, item["end_sample"]) - max(start, item["start_sample"]))
        if overlap > 0:
            overlap_by_rhythm[item["rhythm"]] += overlap
    if not overlap_by_rhythm:
        rhythm = "UNKNOWN"
    else:
        rhythm = overlap_by_rhythm.most_common(1)[0][0]
    if rhythm in AF_RHYTHMS:
        coarse = "af"
    elif rhythm in NORMAL_RHYTHMS:
        coarse = "normal"
    else:
        coarse = "other_rhythm"
    return {"rhythm_label": rhythm, "coarse_rhythm_label": coarse, "rhythm_overlap_samples": dict(overlap_by_rhythm)}


def beat_label(symbol: str) -> str:
    if symbol in NORMAL_BEATS:
        return "normal"
    if symbol in SUPRAVENTRICULAR_BEATS:
        return "supraventricular"
    if symbol in VENTRICULAR_BEATS:
        return "ventricular"
    if symbol in FUSION_PACED_BEATS:
        return "fusion_paced_unknown"
    return "other"


def export_window(record: str, raw_dir: Path, out_dir: Path, start_s: int, seconds: int) -> dict[str, Any] | None:
    header = wfdb.rdheader(str(raw_dir / record))
    start = int(start_s * header.fs)
    stop = int((start_s + seconds) * header.fs)
    signal_record = wfdb.rdrecord(str(raw_dir / record), sampfrom=start, sampto=stop)
    lead = "MLII" if "MLII" in signal_record.sig_name else signal_record.sig_name[0]
    lead_idx = signal_record.sig_name.index(lead)
    ann = wfdb.rdann(str(raw_dir / record), "atr", sampfrom=start, sampto=stop)
    intervals = rhythm_intervals(raw_dir, record, header.sig_len)
    rhythm = rhythm_label_for_window(intervals, start, stop)
    beats = []
    for sample, symbol in zip(ann.sample, ann.symbol):
        if symbol not in BEAT_SYMBOLS:
            continue
        beats.append({"sample": int(sample - start), "absolute_sample": int(sample), "symbol": symbol, "label": beat_label(symbol)})
    if len(beats) < 5:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"mitdb_{record}_{start_s:05d}_{seconds}s_ecg.csv"
    pd.DataFrame({"signal": signal_record.p_signal[:, lead_idx]}).to_csv(out_csv, index=False)
    beat_counts = Counter(item["label"] for item in beats)
    return {
        "dataset": "mitdb_rhythm_beat_windows",
        "record": record,
        "window_start_s": start_s,
        "duration_s": seconds,
        "modality": "ecg",
        "path": str(out_csv),
        "sampling_rate": float(header.fs),
        "source_channel": lead,
        "num_beats": len(beats),
        "beat_label_counts": dict(sorted(beat_counts.items())),
        "beat_annotations": beats,
        **rhythm,
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    for record in args.records:
        header = wfdb.rdheader(str(raw_dir / record))
        total_seconds = int(header.sig_len / header.fs)
        kept = 0
        for start_s in range(0, max(0, total_seconds - args.seconds + 1), args.stride_seconds):
            if args.max_windows_per_record is not None and kept >= args.max_windows_per_record:
                break
            row = export_window(record, raw_dir, out_dir, start_s, args.seconds)
            if row is None:
                continue
            rows.append(row)
            kept += 1
    return {
        "dataset": "mitdb_rhythm_beat_windows",
        "records": rows,
        "num_windows": len(rows),
        "rhythm_counts": dict(Counter(row["coarse_rhythm_label"] for row in rows)),
        "detailed_rhythm_counts": dict(Counter(row["rhythm_label"] for row in rows)),
        "beat_counts": dict(sum((Counter(row["beat_label_counts"]) for row in rows), Counter())),
        "window_seconds": args.seconds,
        "stride_seconds": args.stride_seconds,
        "max_windows_per_record": args.max_windows_per_record,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MIT-BIH windows with rhythm and beat labels.")
    parser.add_argument("--raw-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/raw/mitdb"))
    parser.add_argument("--out-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/ecg_rhythm_beat"))
    parser.add_argument("--manifest", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/ecg_rhythm_beat_manifest.json"))
    parser.add_argument("--records", nargs="*", default=MITDB_RECORDS)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--stride-seconds", type=int, default=60)
    parser.add_argument("--max-windows-per-record", type=int, default=5)
    args = parser.parse_args()
    manifest = build_manifest(args)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: manifest[key] for key in ["num_windows", "rhythm_counts", "detailed_rhythm_counts", "beat_counts"]}, indent=2))


if __name__ == "__main__":
    main()
