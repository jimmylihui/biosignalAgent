#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import wfdb

CLASS_TO_ID = {"background": 0, "p": 1, "qrs": 2, "t": 3}
NUM_TO_CLASS = {0: "p", 1: "qrs", 2: "t"}


def ensure_qtdb(raw_dir: Path, max_records: int | None = None, annotators: tuple[str, ...] = ("q1c", "q2c", "pu0", "pu1")) -> list[str]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    records = wfdb.get_record_list("qtdb")
    if max_records:
        records = records[:max_records]
    missing = []
    for rec in records:
        for file_name in [f"{rec}.hea", f"{rec}.dat", *[f"{rec}.{ann}" for ann in annotators]]:
            local = raw_dir / file_name
            if local.exists():
                continue
            try:
                wfdb.dl_files("qtdb", dl_dir=str(raw_dir), files=[file_name], keep_subdirs=False, overwrite=False)
            except Exception as exc:
                missing.append({"file": file_name, "error": f"{type(exc).__name__}: {exc}"})
    if missing:
        (raw_dir / "download_missing_files.json").write_text(json.dumps(missing, indent=2) + "\n")
    return records


def annotation_to_mask(base: Path, annotator: str, length: int) -> tuple[np.ndarray, dict]:
    ann = wfdb.rdann(str(base), annotator)
    mask = np.zeros(length, dtype=np.uint8)
    open_start: dict[int, int] = {}
    pending_center: dict[int, int] = {}
    spans = []
    centers = {"p": [], "qrs": [], "t": []}
    for sample, sym, num in zip(ann.sample, ann.symbol, ann.num):
        sample = int(sample)
        num = int(num)
        wave = NUM_TO_CLASS.get(num)
        if wave is None:
            continue
        if sym == "(":
            open_start[num] = sample
        elif sym == ")":
            if num in open_start:
                start = max(0, open_start.pop(num))
            elif num in pending_center:
                start = max(0, pending_center[num])
            else:
                start = max(0, sample - 12)
            stop = min(length, sample + 1)
            if stop > start:
                mask[start:stop] = CLASS_TO_ID[wave]
                spans.append({"wave": wave, "start": start, "stop": stop})
        elif sym.lower() in {"p", "n", "t"}:
            centers[wave].append(sample)
            pending_center[num] = sample
    return mask, {"num_spans": len(spans), "spans": spans, "centers": centers}


def iter_windows(signal: np.ndarray, mask: np.ndarray, window: int, stride: int, min_labeled: int):
    n = len(signal)
    starts = set(range(0, max(1, n - window + 1), stride))
    labeled = np.flatnonzero(mask > 0)
    if len(labeled):
        for center in labeled[:: max(1, len(labeled) // 300)]:
            starts.add(int(max(0, min(n - window, center - window // 2))))
    for start in sorted(starts):
        stop = start + window
        if stop > n:
            continue
        y = mask[start:stop]
        if int(np.sum(y > 0)) < min_labeled:
            continue
        x = signal[start:stop].astype(np.float32)
        med = float(np.nanmedian(x))
        iqr = float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25)) + 1e-6
        x = np.clip((x - med) / iqr, -8, 8).astype(np.float32)
        yield start, x, y.astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare QTDB ECG P/QRS/T delineation windows for segmentation training.")
    ap.add_argument("--raw-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/raw/qtdb"))
    ap.add_argument("--out-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/qtdb_delineation"))
    ap.add_argument("--manifest", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/qtdb_delineation_manifest.json"))
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--min-labeled", type=int, default=24)
    ap.add_argument("--annotators", default="q1c,q2c", help="Comma-separated annotators, e.g. q1c,q2c for manual or pu0,pu1 for ecgpuwave pretraining.")
    args = ap.parse_args()

    annotators = tuple(a.strip() for a in args.annotators.split(",") if a.strip())
    records = ensure_qtdb(args.raw_dir, args.max_records, annotators=annotators) if args.download else wfdb.get_record_list("qtdb")
    if args.max_records:
        records = records[: args.max_records]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = {"records_seen": 0, "windows": 0, "annotator_counts": {}, "class_pixels": {k: 0 for k in CLASS_TO_ID}}
    for rec in records:
        base = args.raw_dir / rec
        if not (base.with_suffix(".hea").exists() and base.with_suffix(".dat").exists()):
            continue
        try:
            record = wfdb.rdrecord(str(base))
        except Exception:
            continue
        values = record.p_signal[:, 0].astype(np.float32)
        summary["records_seen"] += 1
        for annotator in annotators:
            if not base.with_suffix(f".{annotator}").exists():
                continue
            try:
                mask, meta = annotation_to_mask(base, annotator, len(values))
            except Exception:
                continue
            count = 0
            for start, x, y in iter_windows(values, mask, args.window, args.stride, args.min_labeled):
                out = args.out_dir / f"{rec}_{annotator}_{start:07d}.npz"
                np.savez_compressed(out, signal=x, mask=y, fs=float(record.fs), record=rec, annotator=annotator, start_sample=int(start))
                rows.append({"path": str(out), "record": rec, "annotator": annotator, "start_sample": int(start), "sampling_rate": float(record.fs), "window": int(args.window)})
                count += 1
                summary["windows"] += 1
                for cls, idx in CLASS_TO_ID.items():
                    summary["class_pixels"][cls] += int(np.sum(y == idx))
            summary["annotator_counts"][annotator] = summary["annotator_counts"].get(annotator, 0) + count
    payload = {"dataset": "QTDB delineation", "class_to_id": CLASS_TO_ID, "annotators": annotators, "summary": summary, "rows": rows}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"manifest": str(args.manifest), **summary}, indent=2))


if __name__ == "__main__":
    main()
