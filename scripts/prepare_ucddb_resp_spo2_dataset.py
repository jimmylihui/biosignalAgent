from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://physionet.org/files/ucddb/1.0.0"
DEFAULT_RECORDS = ["ucddb002"]


def download_file(name: str, raw_dir: Path, download: bool) -> Path:
    path = raw_dir / name
    if path.exists():
        return path
    if not download:
        raise FileNotFoundError(f"Missing {path}. Re-run with --download.")
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{name}"
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return path


def parse_edf_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        fixed = handle.read(256)
        header_bytes = int(fixed[184:192].decode().strip())
        num_records = int(fixed[236:244].decode().strip())
        record_duration = float(fixed[244:252].decode().strip().replace("+", ""))
        num_signals = int(fixed[252:256].decode().strip())
        handle.seek(0)
        header = handle.read(header_bytes)
    pos = 256
    fields = {}
    for name, length, caster in [
        ("labels", 16, str),
        ("transducers", 80, str),
        ("phys_dims", 8, str),
        ("phys_mins", 8, float),
        ("phys_maxs", 8, float),
        ("dig_mins", 8, float),
        ("dig_maxs", 8, float),
        ("prefilters", 80, str),
        ("samples_per_record", 8, int),
        ("reserved", 32, str),
    ]:
        vals = []
        for i in range(num_signals):
            raw = header[pos + i * length: pos + (i + 1) * length].decode(errors="replace").strip()
            vals.append(caster(raw) if raw else caster(0) if caster is not str else "")
        fields[name] = vals
        pos += length * num_signals
    return {"header_bytes": header_bytes, "num_records": num_records, "record_duration": record_duration, "num_signals": num_signals, **fields}


def read_edf_channel(path: Path, channel: str, start_s: int, duration_s: int) -> tuple[np.ndarray, float]:
    meta = parse_edf_header(path)
    labels = meta["labels"]
    if channel not in labels:
        raise ValueError(f"Channel {channel} not found in {path}; available={labels}")
    idx = labels.index(channel)
    samples_per_record = meta["samples_per_record"]
    record_duration = meta["record_duration"]
    fs = samples_per_record[idx] / record_duration
    start_record = int(start_s // record_duration)
    end_record = int(np.ceil((start_s + duration_s) / record_duration))
    bytes_per_record = int(sum(samples_per_record) * 2)
    offset_within_record = int(sum(samples_per_record[:idx]) * 2)
    values = []
    with path.open("rb") as handle:
        for record_idx in range(start_record, min(end_record, meta["num_records"])):
            offset = meta["header_bytes"] + record_idx * bytes_per_record + offset_within_record
            handle.seek(offset)
            raw = handle.read(samples_per_record[idx] * 2)
            digital = np.frombuffer(raw, dtype="<i2").astype(float)
            values.append(digital)
    if not values:
        return np.array([], dtype=float), fs
    digital = np.concatenate(values)
    phys_min = meta["phys_mins"][idx]
    phys_max = meta["phys_maxs"][idx]
    dig_min = meta["dig_mins"][idx]
    dig_max = meta["dig_maxs"][idx]
    physical = (digital - dig_min) / (dig_max - dig_min) * (phys_max - phys_min) + phys_min
    start_offset = int((start_s - start_record * record_duration) * fs)
    length = int(duration_s * fs)
    return physical[start_offset:start_offset + length], fs


def parse_time_to_seconds(value: str) -> int:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


def parse_respiratory_events(path: Path) -> list[dict[str, Any]]:
    events = []
    for line in path.read_text(errors="replace").splitlines():
        match = re.match(r"^(\d\d:\d\d:\d\d)\s+([A-Z-]+)\s+(?:\S+\s+)?(\d+)", line.strip())
        if not match:
            continue
        start_s = parse_time_to_seconds(match.group(1))
        event_type = match.group(2)
        duration_s = int(match.group(3))
        if event_type.startswith("APNEA") or event_type.startswith("HYP"):
            events.append({"start_s": start_s, "end_s": start_s + duration_s, "duration_s": duration_s, "event_type": event_type})
    return events


def label_window(events: list[dict[str, Any]], start_s: int, duration_s: int) -> dict[str, Any]:
    end_s = start_s + duration_s
    overlapping = [event for event in events if event["start_s"] < end_s and event["end_s"] > start_s]
    apnea_events = [event for event in overlapping if event["event_type"].startswith("APNEA")]
    hypopnea_events = [event for event in overlapping if event["event_type"].startswith("HYP")]
    return {
        "label": "respiratory_event" if overlapping else "normal",
        "event_count": len(overlapping),
        "apnea_count": len(apnea_events),
        "hypopnea_count": len(hypopnea_events),
        "event_types": sorted({event["event_type"] for event in overlapping}),
    }


def candidate_starts(events: list[dict[str, Any]], max_windows: int, duration_s: int) -> list[int]:
    positives = sorted({max(0, event["start_s"] - 15) for event in events})
    negatives = []
    cursor = 0
    event_ranges = [(event["start_s"], event["end_s"]) for event in events]
    while len(negatives) < max_windows and cursor < 8 * 3600:
        if all(not (start < cursor + duration_s and end > cursor) for start, end in event_ranges):
            negatives.append(cursor)
        cursor += duration_s * 5
    selected = positives[: max_windows // 2] + negatives[: max_windows - min(len(positives), max_windows // 2)]
    return sorted(set(selected))[:max_windows]


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for record in args.records:
        raw_dir = Path(args.raw_dir)
        rec_path = download_file(f"{record}.rec", raw_dir, args.download)
        evt_path = download_file(f"{record}_respevt.txt", raw_dir, args.download)
        events = parse_respiratory_events(evt_path)
        for start_s in candidate_starts(events, args.max_windows_per_record, args.seconds):
            labels = label_window(events, start_s, args.seconds)
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            flow, flow_fs = read_edf_channel(rec_path, args.resp_channel, start_s, args.seconds)
            spo2, spo2_fs = read_edf_channel(rec_path, args.spo2_channel, start_s, args.seconds)
            resp_csv = out_dir / f"{record}_{start_s:05d}_{args.resp_channel.lower()}_{args.seconds}s.csv"
            spo2_csv = out_dir / f"{record}_{start_s:05d}_{args.spo2_channel.lower()}_{args.seconds}s.csv"
            pd.DataFrame({"signal": flow}).to_csv(resp_csv, index=False)
            pd.DataFrame({"signal": spo2}).to_csv(spo2_csv, index=False)
            rows.append({
                "dataset": "ucddb_resp_spo2_windows",
                "record": record,
                "window_start_s": start_s,
                "duration_s": args.seconds,
                "resp_path": str(resp_csv),
                "resp_sampling_rate": float(flow_fs),
                "spo2_path": str(spo2_csv),
                "spo2_sampling_rate": float(spo2_fs),
                **labels,
            })
    counts = Counter(row["label"] for row in rows)
    return {"dataset": "ucddb_resp_spo2_windows", "records": rows, "num_windows": len(rows), "label_counts": dict(sorted(counts.items()))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare UCDDB RESP/SpO2 windows with respiratory-event labels.")
    parser.add_argument("--raw-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/raw/ucddb"))
    parser.add_argument("--out-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/ucddb_resp_spo2"))
    parser.add_argument("--manifest", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/ucddb_resp_spo2_manifest.json"))
    parser.add_argument("--records", nargs="*", default=DEFAULT_RECORDS)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--max-windows-per-record", type=int, default=40)
    parser.add_argument("--resp-channel", default="Flow")
    parser.add_argument("--spo2-channel", default="SpO2")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest(args)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: manifest[key] for key in ["num_windows", "label_counts"]}, indent=2))


if __name__ == "__main__":
    main()
