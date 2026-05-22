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

BASE_URL = "https://physionet.org/files/chbmit/1.0.0"


def edf_expected_size(path: Path) -> int | None:
    try:
        header = parse_edf_header(path)
    except Exception:
        return None
    return int(header["header_bytes"] + header["num_records"] * int(np.sum(header["samples_per_record"])) * 2)


def download_file(url: str, path: Path, enabled: bool) -> Path:
    if path.exists():
        if path.suffix.lower() == ".edf":
            expected = edf_expected_size(path)
            if expected is not None and path.stat().st_size < expected:
                if not enabled:
                    raise FileNotFoundError(f"Incomplete EDF {path}; re-run with --download to fetch the full file.")
            else:
                return path
        else:
            return path
    if not enabled:
        raise FileNotFoundError(f"Missing {path}. Re-run with --download to fetch the CHB-MIT subset.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp_path.replace(path)
    return path


def parse_summary(path: Path) -> list[dict[str, Any]]:
    records = []
    current: dict[str, Any] | None = None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("File Name:"):
            if current:
                records.append(current)
            current = {"file": line.split(":", 1)[1].strip(), "seizures": []}
        elif current and "Number of Seizures in File" in line:
            current["num_seizures"] = int(re.findall(r"\d+", line)[0])
        elif current and "Seizure Start Time" in line:
            current["seizures"].append({"start_s": int(re.findall(r"\d+", line)[0])})
        elif current and "Seizure End Time" in line and current["seizures"]:
            current["seizures"][-1]["end_s"] = int(re.findall(r"\d+", line)[0])
    if current:
        records.append(current)
    return records


def parse_edf_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        fixed = handle.read(256)
        header_bytes = int(fixed[184:192].decode(errors="ignore").strip())
        num_records = int(float(fixed[236:244].decode(errors="ignore").strip()))
        record_duration = float(fixed[244:252].decode(errors="ignore").strip())
        num_signals = int(fixed[252:256].decode(errors="ignore").strip())
        signal_header = handle.read(header_bytes - 256)
    offset = 0
    def take(width: int) -> list[str]:
        nonlocal offset
        values = [signal_header[offset + i * width: offset + (i + 1) * width].decode(errors="ignore").strip() for i in range(num_signals)]
        offset += width * num_signals
        return values
    labels = take(16)
    take(80)  # transducer
    take(8)   # physical dimension
    physical_min = np.asarray([float(x or 0) for x in take(8)], dtype=float)
    physical_max = np.asarray([float(x or 0) for x in take(8)], dtype=float)
    digital_min = np.asarray([float(x or -32768) for x in take(8)], dtype=float)
    digital_max = np.asarray([float(x or 32767) for x in take(8)], dtype=float)
    take(80)  # prefilter
    samples_per_record = np.asarray([int(x) for x in take(8)], dtype=int)
    return {
        "header_bytes": header_bytes,
        "num_records": num_records,
        "record_duration": record_duration,
        "num_signals": num_signals,
        "labels": labels,
        "samples_per_record": samples_per_record,
        "physical_min": physical_min,
        "physical_max": physical_max,
        "digital_min": digital_min,
        "digital_max": digital_max,
    }


def read_edf_channel(path: Path, channel: str | None, start_s: float, duration_s: float) -> tuple[np.ndarray, float, str]:
    header = parse_edf_header(path)
    labels = header["labels"]
    channel_idx = labels.index(channel) if channel in labels else 0
    samples_per_record = header["samples_per_record"]
    fs = float(samples_per_record[channel_idx] / header["record_duration"])
    start_sample = int(start_s * fs)
    stop_sample = int((start_s + duration_s) * fs)
    values = []
    target_offset = int(np.sum(samples_per_record[:channel_idx]))
    record_width = int(np.sum(samples_per_record))
    with path.open("rb") as handle:
        handle.seek(header["header_bytes"])
        for _ in range(header["num_records"]):
            record = np.fromfile(handle, dtype="<i2", count=record_width)
            if len(record) < record_width:
                break
            segment = record[target_offset: target_offset + samples_per_record[channel_idx]]
            values.append(segment)
    digital = np.concatenate(values).astype(float)
    digital = digital[start_sample:stop_sample]
    dmin = header["digital_min"][channel_idx]
    dmax = header["digital_max"][channel_idx]
    pmin = header["physical_min"][channel_idx]
    pmax = header["physical_max"][channel_idx]
    physical = (digital - dmin) / (dmax - dmin + 1e-12) * (pmax - pmin) + pmin
    return physical, fs, labels[channel_idx]


def non_seizure_start(file_info: dict[str, Any], duration_s: float) -> float:
    intervals = [(item["start_s"], item.get("end_s", item["start_s"] + duration_s)) for item in file_info.get("seizures", [])]
    for candidate in [60.0, 300.0, 600.0, 1200.0, 1800.0]:
        if all(candidate + duration_s < start or candidate > end + duration_s for start, end in intervals):
            return candidate
    return 0.0


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    raw_dir = Path(args.raw_dir) / args.subject
    summary = download_file(f"{BASE_URL}/{args.subject}/{args.subject}-summary.txt", raw_dir / f"{args.subject}-summary.txt", args.download)
    infos = [item for item in parse_summary(summary) if item.get("seizures")]
    rows = []
    out_dir = Path(args.out_dir)
    for info in infos[: args.max_seizure_files]:
        edf_path = download_file(f"{BASE_URL}/{args.subject}/{info['file']}", raw_dir / info["file"], args.download)
        seizure = info["seizures"][0]
        seizure_start = max(0.0, float(seizure["start_s"]) - args.window_seconds / 2.0)
        examples = [(seizure_start, "seizure"), (non_seizure_start(info, args.window_seconds), "non_seizure")]
        for start_s, label in examples:
            values, fs, channel = read_edf_channel(edf_path, args.channel, start_s, args.window_seconds)
            if len(values) < int(fs * args.window_seconds * 0.8):
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            rec_id = f"{Path(info['file']).stem}_{int(start_s):05d}_{label}"
            out_csv = out_dir / f"chbmit_{rec_id}_eeg.csv"
            pd.DataFrame({"signal": values}).to_csv(out_csv, index=False)
            rows.append({
                "dataset": "chbmit_seizure",
                "record": rec_id,
                "subject": args.subject,
                "modality": "eeg",
                "path": str(out_csv),
                "sampling_rate": fs,
                "duration_s": float(len(values) / fs),
                "source_file": str(edf_path),
                "source_channel": channel,
                "window_start_s": start_s,
                "label": label,
            })
    label_counts = Counter(row["label"] for row in rows)
    if not rows:
        raise RuntimeError("No CHB-MIT seizure/non-seizure windows were exported.")
    if label_counts.get("seizure", 0) == 0 or label_counts.get("non_seizure", 0) == 0:
        raise RuntimeError(
            "CHB-MIT export did not include both seizure and non-seizure windows. "
            "The EDF file may be incomplete; remove the partial .edf and re-run with --download."
        )
    return {"dataset": "chbmit_seizure", "records": rows, "num_windows": len(rows), "label_counts": dict(label_counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a small CHB-MIT seizure-window benchmark.")
    parser.add_argument("--raw-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/raw/chbmit"))
    parser.add_argument("--out-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/chbmit_seizure"))
    parser.add_argument("--manifest", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/chbmit_seizure_manifest.json"))
    parser.add_argument("--subject", default="chb01")
    parser.add_argument("--channel", default=None)
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--max-seizure-files", type=int, default=2)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest(args)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: manifest[key] for key in ["num_windows", "label_counts"]}, indent=2))


if __name__ == "__main__":
    main()
