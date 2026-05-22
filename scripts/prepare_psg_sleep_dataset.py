from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typing import Any

import pandas as pd

from scripts.prepare_ucddb_resp_spo2_dataset import download_file, parse_respiratory_events, label_window, read_edf_channel

DEFAULT_RECORDS = ["ucddb002"]
STAGE_MAP = {
    0: "wake",
    1: "n1",
    2: "n2",
    3: "n3",
    4: "n3",
    5: "rem",
}
COARSE_STAGE_MAP = {
    "wake": "wake_rem",
    "rem": "wake_rem",
    "n1": "n1_n2",
    "n2": "n1_n2",
    "n3": "n3",
}


def load_stage_codes(path: Path) -> list[int]:
    codes = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            codes.append(int(line))
        except ValueError:
            continue
    return codes


def candidate_starts(stage_codes: list[int], events: list[dict[str, Any]], epoch_s: int, max_windows: int) -> list[int]:
    starts = []
    # Seed with respiratory-event epochs.
    for event in events:
        starts.append(max(0, int(event["start_s"] // epoch_s) * epoch_s))
    # Add first examples of each sleep stage code.
    seen = set()
    for idx, code in enumerate(stage_codes):
        label = STAGE_MAP.get(code, "unknown")
        if label not in seen:
            starts.append(idx * epoch_s)
            seen.add(label)
    # Add evenly spaced background windows to avoid only event-driven PSG.
    step = max(1, len(stage_codes) // max(1, max_windows))
    for idx in range(0, len(stage_codes), step):
        starts.append(idx * epoch_s)
    return sorted(set(starts))[:max_windows]


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    for record in args.records:
        rec_path = download_file(f"{record}.rec", raw_dir, args.download)
        stage_path = download_file(f"{record}_stage.txt", raw_dir, args.download)
        event_path = download_file(f"{record}_respevt.txt", raw_dir, args.download)
        stages = load_stage_codes(stage_path)
        events = parse_respiratory_events(event_path)
        for start_s in candidate_starts(stages, events, args.epoch_seconds, args.max_windows_per_record):
            epoch_idx = start_s // args.epoch_seconds
            if epoch_idx >= len(stages):
                continue
            stage = STAGE_MAP.get(stages[epoch_idx], "unknown")
            event_label = label_window(events, start_s, args.epoch_seconds)
            out_dir.mkdir(parents=True, exist_ok=True)
            eeg, eeg_fs = read_edf_channel(rec_path, args.eeg_channel, start_s, args.epoch_seconds)
            resp, resp_fs = read_edf_channel(rec_path, args.resp_channel, start_s, args.epoch_seconds)
            spo2, spo2_fs = read_edf_channel(rec_path, args.spo2_channel, start_s, args.epoch_seconds)
            prefix = f"{record}_{start_s:05d}_{args.epoch_seconds}s"
            eeg_csv = out_dir / f"{prefix}_eeg.csv"
            resp_csv = out_dir / f"{prefix}_resp.csv"
            spo2_csv = out_dir / f"{prefix}_spo2.csv"
            pd.DataFrame({"signal": eeg}).to_csv(eeg_csv, index=False)
            pd.DataFrame({"signal": resp}).to_csv(resp_csv, index=False)
            pd.DataFrame({"signal": spo2}).to_csv(spo2_csv, index=False)
            rows.append({
                "dataset": "ucddb_psg_sleep_windows",
                "record": record,
                "window_start_s": start_s,
                "duration_s": args.epoch_seconds,
                "stage_code": stages[epoch_idx],
                "sleep_stage": stage,
                "coarse_sleep_stage": COARSE_STAGE_MAP.get(stage, "unknown"),
                "respiratory_event_label": event_label["label"],
                "event_count": event_label["event_count"],
                "event_types": event_label["event_types"],
                "eeg_path": str(eeg_csv),
                "eeg_sampling_rate": float(eeg_fs),
                "eeg_channel": args.eeg_channel,
                "resp_path": str(resp_csv),
                "resp_sampling_rate": float(resp_fs),
                "resp_channel": args.resp_channel,
                "spo2_path": str(spo2_csv),
                "spo2_sampling_rate": float(spo2_fs),
                "spo2_channel": args.spo2_channel,
            })
    return {
        "dataset": "ucddb_psg_sleep_windows",
        "records": rows,
        "num_windows": len(rows),
        "stage_counts": dict(Counter(row["sleep_stage"] for row in rows)),
        "coarse_stage_counts": dict(Counter(row["coarse_sleep_stage"] for row in rows)),
        "respiratory_event_counts": dict(Counter(row["respiratory_event_label"] for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare UCDDB PSG windows with sleep-stage and respiratory-event labels.")
    parser.add_argument("--raw-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/raw/ucddb"))
    parser.add_argument("--out-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/psg_sleep"))
    parser.add_argument("--manifest", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/psg_sleep_manifest.json"))
    parser.add_argument("--records", nargs="*", default=DEFAULT_RECORDS)
    parser.add_argument("--epoch-seconds", type=int, default=30)
    parser.add_argument("--max-windows-per-record", type=int, default=80)
    parser.add_argument("--eeg-channel", default="C3A2")
    parser.add_argument("--resp-channel", default="Flow")
    parser.add_argument("--spo2-channel", default="SpO2")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest(args)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: manifest[key] for key in ["num_windows", "stage_counts", "coarse_stage_counts", "respiratory_event_counts"]}, indent=2))


if __name__ == "__main__":
    main()
