from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from scipy.io import wavfile

BASE_URL = "https://physionet.org/files/circor-heart-sound/1.0.3"
LOCATIONS = ("AV", "PV", "TV", "MV", "Phc")


def download_file(url: str, path: Path, download: bool) -> Path:
    if path.exists() and path.stat().st_size > 256:
        return path
    if not download:
        raise FileNotFoundError(f"Missing {path}. Re-run with --download.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=90) as response:
        response.raise_for_status()
        tmp = path.with_suffix(path.suffix + ".part")
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
        tmp.replace(path)
    return path


def load_metadata(raw_dir: Path, download: bool) -> pd.DataFrame:
    csv_path = download_file(f"{BASE_URL}/training_data.csv", raw_dir / "training_data.csv", download)
    return pd.read_csv(csv_path)


def load_record_stems(raw_dir: Path, download: bool) -> set[str]:
    records_path = download_file(f"{BASE_URL}/RECORDS", raw_dir / "RECORDS", download)
    stems = set()
    for line in records_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        stems.add(Path(line).name)
    return stems


def selected_patients(df: pd.DataFrame, max_per_class: int | None, seed: int) -> list[dict[str, Any]]:
    rows = []
    for label in ("Present", "Absent"):
        subset = df[df["Murmur"] == label].copy()
        subset = subset.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        if max_per_class is not None:
            subset = subset.head(max_per_class)
        for _, row in subset.iterrows():
            rows.append(row.to_dict())
    return rows


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    df = load_metadata(raw_dir, args.download)
    available_stems = load_record_stems(raw_dir, args.download)
    patients = selected_patients(df, args.max_per_class, args.seed)
    records = []
    for patient in patients:
        patient_id = str(int(patient["Patient ID"]))
        label = str(patient["Murmur"])
        locations_text = str(patient.get("Recording locations:") or "")
        locations = [loc for loc in LOCATIONS if loc in locations_text.split("+")]
        for loc in locations:
            stem = f"{patient_id}_{loc}"
            if stem not in available_stems:
                continue
            wav_path = download_file(f"{BASE_URL}/training_data/{stem}.wav", raw_dir / "training_data" / f"{stem}.wav", args.download)
            sampling_rate, values = wavfile.read(wav_path)
            if values.ndim > 1:
                values = values[:, 0]
            duration_s = float(len(values) / sampling_rate)
            if duration_s < args.min_seconds:
                continue
            murmur_locations = str(patient.get("Murmur locations") or "")
            location_has_murmur = label == "Present" and loc in murmur_locations.split("+")
            records.append({
                "dataset": "circor_heart_sound_1.0.3",
                "patient_id": patient_id,
                "record": stem,
                "location": loc,
                "modality": "pcg",
                "path": str(wav_path),
                "sampling_rate": float(sampling_rate),
                "duration_s": duration_s,
                "patient_murmur_label": label.lower(),
                "record_murmur_label": "present" if location_has_murmur else "absent",
                "label": "abnormal" if location_has_murmur else "normal",
                "patient_label": "abnormal" if label == "Present" else "normal",
                "outcome": patient.get("Outcome"),
                "age": patient.get("Age"),
                "sex": patient.get("Sex"),
                "murmur_locations": patient.get("Murmur locations"),
                "most_audible_location": patient.get("Most audible location"),
            })
    return {
        "dataset": "circor_heart_sound_1.0.3",
        "label_type": "location_aware_murmur_present_absent_recordings",
        "num_records": len(records),
        "num_patients": len(set(r["patient_id"] for r in records)),
        "record_label_counts": dict(Counter(r["label"] for r in records)),
        "patient_label_counts": dict(Counter((str(p["Murmur"]).lower()) for p in patients)),
        "record_murmur_label_counts": dict(Counter(r["record_murmur_label"] for r in records)),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/raw/circor-heart-sound/1.0.3"))
    parser.add_argument("--out-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/circor_pcg"))
    parser.add_argument("--manifest", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/circor_pcg_murmur_manifest.json"))
    parser.add_argument("--max-per-class", type=int, default=120)
    parser.add_argument("--min-seconds", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest(args)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ["num_records", "num_patients", "record_label_counts", "record_murmur_label_counts", "patient_label_counts"]}, indent=2))


if __name__ == "__main__":
    main()
