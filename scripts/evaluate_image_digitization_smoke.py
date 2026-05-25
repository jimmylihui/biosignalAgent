from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.digitize_tools import ML_MODEL_PATH, Signal_digitize_waveform_image, Signal_digitize_waveform_image_ml
from biosignal_agent.tools.digitize_unet_tools import UNET_MODEL_PATH, Signal_digitize_waveform_image_unet
from biosignal_agent.tools.modality_tools import Signal_classify_modality


def run_digitizer(row: dict[str, Any], method: str, out_dir: Path, model_path: str | None, threshold: float) -> dict[str, Any]:
    out_csv = out_dir / f"{row['record']}_{method}_digitized.csv"
    common = dict(
        image_path=row["image_path"],
        sampling_rate=float(row.get("sampling_rate") or 500.0),
        out_csv=str(out_csv),
        value_min=row.get("value_min"),
        value_max=row.get("value_max"),
        crop_left=int(row.get("crop_left") or 0),
        crop_right=int(row.get("crop_right") or 0),
        crop_top=int(row.get("crop_top") or 0),
        crop_bottom=int(row.get("crop_bottom") or 0),
        smooth_window=1,
    )
    if method == "ml":
        result = Signal_digitize_waveform_image_ml(**common, model_path=model_path or str(ML_MODEL_PATH), probability_threshold=threshold)
    elif method == "unet":
        result = Signal_digitize_waveform_image_unet(**common, model_path=model_path or str(UNET_MODEL_PATH), probability_threshold=threshold)
    else:
        result = Signal_digitize_waveform_image(**common, threshold=80)
    out = {
        "record": row["record"],
        "variant": row.get("variant"),
        "method": method,
        "image_path": row["image_path"],
        "digitizer_error": result.get("error"),
        "out_csv": result.get("out_csv"),
        "pixel_coverage": result.get("pixel_coverage"),
        "mask_pixel_fraction": result.get("mask_pixel_fraction"),
        "confidence": result.get("confidence"),
        "num_points": result.get("num_points"),
    }
    if result.get("out_csv"):
        modality = Signal_classify_modality(result["out_csv"], float(row.get("sampling_rate") or 500.0))
        out.update({
            "modality_prediction": modality.get("predicted_modality"),
            "modality_confidence": modality.get("confidence"),
            "modality_correct": modality.get("predicted_modality") == "ecg",
        })
    return out


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test image digitizers on image-only manifests without waveform references.")
    parser.add_argument("--manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/ecg_image_kit_samples_manifest.json")
    parser.add_argument("--method", choices=["rule", "ml", "unet"], default="unet")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--probability-threshold", type=float, default=0.65)
    parser.add_argument("--out-dir", default="/data1/jiahui/biosignal-agent/outputs/digitized/image_smoke")
    parser.add_argument("--out-json", default="/data1/jiahui/biosignal-agent/outputs/ecg_image_kit_smoke_eval.json")
    parser.add_argument("--out-csv", default="/data1/jiahui/biosignal-agent/outputs/ecg_image_kit_smoke_eval.csv")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_digitizer(record, args.method, out_dir, args.model_path, args.probability_threshold) for record in manifest.get("records", [])]
    ok_rows = [row for row in rows if not row.get("digitizer_error")]
    report = {
        "manifest": args.manifest,
        "method": args.method,
        "model_path": args.model_path,
        "num_records": len(rows),
        "num_ok": len(ok_rows),
        "variant_counts": dict(Counter(row.get("variant") for row in rows)),
        "prediction_counts": dict(Counter(row.get("modality_prediction") for row in ok_rows)),
        "metrics": {
            "ok_rate": len(ok_rows) / len(rows) if rows else 0.0,
            "mean_pixel_coverage": mean(ok_rows, "pixel_coverage"),
            "mean_mask_pixel_fraction": mean(ok_rows, "mask_pixel_fraction"),
            "mean_confidence": mean(ok_rows, "confidence"),
            "ecg_modality_retention": sum(1 for row in ok_rows if row.get("modality_correct")) / len(ok_rows) if ok_rows else 0.0,
        },
        "rows": rows,
        "notes": ["Smoke eval only: ECG-Image-Kit sample images in this clone do not include paired waveform references."],
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    fieldnames = sorted({key for row in rows for key in row})
    with Path(args.out_csv).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"num_records": report["num_records"], "num_ok": report["num_ok"], "metrics": report["metrics"], "prediction_counts": report["prediction_counts"]}, indent=2))


if __name__ == "__main__":
    main()
