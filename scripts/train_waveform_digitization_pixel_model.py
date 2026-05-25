from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, jaccard_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.digitize_tools import _crop_rgb_image, pixel_feature_matrix

FEATURE_NAMES = ["gray", "inverse_gray", "red", "green", "blue", "saturation", "blue_excess", "red_excess", "row_norm", "col_norm", "grad_y", "grad_x"]


def load_mask(path: str, crop_left: int, crop_right: int, crop_top: int, crop_bottom: int) -> np.ndarray:
    image = Image.open(path).convert("L")
    width, height = image.size
    left = max(0, int(crop_left))
    right = width - max(0, int(crop_right))
    top = max(0, int(crop_top))
    bottom = height - max(0, int(crop_bottom))
    return np.asarray(image, dtype=np.uint8)[top:bottom, left:right] > 0


def sample_record(record: dict[str, Any], rng: np.random.Generator, max_positive: int, max_negative: int) -> tuple[np.ndarray, np.ndarray]:
    image, _, _ = _crop_rgb_image(record["image_path"], int(record["crop_left"]), int(record["crop_right"]), int(record["crop_top"]), int(record["crop_bottom"]))
    mask = load_mask(record["mask_path"], int(record["crop_left"]), int(record["crop_right"]), int(record["crop_top"]), int(record["crop_bottom"]))
    features = pixel_feature_matrix(image)
    labels = mask.ravel().astype(int)
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    if len(pos) > max_positive:
        pos = rng.choice(pos, size=max_positive, replace=False)
    if len(neg) > max_negative:
        neg = rng.choice(neg, size=max_negative, replace=False)
    idx = np.concatenate([pos, neg])
    rng.shuffle(idx)
    return features[idx], labels[idx]


def train(manifest_path: str, model_path: str, train_variants: set[str] | None, max_positive: int, max_negative: int) -> dict[str, Any]:
    rng = np.random.default_rng(13)
    manifest = json.loads(Path(manifest_path).read_text())
    records = [row for row in manifest.get("records", []) if row.get("mask_path")]
    if train_variants:
        records = [row for row in records if row.get("variant") in train_variants]
    if not records:
        raise ValueError("No records with masks selected for training.")
    x_parts = []
    y_parts = []
    for record in records:
        x, y = sample_record(record, rng, max_positive=max_positive, max_negative=max_negative)
        x_parts.append(x)
        y_parts.append(y)
    x_train = np.vstack(x_parts)
    y_train = np.concatenate(y_parts)
    model = ExtraTreesClassifier(
        n_estimators=80,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=13,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    report = {
        "manifest": manifest_path,
        "model_path": model_path,
        "train_variants": sorted(train_variants) if train_variants else ["all"],
        "num_records": len(records),
        "sample_counts": dict(Counter(y_train.tolist())),
        "train_pixel_iou": float(jaccard_score(y_train, train_pred, zero_division=0)),
        "train_pixel_f1": float(f1_score(y_train, train_pred, zero_division=0)),
        "feature_names": FEATURE_NAMES,
        "model_type": "ExtraTreesClassifier pixel segmentation baseline",
    }
    out = Path(model_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES, "report": report}, out)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight pixel-classifier waveform digitization model from rendered masks.")
    parser.add_argument("--manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark_manifest.json")
    parser.add_argument("--model-path", default="/data1/jiahui/biosignal-agent/outputs/waveform_digitization_pixel_model.joblib")
    parser.add_argument("--out-json", default="/data1/jiahui/biosignal-agent/outputs/waveform_digitization_pixel_model_train.json")
    parser.add_argument("--train-variant", action="append", default=None, help="Rendered variant to train on; repeat for multiple variants, or pass all.")
    parser.add_argument("--max-positive", type=int, default=3000)
    parser.add_argument("--max-negative", type=int, default=6000)
    args = parser.parse_args()
    requested = args.train_variant or ["clean"]
    train_variants = None if "all" in requested else set(requested)
    report = train(args.manifest, args.model_path, train_variants, args.max_positive, args.max_negative)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
