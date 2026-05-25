#!/usr/bin/env python3
"""Evaluate waveform image segmentation masks against mask_path labels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.digitize_tools import _crop_rgb_image
from biosignal_agent.tools.digitize_unet_tools import UNET_MODEL_PATH, build_waveform_segmentation_model


def predict_unet_mask(image_path: str, model_path: str, threshold: float, crop: dict[str, int]) -> np.ndarray:
    import torch

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    input_height, input_width = checkpoint.get('input_size', [128, 384])
    rgb, _, _ = _crop_rgb_image(
        image_path,
        crop.get('left', 0),
        crop.get('right', 0),
        crop.get('top', 0),
        crop.get('bottom', 0),
    )
    height, width = rgb.shape[:2]
    resized = Image.fromarray(rgb).resize((int(input_width), int(input_height)), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    num_classes = int(checkpoint.get('num_classes', 1))
    model = build_waveform_segmentation_model(checkpoint.get('model_type') or checkpoint.get('backbone'), out_channels=num_classes)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        if num_classes > 1:
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_small = np.argmax(probs, axis=0) == 1
        else:
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred_small = prob >= float(threshold)
    pred = Image.fromarray((pred_small.astype(np.uint8) * 255), mode='L').resize((width, height), Image.NEAREST)
    return np.asarray(pred, dtype=np.uint8) > 0


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    if pred.shape != truth.shape:
        truth = np.asarray(Image.fromarray((truth.astype(np.uint8) * 255), mode='L').resize((pred.shape[1], pred.shape[0]), Image.NEAREST), dtype=np.uint8) > 0
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    tn = int(np.logical_and(~pred, ~truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return {
        'true_positive_pixels': tp,
        'false_positive_pixels': fp,
        'false_negative_pixels': fn,
        'true_negative_pixels': tn,
        'precision': precision,
        'recall': recall,
        'dice': dice,
        'iou': iou,
        'pred_mask_fraction': float(pred.mean()) if pred.size else 0.0,
        'truth_mask_fraction': float(truth.mean()) if truth.size else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/datasets/processed/ecg_image_kit_generated_manifest.json')
    ap.add_argument('--method', choices=['unet'], default='unet')
    ap.add_argument('--model-path', default=str(UNET_MODEL_PATH))
    ap.add_argument('--probability-threshold', type=float, default=0.65)
    ap.add_argument('--out-json', default='/data1/jiahui/biosignal-agent/outputs/waveform_segmentation_mask_eval.json')
    ap.add_argument('--out-csv', default='/data1/jiahui/biosignal-agent/outputs/waveform_segmentation_mask_eval.csv')
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    rows = []
    for rec in manifest.get('records', []):
        row = {'record': rec.get('record') or rec.get('id'), 'modality': rec.get('modality'), 'method': args.method}
        mask_path = rec.get('mask_path')
        if not mask_path:
            row['error'] = 'missing mask_path'
            rows.append(row)
            continue
        try:
            crop = {
                'left': int(rec.get('crop_left', 0)),
                'right': int(rec.get('crop_right', 0)),
                'top': int(rec.get('crop_top', 0)),
                'bottom': int(rec.get('crop_bottom', 0)),
            }
            pred = predict_unet_mask(rec['image_path'], args.model_path, args.probability_threshold, crop)
            truth_img = Image.open(mask_path).convert('L')
            w, h = truth_img.size
            left = crop.get('left', 0)
            right = w - crop.get('right', 0)
            top = crop.get('top', 0)
            bottom = h - crop.get('bottom', 0)
            truth_raw = np.asarray(truth_img.crop((left, top, right, bottom)), dtype=np.uint8)
            truth = truth_raw == 1
            row.update(metrics(pred, truth))
        except Exception as exc:
            row['error'] = str(exc)
        rows.append(row)

    ok = [r for r in rows if not r.get('error')]
    def mean(key: str) -> float | None:
        vals = [float(r[key]) for r in ok if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None
    report = {
        'manifest': args.manifest,
        'method': args.method,
        'model_path': args.model_path,
        'probability_threshold': args.probability_threshold,
        'metrics': {
            'num_records': len(rows),
            'num_ok': len(ok),
            'mean_precision': mean('precision'),
            'mean_recall': mean('recall'),
            'mean_dice': mean('dice'),
            'mean_iou': mean('iou'),
            'mean_pred_mask_fraction': mean('pred_mask_fraction'),
            'mean_truth_mask_fraction': mean('truth_mask_fraction'),
        },
        'rows': rows,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    with Path(args.out_csv).open('w', newline='') as f:
        keys = sorted({k for row in rows for k in row})
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'metrics': report['metrics'], 'out_json': args.out_json}, indent=2))


if __name__ == '__main__':
    main()
