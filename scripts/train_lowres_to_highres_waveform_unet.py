from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.digitize_unet_tools import TinyWaveformUNet
from scripts.evaluate_lowres_recovery_digitization import RESAMPLE_METHODS, load_records
from scripts.train_waveform_digitization_unet import dice_iou_from_logits


def load_lowres_to_highres_pair(low_record: dict[str, Any], high_record: dict[str, Any], height: int, width: int, upscale_method: str):
    import torch

    low_image = Image.open(low_record["image_path"]).convert("RGB")
    lw, lh = low_image.size
    ll = int(low_record.get("crop_left") or 0)
    lr = int(low_record.get("crop_right") or 0)
    lt = int(low_record.get("crop_top") or 0)
    lb = int(low_record.get("crop_bottom") or 0)
    low_crop = low_image.crop((ll, lt, lw - lr, lh - lb))
    scale_x = int(round((int(high_record.get("num_points") or 0) + 2 * int(high_record.get("crop_left") or 0)) / max(1, lw)))
    scale = max(1, scale_x)
    low_up = low_crop.resize((low_crop.size[0] * scale, low_crop.size[1] * scale), resample=RESAMPLE_METHODS[upscale_method])
    low_up = low_up.resize((width, height), Image.BILINEAR)

    high_mask = Image.open(high_record["mask_path"]).convert("L")
    hw, hh = high_mask.size
    hl = int(high_record.get("crop_left") or 0)
    hr = int(high_record.get("crop_right") or 0)
    ht = int(high_record.get("crop_top") or 0)
    hb = int(high_record.get("crop_bottom") or 0)
    mask_crop = high_mask.crop((hl, ht, hw - hr, hh - hb)).resize((width, height), Image.NEAREST)

    x = np.asarray(low_up, dtype=np.float32) / 255.0
    y = (np.asarray(mask_crop, dtype=np.float32) > 0).astype(np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1)), torch.from_numpy(y[None, :, :])


class LowresToHighresMaskDataset:
    def __init__(self, pairs: list[tuple[dict[str, Any], dict[str, Any]]], height: int, width: int, upscale_method: str) -> None:
        self.pairs = pairs
        self.height = height
        self.width = width
        self.upscale_method = upscale_method

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        low, high = self.pairs[idx]
        return load_lowres_to_highres_pair(low, high, self.height, self.width, self.upscale_method)


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    low_records = load_records(args.lowres_manifest)
    high_records = load_records(args.highres_manifest)
    wanted = {item.lower() for item in args.include_modality} if args.include_modality else None
    pairs = []
    for record_id, low in low_records.items():
        if record_id not in high_records:
            continue
        if wanted and str(low.get("modality", "")).lower() not in wanted:
            continue
        if not high_records[record_id].get("mask_path"):
            continue
        pairs.append((low, high_records[record_id]))
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise ValueError("No paired low/high records selected.")

    dataset = LowresToHighresMaskDataset(pairs, args.height, args.width, args.upscale_method)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = TinyWaveformUNet.build().to(device)

    pos_pixels = 0.0
    total_pixels = 0.0
    for _, y in loader:
        pos_pixels += float(y.sum().item())
        total_pixels += float(y.numel())
    neg_pixels = max(1.0, total_pixels - pos_pixels)
    pos_weight = torch.tensor([min(25.0, neg_pixels / max(1.0, pos_pixels))], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        dices = []
        ious = []
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            dice, iou = dice_iou_from_logits(logits.detach().cpu(), y.detach().cpu(), threshold=args.report_threshold)
            dices.append(dice)
            ious.append(iou)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "dice": float(np.mean(dices)), "iou": float(np.mean(ious))}
        history.append(row)
        print(json.dumps(row))

    out = Path(args.model_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.cpu().state_dict(),
        "input_size": [args.height, args.width],
        "num_records": len(pairs),
        "pos_weight": float(pos_weight.item()),
        "history": history,
        "model_type": "TinyWaveformUNet",
        "training_task": "lowres_upscaled_image_to_highres_mask",
        "lowres_manifest": args.lowres_manifest,
        "highres_manifest": args.highres_manifest,
        "upscale_method": args.upscale_method,
    }
    torch.save(checkpoint, out)
    report = {"model_path": str(out), **{k: v for k, v in checkpoint.items() if k != "model_state"}, "final": history[-1] if history else {}}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny U-Net from low-resolution waveform images to high-resolution masks.")
    parser.add_argument("--lowres-manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark_manifest.json")
    parser.add_argument("--highres-manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark_highres_aligned_manifest.json")
    parser.add_argument("--model-path", default="/data1/jiahui/biosignal-agent/outputs/lowres_to_highres_waveform_unet.pt")
    parser.add_argument("--out-json", default="/data1/jiahui/biosignal-agent/outputs/lowres_to_highres_waveform_unet_train.json")
    parser.add_argument("--upscale-method", choices=sorted(RESAMPLE_METHODS), default="lanczos")
    parser.add_argument("--include-modality", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--report-threshold", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
