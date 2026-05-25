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


def load_pair(record: dict[str, Any], height: int, width: int):
    import torch

    left, right, top, bottom = int(record["crop_left"]), int(record["crop_right"]), int(record["crop_top"]), int(record["crop_bottom"])
    image = Image.open(record["image_path"]).convert("RGB")
    mask = Image.open(record["mask_path"]).convert("L")
    w, h = image.size
    image = image.crop((left, top, w - right, h - bottom)).resize((width, height), Image.BILINEAR)
    mask = mask.crop((left, top, w - right, h - bottom)).resize((width, height), Image.NEAREST)
    x = np.asarray(image, dtype=np.float32) / 255.0
    y = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1)), torch.from_numpy(y[None, :, :])


class WaveformMaskDataset:
    def __init__(self, records: list[dict[str, Any]], height: int, width: int) -> None:
        self.records = records
        self.height = height
        self.width = width

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        return load_pair(self.records[idx], self.height, self.width)


def dice_iou_from_logits(logits, target, threshold: float = 0.5) -> tuple[float, float]:
    import torch

    pred = (torch.sigmoid(logits) >= threshold).float()
    target = target.float()
    inter = torch.sum(pred * target).item()
    pred_sum = torch.sum(pred).item()
    target_sum = torch.sum(target).item()
    dice = (2 * inter) / (pred_sum + target_sum + 1e-8)
    union = pred_sum + target_sum - inter
    iou = inter / (union + 1e-8)
    return float(dice), float(iou)


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    manifest = json.loads(Path(args.manifest).read_text())
    records = [row for row in manifest.get("records", []) if row.get("mask_path")]
    if args.train_variant and "all" not in args.train_variant:
        allowed = set(args.train_variant)
        records = [row for row in records if row.get("variant") in allowed]
    if not records:
        raise ValueError("No records selected for U-Net training.")
    dataset = WaveformMaskDataset(records, args.height, args.width)
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
            dice, iou = dice_iou_from_logits(logits.detach().cpu(), y.detach().cpu())
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
        "train_variants": args.train_variant or ["all"],
        "num_records": len(records),
        "pos_weight": float(pos_weight.item()),
        "history": history,
        "model_type": "TinyWaveformUNet",
    }
    torch.save(checkpoint, out)
    report = {"model_path": str(out), **{k: v for k, v in checkpoint.items() if k != "model_state"}, "final": history[-1] if history else {}}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny U-Net waveform image digitizer on rendered image/mask pairs.")
    parser.add_argument("--manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark_manifest.json")
    parser.add_argument("--model-path", default="/data1/jiahui/biosignal-agent/outputs/waveform_digitization_unet.pt")
    parser.add_argument("--out-json", default="/data1/jiahui/biosignal-agent/outputs/waveform_digitization_unet_train.json")
    parser.add_argument("--train-variant", action="append", default=None)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    report = train(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
