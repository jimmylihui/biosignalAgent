from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from biosignal_agent.tools.digitize_tools import _crop_rgb_image, _signal_from_mask

UNET_MODEL_PATH = Path("/data1/jiahui/biosignal-agent/outputs/waveform_digitization_tiny_unet_multistyle.pt")


class TinyWaveformUNet:  # factory wrapper avoids importing torch until needed
    @staticmethod
    def build():
        import torch
        from torch import nn

        class ConvBlock(nn.Module):
            def __init__(self, in_channels: int, out_channels: int) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x):
                return self.net(x)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.enc1 = ConvBlock(3, 16)
                self.pool1 = nn.MaxPool2d(2)
                self.enc2 = ConvBlock(16, 32)
                self.pool2 = nn.MaxPool2d(2)
                self.bottleneck = ConvBlock(32, 64)
                self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
                self.dec2 = ConvBlock(64, 32)
                self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
                self.dec1 = ConvBlock(32, 16)
                self.out = nn.Conv2d(16, 1, 1)

            def forward(self, x):
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool1(e1))
                b = self.bottleneck(self.pool2(e2))
                d2 = self.up2(b)
                d2 = self.dec2(torch.cat([d2, e2], dim=1))
                d1 = self.up1(d2)
                d1 = self.dec1(torch.cat([d1, e1], dim=1))
                return self.out(d1)

        return Model()


class TinyWaveformDeepLabV3:
    @staticmethod
    def build():
        from torch import nn
        import torch.nn.functional as F

        class ASPP(nn.Module):
            def __init__(self, channels: int) -> None:
                super().__init__()
                self.branches = nn.ModuleList([
                    nn.Conv2d(channels, 32, 1),
                    nn.Conv2d(channels, 32, 3, padding=2, dilation=2),
                    nn.Conv2d(channels, 32, 3, padding=4, dilation=4),
                    nn.Conv2d(channels, 32, 3, padding=8, dilation=8),
                ])
                self.project = nn.Sequential(nn.Conv2d(128, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

            def forward(self, x):
                return self.project(__import__('torch').cat([branch(x) for branch in self.branches], dim=1))

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv2d(3, 24, 3, stride=2, padding=1), nn.BatchNorm2d(24), nn.ReLU(inplace=True),
                    nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
                    nn.Conv2d(48, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                )
                self.aspp = ASPP(64)
                self.head = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1))

            def forward(self, x):
                size = x.shape[-2:]
                y = self.head(self.aspp(self.stem(x)))
                return F.interpolate(y, size=size, mode="bilinear", align_corners=False)

        return Model()


class TinyWaveformSegFormer:
    @staticmethod
    def build():
        from torch import nn
        import torch.nn.functional as F

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.patch1 = nn.Sequential(nn.Conv2d(3, 24, 7, stride=4, padding=3), nn.BatchNorm2d(24), nn.GELU())
                self.patch2 = nn.Sequential(nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.BatchNorm2d(48), nn.GELU())
                encoder_layer = nn.TransformerEncoderLayer(d_model=48, nhead=4, dim_feedforward=128, batch_first=True, activation="gelu")
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
                self.decode = nn.Sequential(
                    nn.Conv2d(48, 48, 3, padding=1), nn.BatchNorm2d(48), nn.GELU(),
                    nn.Conv2d(48, 24, 3, padding=1), nn.GELU(),
                    nn.Conv2d(24, 1, 1),
                )

            def forward(self, x):
                size = x.shape[-2:]
                y = self.patch2(self.patch1(x))
                b, c, h, w = y.shape
                tokens = y.flatten(2).transpose(1, 2)
                tokens = self.transformer(tokens)
                y = tokens.transpose(1, 2).reshape(b, c, h, w)
                y = self.decode(y)
                return F.interpolate(y, size=size, mode="bilinear", align_corners=False)

        return Model()


def build_waveform_segmentation_model(model_type: str | None = None):
    name = (model_type or "tiny_unet").lower()
    if name in {"tiny_unet", "unet", "tinywaveformunet"}:
        return TinyWaveformUNet.build()
    if name in {"tiny_deeplabv3", "deeplabv3", "deeplab"}:
        return TinyWaveformDeepLabV3.build()
    if name in {"tiny_segformer", "segformer", "segformer_lite"}:
        return TinyWaveformSegFormer.build()
    raise ValueError(f"unknown waveform segmentation model_type: {model_type}")

def Signal_digitize_waveform_image_unet(
    image_path: str,
    sampling_rate: float | None = None,
    out_csv: str | None = None,
    value_min: float | None = None,
    value_max: float | None = None,
    crop_left: int = 0,
    crop_right: int = 0,
    crop_top: int = 0,
    crop_bottom: int = 0,
    model_path: str | None = None,
    probability_threshold: float = 0.5,
    smooth_window: int = 1,
    trace_method: str = "median",
) -> dict[str, Any]:
    model_file = Path(model_path) if model_path else UNET_MODEL_PATH
    if not model_file.exists():
        return {"tool": "Signal_digitize_waveform_image_unet", "error": f"model not found: {model_file}", "confidence": 0.0}
    try:
        import torch

        checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)
        input_height, input_width = checkpoint.get("input_size", [128, 384])
        rgb, crop, _ = _crop_rgb_image(image_path, crop_left, crop_right, crop_top, crop_bottom)
        crop_height, crop_width = rgb.shape[:2]
        resized = Image.fromarray(rgb).resize((int(input_width), int(input_height)), Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
        model = build_waveform_segmentation_model(checkpoint.get("model_type") or checkpoint.get("backbone"))
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        with torch.no_grad():
            logits = model(tensor)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        mask_small = prob >= float(probability_threshold)
        mask = Image.fromarray((mask_small.astype(np.uint8) * 255), mode="L").resize((crop_width, crop_height), Image.NEAREST)
        mask_arr = np.asarray(mask, dtype=np.uint8) > 0
        mean_probability = float(np.mean(prob[mask_small])) if np.any(mask_small) else 0.0
    except Exception as exc:
        return {"tool": "Signal_digitize_waveform_image_unet", "error": str(exc), "confidence": 0.0, "model_source": str(model_file)}
    result = _signal_from_mask(
        mask_arr,
        sampling_rate,
        out_csv,
        image_path,
        value_min,
        value_max,
        smooth_window,
        "Signal_digitize_waveform_image_unet",
        f"{checkpoint.get('model_type', 'tiny_unet')}_waveform_segmentation_path_digitizer" if trace_method == "path" else f"{checkpoint.get('model_type', 'tiny_unet')}_waveform_segmentation_digitizer",
        model_source=str(model_file),
        confidence_scale=max(0.3, mean_probability),
        trace_method=trace_method,
    )
    result["crop"] = {"left": crop[0], "right": crop[1], "top": crop[2], "bottom": crop[3]}
    result["probability_threshold"] = float(probability_threshold)
    result["mask_pixel_fraction"] = float(np.mean(mask_arr)) if mask_arr.size else 0.0
    result["input_size"] = [int(input_height), int(input_width)]
    return result
