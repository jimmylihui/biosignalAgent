from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

CNN_MODEL_PATH = Path('/data1/jiahui/biosignal-agent/outputs/image_modality_classifier_cnn_80e.pt')


class SmallImageModalityCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Dropout(0.25), nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def image_to_cnn_tensor(image_path: str, image_size: tuple[int, int], crop_left: int = 0, crop_right: int = 0, crop_top: int = 0, crop_bottom: int = 0) -> torch.Tensor:
    image = Image.open(image_path).convert('L')
    width, height = image.size
    left = max(0, int(crop_left))
    right = width - max(0, int(crop_right))
    top = max(0, int(crop_top))
    bottom = height - max(0, int(crop_bottom))
    if right > left and bottom > top:
        image = image.crop((left, top, right, bottom))
    image = image.resize(image_size, Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = 1.0 - arr
    return torch.from_numpy(arr[None, None, :, :])


def load_cnn_bundle(model_path: str | None = None) -> dict[str, Any]:
    path = Path(model_path) if model_path else CNN_MODEL_PATH
    return torch.load(path, map_location='cpu', weights_only=False)


def Signal_classify_modality_from_image_cnn(
    image_path: str,
    crop_left: int = 0,
    crop_right: int = 0,
    crop_top: int = 0,
    crop_bottom: int = 0,
    model_path: str | None = None,
) -> dict[str, Any]:
    model_file = Path(model_path) if model_path else CNN_MODEL_PATH
    if not model_file.exists():
        return {'tool': 'Signal_classify_modality_from_image_cnn', 'error': f'model not found: {model_file}', 'confidence': 0.0}
    try:
        bundle = load_cnn_bundle(str(model_file))
        labels = list(bundle['labels'])
        image_size = tuple(bundle.get('image_size', [160, 48]))
        model = SmallImageModalityCNN(num_classes=len(labels))
        model.load_state_dict(bundle['state_dict'])
        model.eval()
        x = image_to_cnn_tensor(image_path, image_size, crop_left, crop_right, crop_top, crop_bottom)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        scores = {label: float(prob) for label, prob in zip(labels, probs)}
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return {
            'tool': 'Signal_classify_modality_from_image_cnn',
            'predicted_modality': ranked[0][0],
            'scores': scores,
            'top_modalities': [{'modality': key, 'score': value} for key, value in ranked[:5]],
            'model_source': str(model_file),
            'confidence': float(ranked[0][1]),
            'method': 'small_cnn_image_modality_classifier',
            'image_size': list(image_size),
            'disclaimer': 'CNN image-level modality classification is a routing baseline; verify with metadata when available.',
        }
    except Exception as exc:
        return {'tool': 'Signal_classify_modality_from_image_cnn', 'error': str(exc), 'confidence': 0.0, 'model_source': str(model_file)}
