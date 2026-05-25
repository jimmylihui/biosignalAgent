from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image
from scipy.fftpack import dct
from scipy.stats import kurtosis, skew

IMAGE_MODEL_PATH = Path('/data1/jiahui/biosignal-agent/outputs/image_modality_classifier_model.joblib')
BASE_IMAGE_FEATURE_NAMES = [
    'width', 'height', 'aspect_ratio',
    'gray_mean', 'gray_std', 'gray_p05', 'gray_p50', 'gray_p95', 'dark_fraction', 'mid_fraction', 'bright_fraction',
    'saturation_mean', 'saturation_std', 'red_mean', 'green_mean', 'blue_mean', 'red_green_diff', 'blue_red_diff',
    'edge_density', 'edge_mean', 'edge_std',
    'row_dark_mean', 'row_dark_std', 'row_dark_max', 'row_dark_entropy',
    'col_dark_mean', 'col_dark_std', 'col_dark_max', 'col_dark_entropy',
    'row_projection_peaks', 'col_projection_peaks',
    'trace_y_mean', 'trace_y_std', 'trace_y_range', 'trace_slope_std', 'trace_slope_abs_mean', 'trace_zero_crossing_rate',
    'trace_fft_centroid', 'trace_fft_entropy', 'trace_low_ratio', 'trace_mid_ratio', 'trace_high_ratio',
    'trace_skew', 'trace_kurtosis',
]
DCT_FEATURE_NAMES = [f'dct_{row}_{col}' for row in range(8) for col in range(8)]
PROFILE_FEATURE_NAMES = [f'row_profile_{idx}' for idx in range(16)] + [f'col_profile_{idx}' for idx in range(32)]
IMAGE_FEATURE_NAMES = BASE_IMAGE_FEATURE_NAMES + DCT_FEATURE_NAMES + PROFILE_FEATURE_NAMES


def _entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    total = float(np.sum(values))
    if total <= 0 or len(values) <= 1:
        return 0.0
    probs = values / total
    return float(-np.sum(probs * np.log2(probs + 1e-12)) / np.log2(len(probs)))


def _count_projection_peaks(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return 0.0
    threshold = float(np.mean(values) + np.std(values))
    peaks = (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:]) & (values[1:-1] > threshold)
    return float(np.sum(peaks))


def _safe_skew(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) <= 2 or float(np.nanstd(values)) < 1e-12:
        return 0.0
    return float(skew(values, nan_policy='omit'))


def _safe_kurtosis(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) <= 3 or float(np.nanstd(values)) < 1e-12:
        return 0.0
    return float(kurtosis(values, nan_policy='omit'))


def extract_image_modality_features(image_path: str, crop_left: int = 0, crop_right: int = 0, crop_top: int = 0, crop_bottom: int = 0) -> dict[str, float]:
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    left = max(0, int(crop_left))
    right = width - max(0, int(crop_right))
    top = max(0, int(crop_top))
    bottom = height - max(0, int(crop_bottom))
    if right <= left or bottom <= top:
        raise ValueError('invalid crop')
    arr = np.asarray(image.crop((left, top, right, bottom)), dtype=float) / 255.0
    h, w = arr.shape[:2]
    gray = np.mean(arr, axis=2)
    max_rgb = np.max(arr, axis=2)
    min_rgb = np.min(arr, axis=2)
    saturation = (max_rgb - min_rgb) / (max_rgb + 1e-12)
    dark = gray < 0.75
    gy, gx = np.gradient(gray)
    edge = np.hypot(gx, gy)
    row_dark = dark.mean(axis=1)
    col_dark = dark.mean(axis=0)
    small_gray = np.asarray(Image.fromarray((gray * 255).astype(np.uint8)).resize((32, 16), Image.Resampling.BILINEAR), dtype=float) / 255.0
    small_dark = small_gray < 0.75
    dct_block = dct(dct((1.0 - small_gray), axis=0, norm='ortho'), axis=1, norm='ortho')[:8, :8]
    row_profile = small_dark.mean(axis=1)
    col_profile = small_dark.mean(axis=0)

    ys = []
    for col in range(w):
        rows = np.flatnonzero(dark[:, col])
        if len(rows):
            ys.append(float(np.median(rows) / max(1, h - 1)))
    trace = np.asarray(ys, dtype=float)
    if len(trace) < 3:
        trace = np.zeros(3, dtype=float)
    trace_centered = trace - np.mean(trace)
    diff = np.diff(trace_centered)
    fft = np.abs(np.fft.rfft(trace_centered)) ** 2
    freqs = np.fft.rfftfreq(len(trace_centered), d=1.0)
    power = float(np.sum(fft))
    fft_norm = fft / (power + 1e-12)
    centroid = float(np.sum(freqs * fft) / (power + 1e-12)) if len(freqs) else 0.0
    entropy = float(-np.sum(fft_norm * np.log2(fft_norm + 1e-12)) / np.log2(len(fft_norm))) if len(fft_norm) > 1 else 0.0

    def band_ratio(lo: float, hi: float) -> float:
        if power <= 0:
            return 0.0
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.sum(fft[mask]) / (power + 1e-12)) if np.any(mask) else 0.0

    features = {
        **{f'dct_{row}_{col}': float(dct_block[row, col]) for row in range(8) for col in range(8)},
        **{f'row_profile_{idx}': float(row_profile[idx]) for idx in range(16)},
        **{f'col_profile_{idx}': float(col_profile[idx]) for idx in range(32)},
        'width': float(w), 'height': float(h), 'aspect_ratio': float(w / max(1, h)),
        'gray_mean': float(np.mean(gray)), 'gray_std': float(np.std(gray)),
        'gray_p05': float(np.percentile(gray, 5)), 'gray_p50': float(np.percentile(gray, 50)), 'gray_p95': float(np.percentile(gray, 95)),
        'dark_fraction': float(np.mean(dark)), 'mid_fraction': float(np.mean((gray >= 0.25) & (gray < 0.85))), 'bright_fraction': float(np.mean(gray >= 0.85)),
        'saturation_mean': float(np.mean(saturation)), 'saturation_std': float(np.std(saturation)),
        'red_mean': float(np.mean(arr[:, :, 0])), 'green_mean': float(np.mean(arr[:, :, 1])), 'blue_mean': float(np.mean(arr[:, :, 2])),
        'red_green_diff': float(np.mean(arr[:, :, 0] - arr[:, :, 1])), 'blue_red_diff': float(np.mean(arr[:, :, 2] - arr[:, :, 0])),
        'edge_density': float(np.mean(edge > np.percentile(edge, 95))), 'edge_mean': float(np.mean(edge)), 'edge_std': float(np.std(edge)),
        'row_dark_mean': float(np.mean(row_dark)), 'row_dark_std': float(np.std(row_dark)), 'row_dark_max': float(np.max(row_dark)), 'row_dark_entropy': _entropy(row_dark),
        'col_dark_mean': float(np.mean(col_dark)), 'col_dark_std': float(np.std(col_dark)), 'col_dark_max': float(np.max(col_dark)), 'col_dark_entropy': _entropy(col_dark),
        'row_projection_peaks': _count_projection_peaks(row_dark), 'col_projection_peaks': _count_projection_peaks(col_dark),
        'trace_y_mean': float(np.mean(trace)), 'trace_y_std': float(np.std(trace)), 'trace_y_range': float(np.percentile(trace, 95) - np.percentile(trace, 5)),
        'trace_slope_std': float(np.std(diff)) if len(diff) else 0.0, 'trace_slope_abs_mean': float(np.mean(np.abs(diff))) if len(diff) else 0.0,
        'trace_zero_crossing_rate': float(np.mean(np.diff(np.signbit(trace_centered)) != 0)) if len(trace_centered) > 1 else 0.0,
        'trace_fft_centroid': centroid, 'trace_fft_entropy': entropy,
        'trace_low_ratio': band_ratio(0.0, 0.03), 'trace_mid_ratio': band_ratio(0.03, 0.12), 'trace_high_ratio': band_ratio(0.12, 0.5),
        'trace_skew': _safe_skew(trace),
        'trace_kurtosis': _safe_kurtosis(trace),
    }
    return {name: float(np.nan_to_num(features[name], nan=0.0, posinf=0.0, neginf=0.0)) for name in IMAGE_FEATURE_NAMES}


def Signal_classify_modality_from_image(image_path: str, crop_left: int = 0, crop_right: int = 0, crop_top: int = 0, crop_bottom: int = 0, model_path: str | None = None) -> dict[str, Any]:
    try:
        features = extract_image_modality_features(image_path, crop_left, crop_right, crop_top, crop_bottom)
    except Exception as exc:
        return {'tool': 'Signal_classify_modality_from_image', 'error': str(exc), 'confidence': 0.0}
    model_file = Path(model_path) if model_path else IMAGE_MODEL_PATH
    if not model_file.exists():
        return {'tool': 'Signal_classify_modality_from_image', 'error': f'model not found: {model_file}', 'confidence': 0.0, 'features': features}
    try:
        bundle = joblib.load(model_file)
        vector = np.asarray([[features[name] for name in bundle['feature_names']]], dtype=float)
        probabilities = bundle['model'].predict_proba(vector)[0]
        scores = {label: float(prob) for label, prob in zip(bundle['model'].classes_, probabilities)}
    except Exception as exc:
        return {'tool': 'Signal_classify_modality_from_image', 'error': str(exc), 'confidence': 0.0, 'features': features, 'model_source': str(model_file)}
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return {
        'tool': 'Signal_classify_modality_from_image',
        'predicted_modality': ranked[0][0],
        'scores': scores,
        'top_modalities': [{'modality': key, 'score': value} for key, value in ranked[:5]],
        'features': features,
        'model_source': str(model_file),
        'confidence': float(ranked[0][1]),
        'method': 'feature_based_image_modality_classifier',
        'disclaimer': 'Image-level modality classification is a routing baseline; verify with metadata when available.',
    }
