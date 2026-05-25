from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew

from .common import load_csv_signal

MODEL_PATH = Path('/data1/jiahui/biosignal-agent/outputs/modality_classifier_model.joblib')


def _trapz(y, x):
    integrate = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
    return integrate(y, x)
FEATURE_NAMES = [
    'sampling_rate',
    'duration_s',
    'num_samples',
    'mean',
    'std',
    'median',
    'iqr',
    'dynamic_range',
    'abs_mean',
    'mad',
    'skew',
    'kurtosis',
    'zero_crossing_rate',
    'slope_std',
    'slope_abs_mean',
    'dominant_frequency_hz',
    'spectral_centroid_hz',
    'spectral_entropy',
    'very_low_ratio',
    'low_ratio',
    'cardiac_ratio',
    'mid_ratio',
    'high_ratio',
    'peak_rate_per_min',
    'peak_interval_cv',
    'value_min',
    'value_max',
]


def extract_modality_features(signal_path: str, sampling_rate: float, column: str | None = None) -> dict[str, float]:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = np.asarray(data.values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError('empty signal')
    centered = values - np.nanmedian(values)
    duration_s = len(values) / float(data.sampling_rate) if data.sampling_rate else 0.0
    q25, q75 = np.nanpercentile(values, [25, 75])
    p1, p5, p95, p99 = np.nanpercentile(values, [1, 5, 95, 99])
    diff = np.diff(centered)
    zcr = float(np.mean(np.diff(np.signbit(centered)) != 0)) if len(centered) > 1 else 0.0
    freqs, psd = scipy_signal.welch(centered, fs=data.sampling_rate, nperseg=min(len(centered), max(8, int(data.sampling_rate * 4))))
    total_power = float(_trapz(psd, freqs)) if len(freqs) else 0.0

    def ratio(low: float, high: float) -> float:
        if total_power <= 0 or not len(freqs):
            return 0.0
        mask = (freqs >= low) & (freqs < min(high, data.sampling_rate * 0.49))
        return float(_trapz(psd[mask], freqs[mask]) / (total_power + 1e-12)) if np.any(mask) else 0.0

    dominant_frequency = float(freqs[np.argmax(psd)]) if len(freqs) and np.any(psd) else 0.0
    spectral_centroid = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-12)) if len(freqs) else 0.0
    psd_norm = psd / (np.sum(psd) + 1e-12) if len(psd) else np.array([])
    spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)) / np.log2(len(psd_norm))) if len(psd_norm) > 1 else 0.0
    distance = max(1, int(data.sampling_rate * 0.25))
    prominence = max(float(np.nanstd(centered)) * 0.5, 1e-8)
    peaks, _ = scipy_signal.find_peaks(centered, distance=distance, prominence=prominence)
    peak_rate = float(len(peaks) / duration_s * 60.0) if duration_s > 0 else 0.0
    intervals = np.diff(peaks) / float(data.sampling_rate) if len(peaks) > 1 else np.array([])
    peak_interval_cv = float(np.nanstd(intervals) / (np.nanmean(intervals) + 1e-12)) if len(intervals) else 0.0
    features = {
        'sampling_rate': float(data.sampling_rate),
        'duration_s': float(duration_s),
        'num_samples': float(len(values)),
        'mean': float(np.nanmean(values)),
        'std': float(np.nanstd(values)),
        'median': float(np.nanmedian(values)),
        'iqr': float(q75 - q25),
        'dynamic_range': float(p95 - p5),
        'abs_mean': float(np.nanmean(np.abs(centered))),
        'mad': float(np.nanmedian(np.abs(centered))),
        'skew': float(skew(values, nan_policy='omit')) if len(values) > 2 else 0.0,
        'kurtosis': float(kurtosis(values, nan_policy='omit')) if len(values) > 3 else 0.0,
        'zero_crossing_rate': zcr,
        'slope_std': float(np.nanstd(diff)) if len(diff) else 0.0,
        'slope_abs_mean': float(np.nanmean(np.abs(diff))) if len(diff) else 0.0,
        'dominant_frequency_hz': dominant_frequency,
        'spectral_centroid_hz': spectral_centroid,
        'spectral_entropy': spectral_entropy,
        'very_low_ratio': ratio(0.0, 0.15),
        'low_ratio': ratio(0.15, 0.5),
        'cardiac_ratio': ratio(0.5, 5.0),
        'mid_ratio': ratio(5.0, 20.0),
        'high_ratio': ratio(20.0, min(100.0, data.sampling_rate * 0.49)),
        'peak_rate_per_min': peak_rate,
        'peak_interval_cv': peak_interval_cv,
        'value_min': float(p1),
        'value_max': float(p99),
    }
    return {name: float(np.nan_to_num(features[name], nan=0.0, posinf=0.0, neginf=0.0)) for name in FEATURE_NAMES}


def _heuristic_scores(features: dict[str, float]) -> dict[str, float]:
    scores = {name: 0.01 for name in ['ecg', 'ppg', 'resp', 'spo2', 'abp', 'pcg', 'acc', 'eda', 'eeg', 'emg', 'bcg', 'scg']}
    sr = features['sampling_rate']
    centroid = features['spectral_centroid_hz']
    high = features['high_ratio']
    cardiac = features['cardiac_ratio']
    low = features['low_ratio'] + features['very_low_ratio']
    dyn = features['dynamic_range']
    vmin = features['value_min']
    vmax = features['value_max']
    if 70 <= vmin <= 105 and 75 <= vmax <= 110 and dyn < 30:
        scores['spo2'] += 2.0
    if centroid < 0.8 and low > 0.4:
        scores['resp'] += 1.2
        scores['eda'] += 0.7
    if cardiac > 0.25 and centroid < 8:
        scores['ppg'] += 1.0
        scores['ecg'] += 0.6
        scores['bcg'] += 0.5
        scores['scg'] += 0.5
    if high > 0.25 or centroid > 20:
        scores['emg'] += 1.0
        scores['eeg'] += 0.7
        scores['pcg'] += 0.5
    if sr >= 500 and high > 0.05:
        scores['pcg'] += 1.0
    if vmax > 80 and dyn > 20:
        scores['abp'] += 1.5
    if sr <= 32 and dyn < 5:
        scores['acc'] += 0.7
        scores['eda'] += 0.5
    total = sum(scores.values())
    return {key: value / total for key, value in scores.items()}


def Signal_classify_modality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    try:
        features = extract_modality_features(signal_path, sampling_rate, column)
    except Exception as exc:
        return {'tool': 'Signal_classify_modality', 'error': str(exc), 'confidence': 0.0}
    model_source = 'heuristic_fallback'
    if MODEL_PATH.exists():
        try:
            bundle = joblib.load(MODEL_PATH)
            vector = np.asarray([[features[name] for name in bundle['feature_names']]], dtype=float)
            probabilities = bundle['model'].predict_proba(vector)[0]
            scores = {label: float(prob) for label, prob in zip(bundle['model'].classes_, probabilities)}
            model_source = str(MODEL_PATH)
        except Exception:
            scores = _heuristic_scores(features)
            model_source = 'heuristic_fallback_after_model_error'
    else:
        scores = _heuristic_scores(features)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return {
        'tool': 'Signal_classify_modality',
        'predicted_modality': ranked[0][0],
        'scores': scores,
        'top_modalities': [{'modality': key, 'score': value} for key, value in ranked[:5]],
        'features': features,
        'model_source': model_source,
        'confidence': float(ranked[0][1]),
        'method': 'feature_based_signal_modality_classifier',
        'disclaimer': 'Modality classification is a routing baseline; verify with metadata or domain review before clinical use.',
    }
