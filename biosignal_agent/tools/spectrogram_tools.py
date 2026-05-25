from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal as scipy_signal

from .common import load_csv_signal


def _safe_entropy(power: np.ndarray) -> float:
    total = float(np.sum(power) + 1e-12)
    p = np.asarray(power, dtype=float) / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)) / np.log2(len(power))) if len(power) > 1 else 0.0


def Signal_extract_spectrogram_features(
    signal_path: str,
    sampling_rate: float,
    column: str | None = None,
    modality: str | None = None,
    window_seconds: float = 2.0,
    overlap: float = 0.5,
    max_frequency_hz: float | None = None,
) -> dict[str, Any]:
    """Extract compact spectrogram features for dense biomedical signals.

    Intended for PCG/EMG-style paths where image-to-waveform recovery is often
    less useful than spectro-temporal task features.
    """
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = np.asarray(data.values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < max(8, int(0.5 * data.sampling_rate)):
        return {"tool": "Signal_extract_spectrogram_features", "error": "signal too short", "confidence": 0.0}
    values = values - np.nanmedian(values)
    nperseg = max(16, min(len(values), int(float(window_seconds) * data.sampling_rate)))
    noverlap = int(max(0.0, min(0.95, float(overlap))) * nperseg)
    freqs, times, spec = scipy_signal.spectrogram(
        values,
        fs=data.sampling_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        mode="magnitude",
    )
    if max_frequency_hz is None:
        mod = str(modality or "").lower()
        max_frequency_hz = 500.0 if mod == "pcg" else 450.0 if mod == "emg" else data.sampling_rate * 0.45
    freq_mask = freqs <= min(float(max_frequency_hz), data.sampling_rate * 0.5)
    freqs = freqs[freq_mask]
    spec = spec[freq_mask]
    power = spec ** 2
    total_power = np.sum(power, axis=0) + 1e-12
    mean_power_by_freq = np.mean(power, axis=1) if power.size else np.array([])
    centroid_by_time = np.sum(freqs[:, None] * power, axis=0) / total_power if len(freqs) else np.array([])
    rolloff = []
    for col in power.T:
        cdf = np.cumsum(col)
        rolloff.append(float(freqs[np.searchsorted(cdf, 0.85 * cdf[-1])]) if len(freqs) and cdf[-1] > 0 else 0.0)
    rolloff = np.asarray(rolloff, dtype=float)

    def band_ratio(low: float, high: float) -> float:
        mask = (freqs >= low) & (freqs < high)
        return float(np.sum(power[mask]) / (np.sum(power) + 1e-12)) if np.any(mask) else 0.0

    mod = str(modality or "").lower()
    if mod == "pcg":
        bands = {"band_20_60_ratio": (20, 60), "band_60_150_ratio": (60, 150), "band_150_400_ratio": (150, 400)}
    elif mod == "emg":
        bands = {"band_20_60_ratio": (20, 60), "band_60_150_ratio": (60, 150), "band_150_300_ratio": (150, 300), "band_300_450_ratio": (300, 450)}
    else:
        bands = {"band_low_ratio": (0, 10), "band_mid_ratio": (10, 40), "band_high_ratio": (40, float(max_frequency_hz))}
    band_features = {name: band_ratio(low, high) for name, (low, high) in bands.items()}
    temporal_energy = np.sum(power, axis=0)
    features = {
        "tool": "Signal_extract_spectrogram_features",
        "source": data.source,
        "modality": mod or None,
        "num_samples": int(len(values)),
        "sampling_rate": float(data.sampling_rate),
        "num_frequency_bins": int(len(freqs)),
        "num_time_bins": int(len(times)),
        "spectrogram_mean_power": float(np.mean(power)) if power.size else 0.0,
        "spectrogram_log_power_mean": float(np.mean(np.log1p(power))) if power.size else 0.0,
        "spectrogram_log_power_std": float(np.std(np.log1p(power))) if power.size else 0.0,
        "spectral_centroid_mean_hz": float(np.mean(centroid_by_time)) if len(centroid_by_time) else None,
        "spectral_centroid_std_hz": float(np.std(centroid_by_time)) if len(centroid_by_time) else None,
        "spectral_rolloff85_mean_hz": float(np.mean(rolloff)) if len(rolloff) else None,
        "spectral_rolloff85_std_hz": float(np.std(rolloff)) if len(rolloff) else None,
        "spectral_entropy": _safe_entropy(mean_power_by_freq) if len(mean_power_by_freq) else 0.0,
        "temporal_energy_cv": float(np.std(temporal_energy) / (np.mean(temporal_energy) + 1e-12)) if len(temporal_energy) else 0.0,
        "temporal_energy_p95_p50_ratio": float(np.percentile(temporal_energy, 95) / (np.percentile(temporal_energy, 50) + 1e-12)) if len(temporal_energy) else 0.0,
        "confidence": 0.65,
        "method": "stft_spectrogram_summary_features",
        **band_features,
    }
    return features



def Signal_render_spectrogram_image(
    signal_path: str,
    sampling_rate: float,
    out_png: str | None = None,
    column: str | None = None,
    modality: str | None = None,
    window_seconds: float = 1.0,
    overlap: float = 0.5,
    max_frequency_hz: float | None = None,
    width: int = 256,
    height: int = 256,
) -> dict[str, Any]:
    """Render a log-power spectrogram image for PCG/EMG-style task models."""
    try:
        from PIL import Image
        data = load_csv_signal(signal_path, sampling_rate, column)
        values = np.asarray(data.values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < max(8, int(0.5 * data.sampling_rate)):
            return {"tool": "Signal_render_spectrogram_image", "error": "signal too short", "confidence": 0.0}
        values = values - np.nanmedian(values)
        nperseg = max(16, min(len(values), int(float(window_seconds) * data.sampling_rate)))
        noverlap = int(max(0.0, min(0.95, float(overlap))) * nperseg)
        freqs, times, spec = scipy_signal.spectrogram(
            values,
            fs=data.sampling_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="density",
            mode="magnitude",
        )
        if max_frequency_hz is None:
            mod = str(modality or "").lower()
            max_frequency_hz = 500.0 if mod == "pcg" else 450.0 if mod == "emg" else data.sampling_rate * 0.45
        mask = freqs <= min(float(max_frequency_hz), data.sampling_rate * 0.5)
        spec = spec[mask]
        log_spec = np.log1p(spec ** 2)
        if log_spec.size == 0:
            return {"tool": "Signal_render_spectrogram_image", "error": "empty spectrogram", "confidence": 0.0}
        lo, hi = np.percentile(log_spec, [1, 99])
        norm = np.clip((log_spec - lo) / max(hi - lo, 1e-12), 0, 1)
        # Flip so low frequencies are at the bottom, as in conventional spectrograms.
        img = Image.fromarray((np.flipud(norm) * 255).astype(np.uint8), mode="L").resize((int(width), int(height)), Image.BILINEAR)
        out_path = Path(out_png) if out_png else Path("/data1/jiahui/biosignal-agent/outputs/spectrogram_images") / f"{Path(signal_path).stem}_spectrogram.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        return {
            "tool": "Signal_render_spectrogram_image",
            "image_path": str(out_path),
            "source": data.source,
            "modality": str(modality).lower() if modality else None,
            "num_samples": int(len(values)),
            "sampling_rate": float(data.sampling_rate),
            "window_seconds": float(window_seconds),
            "overlap": float(overlap),
            "max_frequency_hz": float(max_frequency_hz),
            "image_width": int(width),
            "image_height": int(height),
            "confidence": 0.75,
            "method": "stft_log_power_spectrogram_render",
        }
    except Exception as exc:
        return {"tool": "Signal_render_spectrogram_image", "error": str(exc), "confidence": 0.0}
