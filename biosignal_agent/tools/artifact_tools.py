from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from .common import load_csv_signal


def Signal_detect_artifacts(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    if len(values) == 0:
        return {"tool": "Signal_detect_artifacts", "error": "empty signal", "confidence": 0.0}
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {"tool": "Signal_detect_artifacts", "error": "no finite samples", "confidence": 0.0}
    dynamic_range = float(np.nanpercentile(finite, 95) - np.nanpercentile(finite, 5))
    max_abs = float(np.nanmax(np.abs(finite)))
    clipped_fraction = float((np.abs(finite) >= max_abs * 0.999).mean()) if max_abs > 0 else 1.0
    diffs = np.diff(finite)
    jump_threshold = max(float(np.nanstd(finite)) * 5.0, dynamic_range * 0.5, 1e-8)
    jump_fraction = float((np.abs(diffs) > jump_threshold).mean()) if len(diffs) else 0.0
    if len(finite) >= max(16, int(data.sampling_rate * 2)):
        freqs, psd = scipy_signal.welch(finite - np.nanmedian(finite), fs=data.sampling_rate, nperseg=min(len(finite), int(data.sampling_rate * 4)))
        high_mask = freqs >= min(20.0, data.sampling_rate * 0.25)
        total_power = float(np.trapz(psd, freqs)) if len(freqs) else 0.0
        high_power = float(np.trapz(psd[high_mask], freqs[high_mask])) if np.any(high_mask) else 0.0
        high_frequency_noise_ratio = float(high_power / (total_power + 1e-12)) if total_power > 0 else 0.0
    else:
        high_frequency_noise_ratio = 0.0
    flags = []
    if dynamic_range <= 1e-8:
        flags.append("flatline_or_near_flat")
    if clipped_fraction > 0.05:
        flags.append("clipping_or_saturation")
    if jump_fraction > 0.01:
        flags.append("abrupt_jumps")
    if high_frequency_noise_ratio > 0.45:
        flags.append("high_frequency_noise")
    artifact_level = "high" if len(flags) >= 2 or "flatline_or_near_flat" in flags else "moderate" if flags else "low"
    return {
        "tool": "Signal_detect_artifacts",
        "artifact_level": artifact_level,
        "artifact_flags": flags,
        "dynamic_range": dynamic_range,
        "clipped_fraction": clipped_fraction,
        "jump_fraction": jump_fraction,
        "high_frequency_noise_ratio": high_frequency_noise_ratio,
        "confidence": 0.65,
        "method": "generic_range_jump_noise_artifact_screening",
        "disclaimer": "Generic artifact screen only; modality-specific artifact detection should be validated separately.",
    }
