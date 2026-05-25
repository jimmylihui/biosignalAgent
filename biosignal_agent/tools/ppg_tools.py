from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from .common import bpm_from_peaks, interval_regularity, load_csv_signal, signal_quality_summary
from .peak_detectors import neurokit_nabian2018_peaks, ppg_multiscale_systolic_peaks

PPG_IRREGULARITY_MODEL_PATH = Path("/data1/jiahui/biosignal-agent/outputs/ppg_pulse_irregularity_feature_classifier.joblib")
PPG_QUALITY_MODEL_PATH = Path("/data1/jiahui/biosignal-agent/outputs/ppg_signal_quality_capnobase_classifier.joblib")
PPG_AF_INTERVAL_DL_MODEL_PATH = Path("/data1/jiahui/biosignal-agent/outputs/ppg_af_interval_attention_bilstm.pt")
PPG_QUALITY_DL_MODEL_PATH = Path("/data1/jiahui/biosignal-agent/outputs/ppg_quality_seresnet_moredata.pt")
PPG_RESPIRATION_SELECTOR_MODEL_PATH = Path("/data1/jiahui/biosignal-agent/outputs/ppg_respiration_candidate_selector_multidb.joblib")
PPG_PEAK_DL_MODEL_PATH = Path("/data1/jiahui/biosignal-agent/outputs/ppg_peak_unet_capnobase.pt")
_PPG_AF_INTERVAL_DL_CACHE = None
_PPG_IRREGULARITY_MODEL_CACHE = None
_PPG_QUALITY_MODEL_CACHE = None
_PPG_QUALITY_DL_MODEL_CACHE = None
_PPG_RESPIRATION_SELECTOR_MODEL_CACHE = None
_PPG_PEAK_DL_MODEL_CACHE = None


def _safe_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _ppg_artifact_metrics(values: np.ndarray, sampling_rate: float) -> dict:
    if len(values) == 0:
        return {
            "flatline_fraction": 1.0,
            "saturation_fraction": 1.0,
            "baseline_wander_ratio": None,
            "high_frequency_noise_ratio": None,
            "artifact_score": 1.0,
        }
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {
            "flatline_fraction": 1.0,
            "saturation_fraction": 1.0,
            "baseline_wander_ratio": None,
            "high_frequency_noise_ratio": None,
            "artifact_score": 1.0,
        }
    dynamic_range = float(np.nanpercentile(finite, 95) - np.nanpercentile(finite, 5))
    diff = np.diff(finite)
    flatline_fraction = float(np.mean(np.abs(diff) <= max(dynamic_range * 1e-4, 1e-8))) if len(diff) else 1.0
    lo, hi = np.nanpercentile(finite, [1, 99])
    saturation_fraction = float(np.mean((finite <= lo + dynamic_range * 1e-3) | (finite >= hi - dynamic_range * 1e-3))) if dynamic_range > 0 else 1.0
    baseline_wander_ratio = None
    high_frequency_noise_ratio = None
    if len(finite) >= max(16, int(sampling_rate * 4)) and sampling_rate > 1:
        clean = finite - np.nanmedian(finite)
        freqs, psd = scipy_signal.welch(clean, fs=sampling_rate, nperseg=min(len(clean), int(sampling_rate * 8)))
        total_power = float(np.trapezoid(psd, freqs)) + 1e-12
        baseline_wander_ratio = float(np.trapezoid(psd[freqs < 0.3], freqs[freqs < 0.3]) / total_power) if np.any(freqs < 0.3) else 0.0
        high_mask = freqs > min(8.0, 0.35 * sampling_rate)
        high_frequency_noise_ratio = float(np.trapezoid(psd[high_mask], freqs[high_mask]) / total_power) if np.any(high_mask) else 0.0
    artifact_score = 0.0
    artifact_score += min(0.35, flatline_fraction * 2.0)
    artifact_score += min(0.25, max(0.0, saturation_fraction - 0.08) * 1.5)
    if baseline_wander_ratio is not None:
        artifact_score += min(0.25, max(0.0, baseline_wander_ratio - 0.45) * 0.8)
    if high_frequency_noise_ratio is not None:
        artifact_score += min(0.20, max(0.0, high_frequency_noise_ratio - 0.20) * 0.8)
    return {
        "flatline_fraction": flatline_fraction,
        "saturation_fraction": saturation_fraction,
        "baseline_wander_ratio": baseline_wander_ratio,
        "high_frequency_noise_ratio": high_frequency_noise_ratio,
        "artifact_score": float(min(1.0, artifact_score)),
    }




def _ppg_distribution_metrics(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {
            "skewness": 0.0,
            "kurtosis": 0.0,
            "zero_crossing_rate": 0.0,
            "normalized_dynamic_range": 0.0,
        }
    centered = finite - np.nanmedian(finite)
    scale = np.nanstd(centered) + 1e-8
    z = centered / scale
    return {
        "skewness": float(np.nanmean(z ** 3)),
        "kurtosis": float(np.nanmean(z ** 4) - 3.0),
        "zero_crossing_rate": float(np.mean(np.diff(np.signbit(centered)) != 0)) if len(centered) > 1 else 0.0,
        "normalized_dynamic_range": float((np.nanpercentile(finite, 95) - np.nanpercentile(finite, 5)) / (np.nanmedian(np.abs(finite)) + 1e-8)),
    }


def _ppg_peak_consistency_features(values: np.ndarray, sampling_rate: float) -> dict:
    try:
        peaks, _ = ppg_multiscale_systolic_peaks(values, sampling_rate)
    except Exception:
        peaks = np.asarray([], dtype=int)
    interval_features = _pulse_interval_features(peaks, sampling_rate)
    duration_min = len(values) / float(sampling_rate) / 60.0 if sampling_rate > 0 else 0.0
    peak_rate_per_min = float(len(peaks) / duration_min) if duration_min > 0 else 0.0
    return {
        "num_peaks": int(len(peaks)),
        "peak_rate_per_min": peak_rate_per_min,
        **interval_features,
    }


def _ppg_quality_feature_vector(values: np.ndarray, sampling_rate: float) -> list[float]:
    artifact = _ppg_artifact_metrics(values, sampling_rate)
    distribution = _ppg_distribution_metrics(values)
    peak_features = _ppg_peak_consistency_features(values, sampling_rate)
    keys = [
        "flatline_fraction",
        "saturation_fraction",
        "baseline_wander_ratio",
        "high_frequency_noise_ratio",
        "artifact_score",
        "skewness",
        "kurtosis",
        "zero_crossing_rate",
        "normalized_dynamic_range",
        "num_peaks",
        "peak_rate_per_min",
        "pulse_interval_cv",
        "robust_pulse_interval_cv",
        "normalized_rmssd",
        "successive_change_fraction",
        "turning_point_ratio",
        "short_interval_fraction",
        "long_interval_fraction",
    ]
    merged = {**artifact, **distribution, **peak_features}
    return [_safe_float(merged.get(key)) or 0.0 for key in keys]


def _load_ppg_quality_model() -> dict | object | None:
    global _PPG_QUALITY_MODEL_CACHE
    if joblib is None or not PPG_QUALITY_MODEL_PATH.exists():
        return None
    if _PPG_QUALITY_MODEL_CACHE is None:
        _PPG_QUALITY_MODEL_CACHE = joblib.load(PPG_QUALITY_MODEL_PATH)
    return _PPG_QUALITY_MODEL_CACHE


def _predict_ppg_quality_model(values: np.ndarray, sampling_rate: float) -> dict | None:
    try:
        payload = _load_ppg_quality_model()
        if payload is None:
            return None
        model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        vector = np.asarray([_ppg_quality_feature_vector(values, sampling_rate)], dtype=float)
        if hasattr(model, "predict_proba"):
            classes = list(model.classes_)
            probabilities = model.predict_proba(vector)[0]
            probability_map = {str(label): float(prob) for label, prob in zip(classes, probabilities)}
            good_probability = float(probability_map.get("good", 0.0))
        else:
            label = str(model.predict(vector)[0])
            good_probability = 1.0 if label == "good" else 0.0
            probability_map = {label: good_probability}
        return {
            "quality_model_good_probability": good_probability,
            "quality_model_probabilities": probability_map,
            "quality_model_source": str(PPG_QUALITY_MODEL_PATH),
            "quality_model_cv_metrics": payload.get("cv_metrics") if isinstance(payload, dict) else None,
        }
    except Exception as exc:
        return {"quality_model_error": str(exc)}


class _SEBlock1D(nn.Module if nn is not None else object):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1)


class _SEResBlock1D(nn.Module if nn is not None else object):
    def __init__(self, c_in: int, c_out: int, stride: int = 1, dilation: int = 1, dropout: float = 0.05):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(c_in, c_out, 7, stride=stride, padding=dilation * 3, dilation=dilation),
            nn.BatchNorm1d(c_out),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(c_out, c_out, 5, padding=2),
            nn.BatchNorm1d(c_out),
            _SEBlock1D(c_out),
        )
        self.skip = nn.Identity() if c_in == c_out and stride == 1 else nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride), nn.BatchNorm1d(c_out))
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))



class _PPGPeakConvBlock(nn.Module if nn is not None else object):
    def __init__(self, c_in: int, c_out: int):
        if nn is None:
            return
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, 7, padding=3), nn.BatchNorm1d(c_out), nn.SiLU(),
            nn.Conv1d(c_out, c_out, 5, padding=2), nn.BatchNorm1d(c_out), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class _PPGPeakUNet(nn.Module if nn is not None else object):
    def __init__(self):
        if nn is None:
            return
        super().__init__()
        ch = [1, 24, 48, 96, 128]
        self.e1 = _PPGPeakConvBlock(ch[0], ch[1])
        self.e2 = _PPGPeakConvBlock(ch[1], ch[2])
        self.e3 = _PPGPeakConvBlock(ch[2], ch[3])
        self.b = _PPGPeakConvBlock(ch[3], ch[4])
        self.pool = nn.MaxPool1d(2)
        self.u3 = nn.ConvTranspose1d(ch[4], ch[3], 2, 2)
        self.d3 = _PPGPeakConvBlock(ch[3] * 2, ch[3])
        self.u2 = nn.ConvTranspose1d(ch[3], ch[2], 2, 2)
        self.d2 = _PPGPeakConvBlock(ch[2] * 2, ch[2])
        self.u1 = nn.ConvTranspose1d(ch[2], ch[1], 2, 2)
        self.d1 = _PPGPeakConvBlock(ch[1] * 2, ch[1])
        self.out = nn.Conv1d(ch[1], 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.b(self.pool(e3))
        x = self.u3(b)
        x = torch.cat([x[..., :e3.shape[-1]], e3], 1)
        x = self.d3(x)
        x = self.u2(x)
        x = torch.cat([x[..., :e2.shape[-1]], e2], 1)
        x = self.d2(x)
        x = self.u1(x)
        x = torch.cat([x[..., :e1.shape[-1]], e1], 1)
        x = self.d1(x)
        return self.out(x)


def _load_ppg_peak_dl_model() -> tuple[object, dict] | None:
    global _PPG_PEAK_DL_MODEL_CACHE
    if torch is None or nn is None or not PPG_PEAK_DL_MODEL_PATH.exists():
        return None
    if _PPG_PEAK_DL_MODEL_CACHE is not None:
        return _PPG_PEAK_DL_MODEL_CACHE
    checkpoint = torch.load(PPG_PEAK_DL_MODEL_PATH, map_location="cpu", weights_only=False)
    model = _PPGPeakUNet()
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    _PPG_PEAK_DL_MODEL_CACHE = (model, checkpoint if isinstance(checkpoint, dict) else {})
    return _PPG_PEAK_DL_MODEL_CACHE


def _predict_ppg_peak_dl(values: np.ndarray, sampling_rate: float) -> dict | None:
    try:
        loaded = _load_ppg_peak_dl_model()
        if loaded is None:
            return None
        model, checkpoint = loaded
        target_fs = float(checkpoint.get("target_fs", 125.0))
        threshold = float(checkpoint.get("threshold", 0.28))
        prominence = float(checkpoint.get("prominence", 0.05))
        window = int(checkpoint.get("window", 1024))
        hop = int(checkpoint.get("hop", 512))
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < max(16, int(4 * sampling_rate)):
            return {"peak_dl_error": "signal too short for peak DL model"}
        if abs(float(sampling_rate) - target_fs) > 1e-3:
            target_len = max(16, int(round(len(values) * target_fs / float(sampling_rate))))
            work = scipy_signal.resample(values, target_len).astype(np.float32)
        else:
            work = values.astype(np.float32)
        work = scipy_signal.sosfiltfilt(scipy_signal.butter(3, [0.4 / (0.5 * target_fs), min(8.0, target_fs * 0.45) / (0.5 * target_fs)], btype="bandpass", output="sos"), work).astype(np.float32)
        work = work - np.nanmedian(work)
        work = work / (np.nanpercentile(np.abs(work), 95) + 1e-6)
        acc = np.zeros(len(work), dtype=np.float32)
        wt = np.zeros(len(work), dtype=np.float32)
        starts = list(range(0, max(1, len(work) - window + 1), hop))
        if starts and starts[-1] != len(work) - window:
            starts.append(max(0, len(work) - window))
        if not starts:
            starts = [0]
        win = np.hanning(window).astype(np.float32)
        win = np.maximum(win, 0.05)
        with torch.no_grad():
            for start in starts[:512]:
                seg = work[start:start + window]
                if len(seg) < window:
                    pad = np.zeros(window, dtype=np.float32)
                    pad[:len(seg)] = seg
                    seg = pad
                logits = model(torch.tensor(seg[None, None, :], dtype=torch.float32))
                prob = torch.sigmoid(logits).cpu().numpy()[0, 0]
                end = min(len(work), start + window)
                acc[start:end] += prob[:end - start] * win[:end - start]
                wt[start:end] += win[:end - start]
        prob = acc / np.maximum(wt, 1e-6)
        distance = max(1, int(60.0 / 220.0 * target_fs))
        peaks_rs, props = scipy_signal.find_peaks(prob, distance=distance, height=threshold, prominence=prominence)
        if abs(float(sampling_rate) - target_fs) > 1e-3:
            peaks = np.asarray(np.round(peaks_rs / target_fs * float(sampling_rate)), dtype=int)
            peaks = peaks[(peaks >= 0) & (peaks < len(values))]
        else:
            peaks = peaks_rs.astype(int)
        hr = bpm_from_peaks(peaks, sampling_rate)
        interval = _pulse_interval_features(peaks, sampling_rate)
        return {
            "peak_dl_indices": peaks.tolist(),
            "peak_dl_num_peaks": int(len(peaks)),
            "peak_dl_heart_rate_bpm": hr,
            "peak_dl_probability_mean": float(np.nanmean(prob)) if len(prob) else None,
            "peak_dl_probability_p95": float(np.nanpercentile(prob, 95)) if len(prob) else None,
            "peak_dl_model_source": str(PPG_PEAK_DL_MODEL_PATH),
            "peak_dl_cv_metrics": checkpoint.get("cv_metrics"),
            **{f"peak_dl_{key}": value for key, value in interval.items()},
        }
    except Exception as exc:
        return {"peak_dl_error": str(exc)}


class _PPGQualitySEResNet(nn.Module if nn is not None else object):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 32, 9, padding=4),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            _SEResBlock1D(32, 48, stride=2, dilation=1),
            _SEResBlock1D(48, 64, stride=2, dilation=1),
            _SEResBlock1D(64, 96, stride=2, dilation=2),
            _SEResBlock1D(96, 128, stride=2, dilation=2),
            _SEResBlock1D(128, 160, stride=2, dilation=4),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.25), nn.Linear(160, 64), nn.SiLU(), nn.Dropout(0.15), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)


def _load_ppg_quality_dl_model() -> tuple[object, dict] | None:
    global _PPG_QUALITY_DL_MODEL_CACHE
    if torch is None or nn is None or not PPG_QUALITY_DL_MODEL_PATH.exists():
        return None
    if _PPG_QUALITY_DL_MODEL_CACHE is not None:
        return _PPG_QUALITY_DL_MODEL_CACHE
    checkpoint = torch.load(PPG_QUALITY_DL_MODEL_PATH, map_location="cpu", weights_only=False)
    model = _PPGQualitySEResNet()
    model.load_state_dict(checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint)
    model.eval()
    _PPG_QUALITY_DL_MODEL_CACHE = (model, checkpoint if isinstance(checkpoint, dict) else {})
    return _PPG_QUALITY_DL_MODEL_CACHE


def _quality_dl_windows(values: np.ndarray, sampling_rate: float, target_fs: float = 64.0, target_len: int = 1920) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < max(16, int(8 * sampling_rate)):
        return np.empty((0, 3, target_len), dtype=np.float32)
    window = max(1, int(round(30.0 * sampling_rate)))
    stride = max(1, int(round(15.0 * sampling_rate)))
    starts = list(range(0, max(1, len(values) - window + 1), stride))
    if len(values) < window:
        starts = [0]
    elif starts and starts[-1] != len(values) - window:
        starts.append(len(values) - window)
    rows = []
    for start in starts[:64]:
        segment = values[start:start + window]
        if len(segment) < max(16, int(8 * sampling_rate)):
            continue
        try:
            filtered = scipy_signal.resample(segment, target_len).astype(np.float32) if len(segment) != target_len or abs(sampling_rate - target_fs) > 1e-3 else segment.astype(np.float32)
        except Exception:
            continue
        filtered = filtered - np.nanmedian(filtered)
        filtered = filtered / (np.nanpercentile(np.abs(filtered), 95) + 1e-6)
        d1 = np.gradient(filtered).astype(np.float32)
        d2 = np.gradient(d1).astype(np.float32)
        rows.append(np.stack([filtered, d1 / (np.nanstd(d1) + 1e-6), d2 / (np.nanstd(d2) + 1e-6)], axis=0))
    return np.asarray(rows, dtype=np.float32)


def _predict_ppg_quality_dl_model(values: np.ndarray, sampling_rate: float) -> dict | None:
    try:
        loaded = _load_ppg_quality_dl_model()
        if loaded is None:
            return None
        model, checkpoint = loaded
        windows = _quality_dl_windows(values, sampling_rate, target_fs=float(checkpoint.get("target_fs", 64.0)), target_len=int(checkpoint.get("window_samples", 1920)))
        if len(windows) == 0:
            return {"quality_dl_error": "signal too short for quality DL model"}
        probs = []
        with torch.no_grad():
            for start in range(0, len(windows), 32):
                batch = torch.tensor(windows[start:start + 32], dtype=torch.float32)
                probs.extend(torch.sigmoid(model(batch)).cpu().numpy().astype(float).tolist())
        probs_arr = np.asarray(probs, dtype=float)
        good_probability = float(np.nanmedian(probs_arr))
        return {
            "quality_dl_good_probability": good_probability,
            "quality_dl_window_good_probability_mean": float(np.nanmean(probs_arr)),
            "quality_dl_window_good_probability_min": float(np.nanmin(probs_arr)),
            "quality_dl_bad_window_fraction": float(np.mean(probs_arr < 0.5)),
            "quality_dl_num_windows": int(len(probs_arr)),
            "quality_dl_source": str(PPG_QUALITY_DL_MODEL_PATH),
            "quality_dl_cv_metrics": checkpoint.get("cv_metrics") or checkpoint.get("test_metrics_segade_dalia"),
        }
    except Exception as exc:
        return {"quality_dl_error": str(exc)}


def _pulse_interval_features(peaks: np.ndarray, sampling_rate: float) -> dict:
    peaks = np.asarray(peaks, dtype=float)
    if len(peaks) < 6:
        return {"num_valid_intervals": 0}
    intervals_s = np.diff(peaks) / float(sampling_rate)
    intervals_s = intervals_s[(intervals_s >= 0.25) & (intervals_s <= 3.0)]
    if len(intervals_s) < 5:
        return {"num_valid_intervals": int(len(intervals_s))}
    mean_interval = float(np.mean(intervals_s))
    median_interval = float(np.median(intervals_s))
    successive = np.abs(np.diff(intervals_s))
    rmssd_s = float(np.sqrt(np.mean(np.diff(intervals_s) ** 2))) if len(intervals_s) > 1 else 0.0
    sdnn_s = float(np.std(intervals_s))
    robust_cv = float((1.4826 * np.median(np.abs(intervals_s - median_interval))) / median_interval) if median_interval > 0 else None
    turning_point_ratio = None
    if len(intervals_s) >= 3:
        d1 = np.diff(intervals_s[:-1])
        d2 = np.diff(intervals_s[1:])
        turning_point_ratio = float(np.mean((d1 * d2) < 0))
    return {
        "num_valid_intervals": int(len(intervals_s)),
        "mean_pulse_interval_s": mean_interval,
        "median_pulse_interval_s": median_interval,
        "pulse_interval_cv": float(sdnn_s / mean_interval) if mean_interval > 0 else None,
        "robust_pulse_interval_cv": robust_cv,
        "rmssd_s": rmssd_s,
        "normalized_rmssd": float(rmssd_s / mean_interval) if mean_interval > 0 else None,
        "successive_change_fraction": float(np.mean(successive > 0.12)) if len(successive) else 0.0,
        "pnn80_fraction": float(np.mean(successive > 0.08)) if len(successive) else 0.0,
        "pnn120_fraction": float(np.mean(successive > 0.12)) if len(successive) else 0.0,
        "pnn200_fraction": float(np.mean(successive > 0.20)) if len(successive) else 0.0,
        "turning_point_ratio": turning_point_ratio,
        "short_interval_fraction": float(np.mean(intervals_s < 0.5)),
        "long_interval_fraction": float(np.mean(intervals_s > 1.5)),
    }


def _ppg_irregularity_feature_vector(features: dict, artifact_metrics: dict, heart_rate_bpm: float | None) -> list[float]:
    keys = [
        "pulse_interval_cv",
        "robust_pulse_interval_cv",
        "normalized_rmssd",
        "successive_change_fraction",
        "pnn80_fraction",
        "pnn120_fraction",
        "pnn200_fraction",
        "turning_point_ratio",
        "short_interval_fraction",
        "long_interval_fraction",
    ]
    values = [_safe_float(features.get(key)) or 0.0 for key in keys]
    values.extend([
        _safe_float(heart_rate_bpm) or 0.0,
        _safe_float(features.get("num_valid_intervals")) or 0.0,
        _safe_float(artifact_metrics.get("artifact_score")) or 0.0,
        _safe_float(artifact_metrics.get("baseline_wander_ratio")) or 0.0,
        _safe_float(artifact_metrics.get("high_frequency_noise_ratio")) or 0.0,
    ])
    return values




class _IntervalAttentionBiLSTM(nn.Module if nn is not None else object):
    def __init__(self, seq_features: int = 4, tab_features: int = 14, hidden: int = 48, dropout: float = 0.35) -> None:
        if nn is None:
            return
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(seq_features, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.lstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.attn = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.tab = nn.Sequential(nn.LayerNorm(tab_features), nn.Linear(tab_features, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, seq, mask, feat):
        z = self.input_proj(seq)
        z, _ = self.lstm(z)
        scores = self.attn(z).squeeze(-1)
        scores = scores.masked_fill(mask <= 0, -1e4)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(z * weights.unsqueeze(-1), dim=1)
        tab = self.tab(feat)
        return self.head(torch.cat([pooled, tab], dim=-1)).squeeze(-1)


def _normalize_interval_sequence(intervals: np.ndarray, seq_len: int = 128) -> tuple[np.ndarray, np.ndarray]:
    intervals = np.asarray(intervals, dtype=np.float32)
    intervals = intervals[np.isfinite(intervals)]
    intervals = intervals[(intervals >= 0.25) & (intervals <= 3.0)]
    mask = np.zeros(seq_len, dtype=np.float32)
    seq = np.zeros((seq_len, 4), dtype=np.float32)
    if len(intervals) == 0:
        return seq, mask
    intervals = intervals[:seq_len]
    med = np.median(intervals)
    mad = np.median(np.abs(intervals - med)) + 1e-4
    z = np.clip((intervals - med) / (1.4826 * mad), -8.0, 8.0)
    d = np.concatenate([[0.0], np.diff(intervals)])
    dz = np.clip(d / (med + 1e-4), -4.0, 4.0)
    seq[: len(intervals), 0] = z
    seq[: len(intervals), 1] = dz
    seq[: len(intervals), 2] = np.clip(intervals / (med + 1e-4), 0.2, 3.0)
    seq[: len(intervals), 3] = np.arange(len(intervals), dtype=np.float32) / max(1, seq_len - 1)
    mask[: len(intervals)] = 1.0
    return seq, mask


def _interval_summary_vector(intervals: np.ndarray, artifact_metrics: dict, num_peaks: int, duration_s: float) -> np.ndarray:
    intervals = np.asarray(intervals, dtype=np.float32)
    intervals = intervals[np.isfinite(intervals)]
    intervals = intervals[(intervals >= 0.25) & (intervals <= 3.0)]
    if len(intervals) < 4:
        core = np.zeros(10, dtype=np.float32)
    else:
        mean = float(np.mean(intervals))
        med = float(np.median(intervals))
        diff = np.abs(np.diff(intervals))
        rmssd = float(np.sqrt(np.mean(np.diff(intervals) ** 2))) if len(intervals) > 1 else 0.0
        robust_cv = float((1.4826 * np.median(np.abs(intervals - med))) / (med + 1e-8))
        core = np.asarray([
            mean,
            med,
            float(np.std(intervals) / (mean + 1e-8)),
            robust_cv,
            rmssd / (mean + 1e-8),
            float(np.mean(diff > 0.08)) if len(diff) else 0.0,
            float(np.mean(diff > 0.12)) if len(diff) else 0.0,
            float(np.mean(diff > 0.20)) if len(diff) else 0.0,
            float(np.mean(intervals < 0.5)),
            float(np.mean(intervals > 1.5)),
        ], dtype=np.float32)
    extra = np.asarray([
        _safe_float(artifact_metrics.get("artifact_score")) or 0.0,
        _safe_float(artifact_metrics.get("baseline_wander_ratio")) or 0.0,
        _safe_float(artifact_metrics.get("high_frequency_noise_ratio")) or 0.0,
        float(num_peaks) / max(1e-6, duration_s / 60.0),
    ], dtype=np.float32)
    return np.concatenate([core, extra]).astype(np.float32)


def _load_ppg_interval_dl_model() -> tuple[object, dict, int] | None:
    global _PPG_AF_INTERVAL_DL_CACHE
    if torch is None or nn is None or not PPG_AF_INTERVAL_DL_MODEL_PATH.exists():
        return None
    if _PPG_AF_INTERVAL_DL_CACHE is not None:
        return _PPG_AF_INTERVAL_DL_CACHE
    checkpoint = torch.load(PPG_AF_INTERVAL_DL_MODEL_PATH, map_location="cpu", weights_only=False)
    seq_len = int(checkpoint.get("seq_len", 128)) if isinstance(checkpoint, dict) else 128
    model = _IntervalAttentionBiLSTM(hidden=48, dropout=0.35)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    _PPG_AF_INTERVAL_DL_CACHE = (model, checkpoint if isinstance(checkpoint, dict) else {}, seq_len)
    return _PPG_AF_INTERVAL_DL_CACHE


def _predict_ppg_interval_dl_model(values: np.ndarray, sampling_rate: float, peaks: np.ndarray, artifact_metrics: dict) -> dict | None:
    try:
        loaded = _load_ppg_interval_dl_model()
        if loaded is None:
            return None
        model, checkpoint, seq_len = loaded
        intervals = np.diff(np.asarray(peaks, dtype=float)) / float(sampling_rate)
        seq, mask = _normalize_interval_sequence(intervals, seq_len=seq_len)
        if float(mask.sum()) < 6:
            return {"interval_dl_error": "not enough pulse intervals"}
        feat = _interval_summary_vector(intervals, artifact_metrics, len(peaks), len(values) / float(sampling_rate))
        with torch.no_grad():
            logit = model(
                torch.tensor(seq[None, :, :], dtype=torch.float32),
                torch.tensor(mask[None, :], dtype=torch.float32),
                torch.tensor(feat[None, :], dtype=torch.float32),
            )
            af_probability = float(torch.sigmoid(logit).cpu().numpy()[0])
        return {
            "interval_dl_af_probability": af_probability,
            "interval_dl_predicted_rhythm": "af" if af_probability >= 0.5 else "non_af",
            "interval_dl_model_source": str(PPG_AF_INTERVAL_DL_MODEL_PATH),
            "interval_dl_cv_metrics": checkpoint.get("cv_metrics"),
        }
    except Exception as exc:
        return {"interval_dl_error": str(exc)}


def _load_ppg_irregularity_model() -> dict | object | None:
    global _PPG_IRREGULARITY_MODEL_CACHE
    if joblib is None or not PPG_IRREGULARITY_MODEL_PATH.exists():
        return None
    if _PPG_IRREGULARITY_MODEL_CACHE is None:
        _PPG_IRREGULARITY_MODEL_CACHE = joblib.load(PPG_IRREGULARITY_MODEL_PATH)
    return _PPG_IRREGULARITY_MODEL_CACHE


def _predict_ppg_irregularity_model(features: dict, artifact_metrics: dict, heart_rate_bpm: float | None) -> dict | None:
    try:
        payload = _load_ppg_irregularity_model()
        if payload is None:
            return None
        model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        vector = np.asarray([_ppg_irregularity_feature_vector(features, artifact_metrics, heart_rate_bpm)], dtype=float)
        if hasattr(model, "predict_proba"):
            classes = list(model.classes_)
            probabilities = model.predict_proba(vector)[0]
            probability_map = {str(label): float(prob) for label, prob in zip(classes, probabilities)}
            af_probability = float(probability_map.get("af", 0.0))
        else:
            label = str(model.predict(vector)[0])
            af_probability = 1.0 if label == "af" else 0.0
            probability_map = {label: af_probability}
        return {
            "af_probability": af_probability,
            "model_source": str(PPG_IRREGULARITY_MODEL_PATH),
            "model_cv_metrics": payload.get("cv_metrics") if isinstance(payload, dict) else None,
            "probabilities": probability_map,
        }
    except Exception as exc:
        return {"model_error": str(exc)}


def PPG_assess_quality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    summary = signal_quality_summary(data.values)
    artifact_metrics = _ppg_artifact_metrics(data.values, data.sampling_rate)
    flags = []
    if artifact_metrics["flatline_fraction"] > 0.15:
        flags.append("flatline_or_dropout_segments")
    if artifact_metrics["saturation_fraction"] > 0.20:
        flags.append("possible_saturation_or_clipping")
    if (artifact_metrics.get("baseline_wander_ratio") or 0.0) > 0.65:
        flags.append("strong_baseline_wander")
    if (artifact_metrics.get("high_frequency_noise_ratio") or 0.0) > 0.30:
        flags.append("high_frequency_noise")
    quality = summary.get("quality", "bad")
    if artifact_metrics["artifact_score"] >= 0.55:
        quality = "bad"
    elif artifact_metrics["artifact_score"] >= 0.25 and quality == "good":
        quality = "moderate"
    confidence = float(summary.get("confidence", 0.5)) * (1.0 - 0.45 * artifact_metrics["artifact_score"])
    model_result = _predict_ppg_quality_model(data.values, data.sampling_rate)
    dl_result = _predict_ppg_quality_dl_model(data.values, data.sampling_rate)
    dl_probability = dl_result.get("quality_dl_good_probability") if dl_result else None
    feature_probability = model_result.get("quality_model_good_probability") if model_result else None
    good_probability = dl_probability if dl_probability is not None else feature_probability
    if good_probability is not None:
        good_probability = float(good_probability)
        if good_probability >= 0.72 and artifact_metrics["artifact_score"] < 0.45:
            quality = "good"
        elif good_probability >= 0.40:
            quality = "moderate"
        else:
            quality = "bad"
        confidence = max(confidence, 0.50 + 0.40 * abs(good_probability - 0.5) * 2.0)
        if good_probability < 0.40:
            flags.append("model_low_ppg_quality")
        if dl_probability is not None and dl_result.get("quality_dl_bad_window_fraction", 0.0) > 0.35:
            flags.append("dl_detected_bad_quality_windows")
    result = {
        "tool": "PPG_assess_quality",
        "source": data.source,
        **summary,
        **artifact_metrics,
        "quality": quality,
        "quality_flags": sorted(set(flags)),
        "confidence": float(max(0.05, min(0.9, confidence))),
    }
    if model_result:
        result.update(model_result)
    if dl_result:
        result.update(dl_result)
        if dl_result.get("quality_dl_good_probability") is not None:
            result["quality_model_good_probability"] = dl_result.get("quality_dl_good_probability")
            result["quality_model_source"] = dl_result.get("quality_dl_source")
    return result


def PPG_detect_peaks(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    artifact_metrics = _ppg_artifact_metrics(data.values, data.sampling_rate)
    peaks, details = ppg_multiscale_systolic_peaks(data.values, data.sampling_rate)
    if len(peaks) < 3:
        peaks, details = neurokit_nabian2018_peaks(
            data.values,
            data.sampling_rate,
            low_hz=0.4,
            high_hz=min(8.0, data.sampling_rate * 0.45),
            fallback_threshold_scale=0.35,
        )
    classical_peaks = np.asarray(peaks, dtype=int)
    classical_details = dict(details)
    dl_result = _predict_ppg_peak_dl(data.values, data.sampling_rate)

    def candidate_score(candidate_peaks: np.ndarray, method_name: str) -> float:
        hr = bpm_from_peaks(candidate_peaks, data.sampling_rate)
        feats = _pulse_interval_features(candidate_peaks, data.sampling_rate)
        if hr is None or not 35 <= hr <= 220 or feats.get("num_valid_intervals", 0) < 5:
            return 0.1
        robust_cv = feats.get("robust_pulse_interval_cv")
        cv_penalty = min(0.25, float(robust_cv or 0.0) * 0.4)
        count_rate = len(candidate_peaks) / max(1e-6, len(data.values) / float(data.sampling_rate) / 60.0)
        rate_penalty = 0.2 if not 35 <= count_rate <= 220 else 0.0
        method_bonus = 0.08 if method_name == "ppg_peak_unet_capnobase" else 0.0
        return float(0.7 + method_bonus - cv_penalty - rate_penalty - 0.25 * artifact_metrics["artifact_score"])

    selected_source = "classical"
    if dl_result and dl_result.get("peak_dl_indices"):
        dl_peaks = np.asarray(dl_result.get("peak_dl_indices", []), dtype=int)
        classical_score = candidate_score(classical_peaks, classical_details.get("method", "classical"))
        dl_score = candidate_score(dl_peaks, "ppg_peak_unet_capnobase")
        classical_hr = bpm_from_peaks(classical_peaks, data.sampling_rate)
        dl_hr = bpm_from_peaks(dl_peaks, data.sampling_rate)
        classical_bad = classical_hr is None or not 35 <= classical_hr <= 220 or len(classical_peaks) < 5
        if classical_bad:
            peaks = dl_peaks
            details = {
                "method": "ppg_peak_unet_capnobase",
                "classical_method": classical_details.get("method"),
                "peak_detector_selected": "deep_unet",
                "peak_detector_classical_score": classical_score,
                "peak_detector_dl_score": dl_score,
            }
            selected_source = "deep_unet"
        else:
            peaks = classical_peaks
            details = {
                **classical_details,
                "peak_detector_selected": "classical",
                "peak_detector_classical_score": classical_score,
                "peak_detector_dl_score": dl_score,
                "peak_dl_heart_rate_bpm": dl_hr,
                "peak_dl_num_peaks": int(len(dl_peaks)),
            }
    heart_rate = bpm_from_peaks(peaks, data.sampling_rate)
    regularity = interval_regularity(peaks, data.sampling_rate)
    interval_features = _pulse_interval_features(peaks, data.sampling_rate)
    base_confidence = 0.82 if selected_source == "deep_unet" else (0.78 if details["method"] == "ppg_multiscale_systolic" else (0.72 if details["method"] == "nabian2018" else 0.6))
    confidence = min(base_confidence, regularity["regularity_confidence"])
    if interval_features.get("num_valid_intervals", 0) < 5:
        confidence = min(confidence, 0.25)
    if heart_rate is None or not 35 <= heart_rate <= 220:
        confidence = 0.3
    confidence *= 1.0 - 0.4 * artifact_metrics["artifact_score"]
    result = {
        "tool": "PPG_detect_peaks",
        "peak_indices": peaks.tolist(),
        "num_peaks": int(len(peaks)),
        "heart_rate_bpm": heart_rate,
        "confidence": float(max(0.05, min(0.88, confidence))),
        **regularity,
        **artifact_metrics,
        **details,
    }
    if dl_result:
        for key, value in dl_result.items():
            if key != "peak_dl_indices":
                result.setdefault(key, value)
    return result



def _ppg_pulse_amplitude_features(values: np.ndarray, sampling_rate: float, peaks: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    peaks = np.asarray(peaks, dtype=int)
    peaks = peaks[(peaks >= 0) & (peaks < len(values))]
    finite = values[np.isfinite(values)]
    if len(finite) == 0 or len(peaks) < 3:
        return {
            "median_pulse_amplitude": None,
            "pulse_amplitude_cv": None,
            "perfusion_index_proxy": None,
            "low_amplitude_pulse_fraction": None,
            "num_amplitude_pulses": 0,
        }
    amplitudes = []
    trough_indices = []
    pre_peak_radius = max(1, int(0.55 * sampling_rate))
    for i, peak in enumerate(peaks):
        left = int(peaks[i - 1]) if i > 0 else max(0, int(peak) - pre_peak_radius)
        right = int(peak)
        if right <= left + 1:
            continue
        segment = values[left:right + 1]
        if not np.any(np.isfinite(segment)) or not np.isfinite(values[int(peak)]):
            continue
        trough_offset = int(np.nanargmin(segment))
        trough = left + trough_offset
        amp = float(values[int(peak)] - values[trough])
        if np.isfinite(amp) and amp > 0:
            amplitudes.append(amp)
            trough_indices.append(trough)
    if len(amplitudes) == 0:
        return {
            "median_pulse_amplitude": None,
            "pulse_amplitude_cv": None,
            "perfusion_index_proxy": None,
            "low_amplitude_pulse_fraction": None,
            "num_amplitude_pulses": 0,
        }
    amplitudes = np.asarray(amplitudes, dtype=float)
    median_amp = float(np.nanmedian(amplitudes))
    amp_cv = float(np.nanstd(amplitudes) / (median_amp + 1e-8)) if median_amp > 0 else None
    dc_level = float(np.nanmedian(np.abs(finite))) + 1e-8
    low_threshold = max(median_amp * 0.35, np.nanpercentile(amplitudes, 25) * 0.5)
    return {
        "median_pulse_amplitude": median_amp,
        "pulse_amplitude_iqr": float(np.nanpercentile(amplitudes, 75) - np.nanpercentile(amplitudes, 25)),
        "pulse_amplitude_cv": amp_cv,
        "perfusion_index_proxy": float(median_amp / dc_level),
        "low_amplitude_pulse_fraction": float(np.mean(amplitudes < low_threshold)),
        "num_amplitude_pulses": int(len(amplitudes)),
        "trough_indices": [int(x) for x in trough_indices[:5000]],
    }




def _ppg_pulse_morphology_features(values: np.ndarray, sampling_rate: float, peaks: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    peaks = np.asarray(peaks, dtype=int)
    peaks = peaks[(peaks >= 1) & (peaks < len(values) - 1)]
    if len(values) == 0 or len(peaks) < 3:
        return {
            "median_rise_time_s": None,
            "median_decay_time_s": None,
            "median_pulse_width_half_height_s": None,
            "median_pulse_area": None,
            "median_upstroke_slope": None,
            "pulse_width_cv": None,
            "pulse_area_cv": None,
            "amplitude_alternans_index": None,
            "num_morphology_pulses": 0,
        }
    amplitudes = []
    rise_times = []
    decay_times = []
    widths = []
    areas = []
    slopes = []
    max_left = max(1, int(round(0.9 * sampling_rate)))
    max_right = max(1, int(round(1.2 * sampling_rate)))
    for i, peak in enumerate(peaks):
        left_bound = int(peaks[i - 1]) if i > 0 else max(0, int(peak) - max_left)
        right_bound = int(peaks[i + 1]) if i + 1 < len(peaks) else min(len(values) - 1, int(peak) + max_right)
        left_bound = max(0, min(left_bound, int(peak) - 1))
        right_bound = min(len(values) - 1, max(right_bound, int(peak) + 1))
        left_segment = values[left_bound:int(peak) + 1]
        right_segment = values[int(peak):right_bound + 1]
        if len(left_segment) < 3 or len(right_segment) < 3 or not np.isfinite(values[int(peak)]):
            continue
        left_trough = left_bound + int(np.nanargmin(left_segment))
        right_trough = int(peak) + int(np.nanargmin(right_segment))
        baseline = float(min(values[left_trough], values[right_trough]))
        amp = float(values[int(peak)] - baseline)
        if not np.isfinite(amp) or amp <= 0:
            continue
        half = baseline + 0.5 * amp
        left_cross = left_trough
        for idx in range(int(peak), left_trough, -1):
            if values[idx] <= half:
                left_cross = idx
                break
        right_cross = right_trough
        for idx in range(int(peak), right_trough + 1):
            if values[idx] <= half:
                right_cross = idx
                break
        pulse = values[left_trough:right_trough + 1] - baseline
        pulse = np.maximum(pulse, 0.0)
        amplitudes.append(amp)
        rise = (int(peak) - left_trough) / float(sampling_rate)
        decay = (right_trough - int(peak)) / float(sampling_rate)
        width = max(0.0, (right_cross - left_cross) / float(sampling_rate))
        rise_times.append(rise)
        decay_times.append(decay)
        widths.append(width)
        areas.append(float(np.trapezoid(pulse, dx=1.0 / float(sampling_rate))))
        slopes.append(float(amp / max(rise, 1e-3)))
    if len(amplitudes) == 0:
        return {"num_morphology_pulses": 0}
    amplitudes = np.asarray(amplitudes, dtype=float)
    rise_times = np.asarray(rise_times, dtype=float)
    decay_times = np.asarray(decay_times, dtype=float)
    widths = np.asarray(widths, dtype=float)
    areas = np.asarray(areas, dtype=float)
    slopes = np.asarray(slopes, dtype=float)
    amp_diff = np.abs(np.diff(amplitudes)) if len(amplitudes) > 1 else np.asarray([])
    alternans = float(np.nanmedian(amp_diff[::2]) / (np.nanmedian(amplitudes) + 1e-8)) if len(amp_diff) >= 4 else None
    def cv(x: np.ndarray) -> float | None:
        med = float(np.nanmedian(x)) if len(x) else 0.0
        return float(np.nanstd(x) / (med + 1e-8)) if med > 0 else None
    return {
        "median_rise_time_s": float(np.nanmedian(rise_times)),
        "median_decay_time_s": float(np.nanmedian(decay_times)),
        "median_pulse_width_half_height_s": float(np.nanmedian(widths)),
        "median_pulse_area": float(np.nanmedian(areas)),
        "median_upstroke_slope": float(np.nanmedian(slopes)),
        "pulse_width_cv": cv(widths),
        "pulse_area_cv": cv(areas),
        "rise_time_cv": cv(rise_times),
        "upstroke_slope_cv": cv(slopes),
        "amplitude_alternans_index": alternans,
        "num_morphology_pulses": int(len(amplitudes)),
    }


def _ppg_window_perfusion_stability(values: np.ndarray, sampling_rate: float, window_seconds: float = 30.0) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    window = int(round(window_seconds * sampling_rate))
    if len(values) < max(window, int(10 * sampling_rate)):
        return {"window_perfusion_index_cv": None, "low_perfusion_window_fraction": None, "num_perfusion_windows": 0}
    stride = max(1, window // 2)
    proxies = []
    for start in range(0, len(values) - window + 1, stride):
        seg = values[start:start + window]
        dynamic = float(np.nanpercentile(seg, 95) - np.nanpercentile(seg, 5))
        dc = float(np.nanmedian(np.abs(seg))) + 1e-8
        proxies.append(dynamic / dc)
    proxies = np.asarray(proxies, dtype=float)
    med = float(np.nanmedian(proxies)) if len(proxies) else 0.0
    return {
        "window_perfusion_index_cv": float(np.nanstd(proxies) / (med + 1e-8)) if med > 0 else None,
        "low_perfusion_window_fraction": float(np.mean(proxies < max(0.05, 0.35 * med))) if len(proxies) else None,
        "num_perfusion_windows": int(len(proxies)),
    }

def PPG_assess_perfusion_variability(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    if len(values) == 0:
        return {"tool": "PPG_assess_perfusion_variability", "error": "empty signal", "confidence": 0.0}
    peaks_result = PPG_detect_peaks(signal_path, sampling_rate, column)
    quality_result = PPG_assess_quality(signal_path, sampling_rate, column)
    peaks = np.asarray(peaks_result.get("peak_indices", []), dtype=int)
    finite = values[np.isfinite(values)]
    dynamic_range = float(np.nanpercentile(finite, 95) - np.nanpercentile(finite, 5)) if len(finite) else 0.0
    median_abs = float(np.nanmedian(np.abs(finite))) if len(finite) else 0.0
    amplitude_proxy = float(dynamic_range / (median_abs + 1e-8))
    interval_features = _pulse_interval_features(peaks, data.sampling_rate)
    amplitude_features = _ppg_pulse_amplitude_features(values, data.sampling_rate, peaks)
    morphology_features = _ppg_pulse_morphology_features(values, data.sampling_rate, peaks)
    window_stability = _ppg_window_perfusion_stability(values, data.sampling_rate)
    pulse_interval_cv = interval_features.get("pulse_interval_cv")
    pulse_amplitude_cv = amplitude_features.get("pulse_amplitude_cv")
    pulse_area_cv = morphology_features.get("pulse_area_cv")
    pulse_width_cv = morphology_features.get("pulse_width_cv")
    perfusion_index_proxy = amplitude_features.get("perfusion_index_proxy")
    low_amp_fraction = amplitude_features.get("low_amplitude_pulse_fraction")
    low_window_fraction = window_stability.get("low_perfusion_window_fraction")

    low_perfusion = (
        amplitude_proxy < 0.05
        or dynamic_range < 1e-6
        or (perfusion_index_proxy is not None and perfusion_index_proxy < 0.025)
        or (low_amp_fraction is not None and low_amp_fraction > 0.45)
        or (low_window_fraction is not None and low_window_fraction > 0.45)
    )
    perfusion_level = "low_perfusion_proxy" if low_perfusion else "adequate_perfusion_proxy"
    variability_flags = []
    if pulse_interval_cv is not None and pulse_interval_cv > 0.2:
        variability_flags.append("high_pulse_interval_variability")
    if pulse_amplitude_cv is not None and pulse_amplitude_cv > 0.55:
        variability_flags.append("high_pulse_amplitude_variability")
    if pulse_area_cv is not None and pulse_area_cv > 0.65:
        variability_flags.append("high_pulse_area_variability")
    if pulse_width_cv is not None and pulse_width_cv > 0.45:
        variability_flags.append("high_pulse_width_variability")
    if low_amp_fraction is not None and low_amp_fraction > 0.35:
        variability_flags.append("intermittent_low_amplitude_pulses")
    if low_window_fraction is not None and low_window_fraction > 0.35:
        variability_flags.append("intermittent_low_perfusion_windows")
    if morphology_features.get("amplitude_alternans_index") is not None and morphology_features["amplitude_alternans_index"] > 0.35:
        variability_flags.append("pulse_amplitude_alternans_pattern")
    variability_risk = "high_pulse_variability_proxy" if variability_flags else "no_high_variability_proxy"
    confidence = max(0.35, min(0.72, float(peaks_result.get("confidence", 0.5))))
    quality = quality_result.get("quality")
    if quality == "bad":
        perfusion_level = "artifact_limited_perfusion_proxy"
        variability_flags.append("bad_ppg_quality_limits_perfusion_interpretation")
        confidence = min(confidence, 0.35)
    elif quality == "moderate":
        confidence = min(confidence, 0.55)
    if amplitude_features.get("num_amplitude_pulses", 0) < 5:
        confidence = min(confidence, 0.35)
    if peaks_result.get("artifact_score") is not None and float(peaks_result.get("artifact_score")) > 0.35:
        confidence = min(confidence, 0.45)
    return {
        "tool": "PPG_assess_perfusion_variability",
        "pulse_amplitude_proxy": amplitude_proxy,
        "dynamic_range": dynamic_range,
        **{key: value for key, value in amplitude_features.items() if key != "trough_indices"},
        **morphology_features,
        **window_stability,
        "pulse_interval_cv": pulse_interval_cv,
        "robust_pulse_interval_cv": interval_features.get("robust_pulse_interval_cv"),
        "heart_rate_bpm": peaks_result.get("heart_rate_bpm"),
        "ppg_quality": quality,
        "quality_model_good_probability": quality_result.get("quality_model_good_probability"),
        "perfusion_level": perfusion_level,
        "pulse_variability_risk": variability_risk,
        "pulse_variability_flags": variability_flags,
        "artifact_score": peaks_result.get("artifact_score"),
        "confidence": confidence,
        "method": "ppg_peak_trough_morphology_perfusion_variability_screening",
        "disclaimer": "Screening heuristic only; low perfusion and vascular interpretations require calibrated PPG, sensor gain, and clinical context.",
    }



def PPG_screen_pulse_irregularity(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    peaks_result = PPG_detect_peaks(signal_path, sampling_rate, column)
    peaks = np.asarray(peaks_result.get("peak_indices", []), dtype=float)
    artifact_metrics = _ppg_artifact_metrics(data.values, data.sampling_rate)
    interval_features = _pulse_interval_features(peaks, data.sampling_rate)
    if interval_features.get("num_valid_intervals", 0) < 5:
        return {"tool": "PPG_screen_pulse_irregularity", "error": "not enough valid pulse intervals", "confidence": 0.1, **artifact_metrics}

    pulse_interval_cv = interval_features.get("pulse_interval_cv")
    normalized_rmssd = interval_features.get("normalized_rmssd")
    successive_change_fraction = interval_features.get("successive_change_fraction") or 0.0
    score = 0
    flags = []
    if pulse_interval_cv is not None and pulse_interval_cv > 0.16:
        score += 1
        flags.append("high_pulse_interval_cv")
    if normalized_rmssd is not None and normalized_rmssd > 0.18:
        score += 1
        flags.append("high_pulse_interval_rmssd")
    if successive_change_fraction > 0.25:
        score += 1
        flags.append("frequent_successive_pulse_changes")
    if successive_change_fraction > 0.45 and (normalized_rmssd or 0.0) > 0.14:
        score += 1
        flags.append("dominant_successive_pulse_variability")
    heuristic_risk = "elevated_irregular_pulse_proxy" if score >= 2 else "low_irregular_pulse_proxy"

    feature_model_result = _predict_ppg_irregularity_model(interval_features, artifact_metrics, peaks_result.get("heart_rate_bpm"))
    interval_dl_result = _predict_ppg_interval_dl_model(data.values, data.sampling_rate, peaks, artifact_metrics)

    feature_af_probability = feature_model_result.get("af_probability") if feature_model_result else None
    interval_dl_af_probability = interval_dl_result.get("interval_dl_af_probability") if interval_dl_result else None
    if interval_dl_af_probability is not None:
        af_probability = interval_dl_af_probability
        risk = "elevated_irregular_pulse_proxy" if af_probability >= 0.5 else "low_irregular_pulse_proxy"
        predicted_rhythm = "af" if af_probability >= 0.5 else "non_af"
        method = "ppg_interval_attention_bilstm_classifier"
    elif feature_af_probability is not None:
        af_probability = feature_af_probability
        risk = "elevated_irregular_pulse_proxy" if af_probability >= 0.5 else "low_irregular_pulse_proxy"
        predicted_rhythm = "af" if af_probability >= 0.5 else "non_af"
        method = "ppg_pulse_interval_feature_classifier"
    else:
        af_probability = None
        risk = heuristic_risk
        predicted_rhythm = "af" if risk == "elevated_irregular_pulse_proxy" else "non_af"
        method = "ppg_pulse_interval_irregularity_screening"

    confidence_cap = 0.78 if interval_dl_af_probability is not None else (0.72 if feature_af_probability is not None else 0.62)
    confidence = min(float(peaks_result.get("confidence", 0.5)), confidence_cap)
    if artifact_metrics["artifact_score"] > 0.35:
        confidence = min(confidence, 0.45)
        flags.append("artifact_limited_interpretation")
    result = {
        "tool": "PPG_screen_pulse_irregularity",
        "heart_rate_bpm": peaks_result.get("heart_rate_bpm"),
        **interval_features,
        "irregular_pulse_score": int(score),
        "irregular_pulse_flags": flags,
        "heuristic_irregular_pulse_risk": heuristic_risk,
        "irregular_pulse_risk": risk,
        "predicted_rhythm": predicted_rhythm,
        "af_probability": af_probability,
        "confidence": float(max(0.05, confidence)),
        **artifact_metrics,
        "method": method,
        "disclaimer": "PPG irregular-pulse screening is artifact sensitive; confirm AF/rhythm findings with ECG when clinically relevant.",
    }
    if feature_model_result:
        result["feature_model_af_probability"] = feature_af_probability
        result.update({
            f"feature_{key}": value
            for key, value in feature_model_result.items()
            if key != "af_probability"
        })
    if interval_dl_result:
        result.update(interval_dl_result)
    return result




def _respiratory_band_rate(values: np.ndarray, sampling_rate: float, low_hz: float = 0.08, high_hz: float = 0.7) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < max(16, int(sampling_rate * 12)):
        return {"respiratory_rate_bpm": None, "respiratory_power_ratio": 0.0, "top_respiration_peaks_bpm": []}
    high_hz = min(high_hz, sampling_rate * 0.45)
    if high_hz <= low_hz:
        return {"respiratory_rate_bpm": None, "respiratory_power_ratio": 0.0, "top_respiration_peaks_bpm": []}
    centered = values - np.nanmedian(values)
    sos = scipy_signal.butter(3, [low_hz / (0.5 * sampling_rate), high_hz / (0.5 * sampling_rate)], btype="bandpass", output="sos")
    band = scipy_signal.sosfiltfilt(sos, centered)
    freqs, psd = scipy_signal.welch(band, fs=sampling_rate, nperseg=min(len(band), int(sampling_rate * 32)))
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return {"respiratory_rate_bpm": None, "respiratory_power_ratio": 0.0, "top_respiration_peaks_bpm": []}
    band_freqs = freqs[mask]
    band_psd = psd[mask]
    order = np.argsort(band_psd)[::-1]
    top = []
    for idx in order[:8]:
        bpm = float(band_freqs[idx] * 60.0)
        power_ratio = float(band_psd[idx] / (np.mean(band_psd) + 1e-12))
        top.append({"rate_bpm": bpm, "power_ratio": power_ratio})
    chosen = top[0]["rate_bpm"] if top else None
    # Low respiratory fundamentals in short PPG windows are often motion/baseline half-rate peaks.
    # Prefer a plausible harmonic if it has meaningful spectral support.
    if chosen is not None and chosen < 8.0:
        harmonic_options = [
            item
            for item in top[1:]
            if 9.0 <= item["rate_bpm"] <= 24.0
            and min(abs(item["rate_bpm"] - 2.0 * chosen), abs(item["rate_bpm"] - 3.0 * chosen)) <= 1.5
            and item["power_ratio"] >= max(1.5, top[0]["power_ratio"] * 0.25)
        ]
        if harmonic_options:
            chosen = harmonic_options[0]["rate_bpm"]
    return {
        "respiratory_rate_bpm": chosen,
        "respiratory_power_ratio": top[0]["power_ratio"] if top else 0.0,
        "top_respiration_peaks_bpm": top,
        "respiratory_band_std": float(np.nanstd(band)),
    }


def _interpolate_event_series(event_indices: np.ndarray, event_values: np.ndarray, sampling_rate: float, n_samples: int) -> np.ndarray | None:
    event_indices = np.asarray(event_indices, dtype=float)
    event_values = np.asarray(event_values, dtype=float)
    valid = np.isfinite(event_indices) & np.isfinite(event_values)
    event_indices = event_indices[valid]
    event_values = event_values[valid]
    if len(event_indices) < 4:
        return None
    grid_t = np.arange(n_samples, dtype=float) / float(sampling_rate)
    event_t = event_indices / float(sampling_rate)
    series = np.interp(grid_t, event_t, event_values, left=np.nan, right=np.nan)
    finite = np.isfinite(series)
    if finite.mean() < 0.5:
        return None
    return series[finite]


def _high_rate_consensus_candidate(candidates: list[dict], low_rate_bpm: float) -> dict | None:
    # When a low-frequency baseline peak is selected, PPG often contains a stronger
    # motion/fundamental component plus a respiratory harmonic. Promote a higher
    # candidate only when multiple modulation sources support it.
    bins: dict[float, dict] = {}
    for candidate in candidates:
        for peak in candidate.get("top_respiration_peaks_bpm", [])[:8]:
            rate = peak.get("rate_bpm")
            power = peak.get("power_ratio", 0.0)
            if rate is None or not (13.0 <= rate <= 22.5):
                continue
            if abs(rate - low_rate_bpm) < 3.0:
                continue
            key = round(rate / 1.875) * 1.875
            item = bins.setdefault(key, {"rate_bpm": key, "score": 0.0, "strong_sources": 0, "sources": []})
            item["score"] += float(power)
            if power >= 1.5:
                item["strong_sources"] += 1
            item["sources"].append(candidate.get("source"))
    if not bins:
        return None
    best = max(bins.values(), key=lambda item: (item["strong_sources"], item["score"]))
    chosen_sources = [
        candidate.get("source")
        for candidate in candidates
        if candidate.get("respiratory_rate_bpm") is not None and candidate["respiratory_rate_bpm"] >= 13.0 and abs(candidate["respiratory_rate_bpm"] - best["rate_bpm"]) <= 2.0
    ]
    if best["strong_sources"] >= 2 and best["score"] >= 4.0 and chosen_sources:
        return {
            "respiratory_rate_bpm": float(best["rate_bpm"]),
            "respiratory_power_ratio": float(best["score"]),
            "source": "multi_source_harmonic_consensus",
            "supporting_source": ",".join(sorted(set(str(src) for src in best["sources"] + chosen_sources if src))),
        }
    return None


_RESPIRATION_SELECTOR_SOURCES = ["baseline_wander", "hilbert_envelope", "pulse_amplitude", "pulse_interval"]


def _load_ppg_respiration_selector_model() -> dict | object | None:
    global _PPG_RESPIRATION_SELECTOR_MODEL_CACHE
    if joblib is None or not PPG_RESPIRATION_SELECTOR_MODEL_PATH.exists():
        return None
    if _PPG_RESPIRATION_SELECTOR_MODEL_CACHE is None:
        _PPG_RESPIRATION_SELECTOR_MODEL_CACHE = joblib.load(PPG_RESPIRATION_SELECTOR_MODEL_PATH)
    return _PPG_RESPIRATION_SELECTOR_MODEL_CACHE


def _respiration_candidate_feature_rows(candidates: list[dict], selected_rate: float | None) -> list[dict]:
    bins: dict[float, dict] = {}
    for candidate in candidates:
        source = candidate.get("source")
        candidate_rate = candidate.get("respiratory_rate_bpm")
        if candidate_rate is not None:
            key = round(float(candidate_rate) / 1.875) * 1.875
            item = bins.setdefault(key, {
                "rate_bpm": key,
                "source_scores": {src: 0.0 for src in _RESPIRATION_SELECTOR_SOURCES},
                "source_best_rank": {src: 99 for src in _RESPIRATION_SELECTOR_SOURCES},
                "selected_sources": set(),
                "top_sources": set(),
                "max_power": 0.0,
                "sum_power": 0.0,
            })
            item["selected_sources"].add(source)
        for rank, peak in enumerate(candidate.get("top_respiration_peaks_bpm", []) or []):
            rate = peak.get("rate_bpm")
            power = float(peak.get("power_ratio") or 0.0)
            if rate is None or not (5.0 <= rate <= 35.0):
                continue
            key = round(float(rate) / 1.875) * 1.875
            item = bins.setdefault(key, {
                "rate_bpm": key,
                "source_scores": {src: 0.0 for src in _RESPIRATION_SELECTOR_SOURCES},
                "source_best_rank": {src: 99 for src in _RESPIRATION_SELECTOR_SOURCES},
                "selected_sources": set(),
                "top_sources": set(),
                "max_power": 0.0,
                "sum_power": 0.0,
            })
            if source in _RESPIRATION_SELECTOR_SOURCES:
                item["source_scores"][source] += power * (0.85 ** rank)
                item["source_best_rank"][source] = min(item["source_best_rank"][source], rank)
                if power >= 1.3:
                    item["top_sources"].add(source)
            item["max_power"] = max(float(item["max_power"]), power)
            item["sum_power"] += power * (0.85 ** rank)
    rows = []
    for rate, item in bins.items():
        scores = item["source_scores"]
        ranks = item["source_best_rank"]
        low_neighbor_power = sum(float(v["sum_power"]) for k, v in bins.items() if 5.0 <= k < 12.0 and abs(k - rate) > 1e-6)
        double_support = bins.get(round((rate / 2.0) / 1.875) * 1.875, {}).get("sum_power", 0.0) if rate >= 12.0 else 0.0
        half_support = bins.get(round((rate * 2.0) / 1.875) * 1.875, {}).get("sum_power", 0.0) if rate < 18.0 else 0.0
        rows.append({
            "rate_bpm": float(rate),
            "rate_norm": float(rate / 40.0),
            "is_low_rate": float(rate < 10.0),
            "is_adult_plausible": float(12.0 <= rate <= 24.0),
            "distance_to_18": float(abs(rate - 18.0) / 18.0),
            "sum_power": float(item["sum_power"]),
            "max_power": float(item["max_power"]),
            "num_strong_sources": float(len(item["top_sources"])),
            "num_selected_sources": float(len(item["selected_sources"])),
            "chosen_by_current_rule": float(selected_rate is not None and abs(rate - float(selected_rate)) <= 1.0),
            "low_neighbor_power": float(low_neighbor_power),
            "double_rate_support": float(double_support),
            "half_rate_support": float(half_support),
            **{f"{src}_score": float(scores[src]) for src in _RESPIRATION_SELECTOR_SOURCES},
            **{f"{src}_rank": float(8 if ranks[src] == 99 else ranks[src]) for src in _RESPIRATION_SELECTOR_SOURCES},
            "supporting_sources": ",".join(sorted(str(src) for src in item["top_sources"] if src)),
        })
    return rows


def _select_respiration_candidate_with_model(chosen: dict, candidates: list[dict]) -> dict | None:
    payload = _load_ppg_respiration_selector_model()
    if payload is None:
        return None
    try:
        model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        feature_names = payload.get("feature_names") if isinstance(payload, dict) else None
        if not feature_names:
            return None
        rows = _respiration_candidate_feature_rows(candidates, chosen.get("respiratory_rate_bpm"))
        if not rows:
            return None
        x = np.asarray([[float(row.get(name, 0.0)) for name in feature_names] for row in rows], dtype=float)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(x)[:, 1]
        else:
            probabilities = model.predict(x).astype(float)
        best_index = int(np.argmax(probabilities))
        best = rows[best_index]
        if chosen.get("respiratory_rate_bpm") is not None and abs(float(best["rate_bpm"]) - float(chosen["respiratory_rate_bpm"])) <= 1.0:
            return {**chosen, "respiration_selector_probability": float(probabilities[best_index]), "respiration_selector_source": str(PPG_RESPIRATION_SELECTOR_MODEL_PATH)}
        return {
            **chosen,
            "respiratory_rate_bpm": float(best["rate_bpm"]),
            "source": "learned_multi_source_candidate_selector",
            "supporting_source": best.get("supporting_sources"),
            "respiration_selector_probability": float(probabilities[best_index]),
            "respiration_selector_previous_bpm": chosen.get("respiratory_rate_bpm"),
            "respiration_selector_source": str(PPG_RESPIRATION_SELECTOR_MODEL_PATH),
            "respiration_selector_cv_metrics": payload.get("cv_metrics") if isinstance(payload, dict) else None,
        }
    except Exception as exc:
        return {**chosen, "respiration_selector_error": str(exc)}


def _choose_ppg_respiration_candidate(candidates: list[dict]) -> dict:
    valid = [candidate for candidate in candidates if candidate.get("respiratory_rate_bpm") is not None]
    if not valid:
        return {"respiratory_rate_bpm": None, "respiration_source": None, "respiration_candidate_rates": candidates}
    # Baseline wander is often the most direct respiratory surrogate in clinical PPG.
    baseline = next((candidate for candidate in valid if candidate.get("source") == "baseline_wander"), None)
    amp = next((candidate for candidate in valid if candidate.get("source") == "pulse_amplitude"), None)
    envelope = next((candidate for candidate in valid if candidate.get("source") == "hilbert_envelope"), None)
    if baseline is not None:
        b_rate = baseline["respiratory_rate_bpm"]
        if b_rate is not None and b_rate <= 11.5:
            consensus = _high_rate_consensus_candidate(candidates, float(b_rate))
            if consensus is not None:
                return {**consensus, "respiration_candidate_rates": candidates}
        if amp is not None and abs(amp["respiratory_rate_bpm"] - b_rate) <= 3.0:
            baseline["supporting_source"] = "pulse_amplitude"
            return {**baseline, "respiration_candidate_rates": candidates}
        if envelope is not None and abs(envelope["respiratory_rate_bpm"] - b_rate) <= 3.0:
            baseline["supporting_source"] = "hilbert_envelope"
            return {**baseline, "respiration_candidate_rates": candidates}
        if baseline.get("respiratory_power_ratio", 0.0) >= 2.5:
            return {**baseline, "respiration_candidate_rates": candidates}
    best = max(valid, key=lambda candidate: candidate.get("respiratory_power_ratio", 0.0))
    return {**best, "respiration_candidate_rates": candidates}


def _promote_respiration_harmonic_if_supported(chosen: dict, candidates: list[dict]) -> dict:
    current_rate = chosen.get("respiratory_rate_bpm")
    if current_rate is None or current_rate > 11.5 or chosen.get("source") == "multi_source_harmonic_consensus":
        return chosen
    bins: dict[float, dict] = {}
    for candidate in candidates:
        source = candidate.get("source")
        for rank, peak in enumerate(candidate.get("top_respiration_peaks_bpm", [])[:8]):
            rate = peak.get("rate_bpm")
            power = float(peak.get("power_ratio") or 0.0)
            if rate is None or not (13.0 <= rate <= 22.5) or abs(rate - current_rate) < 3.0:
                continue
            key = round(float(rate) / 1.875) * 1.875
            item = bins.setdefault(key, {"score": 0.0, "sources": set(), "max_power": 0.0})
            item["score"] += power * (0.9 ** rank)
            item["max_power"] = max(float(item["max_power"]), power)
            if power >= 1.3:
                item["sources"].add(source)
    if not bins:
        return chosen

    low_score = 0.0
    low_sources = set()
    low_top_count = 0
    for candidate in candidates:
        for rank, peak in enumerate(candidate.get("top_respiration_peaks_bpm", [])[:4]):
            rate = peak.get("rate_bpm")
            power = float(peak.get("power_ratio") or 0.0)
            if rate is not None and abs(rate - current_rate) <= 1.0:
                low_score += power * (0.9 ** rank)
                if power >= 1.45:
                    low_sources.add(candidate.get("source"))
                if rank <= 1:
                    low_top_count += 1

    best_rate, best = max(bins.items(), key=lambda item: (len(item[1]["sources"]), item[1]["score"]))
    double_rate = round((2.0 * float(current_rate)) / 1.875) * 1.875
    if double_rate in bins and len(bins[double_rate]["sources"]) >= 2 and bins[double_rate]["score"] >= 3.0:
        best_rate, best = double_rate, bins[double_rate]
    elif current_rate <= 8.0:
        high_bins = {
            rate: item
            for rate, item in bins.items()
            if rate >= 15.0 and len(item["sources"]) >= 2 and item["score"] >= 3.0
        }
        if high_bins:
            best_rate, best = max(high_bins.items(), key=lambda item: (len(item[1]["sources"]), item[0], item[1]["score"]))

    slow_breathing_guard = (
        current_rate <= 11.5
        and len(low_sources) >= 2
        and low_score >= float(best["score"]) * 1.35
        and low_top_count >= 2
    )
    if len(best["sources"]) >= 2 and float(best["score"]) >= 3.0 and not slow_breathing_guard:
        return {
            **chosen,
            "respiratory_rate_bpm": float(best_rate),
            "source": "harmonic_promoted_multi_source",
            "supporting_source": ",".join(sorted(str(src) for src in best["sources"] if src)),
            "respiratory_power_ratio": float(best["score"]),
            "harmonic_promoted_from_bpm": float(current_rate),
        }
    return chosen




def PPG_estimate_respiration_modulation(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    if len(values) < data.sampling_rate * 20:
        return {"tool": "PPG_estimate_respiration_modulation", "error": "signal too short", "confidence": 0.0}
    quality = PPG_assess_quality(signal_path, sampling_rate, column)
    peaks_result = PPG_detect_peaks(signal_path, sampling_rate, column)
    peaks = np.asarray(peaks_result.get("peak_indices", []), dtype=int)
    if len(peaks) < 5:
        return {"tool": "PPG_estimate_respiration_modulation", "error": "not enough PPG peaks", "confidence": 0.1}

    candidates = []
    baseline = _respiratory_band_rate(values, data.sampling_rate)
    baseline["source"] = "baseline_wander"
    candidates.append(baseline)

    envelope = np.abs(scipy_signal.hilbert(values - np.nanmedian(values)))
    env_result = _respiratory_band_rate(envelope, data.sampling_rate)
    env_result["source"] = "hilbert_envelope"
    candidates.append(env_result)

    if len(peaks) >= 5:
        amp_series = _interpolate_event_series(peaks, values[peaks], data.sampling_rate, len(values))
        if amp_series is not None:
            amp_result = _respiratory_band_rate(amp_series, data.sampling_rate)
            amp_result["source"] = "pulse_amplitude"
            candidates.append(amp_result)
        intervals = np.diff(peaks) / float(data.sampling_rate)
        interval_series = _interpolate_event_series(peaks[1:], intervals, data.sampling_rate, len(values))
        if interval_series is not None:
            interval_result = _respiratory_band_rate(interval_series, data.sampling_rate)
            interval_result["source"] = "pulse_interval"
            candidates.append(interval_result)

    chosen = _choose_ppg_respiration_candidate(candidates)
    chosen = _promote_respiration_harmonic_if_supported(chosen, candidates)
    selector_chosen = _select_respiration_candidate_with_model(chosen, candidates)
    if selector_chosen is not None:
        chosen = selector_chosen
    resp_rate = chosen.get("respiratory_rate_bpm")
    modulation_index = float((chosen.get("respiratory_band_std") or 0.0) / (np.nanstd(values) + 1e-8))
    confidence = 0.45
    if resp_rate is not None and 5 <= resp_rate <= 40:
        confidence = 0.58
        if chosen.get("supporting_source"):
            confidence += 0.08
        if (chosen.get("respiratory_power_ratio") or 0.0) >= 3.0:
            confidence += 0.06
    if quality.get("quality") == "bad":
        confidence = min(confidence, 0.42)
    elif quality.get("quality") == "good":
        confidence += 0.04
    return {
        "tool": "PPG_estimate_respiration_modulation",
        "respiratory_rate_bpm": resp_rate,
        "respiratory_modulation_index": modulation_index,
        "respiration_source": chosen.get("source"),
        "supporting_source": chosen.get("supporting_source"),
        "harmonic_promoted_from_bpm": chosen.get("harmonic_promoted_from_bpm"),
        "respiration_selector_probability": chosen.get("respiration_selector_probability"),
        "respiration_selector_previous_bpm": chosen.get("respiration_selector_previous_bpm"),
        "respiration_selector_source": chosen.get("respiration_selector_source"),
        "respiration_selector_cv_metrics": chosen.get("respiration_selector_cv_metrics"),
        "respiration_selector_error": chosen.get("respiration_selector_error"),
        "respiration_candidate_rates": chosen.get("respiration_candidate_rates", candidates),
        "heart_rate_bpm": peaks_result.get("heart_rate_bpm"),
        "ppg_quality": quality.get("quality"),
        "quality_model_good_probability": quality.get("quality_model_good_probability"),
        "confidence": float(max(0.05, min(0.82, confidence))),
        "method": "ppg_multi_source_respiration_modulation",
        "disclaimer": "PPG-derived respiration proxy only; validate against respiratory reference signals.",
    }



def PPG_estimate_heart_rate(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    peaks = PPG_detect_peaks(signal_path, sampling_rate, column)
    return {
        "tool": "PPG_estimate_heart_rate",
        "heart_rate_bpm": peaks.get("heart_rate_bpm"),
        "num_pulses": peaks.get("num_peaks"),
        "pulse_peak_indices": peaks.get("peak_indices", []),
        "peak_result": peaks,
        "confidence": peaks.get("confidence", 0.0),
        "method": "ppg_pulse_peak_interval_heart_rate",
    }


def PPG_compute_prv(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    peaks_result = PPG_detect_peaks(signal_path, sampling_rate, column)
    peaks = np.asarray(peaks_result.get("peak_indices", []), dtype=int)
    if len(peaks) < 4:
        return {"tool": "PPG_compute_prv", "error": "not enough pulse peaks", "confidence": 0.1}
    intervals = np.diff(peaks) / float(sampling_rate)
    intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    if len(intervals) < 3:
        return {"tool": "PPG_compute_prv", "error": "not enough valid pulse intervals", "confidence": 0.1}
    ibi_ms = intervals * 1000.0
    diff_ms = np.diff(ibi_ms)
    sdnn = float(np.nanstd(ibi_ms, ddof=1)) if len(ibi_ms) > 1 else 0.0
    rmssd = float(np.sqrt(np.nanmean(diff_ms ** 2))) if len(diff_ms) else 0.0
    pnn50 = float(np.mean(np.abs(diff_ms) > 50.0)) if len(diff_ms) else 0.0
    mean_ibi = float(np.nanmean(ibi_ms))
    mean_pr_bpm = float(60000.0 / mean_ibi) if mean_ibi > 0 else None
    freqs = []
    lf_power = hf_power = lf_hf = None
    if len(ibi_ms) >= 8:
        t = np.cumsum(intervals)
        t = t - t[0]
        if t[-1] > 8:
            grid_fs = 4.0
            grid = np.arange(0, t[-1], 1.0 / grid_fs)
            if len(grid) >= 8:
                interp = np.interp(grid, t, ibi_ms - np.nanmean(ibi_ms))
                freqs, psd = scipy_signal.welch(interp, fs=grid_fs, nperseg=min(len(interp), 256))
                lf_mask = (freqs >= 0.04) & (freqs < 0.15)
                hf_mask = (freqs >= 0.15) & (freqs <= 0.40)
                lf_power = float(np.trapezoid(psd[lf_mask], freqs[lf_mask])) if np.any(lf_mask) else 0.0
                hf_power = float(np.trapezoid(psd[hf_mask], freqs[hf_mask])) if np.any(hf_mask) else 0.0
                lf_hf = float(lf_power / (hf_power + 1e-12)) if hf_power is not None else None
    confidence = min(float(peaks_result.get("confidence", 0.5)), 0.78)
    if peaks_result.get("artifact_score") is not None and float(peaks_result.get("artifact_score")) > 0.35:
        confidence = min(confidence, 0.45)
    return {
        "tool": "PPG_compute_prv",
        "mean_pulse_interval_ms": mean_ibi,
        "median_pulse_interval_ms": float(np.nanmedian(ibi_ms)),
        "mean_pulse_rate_bpm": mean_pr_bpm,
        "sdnn_ms": sdnn,
        "rmssd_ms": rmssd,
        "pnn50": pnn50,
        "pulse_interval_cv": float(np.nanstd(intervals) / (np.nanmean(intervals) + 1e-12)),
        "lf_power": lf_power,
        "hf_power": hf_power,
        "lf_hf_ratio": lf_hf,
        "num_intervals": int(len(intervals)),
        "peak_result": peaks_result,
        "confidence": float(max(0.05, confidence)),
        "method": "ppg_pulse_rate_variability_from_peak_intervals",
        "disclaimer": "PRV is a pulse-interval proxy for HRV and can diverge from ECG HRV during motion, vasoconstriction, arrhythmia, or poor perfusion.",
    }



def _pyppg_fiducial_backend(values: np.ndarray, sampling_rate: float) -> dict | None:
    try:
        import warnings
        import pandas as _pd
        from dotmap import DotMap
        import pyPPG as _pyPPG
        from pyPPG.preproc import Preprocess
        from pyPPG.fiducials import FpCollection
        if not hasattr(np, "NaN"):
            np.NaN = np.nan
        x = np.asarray(values, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) < max(16, int(8 * sampling_rate)):
            return {"pyppg_backend_status": "skipped_signal_too_short"}
        sig = DotMap()
        sig.v = x
        sig.fs = float(sampling_rate)
        sig.name = "biosignal_agent_ppg"
        sig.start_sig = 0
        sig.end_sig = -1
        sig.filtering = True
        sig.correction = []
        prep = Preprocess(fL=0.5, fH=min(12.0, sampling_rate * 0.45), order=4)
        sig.ppg, sig.vpg, sig.apg, sig.jpg = prep.get_signals(sig)
        ppg_obj = _pyPPG.PPG(sig, check_ppg_len=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = FpCollection(ppg_obj).get_fiducials(ppg_obj)
        def clean(col: str) -> list[int]:
            if col not in frame:
                return []
            vals = _pd.to_numeric(frame[col], errors="coerce").dropna().to_numpy(dtype=int)
            vals = vals[(vals >= 0) & (vals < len(x))]
            return [int(v) for v in vals[:5000]]
        onsets = clean("on")
        peaks = clean("sp")
        notches = clean("dn")
        diastolic = clean("dp")
        if len(peaks) < 3 or len(onsets) < 3:
            return {
                "pyppg_backend_status": "empty_or_incompatible_output",
                "pyppg_num_systolic_peaks": int(len(peaks)),
                "pyppg_num_onsets": int(len(onsets)),
                "pyppg_note": "pyPPG 1.0.73 is installed without pinned legacy pandas/numpy; pandas>=3 copy-on-write can leave fiducial frames empty.",
            }
        return {
            "pyppg_backend_status": "ok",
            "pyppg_systolic_peak_indices": peaks,
            "pyppg_pulse_onset_indices": onsets,
            "pyppg_dicrotic_notch_indices": notches,
            "pyppg_diastolic_peak_indices": diastolic,
            "pyppg_num_systolic_peaks": int(len(peaks)),
            "pyppg_num_onsets": int(len(onsets)),
            "pyppg_num_dicrotic_notches": int(len(notches)),
        }
    except Exception as exc:
        return {"pyppg_backend_status": "error", "pyppg_backend_error": f"{type(exc).__name__}: {exc}"}

def PPG_detect_fiducial_points(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    peaks_result = PPG_detect_peaks(signal_path, sampling_rate, column)
    peaks = np.asarray(peaks_result.get("peak_indices", []), dtype=int)
    peaks = peaks[(peaks > 1) & (peaks < len(data.values) - 2)]
    if len(peaks) < 3:
        return {"tool": "PPG_detect_fiducial_points", "error": "not enough pulse peaks", "confidence": 0.1}
    values = data.values
    onsets = []
    notches = []
    radius_left = max(2, int(round(0.65 * sampling_rate)))
    radius_right = max(2, int(round(0.55 * sampling_rate)))
    for i, peak in enumerate(peaks):
        left = int(peaks[i - 1]) if i > 0 else max(0, int(peak) - radius_left)
        right = int(peaks[i + 1]) if i + 1 < len(peaks) else min(len(values) - 1, int(peak) + radius_right)
        left = max(0, min(left, int(peak) - 1))
        right = min(len(values) - 1, max(right, int(peak) + 1))
        pre = values[left:int(peak) + 1]
        post = values[int(peak):right + 1]
        if len(pre) >= 3 and np.any(np.isfinite(pre)):
            onsets.append(left + int(np.nanargmin(pre)))
        if len(post) >= 5 and np.any(np.isfinite(post)):
            deriv = np.gradient(post)
            candidates = np.where((deriv[:-1] < 0) & (deriv[1:] >= 0))[0]
            if len(candidates):
                notch = int(peak) + int(candidates[0] + 1)
            else:
                notch = int(peak) + int(np.nanargmin(post))
            notches.append(notch)
    morphology = _ppg_pulse_morphology_features(values, data.sampling_rate, peaks)
    pyppg_backend = _pyppg_fiducial_backend(values, data.sampling_rate)
    use_pyppg = bool(pyppg_backend and pyppg_backend.get("pyppg_backend_status") == "ok")
    if use_pyppg:
        out_peaks = pyppg_backend.get("pyppg_systolic_peak_indices", peaks.tolist())
        out_onsets = pyppg_backend.get("pyppg_pulse_onset_indices", [int(x) for x in onsets[:5000]])
        out_notches = pyppg_backend.get("pyppg_dicrotic_notch_indices", [int(x) for x in notches[:5000]])
        method = "pyppg_fiducial_backend_with_local_morphology_features"
        confidence = min(0.78, float(max(0.05, peaks_result.get("confidence", 0.5))) + 0.05)
    else:
        out_peaks = peaks.tolist()
        out_onsets = [int(x) for x in onsets[:5000]]
        out_notches = [int(x) for x in notches[:5000]]
        method = "ppg_peak_trough_derivative_fiducial_proxy"
        confidence = float(max(0.05, min(0.7, peaks_result.get("confidence", 0.5)))) if morphology.get("num_morphology_pulses", 0) >= 3 else 0.25
    return {
        "tool": "PPG_detect_fiducial_points",
        "systolic_peak_indices": out_peaks,
        "pulse_onset_indices": out_onsets,
        "dicrotic_notch_indices_proxy": out_notches,
        **morphology,
        "pyppg_backend": pyppg_backend,
        "confidence": confidence,
        "method": method,
        "disclaimer": "Onsets and dicrotic notches are proxy fiducials from a single PPG channel; validated vascular analysis needs calibrated waveform and annotation-specific benchmarking.",
    }


def PPG_estimate_spo2(signal_path: str, sampling_rate: float, red_column: str = "red", infrared_column: str = "ir", column: str | None = None) -> dict:
    frame = pd.read_csv(signal_path)
    if red_column not in frame.columns or infrared_column not in frame.columns:
        return {
            "tool": "PPG_estimate_spo2",
            "error": "red and infrared PPG columns are required for SpO2 estimation",
            "available_columns": list(frame.columns),
            "confidence": 0.0,
            "method": "ratio_of_ratios_requires_dual_wavelength_ppg",
        }
    red = frame[red_column].to_numpy(dtype=float)
    ir = frame[infrared_column].to_numpy(dtype=float)
    mask = np.isfinite(red) & np.isfinite(ir)
    red = red[mask]
    ir = ir[mask]
    if len(red) < max(8, int(5 * sampling_rate)):
        return {"tool": "PPG_estimate_spo2", "error": "signal too short", "confidence": 0.1}
    red_ac = float(np.nanpercentile(red, 95) - np.nanpercentile(red, 5))
    ir_ac = float(np.nanpercentile(ir, 95) - np.nanpercentile(ir, 5))
    red_dc = float(abs(np.nanmedian(red))) + 1e-8
    ir_dc = float(abs(np.nanmedian(ir))) + 1e-8
    ratio = (red_ac / red_dc) / (ir_ac / ir_dc + 1e-12)
    spo2 = float(110.0 - 25.0 * ratio)
    spo2 = float(max(50.0, min(100.0, spo2)))
    quality = "usable_dual_wavelength_proxy" if red_ac > 1e-8 and ir_ac > 1e-8 else "poor_dual_wavelength_proxy"
    return {
        "tool": "PPG_estimate_spo2",
        "spo2_percent_proxy": spo2,
        "ratio_of_ratios": float(ratio),
        "red_ac_dc": float(red_ac / red_dc),
        "ir_ac_dc": float(ir_ac / ir_dc),
        "quality": quality,
        "confidence": 0.45 if quality.startswith("usable") else 0.15,
        "method": "uncalibrated_red_ir_ratio_of_ratios_proxy",
        "disclaimer": "This is an uncalibrated dual-wavelength SpO2 proxy; medical-grade SpO2 requires device-specific calibration and validation.",
    }


def PPG_estimate_bp_proxy(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    perf = PPG_assess_perfusion_variability(signal_path, sampling_rate, column)
    fid = PPG_detect_fiducial_points(signal_path, sampling_rate, column)
    rise = fid.get("median_rise_time_s")
    width = fid.get("median_pulse_width_half_height_s")
    slope = fid.get("median_upstroke_slope")
    stiffness_score = 0.0
    flags = []
    if rise is not None and rise < 0.12:
        stiffness_score += 1.0
        flags.append("short_rise_time_proxy")
    if width is not None and width < 0.22:
        stiffness_score += 0.7
        flags.append("narrow_pulse_width_proxy")
    if slope is not None and slope > 2.0 * max(abs(perf.get("median_pulse_amplitude") or 1.0), 1e-8):
        stiffness_score += 0.5
        flags.append("steep_upstroke_proxy")
    bp_risk = "higher_bp_or_stiffer_pulse_proxy" if stiffness_score >= 1.2 else "no_high_bp_proxy_evidence"
    return {
        "tool": "PPG_estimate_bp_proxy",
        "bp_proxy_risk": bp_risk,
        "bp_proxy_score": float(stiffness_score),
        "bp_proxy_flags": flags,
        "median_rise_time_s": rise,
        "median_pulse_width_half_height_s": width,
        "median_upstroke_slope": slope,
        "perfusion_result": perf,
        "fiducial_result": fid,
        "confidence": float(min(perf.get("confidence", 0.4), fid.get("confidence", 0.4), 0.55)),
        "method": "ppg_morphology_bp_proxy_without_calibration",
        "disclaimer": "Cuffless BP from PPG is calibration-dependent; this tool reports morphology evidence only and does not estimate SBP/DBP.",
    }


def PPG_detect_afib(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    irr = PPG_screen_pulse_irregularity(signal_path, sampling_rate, column)
    return {
        "tool": "PPG_detect_afib",
        "afib_risk": "elevated" if irr.get("predicted_rhythm") == "af" else "low",
        "af_probability": irr.get("af_probability"),
        "predicted_rhythm": irr.get("predicted_rhythm"),
        "irregular_pulse_risk": irr.get("irregular_pulse_risk"),
        "source_irregularity_result": irr,
        "confidence": irr.get("confidence", 0.0),
        "method": "ppg_irregular_pulse_af_wrapper",
        "disclaimer": "PPG AF screening is artifact sensitive and should be confirmed by ECG when clinically relevant.",
    }


def PPG_estimate_sleep_features(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    hr = PPG_estimate_heart_rate(signal_path, sampling_rate, column)
    prv = PPG_compute_prv(signal_path, sampling_rate, column)
    resp = PPG_estimate_respiration_modulation(signal_path, sampling_rate, column)
    perf = PPG_assess_perfusion_variability(signal_path, sampling_rate, column)
    flags = []
    mean_hr = hr.get("heart_rate_bpm")
    rmssd = prv.get("rmssd_ms")
    rr = resp.get("respiratory_rate_bpm")
    if mean_hr is not None and mean_hr < 65:
        flags.append("sleep_like_lower_pulse_rate")
    if rmssd is not None and rmssd > 35:
        flags.append("higher_prv_recovery_proxy")
    if rr is not None and 8 <= rr <= 20:
        flags.append("sleep_plausible_respiration_rate")
    if perf.get("pulse_variability_flags"):
        flags.append("variable_perfusion_or_motion_limits_sleep_proxy")
    sleep_proxy = "sleep_or_rest_compatible_ppg" if len(flags) >= 2 else "wake_or_context_needed"
    return {
        "tool": "PPG_estimate_sleep_features",
        "sleep_proxy": sleep_proxy,
        "sleep_feature_flags": flags,
        "heart_rate_bpm": mean_hr,
        "rmssd_ms": rmssd,
        "sdnn_ms": prv.get("sdnn_ms"),
        "respiratory_rate_bpm": rr,
        "perfusion_level": perf.get("perfusion_level"),
        "confidence": float(min(hr.get("confidence", 0.4), resp.get("confidence", 0.4), 0.58)),
        "method": "ppg_sleep_feature_summary_hr_prv_resp_perfusion",
        "disclaimer": "PPG-only sleep features are context proxies; sleep staging needs PSG or validated multimodal wearable models.",
    }


def PPG_assess_stress_prv(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    prv = PPG_compute_prv(signal_path, sampling_rate, column)
    if "error" in prv:
        return {"tool": "PPG_assess_stress_prv", **prv}
    score = 0.0
    flags = []
    if (prv.get("rmssd_ms") or 0.0) < 20:
        score += 1.0
        flags.append("low_rmssd_prv")
    if (prv.get("sdnn_ms") or 0.0) < 30:
        score += 0.8
        flags.append("low_sdnn_prv")
    if (prv.get("mean_pulse_rate_bpm") or 0.0) > 95:
        score += 0.8
        flags.append("high_pulse_rate")
    if prv.get("lf_hf_ratio") is not None and prv["lf_hf_ratio"] > 2.5:
        score += 0.5
        flags.append("high_lf_hf_proxy")
    level = "elevated_strain_proxy" if score >= 1.5 else "low_strain_proxy"
    return {
        "tool": "PPG_assess_stress_prv",
        "stress_prv_level": level,
        "stress_prv_score": float(score),
        "stress_prv_flags": flags,
        "prv_result": prv,
        "confidence": float(min(prv.get("confidence", 0.4), 0.62)),
        "method": "ppg_prv_stress_recovery_proxy",
        "disclaimer": "Stress/emotion inference from PPG alone is nonspecific and requires context or multimodal validation.",
    }


def PPG_estimate_exercise_intensity(signal_path: str, sampling_rate: float, column: str | None = None, resting_hr_bpm: float | None = None, max_hr_bpm: float | None = None) -> dict:
    hr = PPG_estimate_heart_rate(signal_path, sampling_rate, column)
    bpm = hr.get("heart_rate_bpm")
    if bpm is None:
        return {"tool": "PPG_estimate_exercise_intensity", "error": "heart rate unavailable", "confidence": 0.1}
    if max_hr_bpm is not None and resting_hr_bpm is not None and max_hr_bpm > resting_hr_bpm:
        intensity = (bpm - resting_hr_bpm) / (max_hr_bpm - resting_hr_bpm)
        basis = "heart_rate_reserve"
    elif max_hr_bpm is not None and max_hr_bpm > 0:
        intensity = bpm / max_hr_bpm
        basis = "percent_max_hr"
    else:
        intensity = (bpm - 60.0) / 100.0
        basis = "population_proxy_no_user_calibration"
    intensity = float(max(0.0, min(1.2, intensity)))
    zone = "rest_light" if intensity < 0.45 else ("moderate" if intensity < 0.70 else ("vigorous" if intensity < 0.90 else "near_maximal"))
    return {
        "tool": "PPG_estimate_exercise_intensity",
        "exercise_intensity_zone": zone,
        "exercise_intensity_fraction": intensity,
        "heart_rate_bpm": bpm,
        "calibration_basis": basis,
        "confidence": float(min(hr.get("confidence", 0.4), 0.65 if basis != "population_proxy_no_user_calibration" else 0.45)),
        "method": "ppg_heart_rate_exercise_intensity_proxy",
        "disclaimer": "Exercise intensity from PPG HR is best interpreted with age/resting HR, accelerometry, and motion-quality checks.",
    }


def PPG_assess_vascular_health(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    fid = PPG_detect_fiducial_points(signal_path, sampling_rate, column)
    perf = PPG_assess_perfusion_variability(signal_path, sampling_rate, column)
    rise = fid.get("median_rise_time_s")
    width = fid.get("median_pulse_width_half_height_s")
    decay = fid.get("median_decay_time_s")
    ratio = float(decay / rise) if rise and decay else None
    score = 0.0
    flags = []
    if rise is not None and rise < 0.12:
        score += 0.8
        flags.append("fast_upstroke_stiffness_proxy")
    if width is not None and width < 0.22:
        score += 0.8
        flags.append("narrow_pulse_width_proxy")
    if ratio is not None and ratio > 4.0:
        score += 0.5
        flags.append("prolonged_decay_relative_to_rise_proxy")
    if perf.get("perfusion_level") == "low_perfusion_proxy":
        flags.append("low_perfusion_limits_vascular_interpretation")
    vascular_risk = "elevated_stiffness_proxy" if score >= 1.2 else "no_elevated_stiffness_proxy"
    return {
        "tool": "PPG_assess_vascular_health",
        "vascular_stiffness_proxy": vascular_risk,
        "vascular_proxy_score": float(score),
        "vascular_flags": flags,
        "rise_decay_ratio": ratio,
        "fiducial_result": fid,
        "perfusion_result": perf,
        "confidence": float(min(fid.get("confidence", 0.4), perf.get("confidence", 0.4), 0.55)),
        "method": "ppg_morphology_vascular_health_proxy",
        "disclaimer": "PPG morphology is not a standalone arterial stiffness or vascular-age measurement without calibration and validation.",
    }


def PPG_screen_low_perfusion_shock_risk(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    quality = PPG_assess_quality(signal_path, sampling_rate, column)
    perf = PPG_assess_perfusion_variability(signal_path, sampling_rate, column)
    hr = PPG_estimate_heart_rate(signal_path, sampling_rate, column)
    flags = []
    score = 0.0
    if perf.get("perfusion_level") in {"low_perfusion_proxy", "artifact_limited_perfusion_proxy"}:
        score += 1.0
        flags.append(perf.get("perfusion_level"))
    if (perf.get("low_amplitude_pulse_fraction") or 0.0) > 0.45:
        score += 0.6
        flags.append("many_low_amplitude_pulses")
    if (hr.get("heart_rate_bpm") or 0.0) > 110:
        score += 0.5
        flags.append("tachycardia_with_low_perfusion_proxy")
    if quality.get("quality") == "bad":
        flags.append("bad_signal_quality_limits_shock_screen")
    risk = "elevated_low_perfusion_shock_proxy" if score >= 1.2 else "low_low_perfusion_shock_proxy"
    return {
        "tool": "PPG_screen_low_perfusion_shock_risk",
        "shock_perfusion_risk": risk,
        "shock_perfusion_score": float(score),
        "shock_perfusion_flags": flags,
        "heart_rate_bpm": hr.get("heart_rate_bpm"),
        "perfusion_level": perf.get("perfusion_level"),
        "quality": quality.get("quality"),
        "confidence": float(min(quality.get("confidence", 0.4), perf.get("confidence", 0.4), 0.55)),
        "method": "ppg_low_perfusion_tachycardia_shock_screening_proxy",
        "disclaimer": "Shock risk cannot be diagnosed from PPG alone; use BP, SpO2, clinical context, and perfusion calibration.",
    }
