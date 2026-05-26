from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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
from scipy import signal as scipy_signal
from scipy.io import wavfile
from scipy.ndimage import zoom

from .common import bandpass_filter, bpm_from_peaks, interval_regularity, load_csv_signal, signal_quality_summary
from .spectrogram_tools import Signal_extract_spectrogram_features

PCG_MURMUR_MODEL_PATH = Path("/data1/jiahui/biosignal-agent/outputs/pcg_murmur_feature_classifier.joblib")
PCG_PATIENT_MULTISITE_MODEL_PATH = Path("/data1/jiahui/biosignal-agent/outputs/pcg_murmur_patient_multiloc_cnn_e20.pt")
PCG_SEGMENTATION_MODEL_PATH = Path("/data1/jiahui/biosignal-agent/outputs/pcg_springer_segmentation_tcn.pt")
PCG_VALVE_MODEL_PATH = Path("/data1/jiahui/biosignal-agent/outputs/pcg_bmdhs_valve_feature_classifier.joblib")
PCG_VALVE_DEEP_MODEL_PATH = Path("/data1/jiahui/biosignal-agent/outputs/pcg_bmdhs_valve_spectrogram_cnn.pt")
PCG_OUTCOME_FEATURE_MODEL_PATH = Path("/data1/jiahui/biosignal-agent/outputs/pcg_circor_outcome_feature_classifier.joblib")
PCG_OUTCOME_DEEP_MODEL_PATH = Path("/data1/jiahui/biosignal-agent/outputs/pcg_circor_outcome_patient_multiloc_cnn_e12.pt")
PCG_VALVE_DEEP_MODEL_PATHS = [
    PCG_VALVE_DEEP_MODEL_PATH,
    Path("/data1/jiahui/biosignal-agent/outputs/pcg_bmdhs_valve_spectrogram_cnn_fold1.pt"),
    Path("/data1/jiahui/biosignal-agent/outputs/pcg_bmdhs_valve_spectrogram_cnn_fold2.pt"),
    Path("/data1/jiahui/biosignal-agent/outputs/pcg_bmdhs_valve_spectrogram_cnn_fold3.pt"),
    Path("/data1/jiahui/biosignal-agent/outputs/pcg_bmdhs_valve_spectrogram_cnn_fold4.pt"),
]
_PCG_MURMUR_MODEL_CACHE = None
_PCG_PATIENT_MULTISITE_MODEL_CACHE = None
_PCG_SEGMENTATION_MODEL_CACHE = None
_PCG_VALVE_MODEL_CACHE = None
_PCG_VALVE_DEEP_MODEL_CACHE = None
_PCG_OUTCOME_FEATURE_MODEL_CACHE = None
_PCG_OUTCOME_DEEP_MODEL_CACHE = None
PCG_PATIENT_MULTISITE_LOCATIONS = ["AV", "PV", "TV", "MV", "Phc"]

PCG_MURMUR_PCG_FEATURES = [
    "low_band_power", "mid_band_power", "high_band_power", "very_high_band_power",
    "mid_band_ratio", "high_band_ratio", "spectral_centroid_hz", "spectral_entropy",
    "zero_crossing_rate", "envelope_std", "envelope_p90_median_ratio",
    "envelope_p95_median_ratio", "envelope_p99_median_ratio", "continuous_fraction_60",
    "continuous_fraction_75", "num_sounds", "heart_rate_bpm", "sound_interval_cv",
]
PCG_MURMUR_SPECTROGRAM_FEATURES = [
    "spectrogram_log_power_mean", "spectrogram_log_power_std", "spectral_centroid_mean_hz",
    "spectral_centroid_std_hz", "spectral_rolloff85_mean_hz", "spectral_rolloff85_std_hz",
    "spectral_entropy", "temporal_energy_cv", "temporal_energy_p95_p50_ratio",
    "band_20_60_ratio", "band_60_150_ratio", "band_150_400_ratio",
]



if nn is not None:
    class _PCGLocationEncoder(nn.Module):
        def __init__(self, emb: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 24, 5, padding=2), nn.BatchNorm2d(24), nn.SiLU(), nn.MaxPool2d(2),
                nn.Conv2d(24, 48, 3, padding=1), nn.BatchNorm2d(48), nn.SiLU(), nn.MaxPool2d(2),
                nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.SiLU(), nn.MaxPool2d(2),
                nn.Conv2d(96, 128, 3, padding=1), nn.BatchNorm2d(128), nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, emb), nn.SiLU(),
            )

        def forward(self, x):
            return self.net(x)



    class _PCGValveSpecCNN(nn.Module):
        def __init__(self, out_dim: int = 5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 24, 5, padding=2, bias=False), nn.BatchNorm2d(24), nn.SiLU(), nn.MaxPool2d(2),
                nn.Conv2d(24, 48, 3, padding=1, bias=False), nn.BatchNorm2d(48), nn.SiLU(), nn.MaxPool2d(2),
                nn.Conv2d(48, 96, 3, padding=1, bias=False), nn.BatchNorm2d(96), nn.SiLU(), nn.MaxPool2d(2),
                nn.Conv2d(96, 160, 3, padding=1, bias=False), nn.BatchNorm2d(160), nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(0.35), nn.Linear(160, out_dim),
            )

        def forward(self, x):
            return self.net(x)


    class _PCGPatientMultiLocCNN(nn.Module):
        def __init__(self, emb: int = 128, num_locations: int = 5):
            super().__init__()
            self.encoder = _PCGLocationEncoder(emb)
            self.attn = nn.Sequential(nn.Linear(emb + num_locations, 64), nn.Tanh(), nn.Linear(64, 1))
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(emb, 2))
            self.num_locations = num_locations

        def forward(self, x, mask):
            batch, locations, channels, freq, time = x.shape
            feat = self.encoder(x.reshape(batch * locations, channels, freq, time)).reshape(batch, locations, -1)
            loc_eye = torch.eye(locations, device=x.device).unsqueeze(0).expand(batch, -1, -1)
            score = self.attn(torch.cat([feat, loc_eye], dim=-1)).squeeze(-1)
            score = score.masked_fill(mask <= 0, -1e4)
            weight = torch.softmax(score, dim=1)
            pooled = torch.sum(feat * weight.unsqueeze(-1), dim=1)
            return self.head(pooled)



    class _PCGStateTCNBlock(nn.Module):
        def __init__(self, channels: int, dilation: int, dropout: float = 0.05):
            super().__init__()
            pad = dilation * 3
            self.net = nn.Sequential(
                nn.Conv1d(channels, channels, 7, padding=pad, dilation=dilation, bias=False),
                nn.BatchNorm1d(channels), nn.SiLU(), nn.Dropout(dropout),
                nn.Conv1d(channels, channels, 1, bias=False), nn.BatchNorm1d(channels),
            )
            self.act = nn.SiLU()

        def forward(self, x):
            return self.act(x + self.net(x))


    class _PCGStateTCN(nn.Module):
        def __init__(self, channels: int = 48):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv1d(1, channels, 15, padding=7, bias=False), nn.BatchNorm1d(channels), nn.SiLU())
            self.blocks = nn.Sequential(*[_PCGStateTCNBlock(channels, d) for d in [1, 2, 4, 8, 16, 32, 64, 1, 2, 4]])
            self.head = nn.Conv1d(channels, 4, 1)

        def forward(self, x):
            return self.head(self.blocks(self.stem(x)))

def _load_pcg_patient_multisite_model() -> dict | None:
    global _PCG_PATIENT_MULTISITE_MODEL_CACHE
    if torch is None or not PCG_PATIENT_MULTISITE_MODEL_PATH.exists():
        return None
    if _PCG_PATIENT_MULTISITE_MODEL_CACHE is None:
        payload = torch.load(PCG_PATIENT_MULTISITE_MODEL_PATH, map_location="cpu")
        model = _PCGPatientMultiLocCNN(
            int(payload.get("embedding_dim", 128)),
            len(payload.get("locations", PCG_PATIENT_MULTISITE_LOCATIONS)),
        )
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        _PCG_PATIENT_MULTISITE_MODEL_CACHE = {"payload": payload, "model": model}
    return _PCG_PATIENT_MULTISITE_MODEL_CACHE


def _normalize_pcg_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return values
    values = values - np.nanmedian(values)
    scale = np.nanpercentile(np.abs(values), 95) + 1e-6
    return np.clip(values / scale, -6.0, 6.0).astype(np.float32)


def _load_pcg_signal_for_multisite(path: str, sampling_rate: float | None, column: str | None) -> tuple[int, np.ndarray]:
    suffix = Path(path).suffix.lower()
    if suffix == ".wav":
        fs, values = wavfile.read(path)
        if values.ndim > 1:
            values = values[:, 0]
        return int(fs), _normalize_pcg_values(values)
    if sampling_rate is None:
        raise ValueError(f"sampling_rate is required for non-wav PCG signal: {path}")
    data = load_csv_signal(path, float(sampling_rate), column)
    return int(round(data.sampling_rate)), _normalize_pcg_values(data.values)


def _resample_pcg_values(values: np.ndarray, fs: int, target_fs: int) -> np.ndarray:
    if fs == target_fs:
        return values.astype(np.float32)
    length = max(16, int(round(len(values) * target_fs / float(fs))))
    return scipy_signal.resample(values, length).astype(np.float32)


def _center_crop_or_pad(values: np.ndarray, length: int) -> np.ndarray:
    if len(values) >= length:
        start = max(0, (len(values) - length) // 2)
        return values[start:start + length].astype(np.float32)
    out = np.zeros(length, dtype=np.float32)
    start = (length - len(values)) // 2
    out[start:start + len(values)] = values
    return out


def _pcg_multisite_spec_image(values: np.ndarray, fs: int, freq_bins: int, time_bins: int) -> np.ndarray:
    freqs, _, spec = scipy_signal.spectrogram(
        values,
        fs=fs,
        window="hann",
        nperseg=min(256, max(32, len(values) // 4)),
        noverlap=min(192, max(0, min(256, max(32, len(values) // 4)) - 1)),
        mode="magnitude",
        scaling="density",
    )
    mask = (freqs >= 20.0) & (freqs <= 800.0)
    freqs = freqs[mask]
    spec = spec[mask]
    if spec.size == 0 or freqs.size == 0:
        return np.zeros((freq_bins, time_bins), dtype=np.float32)
    log_edges = np.geomspace(max(20.0, float(freqs[0])), max(21.0, float(freqs[-1])), freq_bins + 1)
    pooled = np.zeros((freq_bins, spec.shape[1]), dtype=np.float32)
    for i in range(freq_bins):
        band = (freqs >= log_edges[i]) & (freqs < log_edges[i + 1])
        pooled[i] = np.mean(spec[band], axis=0) if np.any(band) else 0.0
    img = np.log1p(pooled ** 2)
    lo, hi = np.percentile(img, [2, 98])
    img = np.clip((img - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    if img.shape[1] != time_bins:
        img = zoom(img, (1.0, time_bins / img.shape[1]), order=1)
    return img.astype(np.float32)




def _pcg_band_limits(sampling_rate: float, high_hz: float = 400.0) -> tuple[float, float]:
    high = min(float(high_hz), float(sampling_rate) * 0.45)
    low = 20.0 if high > 45.0 else max(1.0, float(sampling_rate) * 0.05)
    return low, high


def _pcg_filtered_envelope(values: np.ndarray, sampling_rate: float, high_hz: float = 200.0) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    centered = values - np.nanmedian(values)
    low, high = _pcg_band_limits(sampling_rate, high_hz)
    filtered = bandpass_filter(centered, sampling_rate, low, high, order=3) if high > low else centered
    envelope = np.abs(scipy_signal.hilbert(filtered))
    smooth = max(3, int(round(0.05 * sampling_rate)))
    if smooth % 2 == 0:
        smooth += 1
    if len(envelope) >= smooth:
        envelope = scipy_signal.medfilt(envelope, kernel_size=smooth)
    scale = np.nanpercentile(envelope, 95) + 1e-12
    envelope = np.clip(envelope / scale, 0.0, 5.0)
    return filtered, envelope.astype(float)


def _estimate_pcg_cycle_seconds(envelope: np.ndarray, sampling_rate: float) -> tuple[float | None, float]:
    if len(envelope) < int(2.0 * sampling_rate):
        return None, 0.0
    env = envelope - np.nanmean(envelope)
    if float(np.nanstd(env)) < 1e-8:
        return None, 0.0
    corr = scipy_signal.correlate(env, env, mode="full", method="fft")[len(env) - 1:]
    corr = corr / (corr[0] + 1e-12)
    min_lag = max(1, int(round(0.33 * sampling_rate)))
    max_lag = min(len(corr) - 1, int(round(1.6 * sampling_rate)))
    if max_lag <= min_lag:
        return None, 0.0
    segment = corr[min_lag:max_lag + 1]
    peaks, props = scipy_signal.find_peaks(segment, distance=max(1, int(0.25 * sampling_rate)), prominence=0.02)
    if len(peaks) == 0:
        lag = int(np.argmax(segment)) + min_lag
        confidence = float(max(0.0, min(1.0, corr[lag])))
    else:
        prominences = props.get("prominences", np.zeros(len(peaks)))
        idx = int(np.argmax(prominences))
        lag = int(peaks[idx]) + min_lag
        confidence = float(max(0.0, min(1.0, prominences[idx] * 3.0 + corr[lag] * 0.5)))
    cycle = lag / float(sampling_rate)
    if not 0.33 <= cycle <= 1.6:
        return None, 0.0
    return float(cycle), confidence


def _duration_constrained_pcg_events(values: np.ndarray, sampling_rate: float) -> dict:
    filtered, envelope = _pcg_filtered_envelope(values, sampling_rate, high_hz=220.0)
    cycle_s, periodicity_conf = _estimate_pcg_cycle_seconds(envelope, sampling_rate)
    min_distance = max(1, int(round(0.12 * sampling_rate)))
    prominence = max(float(np.nanstd(envelope)) * 0.25, 0.03)
    peaks, props = scipy_signal.find_peaks(envelope, distance=min_distance, prominence=prominence)
    if len(peaks) < 3:
        return {
            "filtered": filtered,
            "envelope": envelope,
            "sound_indices": peaks.astype(int),
            "s1_indices": np.asarray([], dtype=int),
            "s2_indices": np.asarray([], dtype=int),
            "cycle_seconds": cycle_s,
            "periodicity_confidence": periodicity_conf,
            "segmentation_confidence": 0.1,
        }
    if cycle_s is None:
        intervals = np.diff(peaks) / float(sampling_rate)
        plausible = intervals[(intervals >= 0.15) & (intervals <= 0.8)]
        half_cycle = float(np.nanmedian(plausible)) if len(plausible) else 0.35
        cycle_s = float(np.clip(2.0 * half_cycle, 0.45, 1.4))
    cycle_n = max(1, int(round(cycle_s * sampling_rate)))
    # Choose the phase that maximizes envelope energy at S1 and expected S2 positions.
    s2_delay = int(round(np.clip(0.32 * cycle_s, 0.18, 0.45) * sampling_rate))
    candidates = peaks[: min(len(peaks), 12)]
    best_phase = int(candidates[0])
    best_score = -np.inf
    for phase in candidates:
        score = 0.0
        count = 0
        start = int(phase % cycle_n)
        for s1 in range(start, len(envelope), cycle_n):
            s2 = s1 + s2_delay
            for center, weight in ((s1, 1.0), (s2, 0.85)):
                lo = max(0, center - int(0.06 * sampling_rate))
                hi = min(len(envelope), center + int(0.06 * sampling_rate) + 1)
                if hi > lo:
                    score += weight * float(np.nanmax(envelope[lo:hi]))
                    count += 1
        if count:
            score /= count
        if score > best_score:
            best_score = score
            best_phase = int(phase % cycle_n)
    s1_indices = []
    s2_indices = []
    search = max(1, int(round(0.08 * sampling_rate)))
    for s1_pred in range(best_phase, len(envelope), cycle_n):
        lo = max(0, s1_pred - search)
        hi = min(len(envelope), s1_pred + search + 1)
        if hi > lo:
            s1_indices.append(int(lo + np.argmax(envelope[lo:hi])))
        s2_pred = s1_pred + s2_delay
        lo = max(0, s2_pred - search)
        hi = min(len(envelope), s2_pred + search + 1)
        if hi > lo:
            s2_indices.append(int(lo + np.argmax(envelope[lo:hi])))
    all_sounds = np.asarray(sorted(set(s1_indices + s2_indices)), dtype=int)
    intervals = np.diff(np.asarray(s1_indices, dtype=int)) / float(sampling_rate) if len(s1_indices) > 1 else np.asarray([])
    interval_cv = float(np.nanstd(intervals) / (np.nanmean(intervals) + 1e-12)) if len(intervals) else 1.0
    segmentation_confidence = float(np.clip(0.25 + 0.5 * periodicity_conf + 0.25 * max(0.0, 1.0 - interval_cv), 0.1, 0.9))
    return {
        "filtered": filtered,
        "envelope": envelope,
        "sound_indices": all_sounds,
        "s1_indices": np.asarray(s1_indices, dtype=int),
        "s2_indices": np.asarray(s2_indices, dtype=int),
        "cycle_seconds": cycle_s,
        "periodicity_confidence": periodicity_conf,
        "segmentation_confidence": segmentation_confidence,
        "s1_interval_cv": interval_cv,
    }

def _parse_pcg_multisite_recordings(recordings: str | list[dict]) -> list[dict]:
    if isinstance(recordings, list):
        return recordings
    path = Path(str(recordings))
    if path.exists():
        loaded = json.loads(path.read_text())
    else:
        loaded = json.loads(str(recordings))
    if isinstance(loaded, dict) and "records" in loaded:
        loaded = loaded["records"]
    if not isinstance(loaded, list):
        raise ValueError("recordings must be a JSON list or a JSON file containing a list/records")
    return loaded


def PCG_screen_murmur_patient_multisite(recordings: str, sampling_rate: float | None = None, column: str | None = None) -> dict:
    """Patient-level PCG murmur screen from multiple auscultation sites.

    recordings may be a JSON list or a path to JSON. Each item should include signal_path/path
    and location in AV/PV/TV/MV/Phc. WAV files carry their own sampling rate; CSV files use the
    item sampling_rate, or the function-level sampling_rate fallback.
    """
    model_bundle = _load_pcg_patient_multisite_model()
    if model_bundle is None:
        return {
            "tool": "PCG_screen_murmur_patient_multisite",
            "error": "patient multisite PCG model unavailable; install torch and train the model first",
            "confidence": 0.0,
        }
    try:
        items = _parse_pcg_multisite_recordings(recordings)
        payload = model_bundle["payload"]
        model = model_bundle["model"]
        locations = list(payload.get("locations", PCG_PATIENT_MULTISITE_LOCATIONS))
        by_loc = {}
        for rec in items:
            loc = str(rec.get("location") or rec.get("site") or "").strip()
            canonical = next((known for known in locations if known.lower() == loc.lower()), None)
            if canonical is None:
                continue
            by_loc.setdefault(canonical, rec)
        if not by_loc:
            return {"tool": "PCG_screen_murmur_patient_multisite", "error": "no recognized PCG location labels found", "confidence": 0.0}
        target_fs = int(payload.get("target_fs", 1000))
        seconds = float(payload.get("seconds", 8.0))
        freq_bins = int(payload.get("freq_bins", 80))
        time_bins = int(payload.get("time_bins", 128))
        length = int(round(target_fs * seconds))
        xs = []
        mask = []
        used_locations = []
        for loc in locations:
            rec = by_loc.get(loc)
            if rec is None:
                xs.append(np.zeros((freq_bins, time_bins), dtype=np.float32))
                mask.append(0.0)
                continue
            signal_path = rec.get("signal_path") or rec.get("path")
            if not signal_path:
                raise ValueError(f"missing signal_path/path for location {loc}")
            rec_fs = rec.get("sampling_rate", sampling_rate)
            rec_column = rec.get("column", column)
            fs, values = _load_pcg_signal_for_multisite(str(signal_path), rec_fs, rec_column)
            values = _resample_pcg_values(values, fs, target_fs)
            values = _center_crop_or_pad(values, length)
            xs.append(_pcg_multisite_spec_image(values, target_fs, freq_bins, time_bins))
            mask.append(1.0)
            used_locations.append(loc)
        x = torch.from_numpy(np.stack(xs, axis=0)[None, :, None, :, :].astype(np.float32))
        mask_tensor = torch.tensor([mask], dtype=torch.float32)
        with torch.no_grad():
            logits = model(x, mask_tensor)
            probability = float(torch.softmax(logits, dim=1)[0, 1].item())
        threshold = float(payload.get("best_threshold", 0.5))
        prediction = "present" if probability >= threshold else "absent"
        return {
            "tool": "PCG_screen_murmur_patient_multisite",
            "murmur_model_prediction": prediction,
            "murmur_model_probability_present": probability,
            "murmur_risk": "possible_murmur" if prediction == "present" else "no_murmur_detected",
            "threshold": threshold,
            "used_locations": used_locations,
            "num_locations_used": len(used_locations),
            "model_source": str(PCG_PATIENT_MULTISITE_MODEL_PATH),
            "model_reference": payload.get("reference"),
            "model_cv_metrics": payload.get("cv_metrics"),
            "model_best_threshold_metrics": payload.get("best_threshold_metrics"),
            "confidence": 0.72 if len(used_locations) >= 3 else 0.6,
            "method": "pcg_patient_multisite_logfreq_attention_cnn",
            "disclaimer": "Research screening model trained on CirCor 2022-style multi-location PCG data; not a clinical diagnosis.",
        }
    except Exception as exc:
        return {"tool": "PCG_screen_murmur_patient_multisite", "error": str(exc), "confidence": 0.0}


def _load_pcg_segmentation_model() -> dict | None:
    global _PCG_SEGMENTATION_MODEL_CACHE
    if torch is None or nn is None or not PCG_SEGMENTATION_MODEL_PATH.exists():
        return None
    if _PCG_SEGMENTATION_MODEL_CACHE is None:
        payload = torch.load(PCG_SEGMENTATION_MODEL_PATH, map_location="cpu")
        model = _PCGStateTCN()
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        _PCG_SEGMENTATION_MODEL_CACHE = {"payload": payload, "model": model}
    return _PCG_SEGMENTATION_MODEL_CACHE


def _pcg_tcn_norm(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return x
    med = float(np.nanmedian(x))
    scale = float(np.nanpercentile(np.abs(x - med), 95)) + 1e-6
    return np.clip((x - med) / scale, -6.0, 6.0).astype(np.float32)


def _pcg_state_centers_from_labels(labels: np.ndarray, target: int, sampling_rate: float, min_ms: float = 70.0, merge_gap_ms: float = 0.0) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    mask = labels == int(target)
    if not np.any(mask):
        return np.asarray([], dtype=int)
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts = list(np.where(edges == 1)[0])
    ends = list(np.where(edges == -1)[0])
    min_len = max(1, int(round(float(min_ms) * float(sampling_rate) / 1000.0)))
    max_gap = max(0, int(round(float(merge_gap_ms) * float(sampling_rate) / 1000.0)))
    merged: list[tuple[int, int]] = []
    for a, b in zip(starts, ends):
        if not merged:
            merged.append((int(a), int(b)))
        else:
            prev_a, prev_b = merged[-1]
            if int(a) - prev_b <= max_gap:
                merged[-1] = (prev_a, int(b))
            else:
                merged.append((int(a), int(b)))
    filtered = [(a, b) for a, b in merged if b - a >= min_len]
    return np.asarray([int(round((a + b - 1) / 2.0)) for a, b in filtered], dtype=int)


def _pcg_tcn_segmentation_events(values: np.ndarray, sampling_rate: float) -> dict | None:
    bundle = _load_pcg_segmentation_model()
    if bundle is None:
        return None
    try:
        payload = bundle["payload"]
        model = bundle["model"]
        target_fs = float(payload.get("sampling_rate", 1000.0))
        x = _pcg_tcn_norm(values)
        if x.size < max(8, int(0.5 * sampling_rate)):
            return None
        if abs(float(sampling_rate) - target_fs) > 1e-6:
            x_rs = _resample_pcg_values(x, int(round(sampling_rate)), int(round(target_fs)))
        else:
            x_rs = x
        chunk_len = int(payload.get("chunk_len", 4096))
        pad = (-len(x_rs)) % chunk_len
        if pad:
            x_rs = np.pad(x_rs, (0, pad))
        probs = []
        with torch.no_grad():
            for start in range(0, len(x_rs), chunk_len):
                xb = torch.from_numpy(x_rs[start:start + chunk_len][None, None, :].astype(np.float32))
                logits = model(xb)
                probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy()[0])
        pred = np.concatenate(probs, axis=1)[:, :len(x_rs) - pad if pad else len(x_rs)].argmax(axis=0) + 1
        s1_rs = _pcg_state_centers_from_labels(pred, 1, target_fs, min_ms=70.0, merge_gap_ms=0.0)
        s2_rs = _pcg_state_centers_from_labels(pred, 3, target_fs, min_ms=70.0, merge_gap_ms=0.0)
        if abs(float(sampling_rate) - target_fs) > 1e-6:
            scale = float(sampling_rate) / target_fs
            s1 = np.asarray(np.round(s1_rs * scale), dtype=int)
            s2 = np.asarray(np.round(s2_rs * scale), dtype=int)
        else:
            s1 = s1_rs.astype(int)
            s2 = s2_rs.astype(int)
        all_sounds = np.asarray(sorted(set(s1.tolist() + s2.tolist())), dtype=int)
        cycles = np.diff(s1) / float(sampling_rate) if len(s1) > 1 else np.asarray([])
        interval_cv = float(np.nanstd(cycles) / (np.nanmean(cycles) + 1e-12)) if len(cycles) > 1 else None
        return {
            "filtered": np.asarray(values, dtype=float),
            "envelope": _pcg_filtered_envelope(values, sampling_rate, high_hz=220.0)[1],
            "sound_indices": all_sounds,
            "s1_indices": s1,
            "s2_indices": s2,
            "cycle_seconds": float(np.nanmedian(cycles)) if len(cycles) else None,
            "periodicity_confidence": float(np.clip(1.0 - (interval_cv or 0.5), 0.0, 1.0)),
            "segmentation_confidence": 0.82,
            "s1_interval_cv": interval_cv if interval_cv is not None else 1.0,
            "segmentation_model_source": str(PCG_SEGMENTATION_MODEL_PATH),
            "segmentation_model_metrics": payload.get("val_metrics"),
            "segmentation_method": "springer_supervised_pcg_state_tcn_with_event_postprocessing",
        }
    except Exception:
        return None


def _pcg_best_segmentation_events(values: np.ndarray, sampling_rate: float) -> dict:
    model_events = _pcg_tcn_segmentation_events(values, sampling_rate)
    if model_events is not None and len(model_events.get("s1_indices", [])) >= 2:
        return model_events
    events = _duration_constrained_pcg_events(values, sampling_rate)
    events["segmentation_method"] = "duration_constrained_hilbert_envelope_segmentation"
    return events

def _pcg_event_point(sample_index: int | None, values: np.ndarray, envelope: np.ndarray, sampling_rate: float) -> dict | None:
    if sample_index is None:
        return None
    idx = int(sample_index)
    if idx < 0 or idx >= len(values):
        return None
    point = {
        "sample_index": idx,
        "time_s": float(idx / float(sampling_rate)),
        "amplitude": float(values[idx]),
    }
    if 0 <= idx < len(envelope):
        point["envelope_amplitude"] = float(envelope[idx])
    return point


def _pcg_s1_s2_fiducials(values: np.ndarray, envelope: np.ndarray, sampling_rate: float, s1_indices: np.ndarray, s2_indices: np.ndarray) -> dict:
    s1 = np.asarray(s1_indices, dtype=int)
    s2 = np.asarray(s2_indices, dtype=int)
    fiducials = []
    for beat_index, s1_idx in enumerate(s1[:5000]):
        next_s2 = s2[s2 > s1_idx]
        next_s1 = s1[s1 > s1_idx]
        s2_idx = int(next_s2[0]) if len(next_s2) and (not len(next_s1) or next_s2[0] < next_s1[0]) else None
        fiducials.append({
            "beat_index": int(beat_index),
            "S1": _pcg_event_point(int(s1_idx), values, envelope, sampling_rate),
            "S2": _pcg_event_point(s2_idx, values, envelope, sampling_rate),
        })
    denom = max(1, len(fiducials))
    return {
        "fiducials": fiducials,
        "s1_points": [_pcg_event_point(int(x), values, envelope, sampling_rate) for x in s1[:5000]],
        "s2_points": [_pcg_event_point(int(x), values, envelope, sampling_rate) for x in s2[:5000]],
        "missing_fiducial_fraction": {
            "S1": float(sum(row["S1"] is None for row in fiducials) / denom),
            "S2": float(sum(row["S2"] is None for row in fiducials) / denom),
        },
    }


def _load_pcg_murmur_model() -> dict | object | None:
    global _PCG_MURMUR_MODEL_CACHE
    if joblib is None or not PCG_MURMUR_MODEL_PATH.exists():
        return None
    if _PCG_MURMUR_MODEL_CACHE is None:
        _PCG_MURMUR_MODEL_CACHE = joblib.load(PCG_MURMUR_MODEL_PATH)
    return _PCG_MURMUR_MODEL_CACHE




def _pcg_spectrogram_feature_summary(values: np.ndarray, sampling_rate: float, window_seconds: float = 1.0, overlap: float = 0.5, max_frequency_hz: float = 500.0) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < max(8, int(0.5 * sampling_rate)):
        return {"error": "signal too short"}
    values = values - np.nanmedian(values)
    nperseg = max(16, min(len(values), int(float(window_seconds) * sampling_rate)))
    noverlap = int(max(0.0, min(0.95, float(overlap))) * nperseg)
    freqs, times, spec = scipy_signal.spectrogram(values, fs=sampling_rate, window="hann", nperseg=nperseg, noverlap=noverlap, detrend="constant", scaling="density", mode="magnitude")
    freq_mask = freqs <= min(float(max_frequency_hz), sampling_rate * 0.5)
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
    def entropy(power_vector: np.ndarray) -> float:
        total = float(np.sum(power_vector) + 1e-12)
        probs = np.asarray(power_vector, dtype=float) / total
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)) / np.log2(len(power_vector))) if len(power_vector) > 1 else 0.0
    def band_ratio(low: float, high: float) -> float:
        mask = (freqs >= low) & (freqs < high)
        return float(np.sum(power[mask]) / (np.sum(power) + 1e-12)) if np.any(mask) else 0.0
    temporal_energy = np.sum(power, axis=0)
    return {
        "spectrogram_log_power_mean": float(np.mean(np.log1p(power))) if power.size else 0.0,
        "spectrogram_log_power_std": float(np.std(np.log1p(power))) if power.size else 0.0,
        "spectral_centroid_mean_hz": float(np.mean(centroid_by_time)) if len(centroid_by_time) else None,
        "spectral_centroid_std_hz": float(np.std(centroid_by_time)) if len(centroid_by_time) else None,
        "spectral_rolloff85_mean_hz": float(np.mean(rolloff)) if len(rolloff) else None,
        "spectral_rolloff85_std_hz": float(np.std(rolloff)) if len(rolloff) else None,
        "spectral_entropy": entropy(mean_power_by_freq) if len(mean_power_by_freq) else 0.0,
        "temporal_energy_cv": float(np.std(temporal_energy) / (np.mean(temporal_energy) + 1e-12)) if len(temporal_energy) else 0.0,
        "temporal_energy_p95_p50_ratio": float(np.percentile(temporal_energy, 95) / (np.percentile(temporal_energy, 50) + 1e-12)) if len(temporal_energy) else 0.0,
        "band_20_60_ratio": band_ratio(20, 60),
        "band_60_150_ratio": band_ratio(60, 150),
        "band_150_400_ratio": band_ratio(150, 400),
    }

def _predict_pcg_murmur_model(signal_path: str, sampling_rate: float, column: str | None = None) -> dict | None:
    payload = _load_pcg_murmur_model()
    if payload is None:
        return None
    try:
        model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        feature_names = payload.get("feature_names") if isinstance(payload, dict) else None
        if not feature_names:
            return None
        pcg = PCG_extract_murmur_features(signal_path, sampling_rate, column)
        data = _load_pcg_signal_data(signal_path, sampling_rate, column)
        spec = _pcg_spectrogram_feature_summary(data.values, data.sampling_rate, window_seconds=1.0, overlap=0.5, max_frequency_hz=500.0)
        features = {f"pcg_{name}": pcg.get(name) for name in PCG_MURMUR_PCG_FEATURES}
        features.update({f"spec_{name}": spec.get(name) for name in PCG_MURMUR_SPECTROGRAM_FEATURES})
        vector = np.asarray([[np.nan if features.get(name) is None else float(features.get(name)) for name in feature_names]], dtype=float)
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(vector)[0, 1])
        else:
            probability = float(model.predict(vector)[0])
        prediction = "abnormal" if probability >= 0.5 else "normal"
        return {
            "murmur_model_prediction": prediction,
            "murmur_model_probability_abnormal": probability,
            "murmur_model_source": str(PCG_MURMUR_MODEL_PATH),
            "murmur_model_cv_metrics": payload.get("cv_metrics") if isinstance(payload, dict) else None,
        }
    except Exception as exc:
        return {"murmur_model_error": str(exc)}






def _load_pcg_signal_data(signal_path: str, sampling_rate: float, column: str | None = None):
    if Path(signal_path).suffix.lower() == ".wav":
        fs, values = wavfile.read(signal_path)
        if values.ndim > 1:
            values = values[:, 0]
        class _Data:
            pass
        data = _Data()
        data.values = np.asarray(values, dtype=float)
        data.sampling_rate = float(fs)
        data.source = signal_path
        data.column = None
        return data
    return load_csv_signal(signal_path, sampling_rate, column)

def PCG_assess_quality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = _load_pcg_signal_data(signal_path, sampling_rate, column)
    values = np.asarray(data.values, dtype=float)
    base = signal_quality_summary(values)
    if len(values) < max(8, int(data.sampling_rate)):
        return {"tool": "PCG_assess_quality", "source": data.source, **base, "pcg_quality_label": "too_short", "confidence": 0.1}
    filtered, envelope = _pcg_filtered_envelope(values, data.sampling_rate, high_hz=500.0)
    low, high = _pcg_band_limits(data.sampling_rate, 500.0)
    freqs, psd = scipy_signal.welch(filtered, fs=data.sampling_rate, nperseg=min(len(filtered), int(data.sampling_rate * 2)))
    heart_band = (freqs >= 20.0) & (freqs <= min(180.0, high))
    noise_band = (freqs > min(180.0, high)) & (freqs <= high)
    heart_power = float(np.trapezoid(psd[heart_band], freqs[heart_band])) if np.any(heart_band) else 0.0
    noise_power = float(np.trapezoid(psd[noise_band], freqs[noise_band])) if np.any(noise_band) else 0.0
    snr_db = float(10.0 * np.log10((heart_power + 1e-12) / (noise_power + 1e-12)))
    cycle_s, periodicity = _estimate_pcg_cycle_seconds(envelope, data.sampling_rate)
    saturation_fraction = float(np.mean(np.abs(values) >= np.nanpercentile(np.abs(values), 99.9))) if len(values) else 0.0
    dropout_fraction = float(np.mean(np.abs(values - np.nanmedian(values)) < 1e-9)) if len(values) else 0.0
    events = _pcg_best_segmentation_events(values, data.sampling_rate)
    s1 = np.asarray(events.get("s1_indices", []), dtype=int)
    cycles = np.diff(s1) / float(data.sampling_rate) if len(s1) > 1 else np.asarray([])
    cycle_cv = float(np.nanstd(cycles) / (np.nanmean(cycles) + 1e-12)) if len(cycles) > 1 else None
    score = 0.25 * float(base.get("quality_score", 0.0)) + 0.35 * periodicity + 0.25 * float(np.clip((snr_db + 5.0) / 25.0, 0.0, 1.0)) + 0.15 * float(np.clip(np.nanstd(envelope), 0.0, 1.0))
    flags = []
    if snr_db < 0.0:
        flags.append("low_pcg_snr")
    if periodicity < 0.25:
        flags.append("weak_heart_sound_periodicity")
    if cycle_cv is not None and cycle_cv > 0.20:
        flags.append("irregular_cycle_timing")
    if dropout_fraction > 0.2:
        flags.append("dropout_or_flatline_segments")
        score *= 0.6
    if saturation_fraction > 0.05:
        flags.append("possible_saturation_or_clipping")
        score *= 0.8
    if events.get("segmentation_confidence", 0.0) < 0.35:
        flags.append("low_segmentation_confidence")
    label = "good" if score >= 0.65 else "usable" if score >= 0.4 else "poor"
    return {
        "tool": "PCG_assess_quality",
        "source": data.source,
        **base,
        "pcg_quality_score": float(np.clip(score, 0.0, 1.0)),
        "pcg_quality_label": label,
        "pcg_snr_db": snr_db,
        "cycle_seconds": cycle_s,
        "periodicity_confidence": periodicity,
        "dropout_fraction": dropout_fraction,
        "saturation_fraction": saturation_fraction,
        "cycle_duration_cv": cycle_cv,
        "segmentation_confidence": events.get("segmentation_confidence"),
        "pcg_quality_flags": flags,
        "method": "pcg_bandpower_periodicity_segmentation_quality_assessment",
    }


def PCG_detect_heart_sounds(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = _load_pcg_signal_data(signal_path, sampling_rate, column)
    events = _pcg_best_segmentation_events(data.values, data.sampling_rate)
    s1_indices = events["s1_indices"]
    s2_indices = events["s2_indices"]
    sound_indices = events["sound_indices"]
    fiducial_payload = _pcg_s1_s2_fiducials(data.values, events.get("envelope", np.asarray([])), data.sampling_rate, s1_indices, s2_indices)
    heart_rate = bpm_from_peaks(s1_indices, data.sampling_rate) if len(s1_indices) >= 2 else bpm_from_peaks(sound_indices[::2], data.sampling_rate)
    regularity = interval_regularity(s1_indices if len(s1_indices) >= 2 else sound_indices[::2], data.sampling_rate)
    confidence = float(events.get("segmentation_confidence", 0.2)) if heart_rate is not None and 35 <= heart_rate <= 220 else 0.2
    return {
        "tool": "PCG_detect_heart_sounds",
        "fiducials": fiducial_payload["fiducials"],
        "s1_points": fiducial_payload["s1_points"],
        "s2_points": fiducial_payload["s2_points"],
        "missing_fiducial_fraction": fiducial_payload["missing_fiducial_fraction"],
        "sound_indices": sound_indices.tolist(),
        "s1_indices": s1_indices.tolist(),
        "s2_indices": s2_indices.tolist(),
        "num_sounds": int(len(sound_indices)),
        "num_s1": int(len(events["s1_indices"])),
        "num_s2": int(len(events["s2_indices"])),
        "heart_rate_bpm": heart_rate,
        "cycle_seconds": events.get("cycle_seconds"),
        "periodicity_confidence": events.get("periodicity_confidence"),
        "confidence": confidence,
        **regularity,
        "method": events.get("segmentation_method", "duration_constrained_hilbert_envelope_segmentation"),
        "segmentation_model_source": events.get("segmentation_model_source"),
        "segmentation_model_metrics": events.get("segmentation_model_metrics"),
        "reference_method": "Springer/Schmidt-style HSMM-inspired duration-constrained PCG state segmentation",
    }



def PCG_screen_murmur_proxy(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = _load_pcg_signal_data(signal_path, sampling_rate, column)
    values = data.values
    if len(values) < max(8, int(data.sampling_rate)):
        return {"tool": "PCG_screen_murmur_proxy", "error": "signal too short", "confidence": 0.0}
    high = min(400.0, data.sampling_rate * 0.45)
    freqs, psd = scipy_signal.welch(values, fs=data.sampling_rate, nperseg=min(len(values), int(data.sampling_rate * 2)))
    low_mask = (freqs >= 20) & (freqs < min(150, high))
    high_mask = (freqs >= 150) & (freqs <= high)
    low_power = float(np.trapezoid(psd[low_mask], freqs[low_mask])) if np.any(low_mask) else 0.0
    high_power = float(np.trapezoid(psd[high_mask], freqs[high_mask])) if np.any(high_mask) else 0.0
    high_frequency_ratio = float(high_power / (low_power + high_power + 1e-12))
    events = _pcg_best_segmentation_events(values, data.sampling_rate)
    envelope = events["envelope"]
    continuous_fraction = float(np.mean(envelope > np.nanpercentile(envelope, 60))) if len(envelope) else 0.0
    feature_result = PCG_extract_murmur_features(signal_path, sampling_rate, column)
    systolic_score = float(feature_result.get("systolic_murmur_score") or 0.0)
    diastolic_score = float(feature_result.get("diastolic_murmur_score") or 0.0)
    score = min(1.0, max(high_frequency_ratio * 1.5 + max(0.0, continuous_fraction - 0.4), systolic_score, diastolic_score))
    murmur_risk = "possible_murmur_proxy" if score >= 0.55 else "no_murmur_proxy"
    result = {
        "tool": "PCG_screen_murmur_proxy",
        "high_frequency_ratio": high_frequency_ratio,
        "continuous_sound_fraction": continuous_fraction,
        "murmur_proxy_score": float(score),
        "murmur_risk": murmur_risk,
        "systolic_murmur_score": systolic_score,
        "diastolic_murmur_score": diastolic_score,
        "murmur_timing_pattern": feature_result.get("murmur_timing_pattern"),
        "cycle_seconds": events.get("cycle_seconds"),
        "segmentation_confidence": events.get("segmentation_confidence"),
        "confidence": 0.5,
        "method": "pcg_high_frequency_continuity_screening_with_duration_constrained_segmentation",
        "disclaimer": "Screening heuristic only; murmur detection requires validated PCG segmentation and labeled clinical data.",
    }
    model_result = _predict_pcg_murmur_model(signal_path, sampling_rate, column)
    if model_result is not None:
        result.update(model_result)
        if model_result.get("murmur_model_prediction") == "abnormal":
            result["murmur_risk"] = "possible_murmur_proxy"
            result["confidence"] = max(result["confidence"], 0.68)
        elif model_result.get("murmur_model_prediction") == "normal":
            if result.get("murmur_proxy_score", 0.0) >= 0.70:
                result["murmur_risk"] = "possible_murmur_proxy"
                result["model_heuristic_disagreement"] = "model_normal_but_segmentation_score_high"
                result["confidence"] = max(result["confidence"], 0.58)
            else:
                result["murmur_risk"] = "no_murmur_proxy"
                result["confidence"] = max(result["confidence"], 0.62)
        result["method"] = "pcg_feature_spectrogram_murmur_classifier_with_duration_constrained_features"
    return result



def PCG_segment_s1_s2_proxy(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = _load_pcg_signal_data(signal_path, sampling_rate, column)
    events = _pcg_best_segmentation_events(data.values, data.sampling_rate)
    s1_indices = events["s1_indices"]
    s2_indices = events["s2_indices"]
    if len(s1_indices) < 2 or len(s2_indices) < 1:
        return {"tool": "PCG_segment_s1_s2_proxy", "error": "not enough heart sounds", "confidence": 0.1, "method": "duration_constrained_hilbert_envelope_segmentation"}
    s1 = np.asarray(s1_indices, dtype=int)
    s2 = np.asarray(s2_indices, dtype=int)
    fiducial_payload = _pcg_s1_s2_fiducials(data.values, events.get("envelope", np.asarray([])), data.sampling_rate, s1, s2)
    pairs = []
    for idx in s1:
        next_s2 = s2[s2 > idx]
        next_s1 = s1[s1 > idx]
        if len(next_s2) and (not len(next_s1) or next_s2[0] < next_s1[0]):
            pairs.append((idx, int(next_s2[0]), int(next_s1[0]) if len(next_s1) else None))
    systoles = [(b - a) / float(data.sampling_rate) for a, b, _ in pairs]
    diastoles = [(c - b) / float(data.sampling_rate) for _, b, c in pairs if c is not None]
    cycles = np.diff(s1) / float(data.sampling_rate) if len(s1) > 1 else np.asarray([])
    systole_cv = float(np.nanstd(systoles) / (np.nanmean(systoles) + 1e-12)) if len(systoles) > 1 else None
    diastole_cv = float(np.nanstd(diastoles) / (np.nanmean(diastoles) + 1e-12)) if len(diastoles) > 1 else None
    cycle_cv = float(np.nanstd(cycles) / (np.nanmean(cycles) + 1e-12)) if len(cycles) > 1 else None
    systolic_coverage_fraction = float(np.sum([(b - a) for a, b, _ in pairs]) / max(1, len(data.values))) if pairs else 0.0
    diastolic_coverage_fraction = float(np.sum([(c - b) for _, b, c in pairs if c is not None]) / max(1, len(data.values))) if pairs else 0.0
    heart_rate = bpm_from_peaks(s1, data.sampling_rate)
    return {
        "tool": "PCG_segment_s1_s2_proxy",
        "fiducials": fiducial_payload["fiducials"],
        "s1_points": fiducial_payload["s1_points"],
        "s2_points": fiducial_payload["s2_points"],
        "missing_fiducial_fraction": fiducial_payload["missing_fiducial_fraction"],
        "s1_indices": s1.tolist(),
        "s2_indices": s2.tolist(),
        "num_s1": int(len(s1)),
        "num_s2": int(len(s2)),
        "systole_duration_s": float(np.nanmedian(systoles)) if len(systoles) else None,
        "diastole_duration_s": float(np.nanmedian(diastoles)) if len(diastoles) else None,
        "systole_duration_cv": systole_cv,
        "diastole_duration_cv": diastole_cv,
        "cycle_duration_cv": cycle_cv,
        "systolic_coverage_fraction": systolic_coverage_fraction,
        "diastolic_coverage_fraction": diastolic_coverage_fraction,
        "cycle_seconds": events.get("cycle_seconds"),
        "heart_rate_bpm": heart_rate,
        "periodicity_confidence": events.get("periodicity_confidence"),
        "confidence": float(events.get("segmentation_confidence", 0.3)),
        "method": events.get("segmentation_method", "duration_constrained_hilbert_envelope_s1_s2_segmentation"),
        "segmentation_model_source": events.get("segmentation_model_source"),
        "segmentation_model_metrics": events.get("segmentation_model_metrics"),
        "reference_method": "Springer/Schmidt-style HSMM-inspired duration-constrained PCG state segmentation",
        "disclaimer": "S1/S2 segmentation uses the Springer-supervised TCN when available and duration-constrained envelope fallback otherwise; clinical-grade segmentation requires labeled validation.",
    }


def PCG_estimate_heart_rate(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = _load_pcg_signal_data(signal_path, sampling_rate, column)
    events = _pcg_best_segmentation_events(data.values, data.sampling_rate)
    s1 = np.asarray(events.get("s1_indices", []), dtype=int)
    sounds = np.asarray(events.get("sound_indices", []), dtype=int)
    beat_indices = s1 if len(s1) >= 2 else sounds[::2]
    heart_rate = bpm_from_peaks(beat_indices, data.sampling_rate)
    intervals = np.diff(beat_indices) / float(data.sampling_rate) if len(beat_indices) > 1 else np.asarray([])
    regularity = interval_regularity(beat_indices, data.sampling_rate)
    confidence = float(events.get("segmentation_confidence", 0.2)) if heart_rate is not None and 35 <= heart_rate <= 220 else 0.15
    return {
        "tool": "PCG_estimate_heart_rate",
        "heart_rate_bpm": heart_rate,
        "cycle_seconds": float(np.nanmedian(intervals)) if len(intervals) else events.get("cycle_seconds"),
        "num_beats_used": int(len(beat_indices)),
        "beat_source": "s1" if len(s1) >= 2 else "alternate_heart_sound",
        "s1_indices": s1[:20].tolist(),
        "confidence": confidence,
        **regularity,
        "method": "pcg_s1_s1_interval_heart_rate_from_" + events.get("segmentation_method", "duration_constrained_segmentation"),
        "segmentation_model_source": events.get("segmentation_model_source"),
        "disclaimer": "PCG heart-rate estimates depend on heart-sound segmentation quality and can fail with noise, motion, or severe murmur.",
    }


def PCG_assess_rhythm_irregularity(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = _load_pcg_signal_data(signal_path, sampling_rate, column)
    events = _pcg_best_segmentation_events(data.values, data.sampling_rate)
    s1 = np.asarray(events.get("s1_indices", []), dtype=int)
    sounds = np.asarray(events.get("sound_indices", []), dtype=int)
    beat_indices = s1 if len(s1) >= 3 else sounds[::2]
    intervals = np.diff(beat_indices) / float(data.sampling_rate) if len(beat_indices) > 1 else np.asarray([])
    if len(intervals) < 3:
        return {"tool": "PCG_assess_rhythm_irregularity", "error": "not enough PCG cycles", "confidence": 0.1, "method": "pcg_cycle_interval_variability_proxy"}
    mean_interval = float(np.nanmean(intervals))
    sdnn_ms = float(np.nanstd(intervals) * 1000.0)
    rmssd_ms = float(np.sqrt(np.nanmean(np.diff(intervals) ** 2)) * 1000.0) if len(intervals) > 1 else None
    cv = float(np.nanstd(intervals) / (mean_interval + 1e-12))
    irregularity_score = float(np.clip((cv - 0.05) / 0.25, 0.0, 1.0))
    flags = []
    if cv >= 0.20:
        flags.append("marked_cycle_irregularity")
    elif cv >= 0.12:
        flags.append("moderate_cycle_irregularity")
    if rmssd_ms is not None and rmssd_ms >= 120.0:
        flags.append("high_short_term_interval_variability")
    risk = "possible_rhythm_irregularity" if irregularity_score >= 0.45 else "no_major_irregularity_proxy"
    return {
        "tool": "PCG_assess_rhythm_irregularity",
        "heart_rate_bpm": bpm_from_peaks(beat_indices, data.sampling_rate),
        "num_cycles": int(len(intervals)),
        "mean_cycle_seconds": mean_interval,
        "cycle_duration_cv": cv,
        "sdnn_ms": sdnn_ms,
        "rmssd_ms": rmssd_ms,
        "irregularity_score": irregularity_score,
        "rhythm_irregularity_risk": risk,
        "rhythm_flags": flags,
        "segmentation_confidence": events.get("segmentation_confidence"),
        "confidence": float(min(0.7, max(0.25, events.get("segmentation_confidence", 0.3)))) if len(intervals) >= 5 else 0.3,
        "method": "pcg_s1_interval_variability_from_" + events.get("segmentation_method", "duration_constrained_segmentation"),
        "segmentation_model_source": events.get("segmentation_model_source"),
        "disclaimer": "PCG rhythm irregularity is only an auxiliary screen and cannot replace ECG rhythm diagnosis.",
    }


def PCG_detect_s3_s4_proxy(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = _load_pcg_signal_data(signal_path, sampling_rate, column)
    events = _pcg_best_segmentation_events(data.values, data.sampling_rate)
    envelope = np.asarray(events.get("envelope", []), dtype=float)
    s1 = np.asarray(events.get("s1_indices", []), dtype=int)
    s2 = np.asarray(events.get("s2_indices", []), dtype=int)
    if len(s1) < 3 or len(s2) < 2 or len(envelope) == 0:
        return {"tool": "PCG_detect_s3_s4_proxy", "error": "not enough S1/S2 cycles", "confidence": 0.1, "method": "diastolic_extra_sound_envelope_proxy"}
    threshold = float(np.nanpercentile(envelope, 72.0))
    refractory = int(round(0.06 * data.sampling_rate))
    s3_candidates: list[int] = []
    s4_candidates: list[int] = []
    for i, start in enumerate(s1[:-1]):
        next_s1 = int(s1[i + 1])
        next_s2 = s2[(s2 > start) & (s2 < next_s1)]
        if not len(next_s2):
            continue
        s2_i = int(next_s2[0])
        early_a = s2_i + int(round(0.08 * data.sampling_rate))
        early_b = min(next_s1 - refractory, s2_i + int(round(0.28 * data.sampling_rate)))
        late_a = max(s2_i + refractory, next_s1 - int(round(0.24 * data.sampling_rate)))
        late_b = next_s1 - int(round(0.05 * data.sampling_rate))
        for a, b, bucket in ((early_a, early_b, s3_candidates), (late_a, late_b, s4_candidates)):
            if b <= a or a < 0 or b > len(envelope):
                continue
            seg = envelope[a:b]
            if len(seg) and float(np.nanmax(seg)) >= threshold:
                bucket.append(int(a + np.nanargmax(seg)))
    s3_score = float(np.clip(len(s3_candidates) / max(1, len(s1) - 1), 0.0, 1.0))
    s4_score = float(np.clip(len(s4_candidates) / max(1, len(s1) - 1), 0.0, 1.0))
    flags = []
    if s3_score >= 0.35:
        flags.append("possible_s3_extra_sound")
    if s4_score >= 0.35:
        flags.append("possible_s4_extra_sound")
    return {
        "tool": "PCG_detect_s3_s4_proxy",
        "s3_candidate_indices": s3_candidates[:20],
        "s4_candidate_indices": s4_candidates[:20],
        "num_s3_candidates": int(len(s3_candidates)),
        "num_s4_candidates": int(len(s4_candidates)),
        "s3_proxy_score": s3_score,
        "s4_proxy_score": s4_score,
        "extra_sound_flags": flags,
        "segmentation_confidence": events.get("segmentation_confidence"),
        "confidence": float(min(0.55, max(0.2, events.get("segmentation_confidence", 0.25) * 0.65))),
        "method": "diastolic_extra_heart_sound_envelope_timing_proxy",
        "disclaimer": "S3/S4 detection is an unvalidated timing/envelope proxy; true extra-heart-sound detection needs labeled PCG data and expert validation.",
    }




def _load_pcg_valve_deep_model() -> dict | None:
    global _PCG_VALVE_DEEP_MODEL_CACHE
    if torch is None or nn is None:
        return None
    if _PCG_VALVE_DEEP_MODEL_CACHE is None:
        members = []
        for model_path in PCG_VALVE_DEEP_MODEL_PATHS:
            if not model_path.exists():
                continue
            payload = torch.load(model_path, map_location="cpu")
            labels = list(payload.get('labels', ['AS', 'AR', 'MR', 'MS', 'N']))
            model = _PCGValveSpecCNN(len(labels))
            model.load_state_dict(payload['model_state_dict'])
            model.eval()
            members.append({'payload': payload, 'model': model, 'path': str(model_path)})
        if not members:
            return None
        _PCG_VALVE_DEEP_MODEL_CACHE = {'members': members}
    return _PCG_VALVE_DEEP_MODEL_CACHE


def _predict_pcg_valve_deep_model(signal_path: str, sampling_rate: float | None, column: str | None = None) -> dict | None:
    bundle = _load_pcg_valve_deep_model()
    if bundle is None:
        return None
    try:
        members = bundle['members']
        first_payload = members[0]['payload']
        labels = list(first_payload.get('labels', ['AS', 'AR', 'MR', 'MS', 'N']))
        target_fs = int(first_payload.get('target_fs', 1000))
        seconds = float(first_payload.get('seconds', 12.0))
        freq_bins = int(first_payload.get('freq_bins', 80))
        time_bins = int(first_payload.get('time_bins', 128))
        fs, values = _load_pcg_signal_for_multisite(signal_path, sampling_rate, column)
        values = _resample_pcg_values(values, fs, target_fs)
        values = _center_crop_or_pad(values, int(round(target_fs * seconds)))
        img = _pcg_multisite_spec_image(values, target_fs, freq_bins, time_bins)
        x = torch.from_numpy(img[None, None, :, :].astype(np.float32))
        prob_rows = []
        threshold_rows = []
        metric_rows = []
        with torch.no_grad():
            for member in members:
                payload = member['payload']
                probs = torch.sigmoid(member['model'](x)).detach().cpu().numpy()[0]
                prob_rows.append(probs)
                val_metrics = payload.get('val_metrics', {})
                threshold_rows.append([float(val_metrics.get(lab, {}).get('threshold', 0.5)) for lab in labels])
                metric_rows.append(val_metrics)
        mean_probs = np.mean(np.vstack(prob_rows), axis=0)
        mean_thresholds = np.mean(np.vstack(threshold_rows), axis=0)
        probabilities = {lab: float(mean_probs[i]) for i, lab in enumerate(labels)}
        thresholds = {lab: float(mean_thresholds[i]) for i, lab in enumerate(labels)}
        predictions = {lab: bool(probabilities[lab] >= thresholds[lab]) for lab in labels}
        positives = [lab for lab, pred in predictions.items() if pred]
        return {
            'valve_deep_model_probabilities': probabilities,
            'valve_deep_model_thresholds': thresholds,
            'valve_deep_model_predictions': predictions,
            'valve_deep_model_positive_labels': positives,
            'valve_deep_model_source': [member['path'] for member in members],
            'valve_deep_model_val_metrics': {'fold_metrics': metric_rows, 'mean_macro_f1': 0.6375593188659422, 'mean_macro_auroc': 0.7541562445673199},
            'valve_deep_model_reference': first_payload.get('reference'),
        }
    except Exception as exc:
        return {'valve_deep_model_error': f'{type(exc).__name__}:{str(exc)[:160]}'}

def _load_pcg_valve_model() -> dict | None:
    global _PCG_VALVE_MODEL_CACHE
    if joblib is None or not PCG_VALVE_MODEL_PATH.exists():
        return None
    if _PCG_VALVE_MODEL_CACHE is None:
        _PCG_VALVE_MODEL_CACHE = joblib.load(PCG_VALVE_MODEL_PATH)
    return _PCG_VALVE_MODEL_CACHE


def _infer_bmdhs_site_posture(signal_path: str) -> tuple[str, str]:
    name = Path(str(signal_path)).stem
    parts = name.split('_')
    posture = parts[2] if len(parts) > 3 else 'unknown'
    site = parts[3] if len(parts) > 3 else 'unknown'
    return site, posture


def _predict_pcg_valve_model(signal_path: str, sampling_rate: float, column: str | None = None) -> dict | None:
    payload = _load_pcg_valve_model()
    if payload is None:
        return None
    try:
        labels = list(payload.get('labels', []))
        numeric_features = list(payload.get('numeric_features', []))
        categorical_features = list(payload.get('categorical_features', []))
        features = PCG_extract_murmur_features(signal_path, sampling_rate, column)
        row = {}
        for name in numeric_features:
            value = features.get(name)
            row[name] = np.nan if value is None else float(value)
        site, posture = _infer_bmdhs_site_posture(signal_path)
        if 'site' in categorical_features:
            row['site'] = site
        if 'posture' in categorical_features:
            row['posture'] = posture
        import pandas as pd
        x = pd.DataFrame([row], columns=numeric_features + categorical_features)
        probabilities = {}
        predictions = {}
        thresholds = {}
        cv_metrics = {}
        for label in labels:
            model = payload['models'][label]
            probability = float(model.predict_proba(x)[0, 1])
            metric = payload.get('report', {}).get('targets', {}).get(label, {}).get('cv_metrics', {})
            threshold = float(metric.get('threshold', 0.5))
            probabilities[label] = probability
            thresholds[label] = threshold
            predictions[label] = bool(probability >= threshold)
            cv_metrics[label] = metric
        positive = [label for label, is_pos in predictions.items() if is_pos]
        return {
            'valve_model_probabilities': probabilities,
            'valve_model_thresholds': thresholds,
            'valve_model_positive_labels': positive,
            'valve_model_predictions': predictions,
            'valve_model_source': str(PCG_VALVE_MODEL_PATH),
            'valve_model_cv_metrics': cv_metrics,
            'valve_model_reference': payload.get('reference'),
        }
    except Exception as exc:
        return {'valve_model_error': f'{type(exc).__name__}:{str(exc)[:160]}'}



def _infer_pcg_circor_location(signal_path: str) -> str:
    stem = Path(str(signal_path)).stem
    for loc in PCG_PATIENT_MULTISITE_LOCATIONS:
        if stem.endswith('_' + loc) or ('_' + loc + '_') in stem or stem.lower().endswith('_' + loc.lower()):
            return loc
    site, _ = _infer_bmdhs_site_posture(signal_path)
    mapping = {'Aor': 'AV', 'Pul': 'PV', 'Tri': 'TV', 'Mit': 'MV'}
    return mapping.get(site, site if site in PCG_PATIENT_MULTISITE_LOCATIONS else '')


def _load_pcg_outcome_feature_model() -> dict | None:
    global _PCG_OUTCOME_FEATURE_MODEL_CACHE
    if joblib is None or not PCG_OUTCOME_FEATURE_MODEL_PATH.exists():
        return None
    if _PCG_OUTCOME_FEATURE_MODEL_CACHE is None:
        _PCG_OUTCOME_FEATURE_MODEL_CACHE = joblib.load(PCG_OUTCOME_FEATURE_MODEL_PATH)
    return _PCG_OUTCOME_FEATURE_MODEL_CACHE


def _pcg_circor_outcome_feature_row(signal_path: str, sampling_rate: float | None, column: str | None = None) -> dict[str, float]:
    fs, values = _load_pcg_signal_for_multisite(signal_path, sampling_rate, column)
    duration = len(values) / float(fs) if fs else 0.0
    high = min(800.0, fs * 0.45)
    if len(values) >= fs and high > 30:
        sos = scipy_signal.butter(3, [20.0 / (0.5 * fs), high / (0.5 * fs)], btype='bandpass', output='sos')
        filt = scipy_signal.sosfiltfilt(sos, values)
    else:
        filt = values
    env = np.abs(scipy_signal.hilbert(filt)) if len(filt) else np.asarray([])
    freqs, psd = scipy_signal.welch(filt, fs=fs, nperseg=min(len(filt), int(fs * 2))) if len(filt) > 16 else (np.asarray([]), np.asarray([]))
    total = float(np.trapezoid(psd, freqs) + 1e-12) if len(freqs) else 1e-12
    def band(low: float, hi: float) -> float:
        mask = (freqs >= low) & (freqs < min(hi, high))
        return float(np.trapezoid(psd[mask], freqs[mask]) / total) if np.any(mask) else 0.0
    def entropy(power: np.ndarray) -> float:
        p = power / (np.sum(power) + 1e-12)
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)) / np.log2(len(power))) if len(power) > 1 else 0.0
    centroid = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-12)) if len(freqs) else 0.0
    cdf = np.cumsum(psd) if len(psd) else np.asarray([])
    rolloff = float(freqs[np.searchsorted(cdf, 0.85 * cdf[-1])]) if len(freqs) and cdf[-1] > 0 else 0.0
    temporal_cv = 0.0
    if len(filt) > fs:
        nper = max(128, min(len(filt), int(0.5 * fs)))
        _, _, spec = scipy_signal.spectrogram(filt, fs=fs, nperseg=nper, noverlap=nper // 2, mode='magnitude')
        energy = np.sum(spec ** 2, axis=0)
        temporal_cv = float(np.std(energy) / (np.mean(energy) + 1e-12)) if len(energy) else 0.0
    peaks = np.asarray([], dtype=int)
    if len(env):
        peaks, _ = scipy_signal.find_peaks(env, distance=max(1, int(0.18 * fs)), prominence=max(float(np.std(env)) * 0.25, 1e-8))
    intervals = np.diff(peaks) / float(fs) if len(peaks) > 1 else np.asarray([])
    loc = _infer_pcg_circor_location(signal_path)
    return {
        'duration_s': float(duration),
        'rms': float(np.sqrt(np.mean(filt ** 2))) if len(filt) else 0.0,
        'zcr': float(np.mean(np.diff(np.signbit(filt)) != 0)) if len(filt) > 1 else 0.0,
        'envelope_cv': float(np.std(env) / (np.mean(env) + 1e-12)) if len(env) else 0.0,
        'envelope_p95_p50': float(np.percentile(env, 95) / (np.percentile(env, 50) + 1e-12)) if len(env) else 0.0,
        'envelope_p99_p50': float(np.percentile(env, 99) / (np.percentile(env, 50) + 1e-12)) if len(env) else 0.0,
        'spectral_centroid': centroid,
        'spectral_entropy': entropy(psd) if len(psd) else 0.0,
        'rolloff85': rolloff,
        'band_20_60': band(20, 60),
        'band_60_150': band(60, 150),
        'band_150_400': band(150, 400),
        'band_400_800': band(400, 800),
        'temporal_energy_cv': temporal_cv,
        'heart_sound_rate': float(len(peaks) / duration * 60.0) if duration > 0 else 0.0,
        'peak_interval_cv': float(np.std(intervals) / (np.mean(intervals) + 1e-12)) if len(intervals) else 0.0,
        'loc_AV': float(loc == 'AV'), 'loc_PV': float(loc == 'PV'), 'loc_TV': float(loc == 'TV'), 'loc_MV': float(loc == 'MV'), 'loc_Phc': float(loc == 'Phc'),
    }


def _predict_pcg_outcome_feature_model(signal_path: str, sampling_rate: float | None, column: str | None = None) -> dict | None:
    payload = _load_pcg_outcome_feature_model()
    if payload is None:
        return None
    try:
        feature_names = list(payload.get('feature_names', []))
        row = _pcg_circor_outcome_feature_row(signal_path, sampling_rate, column)
        x = np.asarray([[np.nan if row.get(name) is None else row.get(name) for name in feature_names]], dtype=float)
        probability = float(payload['model'].predict_proba(x)[0, 1])
        threshold = float(payload.get('cv_metrics', {}).get('patient_metrics', {}).get('threshold', 0.5))
        return {
            'outcome_feature_model_probability_abnormal': probability,
            'outcome_feature_model_prediction': 'abnormal' if probability >= threshold else 'normal',
            'outcome_feature_model_threshold': threshold,
            'outcome_feature_model_source': str(PCG_OUTCOME_FEATURE_MODEL_PATH),
            'outcome_feature_model_name': payload.get('model_name'),
            'outcome_feature_model_cv_metrics': payload.get('cv_metrics'),
            'outcome_feature_model_reference': payload.get('reference'),
        }
    except Exception as exc:
        return {'outcome_feature_model_error': f'{type(exc).__name__}:{str(exc)[:160]}'}


def _load_pcg_outcome_deep_model() -> dict | None:
    global _PCG_OUTCOME_DEEP_MODEL_CACHE
    if torch is None or nn is None or not PCG_OUTCOME_DEEP_MODEL_PATH.exists():
        return None
    if _PCG_OUTCOME_DEEP_MODEL_CACHE is None:
        payload = torch.load(PCG_OUTCOME_DEEP_MODEL_PATH, map_location='cpu')
        model = _PCGPatientMultiLocCNN(int(payload.get('embedding_dim', 128)), len(payload.get('locations', PCG_PATIENT_MULTISITE_LOCATIONS)))
        model.load_state_dict(payload['model_state_dict'])
        model.eval()
        _PCG_OUTCOME_DEEP_MODEL_CACHE = {'payload': payload, 'model': model}
    return _PCG_OUTCOME_DEEP_MODEL_CACHE


def _predict_pcg_outcome_deep_model(signal_path: str, sampling_rate: float | None, column: str | None = None) -> dict | None:
    bundle = _load_pcg_outcome_deep_model()
    if bundle is None:
        return None
    try:
        payload = bundle['payload']; model = bundle['model']
        locations = list(payload.get('locations', PCG_PATIENT_MULTISITE_LOCATIONS))
        loc = _infer_pcg_circor_location(signal_path)
        target_fs = int(payload.get('target_fs', 1000)); seconds = float(payload.get('seconds', 8.0))
        freq_bins = int(payload.get('freq_bins', 80)); time_bins = int(payload.get('time_bins', 128))
        fs, values = _load_pcg_signal_for_multisite(signal_path, sampling_rate, column)
        values = _resample_pcg_values(values, fs, target_fs)
        values = _center_crop_or_pad(values, int(round(target_fs * seconds)))
        img = _pcg_multisite_spec_image(values, target_fs, freq_bins, time_bins)
        xs = []; mask = []
        for known in locations:
            if loc and known.lower() == loc.lower():
                xs.append(img); mask.append(1.0)
            else:
                xs.append(np.zeros((freq_bins, time_bins), dtype=np.float32)); mask.append(0.0)
        if not any(mask):
            xs[0] = img; mask[0] = 1.0
        x = torch.from_numpy(np.stack(xs, axis=0)[None, :, None, :, :].astype(np.float32))
        mask_tensor = torch.tensor([mask], dtype=torch.float32)
        with torch.no_grad():
            probability = float(torch.softmax(model(x, mask_tensor), dim=1)[0, 1].item())
        threshold = 0.5
        return {
            'outcome_deep_model_probability_abnormal': probability,
            'outcome_deep_model_prediction': 'abnormal' if probability >= threshold else 'normal',
            'outcome_deep_model_threshold': threshold,
            'outcome_deep_model_source': str(PCG_OUTCOME_DEEP_MODEL_PATH),
            'outcome_deep_model_cv_metrics': payload.get('cv_metrics'),
            'outcome_deep_model_reference': payload.get('reference'),
        }
    except Exception as exc:
        return {'outcome_deep_model_error': f'{type(exc).__name__}:{str(exc)[:160]}'}

def PCG_screen_valve_disease_proxy(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    murmur = PCG_screen_murmur_proxy(signal_path, sampling_rate, column)
    features = PCG_extract_murmur_features(signal_path, sampling_rate, column)
    systolic = float(features.get("systolic_murmur_score") or 0.0)
    diastolic = float(features.get("diastolic_murmur_score") or 0.0)
    model_prob = murmur.get("murmur_model_probability_abnormal")
    base_score = max(systolic, diastolic, float(model_prob) if model_prob is not None else 0.0)
    pattern = features.get("murmur_timing_pattern")
    if pattern == "diastolic_dominant":
        candidates = ["aortic_regurgitation_pattern", "mitral_stenosis_pattern"]
    elif pattern == "systolic_dominant":
        candidates = ["aortic_stenosis_pattern", "mitral_regurgitation_pattern", "tricuspid_regurgitation_pattern", "ventricular_septal_defect_pattern"]
    else:
        candidates = ["non_specific_murmur_pattern"] if base_score >= 0.45 else []
    risk = "possible_valvular_murmur_pattern" if base_score >= 0.55 else "no_strong_valve_disease_proxy"
    result = {
        "tool": "PCG_screen_valve_disease_proxy",
        "valve_disease_proxy_score": float(np.clip(base_score, 0.0, 1.0)),
        "valve_pattern_candidates": candidates,
        "murmur_timing_pattern": pattern,
        "systolic_murmur_score": systolic,
        "diastolic_murmur_score": diastolic,
        "murmur_result": murmur,
        "valve_screen_risk": risk,
        "confidence": float(min(0.62, max(0.25, features.get("confidence", 0.35)))),
        "method": "pcg_murmur_timing_valve_disease_pattern_proxy",
        "disclaimer": "This does not diagnose valve disease or subtype; location, radiation, calibrated auscultation, echo, and clinical context are required.",
    }
    deep_result = _predict_pcg_valve_deep_model(signal_path, sampling_rate, column)
    if deep_result is not None:
        result.update(deep_result)
        positives = deep_result.get("valve_deep_model_positive_labels") or []
        disease_positives = [label for label in positives if label != "N"]
        probabilities = deep_result.get("valve_deep_model_probabilities") or {}
        if disease_positives:
            result["valve_screen_risk"] = "possible_valve_disease_deep_model_positive"
            result["valve_pattern_candidates"] = disease_positives
            result["valve_disease_proxy_score"] = float(max(probabilities.get(label, 0.0) for label in disease_positives))
            result["confidence"] = max(result["confidence"], 0.72)
        elif "N" in positives:
            result["valve_screen_risk"] = "normal_deep_model_positive_no_valve_subtype"
            result["valve_pattern_candidates"] = []
            result["confidence"] = max(result["confidence"], 0.66)
        result["method"] = "bmdhs_patient_heldout_valve_spectrogram_cnn_plus_murmur_timing_proxy"
    model_result = _predict_pcg_valve_model(signal_path, sampling_rate, column)
    if model_result is not None:
        result.update(model_result)
        if deep_result is None:
            positives = model_result.get("valve_model_positive_labels") or []
            disease_positives = [label for label in positives if label != "N"]
            probabilities = model_result.get("valve_model_probabilities") or {}
            if disease_positives:
                result["valve_screen_risk"] = "possible_valve_disease_model_positive"
                result["valve_pattern_candidates"] = disease_positives
                result["valve_disease_proxy_score"] = float(max(probabilities.get(label, 0.0) for label in disease_positives))
                result["confidence"] = max(result["confidence"], 0.66)
            elif "N" in positives:
                result["valve_screen_risk"] = "normal_model_positive_no_valve_subtype"
                result["valve_pattern_candidates"] = []
                result["confidence"] = max(result["confidence"], 0.62)
            result["method"] = "bmdhs_patient_grouped_valve_feature_classifier_plus_murmur_timing_proxy"
    return result


def PCG_screen_congenital_abnormality_proxy(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    murmur = PCG_screen_murmur_proxy(signal_path, sampling_rate, column)
    rhythm = PCG_assess_rhythm_irregularity(signal_path, sampling_rate, column)
    quality = PCG_assess_quality(signal_path, sampling_rate, column)
    outcome_feature = _predict_pcg_outcome_feature_model(signal_path, sampling_rate, column)
    outcome_deep = _predict_pcg_outcome_deep_model(signal_path, sampling_rate, column)
    murmur_score = float(murmur.get("murmur_proxy_score") or murmur.get("murmur_model_probability_abnormal") or 0.0)
    rhythm_score = float(rhythm.get("irregularity_score") or 0.0)
    quality_penalty = 0.15 if quality.get("pcg_quality_label") == "poor" else 0.0
    proxy_score = float(np.clip(0.75 * murmur_score + 0.25 * rhythm_score - quality_penalty, 0.0, 1.0))
    supervised_scores = []
    if isinstance(outcome_feature, dict) and outcome_feature.get("outcome_feature_model_probability_abnormal") is not None:
        supervised_scores.append(float(outcome_feature["outcome_feature_model_probability_abnormal"]))
    if isinstance(outcome_deep, dict) and outcome_deep.get("outcome_deep_model_probability_abnormal") is not None:
        supervised_scores.append(float(outcome_deep["outcome_deep_model_probability_abnormal"]))
    supervised_score = float(np.nanmean(supervised_scores)) if supervised_scores else None
    score = float(np.clip(0.60 * proxy_score + 0.40 * supervised_score, 0.0, 1.0)) if supervised_score is not None else proxy_score
    flags = []
    if murmur_score >= 0.55:
        flags.append("murmur_or_abnormal_sound_pattern")
    if rhythm_score >= 0.45:
        flags.append("irregular_cycle_timing")
    if supervised_score is not None and supervised_score >= 0.5:
        flags.append("circor_outcome_model_abnormal_signal")
    risk = "possible_congenital_or_structural_abnormality_screen_positive" if score >= 0.5 else "no_strong_congenital_abnormality_screen_signal"
    confidence = float(min(0.64, max(0.2, murmur.get("confidence", 0.3))))
    if supervised_score is not None:
        confidence = max(confidence, 0.58)
    return {
        "tool": "PCG_screen_congenital_abnormality_proxy",
        "congenital_abnormality_proxy_score": score,
        "proxy_only_score": proxy_score,
        "supervised_outcome_probability_abnormal": supervised_score,
        "congenital_screen_risk": risk,
        "screen_flags": flags,
        "murmur_result": murmur,
        "rhythm_result": rhythm,
        "quality_result": quality,
        "outcome_feature_model_result": outcome_feature,
        "outcome_deep_model_result": outcome_deep,
        "confidence": confidence,
        "method": "circor_outcome_feature_and_cnn_screen_plus_murmur_rhythm_proxy",
        "disclaimer": "Research screening output for pediatric/structural PCG abnormality; CirCor outcome models have modest AUROC and this is not a CHD diagnosis or substitute for echocardiography/clinical review.",
    }


def _parse_pcg_monitor_recordings(recordings: str, sampling_rate: float | None, column: str | None) -> list[dict]:
    path = Path(str(recordings))
    if path.exists() and path.suffix.lower() == ".json":
        loaded = json.loads(path.read_text())
    else:
        try:
            loaded = json.loads(str(recordings))
        except Exception:
            loaded = [{"signal_path": str(recordings), "sampling_rate": sampling_rate, "column": column}]
    if isinstance(loaded, dict) and "records" in loaded:
        loaded = loaded["records"]
    if isinstance(loaded, dict):
        loaded = [loaded]
    if not isinstance(loaded, list):
        raise ValueError("recordings must be a signal path, JSON list, or JSON file")
    return loaded


def PCG_monitor_heart_function_proxy(recordings: str, sampling_rate: float | None = None, column: str | None = None) -> dict:
    items = _parse_pcg_monitor_recordings(recordings, sampling_rate, column)
    summaries = []
    for i, item in enumerate(items):
        signal_path = item.get("signal_path") or item.get("path") if isinstance(item, dict) else str(item)
        if not signal_path:
            continue
        fs = item.get("sampling_rate", sampling_rate) if isinstance(item, dict) else sampling_rate
        col = item.get("column", column) if isinstance(item, dict) else column
        if fs is None and Path(str(signal_path)).suffix.lower() != ".wav":
            continue
        seg = PCG_segment_s1_s2_proxy(str(signal_path), float(fs or 0.0), col)
        feat = PCG_extract_murmur_features(str(signal_path), float(fs or 0.0), col)
        qual = PCG_assess_quality(str(signal_path), float(fs or 0.0), col)
        summaries.append({
            "record_index": i,
            "source": str(signal_path),
            "timestamp": item.get("timestamp") if isinstance(item, dict) else None,
            "heart_rate_bpm": seg.get("heart_rate_bpm"),
            "systole_duration_s": seg.get("systole_duration_s"),
            "diastole_duration_s": seg.get("diastole_duration_s"),
            "cycle_duration_cv": seg.get("cycle_duration_cv"),
            "systolic_murmur_score": feat.get("systolic_murmur_score"),
            "diastolic_murmur_score": feat.get("diastolic_murmur_score"),
            "pcg_quality_label": qual.get("pcg_quality_label"),
            "segmentation_confidence": seg.get("confidence"),
        })
    if not summaries:
        return {"tool": "PCG_monitor_heart_function_proxy", "error": "no usable PCG recordings", "confidence": 0.0}
    def vals(name: str) -> np.ndarray:
        return np.asarray([x[name] for x in summaries if x.get(name) is not None], dtype=float)
    hr = vals("heart_rate_bpm")
    murmur_scores = np.asarray([max(float(x.get("systolic_murmur_score") or 0.0), float(x.get("diastolic_murmur_score") or 0.0)) for x in summaries], dtype=float)
    syst = vals("systole_duration_s")
    trend_flags = []
    if len(hr) >= 2 and abs(float(hr[-1] - hr[0])) >= 20.0:
        trend_flags.append("large_heart_rate_change")
    if len(murmur_scores) >= 2 and float(murmur_scores[-1] - murmur_scores[0]) >= 0.2:
        trend_flags.append("increasing_murmur_proxy_score")
    if len(syst) >= 2 and abs(float(syst[-1] - syst[0])) >= 0.06:
        trend_flags.append("changed_systolic_timing_proxy")
    risk = "possible_change_in_pcg_function_features" if trend_flags or float(np.nanmax(murmur_scores)) >= 0.65 else "no_major_change_proxy"
    return {
        "tool": "PCG_monitor_heart_function_proxy",
        "num_recordings": int(len(summaries)),
        "record_summaries": summaries,
        "heart_rate_range_bpm": [float(np.nanmin(hr)), float(np.nanmax(hr))] if len(hr) else None,
        "max_murmur_proxy_score": float(np.nanmax(murmur_scores)) if len(murmur_scores) else None,
        "trend_flags": trend_flags,
        "heart_function_monitoring_risk": risk,
        "confidence": 0.45 if len(summaries) == 1 else 0.58,
        "method": "longitudinal_pcg_interval_intensity_murmur_feature_monitoring_proxy",
        "disclaimer": "Heart-function monitoring from PCG is a longitudinal feature proxy, not a diagnosis; validated cohorts and clinical endpoints are needed.",
    }




def _pcg_state_segments(s1_indices: np.ndarray, s2_indices: np.ndarray, n_samples: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    s1 = np.asarray(s1_indices, dtype=int)
    s2 = np.asarray(s2_indices, dtype=int)
    s1 = s1[(s1 >= 0) & (s1 < n_samples)]
    s2 = s2[(s2 >= 0) & (s2 < n_samples)]
    systole: list[tuple[int, int]] = []
    diastole: list[tuple[int, int]] = []
    for start in s1:
        next_s2 = s2[s2 > start]
        next_s1 = s1[s1 > start]
        if len(next_s2) and (not len(next_s1) or next_s2[0] < next_s1[0]):
            end_s2 = int(next_s2[0])
            if end_s2 > start:
                systole.append((int(start), min(n_samples, end_s2)))
            if len(next_s1) and int(next_s1[0]) > end_s2:
                diastole.append((end_s2, min(n_samples, int(next_s1[0]))))
    return systole, diastole


def _concat_segments(values: np.ndarray, segments: list[tuple[int, int]], min_samples: int) -> np.ndarray:
    chunks = [values[max(0, a):min(len(values), b)] for a, b in segments if b > a]
    chunks = [chunk for chunk in chunks if len(chunk) >= max(4, min_samples)]
    return np.concatenate(chunks) if chunks else np.asarray([], dtype=float)


def _pcg_band_ratios_for_values(values: np.ndarray, sampling_rate: float, high_limit: float = 500.0) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < max(16, int(0.15 * sampling_rate)):
        return {
            "low_band_power": 0.0,
            "mid_band_power": 0.0,
            "high_band_power": 0.0,
            "very_high_band_power": 0.0,
            "mid_band_ratio": 0.0,
            "high_band_ratio": 0.0,
            "spectral_centroid_hz": None,
        }
    high = min(float(high_limit), sampling_rate * 0.45)
    freqs, psd = scipy_signal.welch(values, fs=sampling_rate, nperseg=min(len(values), int(sampling_rate * 2)))
    def band_power(low: float, high_band: float) -> float:
        mask = (freqs >= low) & (freqs < min(high_band, high))
        return float(np.trapezoid(psd[mask], freqs[mask])) if np.any(mask) else 0.0
    low_power = band_power(20.0, 60.0)
    mid_power = band_power(60.0, 150.0)
    high_power = band_power(150.0, 400.0)
    very_high_power = band_power(400.0, high)
    total = low_power + mid_power + high_power + very_high_power + 1e-12
    centroid = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-12)) if len(freqs) else None
    return {
        "low_band_power": low_power,
        "mid_band_power": mid_power,
        "high_band_power": high_power,
        "very_high_band_power": very_high_power,
        "mid_band_ratio": float(mid_power / total),
        "high_band_ratio": float((high_power + very_high_power) / total),
        "spectral_centroid_hz": centroid,
    }


def _pcg_segment_continuity(envelope: np.ndarray, segments: list[tuple[int, int]], percentile: float = 65.0) -> float | None:
    if len(envelope) == 0 or not segments:
        return None
    threshold = float(np.nanpercentile(envelope, percentile))
    values = []
    for a, b in segments:
        seg = envelope[max(0, a):min(len(envelope), b)]
        if len(seg):
            values.append(float(np.mean(seg > threshold)))
    return float(np.nanmedian(values)) if values else None

def PCG_extract_murmur_features(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = _load_pcg_signal_data(signal_path, sampling_rate, column)
    values = np.asarray(data.values, dtype=float)
    if len(values) < max(8, int(data.sampling_rate)):
        return {"tool": "PCG_extract_murmur_features", "error": "signal too short", "confidence": 0.0}
    centered = values - np.nanmedian(values)
    high = min(500.0, data.sampling_rate * 0.45)
    low_cut = 20.0 if high > 40 else max(1.0, data.sampling_rate * 0.05)
    filtered = bandpass_filter(centered, data.sampling_rate, low_cut, high, order=3) if high > low_cut else centered
    envelope = np.abs(scipy_signal.hilbert(filtered))
    global_features = _pcg_band_ratios_for_values(filtered, data.sampling_rate, high_limit=500.0)
    freqs, psd = scipy_signal.welch(filtered, fs=data.sampling_rate, nperseg=min(len(filtered), int(data.sampling_rate * 2)))
    psd_norm = psd / (np.sum(psd) + 1e-12) if len(psd) else np.array([])
    spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)) / np.log2(len(psd_norm))) if len(psd_norm) > 1 else 0.0
    zcr = float(np.mean(np.diff(np.signbit(filtered)) != 0)) if len(filtered) > 1 else 0.0
    env_median = float(np.nanmedian(envelope)) if len(envelope) else 0.0
    env_p90 = float(np.nanpercentile(envelope, 90)) if len(envelope) else 0.0
    env_p95 = float(np.nanpercentile(envelope, 95)) if len(envelope) else 0.0
    env_p99 = float(np.nanpercentile(envelope, 99)) if len(envelope) else 0.0
    env_std = float(np.nanstd(envelope)) if len(envelope) else 0.0
    continuous_fraction_60 = float(np.mean(envelope > np.nanpercentile(envelope, 60))) if len(envelope) else 0.0
    continuous_fraction_75 = float(np.mean(envelope > np.nanpercentile(envelope, 75))) if len(envelope) else 0.0

    events = _pcg_best_segmentation_events(values, data.sampling_rate)
    s1 = np.asarray(events.get("s1_indices", []), dtype=int)
    s2 = np.asarray(events.get("s2_indices", []), dtype=int)
    systole_segments, diastole_segments = _pcg_state_segments(s1, s2, len(filtered))
    min_seg = max(4, int(0.05 * data.sampling_rate))
    systole_values = _concat_segments(filtered, systole_segments, min_seg)
    diastole_values = _concat_segments(filtered, diastole_segments, min_seg)
    systole_features = _pcg_band_ratios_for_values(systole_values, data.sampling_rate, high_limit=500.0)
    diastole_features = _pcg_band_ratios_for_values(diastole_values, data.sampling_rate, high_limit=500.0)
    systolic_continuity = _pcg_segment_continuity(envelope, systole_segments, percentile=65.0)
    diastolic_continuity = _pcg_segment_continuity(envelope, diastole_segments, percentile=65.0)
    systolic_murmur_score = float(min(1.0, 1.5 * systole_features["high_band_ratio"] + 0.8 * (systolic_continuity or 0.0)))
    diastolic_murmur_score = float(min(1.0, 1.7 * diastole_features["high_band_ratio"] + 0.8 * (diastolic_continuity or 0.0)))
    timing_pattern = "diastolic_dominant" if diastolic_murmur_score > systolic_murmur_score + 0.12 else "systolic_dominant" if systolic_murmur_score > diastolic_murmur_score + 0.12 else "mixed_or_uncertain"

    peaks = np.asarray(events.get("sound_indices", []), dtype=int)
    intervals = np.diff(peaks) / float(data.sampling_rate) if len(peaks) > 1 else np.array([])
    interval_cv = float(np.nanstd(intervals) / (np.nanmean(intervals) + 1e-12)) if len(intervals) else None
    result = {
        "tool": "PCG_extract_murmur_features",
        "low_band_power": global_features["low_band_power"],
        "mid_band_power": global_features["mid_band_power"],
        "high_band_power": global_features["high_band_power"],
        "very_high_band_power": global_features["very_high_band_power"],
        "mid_band_ratio": global_features["mid_band_ratio"],
        "high_band_ratio": global_features["high_band_ratio"],
        "spectral_centroid_hz": global_features["spectral_centroid_hz"],
        "spectral_entropy": spectral_entropy,
        "zero_crossing_rate": zcr,
        "envelope_std": env_std,
        "envelope_p90_median_ratio": float(env_p90 / (env_median + 1e-12)),
        "envelope_p95_median_ratio": float(env_p95 / (env_median + 1e-12)),
        "envelope_p99_median_ratio": float(env_p99 / (env_median + 1e-12)),
        "continuous_fraction_60": continuous_fraction_60,
        "continuous_fraction_75": continuous_fraction_75,
        "systolic_high_band_ratio": systole_features["high_band_ratio"],
        "diastolic_high_band_ratio": diastole_features["high_band_ratio"],
        "systolic_mid_band_ratio": systole_features["mid_band_ratio"],
        "diastolic_mid_band_ratio": diastole_features["mid_band_ratio"],
        "systolic_spectral_centroid_hz": systole_features["spectral_centroid_hz"],
        "diastolic_spectral_centroid_hz": diastole_features["spectral_centroid_hz"],
        "systolic_continuous_fraction": systolic_continuity,
        "diastolic_continuous_fraction": diastolic_continuity,
        "systolic_murmur_score": systolic_murmur_score,
        "diastolic_murmur_score": diastolic_murmur_score,
        "murmur_timing_pattern": timing_pattern,
        "num_systolic_segments": int(len(systole_segments)),
        "num_diastolic_segments": int(len(diastole_segments)),
        "segmentation_confidence": events.get("segmentation_confidence"),
        "num_sounds": int(len(peaks)),
        "heart_rate_bpm": bpm_from_peaks(s1, data.sampling_rate) if len(s1) >= 2 else bpm_from_peaks(peaks[::2], data.sampling_rate),
        "sound_interval_cv": interval_cv,
        "confidence": float(min(0.78, max(0.45, events.get("segmentation_confidence", 0.4)))) if len(systole_segments) >= 2 else 0.4,
        "method": "pcg_segmentation_aware_spectral_envelope_murmur_features",
        "disclaimer": "Feature extraction only; classification requires labeled training/evaluation.",
    }
    return result
