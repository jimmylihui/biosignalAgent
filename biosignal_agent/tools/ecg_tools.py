from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None
from scipy import signal as scipy_signal
try:
    from pandas.errors import ChainedAssignmentError
    warnings.filterwarnings("ignore", category=ChainedAssignmentError)
except Exception:  # pragma: no cover
    pass

from .common import bpm_from_peaks, interval_regularity, load_csv_signal, signal_quality_summary
from .peak_detectors import neurokit_nabian2018_peaks

try:
    import neurokit2 as nk
except ImportError:  # pragma: no cover
    nk = None


ECG_ARRHYTHMIA_MODEL_PATH = Path('/data1/jiahui/biosignal-agent/outputs/ecg_arrhythmia_feature_model.joblib')
ECG_APNEA_MODEL_PATH = Path('/data1/jiahui/biosignal-agent/outputs/ecg_apnea_feature_model.joblib')
ECG_ARRHYTHMIA_DEEP_MODEL_PATH = Path('/data1/jiahui/biosignal-agent/outputs/ecg_arrhythmia_1dcnn_model.pt')
ECG_APNEA_DEEP_MODEL_PATH = Path('/data1/jiahui/biosignal-agent/outputs/ecg_apnea_1dcnn_model.pt')
ECG_ARRHYTHMIA_BEAT_MODEL_PATH = Path('/data1/jiahui/biosignal-agent/outputs/ecg_arrhythmia_beat_cnn_model.pt')
ECG_APNEA_RR_EDR_MODEL_PATH = Path('/data1/jiahui/biosignal-agent/outputs/ecg_apnea_rr_edr_cnn_mixed_short_model.pt')
ECG_APNEA_RR_EDR_CONTEXT_MODEL_PATH = Path('/data1/jiahui/biosignal-agent/outputs/ecg_apnea_rr_edr_context_cnn_model.pt')
ECG_APNEA_RR_EDR_SEQUENCE_CONTEXT_MODEL_PATH = Path('/data1/jiahui/biosignal-agent/outputs/ecg_apnea_rr_edr_sequence_context_three_dataset_model.pt')
ECG_ARRHYTHMIA_SUBTYPE_MODEL_PATH = Path('/data1/jiahui/biosignal-agent/outputs/ecg_arrhythmia_abnormal_subtype_mitdb_incart_svdb_cnn_model.pt')
ECG_RPEAK_DEEP_MODEL_PATH = Path('/data1/jiahui/biosignal-agent/outputs/ecg_rpeak_segmentation_cnn_model.pt')
ECG_RHYTHM_MODEL_PATH = Path('/data1/jiahui/biosignal-agent/outputs/ecg_rhythm_feature_classifier_plus_afdb_partial.joblib')
ECG_AF_MODEL_PATH = Path('/data1/jiahui/biosignal-agent/outputs/ecg_af_feature_classifier_partial.joblib')
ECG_DELINEATION_MODEL_PATH = Path('/data1/jiahui/biosignal-agent/outputs/ecg_delineation_qtdb_cached90_unet.pt')
ECG_ST_MODEL_PATH = Path('/data1/jiahui/biosignal-agent/outputs/ecg_st_feature_classifier_edb12.joblib')
ECG_PTBXL_CD_MODEL_PATH = Path('/data1/jiahui/biosignal-agent/outputs/ptbxl_superclass_models/ecg_ptbxl_cd_lead2_feature_classifier.joblib')
ECG_PTBXL_STTC_MODEL_PATH = Path('/data1/jiahui/biosignal-agent/outputs/ptbxl_superclass_models/ecg_ptbxl_sttc_lead2_feature_classifier_balanced780.joblib')
ECG_PTBXL_12LEAD_MODEL_DIR = Path('/data1/jiahui/biosignal-agent/outputs/ptbxl_full_12lead_resnet')
ECG_PTBXL_12LEAD_REPORT_PATH = ECG_PTBXL_12LEAD_MODEL_DIR / 'ecg_ptbxl_full_12lead_resnet_train_report.json'
ECG_PTBXL_12LEAD_TARGETS = ('norm', 'mi', 'sttc', 'cd', 'hyp')
_TORCH_MODEL_CACHE = {}


def _safe_trapz(y: np.ndarray, x: np.ndarray) -> float:
    integrate = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
    return float(integrate(y, x)) if len(y) and len(x) else 0.0


def _robust_resample_ecg(values: np.ndarray, target_len: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.zeros(target_len, dtype=np.float32)
    med = float(np.median(values))
    q75, q25 = np.percentile(values, [75, 25])
    scale = float(q75 - q25)
    if scale < 1e-8:
        scale = float(np.std(values)) + 1e-8
    values = np.clip((values - med) / scale, -8.0, 8.0)
    if len(values) == target_len:
        return values.astype(np.float32)
    return scipy_signal.resample(values, target_len).astype(np.float32)


if nn is not None:
    class _RPeakSegCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, 32, 9, padding=4, bias=False), nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 48, 9, padding=8, dilation=2, bias=False), nn.BatchNorm1d(48), nn.ReLU(),
                nn.Conv1d(48, 64, 9, padding=16, dilation=4, bias=False), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 64, 7, padding=18, dilation=6, bias=False), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 32, 5, padding=2, bias=False), nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 1, 1),
            )

        def forward(self, x):
            if x.ndim == 2:
                x = x[:, None, :]
            return self.net(x).squeeze(1)


if nn is not None:
    class _ECGSubtypeCNN(nn.Module):
        def __init__(self, feature_dim: int = 3, num_classes: int = 4, dropout: float = 0.25):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv1d(1, 32, 9, padding=4, bias=False), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 7, padding=3, bias=False), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(64, 128, 5, padding=2, bias=False), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(128, 192, 3, padding=1, bias=False), nn.BatchNorm1d(192), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            )
            self.head = nn.Sequential(nn.Linear(192 + feature_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, num_classes))

        def forward(self, x, f):
            if x.ndim == 2:
                x = x[:, None, :]
            return self.head(torch.cat([self.cnn(x), f], dim=1))


if nn is not None:
    class _ECGBeatCNN(nn.Module):
        def __init__(self, feature_dim: int = 3, dropout: float = 0.25):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv1d(1, 32, 9, padding=4, bias=False), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 7, padding=3, bias=False), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(64, 128, 5, padding=2, bias=False), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(128, 160, 3, padding=1, bias=False), nn.BatchNorm1d(160), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            )
            self.head = nn.Sequential(nn.Linear(160 + feature_dim, 96), nn.ReLU(), nn.Dropout(dropout), nn.Linear(96, 1))

        def forward(self, x, f):
            if x.ndim == 2:
                x = x[:, None, :]
            z = self.cnn(x)
            return self.head(torch.cat([z, f], dim=1)).squeeze(-1)


if nn is not None:
    class _SeqContextCNN(nn.Module):
        def __init__(self, in_channels: int = 6, dropout: float = 0.35):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv1d(in_channels, 32, 11, padding=5, bias=False), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 9, padding=4, bias=False), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(64, 96, 7, padding=3, bias=False), nn.BatchNorm1d(96), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(96, 128, 5, padding=2, bias=False), nn.BatchNorm1d(128), nn.ReLU(),
            )
            self.rnn = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
            self.head = nn.Sequential(nn.LayerNorm(256), nn.Dropout(dropout), nn.Linear(256, 96), nn.ReLU(), nn.Dropout(dropout), nn.Linear(96, 1))

        def forward(self, x):
            z = self.cnn(x).transpose(1, 2)
            z, _ = self.rnn(z)
            pooled = torch.cat([z.mean(dim=1), z.amax(dim=1)], dim=1)
            return self.head(pooled).squeeze(-1)


if nn is not None:
    class _RREdrCNN(nn.Module):
        def __init__(self, in_channels: int = 2, dropout: float = 0.25, use_lstm: bool = False):
            super().__init__()
            self.use_lstm = use_lstm
            if use_lstm:
                self.cnn = nn.Sequential(
                    nn.Conv1d(in_channels, 32, 9, padding=4, bias=False), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                    nn.Conv1d(32, 64, 7, padding=3, bias=False), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                    nn.Conv1d(64, 96, 5, padding=2, bias=False), nn.BatchNorm1d(96), nn.ReLU(), nn.MaxPool1d(2),
                )
                self.rnn = nn.LSTM(96, 64, batch_first=True, bidirectional=True)
                self.head = nn.Sequential(nn.LayerNorm(256), nn.Dropout(0.30), nn.Linear(256, 96), nn.ReLU(), nn.Dropout(0.30), nn.Linear(96, 1))
            else:
                self.net = nn.Sequential(
                    nn.Conv1d(in_channels, 32, 9, padding=4, bias=False), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                    nn.Conv1d(32, 64, 7, padding=3, bias=False), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                    nn.Conv1d(64, 128, 5, padding=2, bias=False), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
                    nn.Conv1d(128, 128, 3, padding=1, bias=False), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                    nn.Dropout(dropout), nn.Linear(128, 1),
                )

        def forward(self, x):
            if self.use_lstm:
                z = self.cnn(x).transpose(1, 2)
                z, _ = self.rnn(z)
                pooled = torch.cat([z.mean(dim=1), z.amax(dim=1)], dim=1)
                return self.head(pooled).squeeze(-1)
            return self.net(x).squeeze(-1)


    class _ECGTinyCNN(nn.Module):
        def __init__(self, dropout: float = 0.20):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, 24, kernel_size=15, padding=7, bias=False),
                nn.BatchNorm1d(24),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(4),
                nn.Conv1d(24, 48, kernel_size=9, padding=4, bias=False),
                nn.BatchNorm1d(48),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(4),
                nn.Conv1d(48, 96, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm1d(96),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(4),
                nn.Conv1d(96, 128, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            if x.ndim == 2:
                x = x[:, None, :]
            return self.net(x).squeeze(-1)
    class _ECGTwelveLeadResBlock(nn.Module):
        def __init__(self, c_in: int, c_out: int, stride: int = 1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(c_in, c_out, 7, stride=stride, padding=3, bias=False), nn.BatchNorm1d(c_out), nn.ReLU(inplace=True),
                nn.Conv1d(c_out, c_out, 7, padding=3, bias=False), nn.BatchNorm1d(c_out),
            )
            self.skip = nn.Identity() if c_in == c_out and stride == 1 else nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm1d(c_out))
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.act(self.net(x) + self.skip(x))


    class _ECGTwelveLeadResNet(nn.Module):
        def __init__(self, dropout: float = 0.25):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv1d(12, 32, 15, padding=7, bias=False), nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(2))
            self.blocks = nn.Sequential(
                _ECGTwelveLeadResBlock(32, 48, 2),
                _ECGTwelveLeadResBlock(48, 64, 2),
                _ECGTwelveLeadResBlock(64, 96, 2),
                _ECGTwelveLeadResBlock(96, 128, 2),
                _ECGTwelveLeadResBlock(128, 160, 2),
            )
            self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(160, 1))

        def forward(self, x):
            return self.head(self.blocks(self.stem(x))).squeeze(-1)
else:
    _ECGTinyCNN = None
    _ECGBeatCNN = None
    _RREdrCNN = None
    _SeqContextCNN = None
    _ECGSubtypeCNN = None
    _RPeakSegCNN = None
    _ECGTwelveLeadResNet = None


def _robust_beat_segment(values: np.ndarray, center: int, sampling_rate: float, beat_len: int) -> np.ndarray:
    pre = int(0.25 * sampling_rate)
    post = int(0.45 * sampling_rate)
    start = center - pre
    stop = center + post
    seg = np.zeros(pre + post, dtype=np.float32)
    src_start = max(start, 0)
    src_stop = min(stop, len(values))
    dst_start = src_start - start
    dst_stop = dst_start + max(0, src_stop - src_start)
    if src_stop > src_start:
        seg[dst_start:dst_stop] = np.asarray(values[src_start:src_stop], dtype=np.float32)
    med = float(np.median(seg))
    q75, q25 = np.percentile(seg, [75, 25])
    scale = float(q75 - q25)
    if scale < 1e-8:
        scale = float(np.std(seg)) + 1e-8
    seg = np.clip((seg - med) / scale, -8.0, 8.0)
    if len(seg) != beat_len:
        seg = scipy_signal.resample(seg, beat_len)
    return seg.astype(np.float32)


def _predict_arrhythmia_beat_model(model_path: Path, values: np.ndarray, sampling_rate: float, peaks: np.ndarray) -> tuple[float | None, dict | None, str | None, str | None, float | None]:
    if torch is None or _ECGBeatCNN is None or not model_path.exists():
        return None, None, None, None, None
    peaks = np.asarray(peaks, dtype=int)
    peaks = peaks[(peaks > 0) & (peaks < len(values))]
    if len(peaks) < 3:
        return None, None, None, None, None
    try:
        bundle = torch.load(model_path, map_location='cpu', weights_only=False)
        beat_len = int(bundle.get('beat_len', 256))
        feature_dim = int(bundle.get('feature_dim', 3))
        segments = []
        feats = []
        rr_prev = np.r_[np.nan, np.diff(peaks) / float(sampling_rate)]
        rr_next = np.r_[np.diff(peaks) / float(sampling_rate), np.nan]
        med_rr = float(np.nanmedian(np.r_[rr_prev, rr_next])) if len(peaks) > 1 else 0.8
        rr_prev = np.nan_to_num(rr_prev, nan=med_rr, posinf=med_rr, neginf=med_rr)
        rr_next = np.nan_to_num(rr_next, nan=med_rr, posinf=med_rr, neginf=med_rr)
        for i, peak in enumerate(peaks):
            segments.append(_robust_beat_segment(values, int(peak), sampling_rate, beat_len))
            local_hr = 60.0 / max(float((rr_prev[i] + rr_next[i]) / 2.0), 0.25)
            feats.append([float(np.clip(rr_prev[i], 0.25, 3.0)), float(np.clip(rr_next[i], 0.25, 3.0)), float(np.clip(local_hr / 100.0, 0.2, 2.5))])
        X = torch.tensor(np.asarray(segments, dtype=np.float32), dtype=torch.float32)
        F = torch.tensor(np.asarray(feats, dtype=np.float32)[:, :feature_dim], dtype=torch.float32)
        model = _ECGBeatCNN(feature_dim=feature_dim)
        model.load_state_dict(bundle['state_dict'])
        model.eval()
        probs = []
        with torch.no_grad():
            for start in range(0, len(X), 512):
                probs.append(torch.sigmoid(model(X[start:start + 512], F[start:start + 512])).numpy())
        beat_probs = np.concatenate(probs).astype(float)
        subtype_details = _predict_arrhythmia_subtypes(ECG_ARRHYTHMIA_SUBTYPE_MODEL_PATH, X, F, beat_probs, bundle_threshold=float(bundle.get('threshold', 0.83)))
        threshold = float(bundle.get('threshold', 0.83))
        high = beat_probs >= threshold
        top_k = min(5, len(beat_probs))
        score = float(max(np.max(beat_probs), np.mean(np.sort(beat_probs)[-top_k:])))
        details = {
            'num_screened_beats': int(len(beat_probs)),
            'screened_peak_indices': [int(p) for p in peaks.tolist()],
            'beat_abnormal_probabilities': [float(x) for x in beat_probs.tolist()],
            'max_beat_abnormal_probability': float(np.max(beat_probs)),
            'mean_top5_beat_abnormal_probability': float(np.mean(np.sort(beat_probs)[-top_k:])),
            'beat_abnormal_fraction_at_threshold': float(np.mean(high)),
            'num_abnormal_beats_at_threshold': int(np.sum(high)),
            'subtype_details': subtype_details,
        }
        return score, details, str(bundle.get('architecture', type(model).__name__)), str(model_path), threshold
    except Exception as exc:
        return None, {'error': f'beat_model_error:{type(exc).__name__}:{str(exc)[:80]}'}, str(model_path), str(model_path), None


def _predict_arrhythmia_subtypes(model_path: Path, X, F, beat_probs: np.ndarray, bundle_threshold: float) -> dict | None:
    if torch is None or _ECGSubtypeCNN is None or not model_path.exists():
        return None
    try:
        high = np.asarray(beat_probs) >= float(bundle_threshold)
        if not np.any(high):
            top_n = min(3, len(beat_probs))
            high[np.argsort(beat_probs)[-top_n:]] = True
        X_sel = X[high]
        F_sel = F[high]
        bundle = torch.load(model_path, map_location='cpu', weights_only=False)
        classes = list(bundle.get('classes', ['S', 'V', 'F', 'Q']))
        model = _ECGSubtypeCNN(feature_dim=int(bundle.get('feature_dim', 3)), num_classes=len(classes))
        model.load_state_dict(bundle['state_dict'])
        model.eval()
        subtype_probs = []
        with torch.no_grad():
            for start in range(0, len(X_sel), 512):
                subtype_probs.append(torch.softmax(model(X_sel[start:start + 512], F_sel[start:start + 512]), dim=1).numpy())
        probs = np.concatenate(subtype_probs).astype(float)
        pred = np.argmax(probs, axis=1)
        counts = {classes[i]: int(np.sum(pred == i)) for i in range(len(classes))}
        mean_probs = {classes[i]: float(np.mean(probs[:, i])) for i in range(len(classes))}
        dominant = max(counts, key=counts.get) if counts else None
        return {
            'model_source': str(model_path),
            'model_name': str(bundle.get('architecture', type(model).__name__)),
            'cv_metrics': bundle.get('cv_metrics'),
            'num_subtyped_beats': int(len(pred)),
            'subtype_counts': counts,
            'mean_subtype_probabilities': mean_probs,
            'dominant_subtype': dominant,
        }
    except Exception as exc:
        return {'error': f'subtype_model_error:{type(exc).__name__}:{str(exc)[:80]}', 'model_source': str(model_path)}


def _robust_norm_sequence(seq: np.ndarray) -> np.ndarray:
    seq = np.asarray(seq, dtype=np.float32)
    seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
    med = float(np.median(seq))
    scale = float(np.percentile(seq, 75) - np.percentile(seq, 25))
    if scale < 1e-8:
        scale = float(np.std(seq) + 1e-8)
    return np.nan_to_num(np.clip((seq - med) / scale, -6, 6), nan=0.0, posinf=6.0, neginf=-6.0).astype(np.float32)


def _smooth_sequence(seq: np.ndarray, width: int) -> np.ndarray:
    seq = np.asarray(seq, dtype=np.float32)
    width = max(3, int(width))
    if width % 2 == 0:
        width += 1
    if len(seq) < width:
        return seq.copy()
    return scipy_signal.savgol_filter(seq, width, 2).astype(np.float32)



def _safe_float(value, default: float | None = None) -> float | None:
    try:
        value = float(value)
    except Exception:
        return default
    return value if np.isfinite(value) else default


def _ecg_artifact_metrics(values: np.ndarray, sampling_rate: float) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return {
            'saturation_fraction': 1.0,
            'flatline_fraction': 1.0,
            'baseline_wander_ratio': None,
            'high_frequency_noise_ratio': None,
            'powerline_noise_ratio': None,
            'signal_dynamic_range': 0.0,
        }
    centered = values - float(np.median(values))
    dynamic = float(np.percentile(values, 95) - np.percentile(values, 5))
    diffs = np.diff(values)
    flat_tol = max(1e-8, dynamic * 1e-4)
    flatline_fraction = float(np.mean(np.abs(diffs) <= flat_tol)) if len(diffs) else 1.0
    lo, hi = np.percentile(values, [0.5, 99.5])
    edge_tol = max(1e-8, dynamic * 0.01)
    saturation_fraction = float(np.mean((values <= lo + edge_tol) | (values >= hi - edge_tol))) if dynamic > 0 else 1.0
    fs = float(sampling_rate)
    baseline_ratio = None
    hf_ratio = None
    powerline_ratio = None
    if len(centered) >= max(32, int(fs * 4)) and fs > 2:
        freqs, psd = scipy_signal.welch(centered, fs=fs, nperseg=min(len(centered), max(64, int(fs * 4))))
        total = _safe_trapz(psd, freqs) + 1e-12
        baseline_ratio = float(_safe_trapz(psd[freqs < 0.5], freqs[freqs < 0.5]) / total) if np.any(freqs < 0.5) else 0.0
        hf_mask = freqs > min(40.0, 0.35 * fs)
        hf_ratio = float(_safe_trapz(psd[hf_mask], freqs[hf_mask]) / total) if np.any(hf_mask) else 0.0
        pl_mask = ((freqs >= 49.0) & (freqs <= 51.0)) | ((freqs >= 59.0) & (freqs <= 61.0))
        powerline_ratio = float(_safe_trapz(psd[pl_mask], freqs[pl_mask]) / total) if np.any(pl_mask) else 0.0
    return {
        'saturation_fraction': saturation_fraction,
        'flatline_fraction': flatline_fraction,
        'baseline_wander_ratio': baseline_ratio,
        'high_frequency_noise_ratio': hf_ratio,
        'powerline_noise_ratio': powerline_ratio,
        'signal_dynamic_range': dynamic,
    }


def _interval_summary_ms(start: np.ndarray, stop: np.ndarray, fs: float, lo_ms: float, hi_ms: float) -> dict:
    start = np.asarray(start, dtype=float)
    stop = np.asarray(stop, dtype=float)
    n = min(len(start), len(stop))
    if n == 0:
        return {'median_ms': None, 'iqr_ms': None, 'valid_count': 0, 'total_count': 0, 'valid_fraction': 0.0}
    vals = (stop[:n] - start[:n]) / float(fs) * 1000.0
    vals = vals[np.isfinite(vals)]
    total = int(len(vals))
    vals = vals[(vals >= lo_ms) & (vals <= hi_ms)]
    if len(vals) == 0:
        return {'median_ms': None, 'iqr_ms': None, 'valid_count': 0, 'total_count': total, 'valid_fraction': 0.0}
    return {
        'median_ms': float(np.median(vals)),
        'iqr_ms': float(np.percentile(vals, 75) - np.percentile(vals, 25)) if len(vals) > 1 else 0.0,
        'valid_count': int(len(vals)),
        'total_count': total,
        'valid_fraction': float(len(vals) / max(total, 1)),
    }


def _qt_correct(qt_ms: float | None, rr_s: np.ndarray) -> tuple[float | None, float | None]:
    if qt_ms is None or len(rr_s) == 0:
        return None, None
    rr = float(np.nanmedian(rr_s))
    if not np.isfinite(rr) or rr <= 0:
        return None, None
    return float(qt_ms / np.sqrt(rr)), float(qt_ms / np.cbrt(rr))


def _rr_edr_sequence(values: np.ndarray, sampling_rate: float, seq_len: int = 256, num_channels: int = 2, peaks: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.zeros((num_channels, seq_len), dtype=np.float32)
    if peaks is None:
        peaks, _ = neurokit_nabian2018_peaks(values, sampling_rate, low_hz=None, high_hz=None, fallback_threshold_scale=0.6)
    peaks = np.asarray(peaks, dtype=int)
    peaks = peaks[(peaks >= 0) & (peaks < len(values))]
    duration = len(values) / float(sampling_rate)
    grid = np.linspace(0, duration, seq_len)
    if len(peaks) >= 4:
        t = peaks / float(sampling_rate)
        rr = np.diff(t)
        rt = (t[:-1] + t[1:]) / 2.0
        keep = np.isfinite(rr) & (rr >= 0.25) & (rr <= 3.0)
        rr = rr[keep]
        rt = rt[keep]
        rr_seq = np.interp(grid, rt, rr, left=float(np.median(rr)), right=float(np.median(rr))) if len(rr) >= 2 else np.full(seq_len, float(np.median(rr)) if len(rr) else 0.8)
        amp = values[peaks]
        amp = (amp - np.median(amp)) / (np.percentile(amp, 75) - np.percentile(amp, 25) + 1e-8)
        edr = np.interp(grid, t, amp, left=float(amp[0]), right=float(amp[-1])) if len(amp) >= 2 else np.zeros(seq_len)
    else:
        rr_seq = np.full(seq_len, 0.8)
        edr = np.zeros(seq_len)
    rr_n = _robust_norm_sequence(rr_seq)
    edr_n = _robust_norm_sequence(edr)
    chans = [rr_n, edr_n]
    if num_channels > 2:
        chans.extend([_robust_norm_sequence(np.gradient(rr_n)), _robust_norm_sequence(np.gradient(edr_n))])
    if num_channels > 4:
        slow_width = max(9, int(seq_len / 10))
        chans.extend([_robust_norm_sequence(_smooth_sequence(rr_n, slow_width)), _robust_norm_sequence(_smooth_sequence(edr_n, slow_width))])
    return np.nan_to_num(np.stack(chans[:num_channels]), nan=0.0, posinf=6.0, neginf=-6.0).astype(np.float32)



def _predict_apnea_sequence_context_model(model_path: Path, values: np.ndarray, sampling_rate: float) -> tuple[float | None, str | None, str | None, float | None, dict | None, dict | None]:
    if torch is None or _SeqContextCNN is None or not model_path.exists():
        return None, None, None, None, None, None
    try:
        cache_key = ('apnea_sequence_context', str(model_path), float(model_path.stat().st_mtime))
        cached = _TORCH_MODEL_CACHE.get(cache_key)
        if cached is None:
            bundle = torch.load(model_path, map_location='cpu', weights_only=False)
            num_channels = int(bundle.get('num_channels', 6))
            model = _SeqContextCNN(in_channels=num_channels)
            model.load_state_dict(bundle['state_dict'])
            model.eval()
            cached = (bundle, model)
            _TORCH_MODEL_CACHE[cache_key] = cached
        bundle, model = cached
        per_minute_len = int(bundle.get('per_minute_len', 64))
        num_channels = int(bundle.get('num_channels', 6))
        radius = int(bundle.get('context_radius', 10))
        samples_per_epoch = max(1, int(round(60.0 * float(sampling_rate))))
        clean = np.asarray(values, dtype=float)
        clean = clean[np.isfinite(clean)]
        n_epochs = len(clean) // samples_per_epoch
        if n_epochs < 3:
            return None, None, None, None, None, None
        minute_features = []
        for i in range(n_epochs):
            chunk = clean[i * samples_per_epoch:(i + 1) * samples_per_epoch]
            minute_features.append(_rr_edr_sequence(chunk, sampling_rate, seq_len=per_minute_len, num_channels=num_channels))
        minute_features = np.asarray(minute_features, dtype=np.float32)
        X = []
        for i in range(n_epochs):
            parts = []
            for off in range(-radius, radius + 1):
                j = min(max(i + off, 0), n_epochs - 1)
                parts.append(minute_features[j])
            X.append(np.concatenate(parts, axis=1))
        x = torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32)
        probs = []
        with torch.no_grad():
            for start in range(0, len(x), 128):
                probs.append(torch.sigmoid(model(x[start:start + 128])).cpu().numpy())
        epoch_probs = np.concatenate(probs).astype(float)
        threshold = bundle.get('threshold')
        threshold = float(threshold) if threshold is not None else 0.68
        high = epoch_probs[epoch_probs >= threshold]
        if len(high):
            segment_score = float(np.percentile(high, 75))
        else:
            segment_score = float(np.percentile(epoch_probs, 95))
        details = {
            'num_epochs': int(n_epochs),
            'context_radius_epochs': int(radius),
            'context_minutes': int(2 * radius + 1),
            'epoch_probability_mean': float(np.mean(epoch_probs)),
            'epoch_probability_max': float(np.max(epoch_probs)),
            'epoch_probability_p95': float(np.percentile(epoch_probs, 95)),
            'elevated_epoch_count': int(np.sum(epoch_probs >= threshold)),
            'elevated_epoch_fraction': float(np.mean(epoch_probs >= threshold)),
            'source_metrics': bundle.get('source_metrics'),
        }
        return segment_score, str(bundle.get('architecture', type(model).__name__)), str(model_path), threshold, bundle.get('cv_metrics'), details
    except Exception as exc:
        return None, f'sequence_context_model_error:{type(exc).__name__}:{str(exc)[:80]}', str(model_path), None, None, None


def _predict_apnea_rr_edr_model(model_path: Path, values: np.ndarray, sampling_rate: float, peaks: np.ndarray | None = None) -> tuple[float | None, str | None, str | None, float | None, dict | None]:
    if torch is None or _RREdrCNN is None or not model_path.exists():
        return None, None, None, None, None
    try:
        cache_key = ('apnea_rr_edr', str(model_path), float(model_path.stat().st_mtime))
        cached = _TORCH_MODEL_CACHE.get(cache_key)
        if cached is None:
            bundle = torch.load(model_path, map_location='cpu', weights_only=False)
            num_channels = int(bundle.get('num_channels', 2))
            use_lstm = 'BiLSTM' in str(bundle.get('architecture', ''))
            model = _RREdrCNN(in_channels=num_channels, use_lstm=use_lstm)
            model.load_state_dict(bundle['state_dict'])
            model.eval()
            cached = (bundle, model)
            _TORCH_MODEL_CACHE[cache_key] = cached
        bundle, model = cached
        seq_len = int(bundle.get('seq_len', 256))
        num_channels = int(bundle.get('num_channels', 2))
        x = _rr_edr_sequence(values, sampling_rate, seq_len, num_channels=num_channels, peaks=peaks)
        with torch.no_grad():
            score = float(torch.sigmoid(model(torch.tensor(x[None, :, :], dtype=torch.float32))).item())
        threshold = bundle.get('threshold')
        threshold = float(threshold) if threshold is not None else None
        return score, str(bundle.get('architecture', type(model).__name__)), str(model_path), threshold, bundle.get('cv_metrics')
    except Exception as exc:
        return None, f'rr_edr_model_error:{type(exc).__name__}:{str(exc)[:80]}', str(model_path), None, None


def _predict_deep_ecg_model(model_path: Path, values: np.ndarray) -> tuple[float | None, str | None, str | None, float | None, dict | None]:
    if torch is None or _ECGTinyCNN is None or not model_path.exists():
        return None, None, None, None, None
    try:
        bundle = torch.load(model_path, map_location='cpu', weights_only=False)
        target_len = int(bundle.get('target_len', 4096))
        x = _robust_resample_ecg(values, target_len)
        model = _ECGTinyCNN()
        model.load_state_dict(bundle['state_dict'])
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(x[None, :], dtype=torch.float32))
            score = float(torch.sigmoid(logits).item())
        threshold = bundle.get('threshold')
        threshold = float(threshold) if threshold is not None else None
        return score, str(bundle.get('architecture', type(model).__name__)), str(model_path), threshold, bundle.get('cv_metrics')
    except Exception as exc:
        return None, f'deep_model_error:{type(exc).__name__}:{str(exc)[:80]}', str(model_path), None, None


def _predict_rpeaks_deep_model(model_path: Path, values: np.ndarray, sampling_rate: float) -> tuple[np.ndarray | None, dict]:
    if torch is None or _RPeakSegCNN is None or not model_path.exists():
        return None, {'method': 'deep_rpeak_unavailable'}
    try:
        cache_key = ('rpeak_seg', str(model_path), float(model_path.stat().st_mtime))
        cached = _TORCH_MODEL_CACHE.get(cache_key)
        if cached is None:
            bundle = torch.load(model_path, map_location='cpu', weights_only=False)
            model = _RPeakSegCNN(); model.load_state_dict(bundle['state_dict']); model.eval()
            cached = (bundle, model)
            _TORCH_MODEL_CACHE[cache_key] = cached
        bundle, model = cached
        target_fs = float(bundle.get('target_fs', 250.0))
        win_len = int(bundle.get('win_len', 2048))
        threshold = float(bundle.get('threshold', 0.5))
        raw = np.asarray(values, dtype=np.float32)
        raw = raw[np.isfinite(raw)]
        if len(raw) < max(16, int(1.0 * sampling_rate)):
            return None, {'method': 'deep_rpeak_unavailable', 'reason': 'signal_too_short'}
        if abs(float(sampling_rate) - target_fs) > 1e-6:
            new_len = int(round(len(raw) * target_fs / float(sampling_rate)))
            x_rs = scipy_signal.resample(raw, new_len).astype(np.float32)
        else:
            x_rs = raw
        med = float(np.median(x_rs)); q75, q25 = np.percentile(x_rs, [75, 25]); scale = float(q75 - q25)
        if scale < 1e-8: scale = float(np.std(x_rs)) + 1e-8
        x_rs = np.clip((x_rs - med) / scale, -8.0, 8.0).astype(np.float32)
        probs_sum = np.zeros(len(x_rs), dtype=np.float32)
        counts = np.zeros(len(x_rs), dtype=np.float32)
        stride = win_len // 2
        starts = list(range(0, max(1, len(x_rs) - win_len + 1), stride))
        if not starts or starts[-1] + win_len < len(x_rs):
            starts.append(max(0, len(x_rs) - win_len))
        with torch.no_grad():
            for start in starts:
                seg = np.zeros(win_len, dtype=np.float32)
                part = x_rs[start:start + win_len]
                seg[:len(part)] = part
                logits = model(torch.tensor(seg[None, :], dtype=torch.float32))
                prob = torch.sigmoid(logits).cpu().numpy()[0][:len(part)]
                probs_sum[start:start + len(part)] += prob
                counts[start:start + len(part)] += 1.0
        prob = probs_sum / np.maximum(counts, 1.0)
        min_distance = max(1, int(0.24 * target_fs))
        peaks, props = scipy_signal.find_peaks(prob, height=threshold, distance=min_distance)
        if len(peaks) < 2:
            adaptive = max(0.25, float(np.percentile(prob, 99.2)))
            peaks, props = scipy_signal.find_peaks(prob, height=adaptive, distance=min_distance)
            threshold = adaptive
        if abs(float(sampling_rate) - target_fs) > 1e-6:
            peaks_orig = np.rint(peaks * float(sampling_rate) / target_fs).astype(int)
        else:
            peaks_orig = peaks.astype(int)
        peaks_orig = peaks_orig[(peaks_orig >= 0) & (peaks_orig < len(raw))]
        return peaks_orig.astype(int), {
            'method': 'deep_rpeak_segmentation_cnn',
            'deep_model_source': str(model_path),
            'deep_threshold': float(threshold),
            'deep_probability_max': float(np.max(prob)) if len(prob) else 0.0,
            'deep_probability_mean': float(np.mean(prob)) if len(prob) else 0.0,
            'num_deep_peaks': int(len(peaks_orig)),
            'cv_fold_losses': bundle.get('fold_reports'),
        }
    except Exception as exc:
        return None, {'method': 'deep_rpeak_error', 'reason': f'{type(exc).__name__}:{str(exc)[:120]}'}


def _clean_rr_intervals(peaks: np.ndarray, sampling_rate: float) -> np.ndarray:
    rr = np.diff(np.asarray(peaks, dtype=float)) / float(sampling_rate)
    rr = rr[np.isfinite(rr) & (rr >= 0.25) & (rr <= 3.0)]
    if len(rr) < 3:
        return rr
    med = float(np.median(rr))
    mad = float(np.median(np.abs(rr - med))) + 1e-8
    keep = np.abs(rr - med) <= max(0.25, 4.5 * mad)
    return rr[keep]


def _ecg_feature_dict(values: np.ndarray, sampling_rate: float, peaks: np.ndarray | None = None) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if peaks is None:
        peaks, _ = neurokit_nabian2018_peaks(values, sampling_rate, low_hz=None, high_hz=None, fallback_threshold_scale=0.6)
    peaks = np.asarray(peaks, dtype=int)
    peaks = peaks[(peaks >= 0) & (peaks < len(values))]
    rr = _clean_rr_intervals(peaks, sampling_rate)
    finite = values[np.isfinite(values)]
    centered = finite - float(np.nanmedian(finite)) if len(finite) else np.zeros(1)
    duration_s = float(len(finite) / sampling_rate) if sampling_rate else 0.0
    features: dict[str, float] = {
        'duration_s': duration_s,
        'num_samples': float(len(finite)),
        'num_peaks': float(len(peaks)),
        'peaks_per_minute': float(len(peaks) / max(duration_s, 1e-8) * 60.0),
        'signal_mean': float(np.nanmean(finite)) if len(finite) else 0.0,
        'signal_std': float(np.nanstd(finite)) if len(finite) else 0.0,
        'signal_range_p95_p05': float(np.nanpercentile(finite, 95) - np.nanpercentile(finite, 5)) if len(finite) else 0.0,
        'signal_mad': float(np.nanmedian(np.abs(centered - np.nanmedian(centered)))) if len(centered) else 0.0,
        'flat_fraction': float(np.mean(np.abs(np.diff(finite)) < 1e-8)) if len(finite) > 1 else 1.0,
    }
    if len(rr):
        diff_rr = np.diff(rr)
        mean_rr = float(np.mean(rr))
        med_rr = float(np.median(rr))
        features.update({
            'rr_count': float(len(rr)),
            'heart_rate_bpm': float(60.0 / med_rr) if med_rr > 0 else 0.0,
            'rr_mean_s': mean_rr,
            'rr_median_s': med_rr,
            'rr_std_s': float(np.std(rr)),
            'rr_cv': float(np.std(rr) / mean_rr) if mean_rr > 0 else 0.0,
            'rr_iqr_s': float(np.percentile(rr, 75) - np.percentile(rr, 25)),
            'rr_min_s': float(np.min(rr)),
            'rr_max_s': float(np.max(rr)),
            'rr_range_s': float(np.max(rr) - np.min(rr)),
            'rmssd_s': float(np.sqrt(np.mean(diff_rr ** 2))) if len(diff_rr) else 0.0,
            'pnn50': float(np.mean(np.abs(diff_rr) > 0.05)) if len(diff_rr) else 0.0,
            'pnn120': float(np.mean(np.abs(diff_rr) > 0.12)) if len(diff_rr) else 0.0,
            'successive_change_fraction': float(np.mean(np.abs(diff_rr) / np.maximum(rr[:-1], 1e-8) > 0.18)) if len(diff_rr) else 0.0,
            'pause_fraction': float(np.mean(rr > 2.0)),
            'short_rr_fraction': float(np.mean(rr < 0.45)),
            'long_rr_fraction': float(np.mean(rr > 1.2)),
        })
        if len(rr) >= 4:
            rr_center = rr - np.mean(rr)
            freqs, psd = scipy_signal.welch(rr_center, fs=1.0 / mean_rr if mean_rr > 0 else 1.0, nperseg=min(len(rr_center), 64))
            lf = (freqs >= 0.04) & (freqs < 0.15)
            hf = (freqs >= 0.15) & (freqs <= 0.40)
            total = _safe_trapz(psd, freqs) + 1e-12
            lf_power = _safe_trapz(psd[lf], freqs[lf]) if np.any(lf) else 0.0
            hf_power = _safe_trapz(psd[hf], freqs[hf]) if np.any(hf) else 0.0
            features.update({
                'rr_lf_power_ratio': float(lf_power / total),
                'rr_hf_power_ratio': float(hf_power / total),
                'rr_lf_hf_ratio': float(lf_power / (hf_power + 1e-12)),
                'rr_dominant_freq_hz': float(freqs[int(np.argmax(psd))]) if len(freqs) else 0.0,
            })
    for name in [
        'rr_count','heart_rate_bpm','rr_mean_s','rr_median_s','rr_std_s','rr_cv','rr_iqr_s','rr_min_s','rr_max_s','rr_range_s','rmssd_s','pnn50','pnn120',
        'successive_change_fraction','pause_fraction','short_rr_fraction','long_rr_fraction','rr_lf_power_ratio','rr_hf_power_ratio','rr_lf_hf_ratio','rr_dominant_freq_hz'
    ]:
        features.setdefault(name, 0.0)
    return {key: float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)) for key, value in features.items()}


def _predict_feature_model(model_path: Path, features: dict[str, float]) -> tuple[float | None, str | None, str | None, float | None]:
    if not model_path.exists():
        return None, None, None, None
    try:
        bundle = joblib.load(model_path)
        names = bundle['feature_names']
        X = np.asarray([[float(features.get(name, 0.0)) for name in names]], dtype=float)
        model = bundle['model']
        classes = list(getattr(model, 'classes_', []))
        proba = model.predict_proba(X)[0]
        if 1 in classes:
            score = float(proba[classes.index(1)])
        else:
            score = float(np.max(proba))
        threshold = bundle.get('threshold')
        threshold = float(threshold) if threshold is not None else None
        return score, str(bundle.get('model_name', type(model).__name__)), str(model_path), threshold
    except Exception as exc:
        return None, f'model_error:{type(exc).__name__}:{str(exc)[:80]}', str(model_path), None


def ECG_assess_quality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = np.asarray(data.values, dtype=float)
    base = signal_quality_summary(values)
    peak_result = ECG_detect_r_peaks(signal_path, sampling_rate, column)
    peaks = np.asarray(peak_result.get("r_peak_indices", []), dtype=int)
    features = _ecg_feature_dict(values, data.sampling_rate, peaks if len(peaks) else None)
    regularity = interval_regularity(peaks, data.sampling_rate) if len(peaks) >= 2 else {'regularity_confidence': 0.15}
    artifacts = _ecg_artifact_metrics(values, data.sampling_rate)
    duration_s = float(features.get('duration_s', 0.0))
    expected_min_peaks = max(3, int(duration_s * 35.0 / 60.0)) if duration_s >= 10 else 2
    peak_density_ok = len(peaks) >= expected_min_peaks
    rr_cv = _safe_float(features.get('rr_cv'), None)
    heart_rate = _safe_float(features.get('heart_rate_bpm'), None)
    flags = []
    if artifacts['signal_dynamic_range'] <= 1e-8 or artifacts['flatline_fraction'] > 0.35:
        flags.append('flat_or_near_flat_signal')
    if artifacts['saturation_fraction'] > 0.08:
        flags.append('possible_clipping_or_saturation')
    if artifacts['baseline_wander_ratio'] is not None and artifacts['baseline_wander_ratio'] > 0.55:
        flags.append('baseline_wander_dominant')
    if artifacts['high_frequency_noise_ratio'] is not None and artifacts['high_frequency_noise_ratio'] > 0.35:
        flags.append('high_frequency_noise')
    if artifacts['powerline_noise_ratio'] is not None and artifacts['powerline_noise_ratio'] > 0.18:
        flags.append('powerline_noise')
    if not peak_density_ok:
        flags.append('too_few_detected_r_peaks')
    if heart_rate is not None and not 35 <= heart_rate <= 220:
        flags.append('implausible_heart_rate')
    if rr_cv is not None and rr_cv > 0.35:
        flags.append('very_irregular_rr_or_peak_errors')
    sqi_score = float(base.get('confidence', 0.5))
    sqi_score *= float(peak_result.get('confidence', 0.5))
    sqi_score = min(1.0, sqi_score + 0.15 * float(regularity.get('regularity_confidence', 0.5)))
    if 'flat_or_near_flat_signal' in flags:
        sqi_score *= 0.20
    if 'possible_clipping_or_saturation' in flags:
        sqi_score *= 0.70
    if 'baseline_wander_dominant' in flags:
        sqi_score *= 0.75
    if 'high_frequency_noise' in flags:
        sqi_score *= 0.80
    if 'too_few_detected_r_peaks' in flags or 'implausible_heart_rate' in flags:
        sqi_score *= 0.55
    if len(flags) >= 3:
        sqi_score *= 0.80
    sqi_score = float(max(0.0, min(1.0, sqi_score)))
    quality = 'good' if sqi_score >= 0.72 and not flags else 'fair' if sqi_score >= 0.45 else 'poor'
    return {
        "tool": "ECG_assess_quality",
        "source": data.source,
        **base,
        **artifacts,
        "quality": quality,
        "ecg_sqi_score": sqi_score,
        "quality_flags": flags,
        "num_detected_r_peaks": int(len(peaks)),
        "heart_rate_bpm": heart_rate,
        "rr_cv": rr_cv,
        "peak_detection_method": peak_result.get('method'),
        "used_deep_peak_model": bool(peak_result.get('used_deep_model')),
        "method": "signal_quality_summary_plus_deep_rpeak_sqi_artifact_metrics",
    }


def ECG_detect_r_peaks(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    deep_peaks, deep_details = _predict_rpeaks_deep_model(ECG_RPEAK_DEEP_MODEL_PATH, values, data.sampling_rate)
    try:
        if nk is None:
            raise RuntimeError("neurokit2 is not installed")
        cleaned = nk.ecg_clean(values, sampling_rate=data.sampling_rate, method="pantompkins1985")
        _, info = nk.ecg_peaks(cleaned, sampling_rate=data.sampling_rate, method="pantompkins1985", correct_artifacts=True)
        classical_peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
        classical_details = {"classical_method": "pantompkins1985", "median_prominence": None}
    except Exception as exc:
        classical_peaks, classical_details = neurokit_nabian2018_peaks(values, data.sampling_rate, low_hz=None, high_hz=None, fallback_threshold_scale=0.6)
        classical_details["fallback_reason"] = str(exc)
        classical_details["classical_method"] = classical_details.pop("method", "nabian2018")
    classical_hr = bpm_from_peaks(classical_peaks, data.sampling_rate)
    deep_hr = bpm_from_peaks(deep_peaks, data.sampling_rate) if deep_peaks is not None else None
    use_deep = bool(deep_peaks is not None and len(deep_peaks) >= 2 and deep_hr is not None and 35 <= deep_hr <= 220)
    peaks = np.asarray(deep_peaks if use_deep else classical_peaks, dtype=int)
    heart_rate = bpm_from_peaks(peaks, data.sampling_rate)
    method = "deep_rpeak_segmentation_cnn" if use_deep else classical_details.get("classical_method", "pantompkins1985")
    confidence = 0.88 if use_deep else 0.82 if method == "pantompkins1985" else 0.65
    if heart_rate is None or not 35 <= heart_rate <= 220:
        confidence = 0.3
    return {"tool": "ECG_detect_r_peaks", "r_peak_indices": peaks.tolist(), "num_peaks": int(len(peaks)), "heart_rate_bpm": heart_rate, "confidence": confidence, "method": method, "used_deep_model": bool(use_deep), "deep_num_peaks": int(len(deep_peaks)) if deep_peaks is not None else None, "classical_num_peaks": int(len(classical_peaks)), **classical_details, **deep_details}


def ECG_compute_hrv(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    peak_result = ECG_detect_r_peaks(signal_path, sampling_rate, column)
    peaks = np.asarray(peak_result["r_peak_indices"], dtype=float)
    if len(peaks) < 3:
        return {"tool": "ECG_compute_hrv", "error": "not enough R peaks", "confidence": 0.1}
    raw_rr_s = np.diff(peaks) / float(data.sampling_rate)
    raw_rr_s = raw_rr_s[np.isfinite(raw_rr_s) & (raw_rr_s > 0)]
    rr_s = _clean_rr_intervals(peaks, data.sampling_rate)
    if len(rr_s) < 2:
        return {"tool": "ECG_compute_hrv", "error": "not enough clean RR intervals", "confidence": 0.1, "num_raw_rr": int(len(raw_rr_s))}
    rr_ms = rr_s * 1000.0
    diff_ms = np.diff(rr_ms)
    duration_s = float((peaks[-1] - peaks[0]) / float(data.sampling_rate)) if len(peaks) > 1 else 0.0
    mean_rr = float(np.mean(rr_ms))
    median_rr = float(np.median(rr_ms))
    mean_hr = float(60000.0 / mean_rr) if mean_rr > 0 else None
    median_hr = float(60000.0 / median_rr) if median_rr > 0 else None
    sdnn = float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else 0.0
    sdsd = float(np.std(diff_ms, ddof=1)) if len(diff_ms) > 1 else 0.0
    rmssd = float(np.sqrt(np.mean(diff_ms ** 2))) if len(diff_ms) else None
    nn20 = int(np.sum(np.abs(diff_ms) > 20.0)) if len(diff_ms) else 0
    nn50 = int(np.sum(np.abs(diff_ms) > 50.0)) if len(diff_ms) else 0
    pnn20 = float(nn20 / len(diff_ms)) if len(diff_ms) else None
    pnn50 = float(nn50 / len(diff_ms)) if len(diff_ms) else None
    sd1_ms = float(np.sqrt(0.5) * np.std(diff_ms, ddof=1)) if len(diff_ms) > 1 else None
    sd2_ms = float(np.sqrt(max(0.0, 2.0 * sdnn ** 2 - (sd1_ms or 0.0) ** 2))) if sd1_ms is not None else None
    sd1_sd2_ratio = float(sd1_ms / sd2_ms) if sd1_ms is not None and sd2_ms and sd2_ms > 0 else None
    triangular_index = None
    tinn_proxy_ms = None
    if len(rr_ms) >= 10:
        hist, edges = np.histogram(rr_ms, bins=max(8, min(64, int(np.sqrt(len(rr_ms))))))
        max_bin = int(np.max(hist)) if len(hist) else 0
        triangular_index = float(len(rr_ms) / max_bin) if max_bin > 0 else None
        nonzero = np.flatnonzero(hist)
        if len(nonzero):
            tinn_proxy_ms = float(edges[nonzero[-1] + 1] - edges[nonzero[0]])
    frequency_metrics = {
        'vlf_power_ms2': None, 'lf_power_ms2': None, 'hf_power_ms2': None, 'total_power_ms2': None, 'lf_hf_ratio': None, 'lf_norm': None, 'hf_norm': None
    }
    frequency_method = None
    frequency_reliability = 'not_computed'
    if len(rr_s) >= 8 and duration_s >= 30.0:
        rr_t = np.cumsum(rr_s)
        rr_t = rr_t - rr_t[0]
        fs_interp = 4.0
        grid = np.arange(0, rr_t[-1], 1.0 / fs_interp)
        if len(grid) >= 16 and rr_t[-1] > 0:
            tach = np.interp(grid, rr_t, rr_ms)
            tach = scipy_signal.detrend(tach, type='constant')
            freqs, psd = scipy_signal.welch(tach, fs=fs_interp, nperseg=min(256, len(tach)))
            def band(lo, hi):
                mask = (freqs >= lo) & (freqs < hi)
                return _safe_trapz(psd[mask], freqs[mask]) if np.any(mask) else 0.0
            vlf = band(0.0033, 0.04)
            lf = band(0.04, 0.15)
            hf = band(0.15, 0.40)
            total = band(0.0033, 0.40)
            denom = lf + hf
            frequency_metrics = {
                'vlf_power_ms2': float(vlf),
                'lf_power_ms2': float(lf),
                'hf_power_ms2': float(hf),
                'total_power_ms2': float(total),
                'lf_hf_ratio': float(lf / (hf + 1e-12)),
                'lf_norm': float(lf / denom) if denom > 0 else None,
                'hf_norm': float(hf / denom) if denom > 0 else None,
            }
            frequency_method = 'welch_interpolated_nn_tachogram_4hz'
            frequency_reliability = 'standard_short_term' if duration_s >= 300 and len(rr_s) >= 240 else 'limited_short_recording'
    clean_ratio = float(len(rr_s) / len(raw_rr_s)) if len(raw_rr_s) else 0.0
    removed_rr = int(max(0, len(raw_rr_s) - len(rr_s)))
    artifacts = _ecg_artifact_metrics(data.values, data.sampling_rate)
    reliability_flags = []
    if duration_s < 30.0:
        reliability_flags.append('very_short_recording')
    elif duration_s < 120.0:
        reliability_flags.append('short_term_hrv_only')
    if duration_s < 300.0:
        reliability_flags.append('frequency_domain_limited_below_5_min')
    if len(rr_s) < 20:
        reliability_flags.append('few_nn_intervals')
    if clean_ratio < 0.85:
        reliability_flags.append('many_rr_intervals_removed')
    if artifacts.get('flatline_fraction', 0.0) > 0.25 or artifacts.get('signal_dynamic_range', 0.0) <= 1e-8:
        reliability_flags.append('poor_signal_quality')
    if not peak_result.get('used_deep_model') and float(peak_result.get('confidence', 0.5)) < 0.7:
        reliability_flags.append('lower_confidence_peak_detection')
    confidence = float(peak_result.get('confidence', 0.5))
    if 'few_nn_intervals' in reliability_flags:
        confidence *= 0.65
    if 'many_rr_intervals_removed' in reliability_flags:
        confidence *= 0.75
    if 'poor_signal_quality' in reliability_flags:
        confidence *= 0.55
    confidence = float(max(0.1, min(0.95, confidence)))
    return {
        "tool": "ECG_compute_hrv",
        "method": "deep_r_peak_nn_cleaning_standard_hrv" if peak_result.get('used_deep_model') else "r_peak_nn_cleaning_standard_hrv",
        "peak_detection_method": peak_result.get('method'),
        "num_r_peaks": int(len(peaks)),
        "num_raw_rr": int(len(raw_rr_s)),
        "num_clean_nn": int(len(rr_ms)),
        "num_removed_rr": removed_rr,
        "clean_nn_ratio": clean_ratio,
        "duration_s": duration_s,
        "mean_rr_ms": mean_rr,
        "median_rr_ms": median_rr,
        "mean_heart_rate_bpm": mean_hr,
        "median_heart_rate_bpm": median_hr,
        "min_heart_rate_bpm": float(60000.0 / np.max(rr_ms)) if len(rr_ms) else None,
        "max_heart_rate_bpm": float(60000.0 / np.min(rr_ms)) if len(rr_ms) else None,
        "sdnn_ms": sdnn,
        "sdsd_ms": sdsd,
        "rmssd_ms": rmssd,
        "nn20": nn20,
        "nn50": nn50,
        "pnn20": pnn20,
        "pnn50": pnn50,
        "sd1_ms": sd1_ms,
        "sd2_ms": sd2_ms,
        "sd1_sd2_ratio": sd1_sd2_ratio,
        "triangular_index": triangular_index,
        "tinn_proxy_ms": tinn_proxy_ms,
        **frequency_metrics,
        "frequency_method": frequency_method,
        "frequency_reliability": frequency_reliability,
        "reliability_flags": reliability_flags,
        "confidence": confidence,
        "disclaimer": "HRV metrics are screening/research summaries; frequency-domain values need sufficiently long, clean, stationary NN intervals.",
    }



def _rhythm_feature_values(values: np.ndarray, sampling_rate: float, peaks: np.ndarray, features: dict, beat_details: dict | None, feature_names: list[str]) -> np.ndarray:
    beat_details = beat_details or {}
    subtype = beat_details.get('subtype_details') or {}
    subtype_counts = subtype.get('subtype_counts') or {}
    total_sub = max(1, sum(int(v) for v in subtype_counts.values()))
    nbeats = max(1, int(len(peaks)))
    rr = _clean_rr_intervals(peaks, sampling_rate) if len(peaks) > 1 else np.asarray([])
    diff = np.diff(rr) if len(rr) > 1 else np.asarray([])
    artifacts = _ecg_artifact_metrics(values, sampling_rate)
    vals = {
        'duration_s': _safe_float(features.get('duration_s'), 0.0),
        'heart_rate_bpm': _safe_float(features.get('heart_rate_bpm'), 0.0),
        'peaks_per_minute': _safe_float(features.get('peaks_per_minute'), 0.0),
        'rr_count': _safe_float(features.get('rr_count'), 0.0),
        'rr_mean_s': _safe_float(features.get('rr_mean_s'), 0.0),
        'rr_median_s': _safe_float(features.get('rr_median_s'), 0.0),
        'rr_std_s': _safe_float(features.get('rr_std_s'), 0.0),
        'rr_cv': _safe_float(features.get('rr_cv'), 0.0),
        'rr_iqr_s': _safe_float(features.get('rr_iqr_s'), 0.0),
        'rr_range_s': _safe_float(features.get('rr_range_s'), 0.0),
        'rmssd_s': _safe_float(features.get('rmssd_s'), 0.0),
        'pnn50': _safe_float(features.get('pnn50'), 0.0),
        'pnn120': _safe_float(features.get('pnn120'), 0.0),
        'successive_change_fraction': _safe_float(features.get('successive_change_fraction'), 0.0),
        'pause_fraction': _safe_float(features.get('pause_fraction'), 0.0),
        'short_rr_fraction': _safe_float(features.get('short_rr_fraction'), 0.0),
        'long_rr_fraction': _safe_float(features.get('long_rr_fraction'), 0.0),
        'rr_lf_hf_ratio': _safe_float(features.get('rr_lf_hf_ratio'), 0.0),
        'rr_dominant_freq_hz': _safe_float(features.get('rr_dominant_freq_hz'), 0.0),
        'rr_diff_mean_abs': float(np.mean(np.abs(diff))) if len(diff) else 0.0,
        'rr_diff_std': float(np.std(diff)) if len(diff) else 0.0,
        'rr_diff_sign_changes': float(np.mean(np.diff(np.sign(diff)) != 0)) if len(diff) > 2 else 0.0,
        'signal_std': _safe_float(features.get('signal_std'), 0.0),
        'signal_mad': _safe_float(features.get('signal_mad'), 0.0),
        'flat_fraction': _safe_float(features.get('flat_fraction'), 0.0),
        'baseline_wander_ratio': _safe_float(artifacts.get('baseline_wander_ratio'), 0.0),
        'high_frequency_noise_ratio': _safe_float(artifacts.get('high_frequency_noise_ratio'), 0.0),
        'powerline_noise_ratio': _safe_float(artifacts.get('powerline_noise_ratio'), 0.0),
        'beat_max_abnormal_probability': _safe_float(beat_details.get('max_beat_abnormal_probability'), 0.0),
        'beat_top5_abnormal_probability': _safe_float(beat_details.get('mean_top5_beat_abnormal_probability'), 0.0),
        'beat_abnormal_fraction': _safe_float(beat_details.get('beat_abnormal_fraction_at_threshold'), 0.0),
        'beat_abnormal_count_per_beat': _safe_float(beat_details.get('num_abnormal_beats_at_threshold'), 0.0) / nbeats,
        'subtype_s_fraction': _safe_float(subtype_counts.get('S'), 0.0) / total_sub,
        'subtype_v_fraction': _safe_float(subtype_counts.get('V'), 0.0) / total_sub,
        'subtype_f_fraction': _safe_float(subtype_counts.get('F'), 0.0) / total_sub,
        'subtype_q_fraction': _safe_float(subtype_counts.get('Q'), 0.0) / total_sub,
    }
    return np.asarray([[vals.get(name, 0.0) for name in feature_names]], dtype=float)


def _predict_rhythm_feature_model(model_path: Path, values: np.ndarray, sampling_rate: float, peaks: np.ndarray, features: dict, beat_details: dict | None) -> tuple[str | None, dict | None, str | None, dict | None]:
    if not model_path.exists():
        return None, None, None, None
    try:
        cache_key = ('ecg_rhythm_feature_classifier', str(model_path), float(model_path.stat().st_mtime))
        cached = _TORCH_MODEL_CACHE.get(cache_key)
        if cached is None:
            bundle = joblib.load(model_path)
            cached = bundle
            _TORCH_MODEL_CACHE[cache_key] = cached
        bundle = cached
        model = bundle['model']
        feature_names = list(bundle.get('feature_names', []))
        x = _rhythm_feature_values(values, sampling_rate, peaks, features, beat_details, feature_names)
        pred = str(model.predict(x)[0])
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(x)[0]
            classes = list(getattr(model, 'classes_', bundle.get('classes', [])))
            probabilities = {str(cls): float(prob) for cls, prob in zip(classes, probs)}
        return pred, probabilities, str(model_path), bundle.get('cv_metrics')
    except Exception as exc:
        return None, {'error': f'rhythm_model_error:{type(exc).__name__}:{str(exc)[:100]}'}, str(model_path), None



def _predict_af_feature_model(model_path: Path, values: np.ndarray, sampling_rate: float, peaks: np.ndarray, features: dict, beat_details: dict | None) -> tuple[float | None, float | None, str | None, dict | None]:
    if not model_path.exists():
        return None, None, None, None
    try:
        cache_key = ('ecg_af_feature_classifier', str(model_path), float(model_path.stat().st_mtime))
        cached = _TORCH_MODEL_CACHE.get(cache_key)
        if cached is None:
            bundle = joblib.load(model_path)
            cached = bundle
            _TORCH_MODEL_CACHE[cache_key] = cached
        bundle = cached
        model = bundle['model']
        feature_names = list(bundle.get('feature_names', []))
        x = _rhythm_feature_values(values, sampling_rate, peaks, features, beat_details, feature_names)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(x)[0]
            classes = list(getattr(model, 'classes_', bundle.get('classes', [0, 1])))
            score = float(proba[classes.index(1)]) if 1 in classes else float(np.max(proba))
        else:
            score = float(model.predict(x)[0])
        threshold = bundle.get('threshold')
        threshold = float(threshold) if threshold is not None else 0.5
        return score, threshold, str(model_path), bundle.get('cv_metrics')
    except Exception as exc:
        return None, None, str(model_path), {'error': f'af_model_error:{type(exc).__name__}:{str(exc)[:100]}'}

def ECG_screen_arrhythmia(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    peak_result = ECG_detect_r_peaks(signal_path, sampling_rate, column)
    peaks = np.asarray(peak_result.get("r_peak_indices", []), dtype=float)
    if len(peaks) < 4:
        return {"tool": "ECG_screen_arrhythmia", "error": "not enough R peaks", "confidence": 0.1}
    rr_s = _clean_rr_intervals(peaks, sampling_rate)
    if len(rr_s) < 3:
        return {"tool": "ECG_screen_arrhythmia", "error": "not enough valid RR intervals", "confidence": 0.1}
    features = _ecg_feature_dict(data.values, data.sampling_rate, peaks.astype(int))
    beat_score, beat_details, beat_model_name, beat_model_source, beat_threshold = _predict_arrhythmia_beat_model(ECG_ARRHYTHMIA_BEAT_MODEL_PATH, data.values, data.sampling_rate, peaks.astype(int))
    rhythm_prediction, rhythm_probabilities, rhythm_model_source, rhythm_cv_metrics = _predict_rhythm_feature_model(ECG_RHYTHM_MODEL_PATH, data.values, data.sampling_rate, peaks.astype(int), features, beat_details)
    af_binary_score, af_binary_threshold, af_binary_model_source, af_binary_cv_metrics = _predict_af_feature_model(ECG_AF_MODEL_PATH, data.values, data.sampling_rate, peaks.astype(int), features, beat_details)
    deep_score, deep_model_name, deep_model_source, deep_threshold, deep_cv_metrics = _predict_deep_ecg_model(ECG_ARRHYTHMIA_DEEP_MODEL_PATH, data.values)
    model_score, model_name, model_source, model_threshold = _predict_feature_model(ECG_ARRHYTHMIA_MODEL_PATH, features)
    heart_rate = float(features.get('heart_rate_bpm', 0.0)) or None
    rr_cv = float(features.get('rr_cv', 0.0))
    pause_count = int(np.sum(rr_s > 2.0))
    ectopy_proxy_fraction = float(features.get('successive_change_fraction', 0.0))
    flags = []
    if heart_rate is not None and heart_rate < 50:
        flags.append("bradycardia_pattern")
    if heart_rate is not None and heart_rate > 110:
        flags.append("tachycardia_pattern")
    if rr_cv > 0.14:
        flags.append("irregular_rr_pattern")
    if pause_count:
        flags.append("long_pause_pattern")
    if ectopy_proxy_fraction > 0.10:
        flags.append("ectopy_proxy_pattern")
    heuristic_score = min(1.0, 0.22 * len(flags) + 0.9 * max(0.0, rr_cv - 0.08) + 0.5 * ectopy_proxy_fraction)
    if deep_score is not None:
        score = float(deep_score)
        threshold = deep_threshold if deep_threshold is not None else 0.50
        active_model_name = deep_model_name
        active_model_source = deep_model_source
        active_method = 'ecg_1dcnn_arrhythmia_screening'
    elif beat_score is not None:
        score = float(beat_score)
        threshold = beat_threshold if beat_threshold is not None else 0.83
        active_model_name = beat_model_name
        active_model_source = beat_model_source
        active_method = 'ecg_beat_cnn_arrhythmia_screening'
    elif model_score is not None:
        score = float(model_score)
        threshold = model_threshold if model_threshold is not None else 0.36
        active_model_name = model_name
        active_model_source = model_source
        active_method = 'ecg_feature_model_arrhythmia_screening'
    else:
        score = heuristic_score
        threshold = 0.36
        active_model_name = model_name
        active_model_source = model_source
        active_method = 'rr_interval_screening_v2'
    risk = "elevated" if score >= threshold else "low"
    confidence = min(0.85, max(0.35, abs(score - threshold) * 1.6 + 0.45))
    return {
        "tool": "ECG_screen_arrhythmia",
        "heart_rate_bpm": heart_rate,
        "rr_cv": rr_cv,
        "pause_count": pause_count,
        "ectopy_proxy_fraction": ectopy_proxy_fraction,
        "arrhythmia_score": score,
        "decision_threshold": threshold,
        "arrhythmia_flags": flags,
        "arrhythmia_risk": risk,
        "confidence": confidence,
        "model_source": active_model_source,
        "method": active_method,
        "model_name": active_model_name,
        "deep_cv_metrics": deep_cv_metrics,
        "beat_model_details": beat_details,
        "predicted_rhythm": rhythm_prediction,
        "rhythm_probabilities": rhythm_probabilities,
        "rhythm_model_source": rhythm_model_source,
        "rhythm_cv_metrics": rhythm_cv_metrics,
        "af_binary_probability": af_binary_score,
        "af_binary_decision_threshold": af_binary_threshold,
        "af_binary_model_source": af_binary_model_source,
        "af_binary_cv_metrics": af_binary_cv_metrics,
        "disclaimer": "Screening heuristic/model only; not a diagnostic rhythm classifier.",
    }


def _apnea_evidence_summary(values: np.ndarray, sampling_rate: float, peaks: np.ndarray, seq_len: int = 256) -> dict:
    values = np.asarray(values, dtype=float)
    peaks = np.asarray(peaks, dtype=int)
    rr_raw = np.diff(peaks) / float(sampling_rate) if len(peaks) > 1 else np.array([])
    rr_clean = _clean_rr_intervals(peaks, sampling_rate) if len(peaks) > 1 else np.array([])
    clean_ratio = float(len(rr_clean) / len(rr_raw)) if len(rr_raw) else 0.0
    evidence_flags = []
    reliability_flags = []
    rr_cv = float(np.std(rr_clean) / np.mean(rr_clean)) if len(rr_clean) and np.mean(rr_clean) > 0 else None
    rr_range_s = float(np.max(rr_clean) - np.min(rr_clean)) if len(rr_clean) else None
    if rr_cv is not None and rr_cv > 0.10:
        evidence_flags.append('rr_variability_supports_apnea')
    if rr_range_s is not None and rr_range_s > 0.45:
        evidence_flags.append('wide_rr_range_supports_apnea')
    if clean_ratio < 0.85:
        reliability_flags.append('many_rr_intervals_removed')
    if len(rr_clean) < 20:
        reliability_flags.append('few_rr_intervals')
    seq = _rr_edr_sequence(values, sampling_rate, seq_len=seq_len, peaks=peaks)
    rr_seq = seq[0]
    edr_seq = seq[1]
    fs_seq = seq_len / max(len(values) / float(sampling_rate), 1e-8)
    freqs, rr_psd = scipy_signal.welch(rr_seq - np.mean(rr_seq), fs=fs_seq, nperseg=min(128, len(rr_seq)))
    _, edr_psd = scipy_signal.welch(edr_seq - np.mean(edr_seq), fs=fs_seq, nperseg=min(128, len(edr_seq)))
    def rel_power(psd, lo, hi):
        total = _safe_trapz(psd, freqs) + 1e-12
        mask = (freqs >= lo) & (freqs < hi)
        return float(_safe_trapz(psd[mask], freqs[mask]) / total) if np.any(mask) else 0.0
    rr_apnea_band = rel_power(rr_psd, 0.01, 0.05)
    edr_apnea_band = rel_power(edr_psd, 0.01, 0.05)
    rr_dominant = float(freqs[int(np.argmax(rr_psd))]) if len(freqs) else 0.0
    edr_dominant = float(freqs[int(np.argmax(edr_psd))]) if len(freqs) else 0.0
    edr_variability = float(np.std(edr_seq))
    if rr_apnea_band > 0.25:
        evidence_flags.append('rr_low_frequency_oscillation')
    if edr_apnea_band > 0.25:
        evidence_flags.append('edr_low_frequency_oscillation')
    if edr_variability > 0.75:
        evidence_flags.append('edr_amplitude_variability')
    return {
        'rr_clean_ratio': clean_ratio,
        'rr_interval_count': int(len(rr_clean)),
        'rr_cv_evidence': rr_cv,
        'rr_range_s': rr_range_s,
        'rr_apnea_band_power_ratio': rr_apnea_band,
        'edr_apnea_band_power_ratio': edr_apnea_band,
        'rr_dominant_frequency_hz': rr_dominant,
        'edr_dominant_frequency_hz': edr_dominant,
        'edr_variability': edr_variability,
        'evidence_flags': evidence_flags,
        'reliability_flags': reliability_flags,
    }


def ECG_screen_sleep_apnea(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    peak_result = ECG_detect_r_peaks(signal_path, sampling_rate, column)
    peaks = np.asarray(peak_result.get("r_peak_indices", []), dtype=int)
    if len(peaks) < 4:
        return {"tool": "ECG_screen_sleep_apnea", "error": "not enough R peaks", "confidence": 0.1}
    features = _ecg_feature_dict(data.values, data.sampling_rate, peaks)
    duration_s = len(data.values) / float(data.sampling_rate) if data.sampling_rate else 0.0
    sequence_context_score = sequence_context_model_name = sequence_context_model_source = sequence_context_threshold = sequence_context_cv_metrics = sequence_context_details = None
    if duration_s >= 600.0:
        sequence_context_score, sequence_context_model_name, sequence_context_model_source, sequence_context_threshold, sequence_context_cv_metrics, sequence_context_details = _predict_apnea_sequence_context_model(ECG_APNEA_RR_EDR_SEQUENCE_CONTEXT_MODEL_PATH, data.values, data.sampling_rate)
    context_score = context_model_name = context_model_source = context_threshold = context_cv_metrics = None
    if duration_s >= 180.0:
        context_score, context_model_name, context_model_source, context_threshold, context_cv_metrics = _predict_apnea_rr_edr_model(ECG_APNEA_RR_EDR_CONTEXT_MODEL_PATH, data.values, data.sampling_rate)
    rr_edr_score, rr_edr_model_name, rr_edr_model_source, rr_edr_threshold, rr_edr_cv_metrics = _predict_apnea_rr_edr_model(ECG_APNEA_RR_EDR_MODEL_PATH, data.values, data.sampling_rate)
    deep_score, deep_model_name, deep_model_source, deep_threshold, deep_cv_metrics = _predict_deep_ecg_model(ECG_APNEA_DEEP_MODEL_PATH, data.values)
    model_score, model_name, model_source, model_threshold = _predict_feature_model(ECG_APNEA_MODEL_PATH, features)
    score = 0
    flags = []
    heart_rate = features.get('heart_rate_bpm') or None
    rr_cv = features.get('rr_cv')
    rmssd_ms = features.get('rmssd_s', 0.0) * 1000.0
    sdnn_ms = features.get('rr_std_s', 0.0) * 1000.0
    lf_hf = features.get('rr_lf_hf_ratio', 0.0)
    dom_freq = features.get('rr_dominant_freq_hz', 0.0)
    if heart_rate is not None and (heart_rate < 55 or heart_rate > 95):
        score += 1; flags.append("sleep_epoch_heart_rate_extreme")
    if rr_cv is not None and rr_cv > 0.08:
        score += 1; flags.append("elevated_rr_variability")
    if rmssd_ms > 55 or sdnn_ms > 65:
        score += 1; flags.append("high_short_term_hrv")
    if 0.01 <= dom_freq <= 0.05 and lf_hf > 1.0:
        score += 1; flags.append("apnea_band_rr_oscillation_proxy")
    heuristic_score = min(1.0, score / 4.0)
    evidence = _apnea_evidence_summary(data.values, data.sampling_rate, peaks)
    if sequence_context_score is not None:
        apnea_probability = float(sequence_context_score)
        threshold = sequence_context_threshold if sequence_context_threshold is not None else 0.68
        active_model_name = sequence_context_model_name
        active_model_source = sequence_context_model_source
        active_method = 'ecg_rr_edr_sequence_context_cnn_apnea_screening'
        deep_cv_metrics = sequence_context_cv_metrics
    elif context_score is not None:
        apnea_probability = float(context_score)
        threshold = context_threshold if context_threshold is not None else 0.58
        active_model_name = context_model_name
        active_model_source = context_model_source
        active_method = 'ecg_rr_edr_context_cnn_apnea_screening'
        deep_cv_metrics = context_cv_metrics
    elif rr_edr_score is not None:
        apnea_probability = float(rr_edr_score)
        threshold = rr_edr_threshold if rr_edr_threshold is not None else 0.65
        active_model_name = rr_edr_model_name
        active_model_source = rr_edr_model_source
        active_method = 'ecg_rr_edr_cnn_apnea_screening'
        deep_cv_metrics = rr_edr_cv_metrics
    elif deep_score is not None:
        apnea_probability = float(deep_score)
        threshold = deep_threshold if deep_threshold is not None else 0.50
        active_model_name = deep_model_name
        active_model_source = deep_model_source
        active_method = 'ecg_1dcnn_apnea_screening'
    elif model_score is not None:
        apnea_probability = float(model_score)
        threshold = model_threshold if model_threshold is not None else 0.50
        active_model_name = model_name
        active_model_source = model_source
        active_method = 'ecg_rr_feature_model_apnea_screening'
    else:
        apnea_probability = heuristic_score
        threshold = 0.50
        active_model_name = model_name
        active_model_source = model_source
        active_method = 'apdet_inspired_rr_oscillation_proxy'
    apnea_risk = "elevated" if apnea_probability >= threshold else "low"
    model_margin = float(apnea_probability - threshold)
    reliability_flags = list(evidence.get('reliability_flags', []))
    if len(peaks) < 20:
        reliability_flags.append('short_epoch_few_beats')
    if active_method not in {'ecg_rr_edr_cnn_apnea_screening', 'ecg_rr_edr_context_cnn_apnea_screening', 'ecg_rr_edr_sequence_context_cnn_apnea_screening'}:
        reliability_flags.append('rr_edr_model_unavailable_fallback_used')
    confidence = min(0.88, max(0.30, abs(model_margin) * 1.4 + 0.42))
    if reliability_flags:
        confidence = max(0.25, confidence * 0.85)
    return {
        "tool": "ECG_screen_sleep_apnea",
        "apnea_risk": apnea_risk,
        "apnea_probability": apnea_probability,
        "decision_threshold": threshold,
        "model_margin": model_margin,
        "apnea_proxy_score": score,
        "apnea_proxy_flags": flags,
        "evidence_flags": evidence.get('evidence_flags', []),
        "reliability_flags": reliability_flags,
        "heart_rate_bpm": heart_rate,
        "mean_rr_ms": features.get('rr_mean_s', 0.0) * 1000.0,
        "sdnn_ms": sdnn_ms,
        "rmssd_ms": rmssd_ms,
        "rr_cv": rr_cv,
        "rr_dominant_freq_hz": dom_freq,
        "rr_lf_hf_ratio": lf_hf,
        "input_duration_s": duration_s,
        "sequence_context_details": sequence_context_details,
        "rr_clean_ratio": evidence.get('rr_clean_ratio'),
        "rr_interval_count": evidence.get('rr_interval_count'),
        "rr_range_s": evidence.get('rr_range_s'),
        "rr_apnea_band_power_ratio": evidence.get('rr_apnea_band_power_ratio'),
        "edr_apnea_band_power_ratio": evidence.get('edr_apnea_band_power_ratio'),
        "edr_dominant_frequency_hz": evidence.get('edr_dominant_frequency_hz'),
        "edr_variability": evidence.get('edr_variability'),
        "confidence": confidence,
        "model_source": active_model_source,
        "model_name": active_model_name,
        "deep_cv_metrics": deep_cv_metrics,
        "external_generalization_note": "The default ECG apnea backends are mixed Apnea-ECG+UCDDB+SLPDB models; expect lower Apnea-ECG-only scores but more realistic cross-dataset behavior." if active_method in {'ecg_rr_edr_sequence_context_cnn_apnea_screening', 'ecg_rr_edr_cnn_apnea_screening'} else None,
        "method": active_method,
        "disclaimer": "ECG-only apnea screening proxy; validate against respiratory effort, airflow, and SpO2 labels before clinical use.",
    }


def ECG_measure_morphology_intervals(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    peak_result = ECG_detect_r_peaks(signal_path, sampling_rate, column)
    peaks = np.asarray(peak_result.get("r_peak_indices", []), dtype=int)
    if len(peaks) < 3:
        return {"tool": "ECG_measure_morphology_intervals", "error": "not enough R peaks", "confidence": 0.1}
    values = np.asarray(data.values, dtype=float)
    artifacts = _ecg_artifact_metrics(values, data.sampling_rate)
    result = {
        "tool": "ECG_measure_morphology_intervals",
        "heart_rate_bpm": peak_result.get("heart_rate_bpm"),
        "confidence": min(0.55, float(peak_result.get("confidence", 0.5))),
        "method": "ecg_dwt_delineation_interval_screening_with_validity_filters",
        "peak_detection_method": peak_result.get('method'),
        "disclaimer": "Screening heuristic only; ECG intervals require validated delineation and lead-specific clinical review.",
    }
    interval_quality_flags = []
    qrs_summary = {'median_ms': None, 'iqr_ms': None, 'valid_count': 0, 'total_count': 0, 'valid_fraction': 0.0}
    pr_summary = {'median_ms': None, 'iqr_ms': None, 'valid_count': 0, 'total_count': 0, 'valid_fraction': 0.0}
    qt_summary = {'median_ms': None, 'iqr_ms': None, 'valid_count': 0, 'total_count': 0, 'valid_fraction': 0.0}
    try:
        if nk is None:
            raise RuntimeError("neurokit2 is not installed")
        cleaned = nk.ecg_clean(values, sampling_rate=data.sampling_rate, method="pantompkins1985")
        _, waves = nk.ecg_delineate(cleaned, rpeaks=peaks, sampling_rate=data.sampling_rate, method="dwt", show=False, show_type="all")
        def valid(name: str) -> np.ndarray:
            arr = np.asarray(waves.get(name, []), dtype=float)
            return arr[np.isfinite(arr)]
        q = valid("ECG_Q_Peaks")
        s_peaks = valid("ECG_S_Peaks")
        p_on = valid("ECG_P_Onsets")
        qrs_on = valid("ECG_R_Onsets")
        qrs_off = valid("ECG_R_Offsets")
        t_off = valid("ECG_T_Offsets")
        qrs_summary = _interval_summary_ms(qrs_on, qrs_off, data.sampling_rate, 40.0, 180.0)
        if qrs_summary['valid_count'] == 0:
            qrs_summary = _interval_summary_ms(q, s_peaks, data.sampling_rate, 30.0, 180.0)
        pr_summary = _interval_summary_ms(p_on, qrs_on, data.sampling_rate, 60.0, 320.0)
        qt_summary = _interval_summary_ms(qrs_on, t_off, data.sampling_rate, 180.0, 650.0)
    except Exception as exc:
        result["fallback_reason"] = str(exc)
        interval_quality_flags.append('delineation_failed')
    qrs_ms = qrs_summary['median_ms']
    pr_ms = pr_summary['median_ms']
    qt_ms = qt_summary['median_ms']
    rr_s = np.diff(peaks) / data.sampling_rate
    rr_s = rr_s[np.isfinite(rr_s) & (rr_s >= 0.3) & (rr_s <= 2.5)]
    qtc_ms, qtc_fridericia_ms = _qt_correct(qt_ms, rr_s)
    st_values = []
    offset = int(0.08 * data.sampling_rate)
    baseline_offset = int(0.04 * data.sampling_rate)
    for peak in peaks:
        st_idx = peak + offset
        base_idx = peak - baseline_offset
        if 0 <= st_idx < len(values) and 0 <= base_idx < len(values):
            st_values.append(values[st_idx] - values[base_idx])
    st_deviation_proxy = float(np.nanmedian(st_values)) if st_values else None
    signal_scale = max(float(np.nanpercentile(values, 95) - np.nanpercentile(values, 5)) if len(values) else 0.0, 1e-8)
    st_deviation_normalized = float(st_deviation_proxy / signal_scale) if st_deviation_proxy is not None else None
    if qrs_summary['valid_count'] < max(3, int(0.25 * len(peaks))):
        interval_quality_flags.append('few_valid_qrs_intervals')
    if qt_summary['valid_count'] < max(3, int(0.20 * len(peaks))):
        interval_quality_flags.append('few_valid_qt_intervals')
    if artifacts.get('flatline_fraction', 0.0) > 0.25 or artifacts.get('signal_dynamic_range', 0.0) <= 1e-8:
        interval_quality_flags.append('poor_signal_quality')
    flags = []
    if qrs_ms is not None and qrs_ms > 120:
        flags.append("wide_qrs_proxy")
    if qtc_ms is not None and qtc_ms > 470:
        flags.append("long_qtc_proxy")
    if pr_ms is not None and pr_ms > 220:
        flags.append("prolonged_pr_proxy")
    if st_deviation_normalized is not None and abs(st_deviation_normalized) > 0.18:
        flags.append("st_deviation_proxy")
    confidence = min(0.70, float(peak_result.get('confidence', 0.5)))
    valid_fracs = [qrs_summary['valid_fraction'], qt_summary['valid_fraction']]
    if pr_summary['total_count']:
        valid_fracs.append(pr_summary['valid_fraction'])
    confidence *= max(0.35, min(1.0, float(np.mean(valid_fracs)) if valid_fracs else 0.35))
    if interval_quality_flags:
        confidence *= 0.75 if len(interval_quality_flags) == 1 else 0.55
    result.update({
        "pr_interval_ms": pr_ms,
        "pr_interval_iqr_ms": pr_summary['iqr_ms'],
        "qrs_duration_ms": qrs_ms,
        "qrs_duration_iqr_ms": qrs_summary['iqr_ms'],
        "qt_interval_ms": qt_ms,
        "qt_interval_iqr_ms": qt_summary['iqr_ms'],
        "qtc_interval_ms": qtc_ms,
        "qtc_fridericia_ms": qtc_fridericia_ms,
        "st_deviation_proxy": st_deviation_proxy,
        "st_deviation_normalized": st_deviation_normalized,
        "valid_interval_counts": {
            "pr": {k: pr_summary[k] for k in ['valid_count', 'total_count', 'valid_fraction']},
            "qrs": {k: qrs_summary[k] for k in ['valid_count', 'total_count', 'valid_fraction']},
            "qt": {k: qt_summary[k] for k in ['valid_count', 'total_count', 'valid_fraction']},
        },
        "interval_quality_flags": interval_quality_flags,
        "morphology_flags": flags,
        "morphology_risk": "elevated" if flags and 'poor_signal_quality' not in interval_quality_flags else "low",
        "confidence": float(max(0.1, min(0.85, confidence))),
    })
    return result





def ECG_estimate_heart_rate(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    peaks = ECG_detect_r_peaks(signal_path, sampling_rate, column)
    return {
        "tool": "ECG_estimate_heart_rate",
        "heart_rate_bpm": peaks.get("heart_rate_bpm"),
        "num_beats": peaks.get("num_peaks"),
        "r_peak_indices": peaks.get("r_peak_indices", []),
        "peak_result": peaks,
        "confidence": peaks.get("confidence", 0.0),
        "method": "r_peak_interval_heart_rate",
    }


def ECG_classify_beats(signal_path: str, sampling_rate: float, column: str | None = None, max_beats: int = 200) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    peak_result = ECG_detect_r_peaks(signal_path, sampling_rate, column)
    peaks = np.asarray(peak_result.get("r_peak_indices", []), dtype=int)
    if len(peaks) < 3:
        return {"tool": "ECG_classify_beats", "error": "not enough R peaks", "confidence": 0.1}
    beat_score, beat_details, model_name, model_source, threshold = _predict_arrhythmia_beat_model(ECG_ARRHYTHMIA_BEAT_MODEL_PATH, data.values, data.sampling_rate, peaks)
    threshold = float(threshold if threshold is not None else 0.83)
    probs = np.asarray((beat_details or {}).get("beat_abnormal_probabilities", []), dtype=float)
    screened_peaks = np.asarray((beat_details or {}).get("screened_peak_indices", peaks.tolist()), dtype=int)
    rr_prev = np.r_[np.nan, np.diff(peaks) / float(data.sampling_rate)]
    rr_next = np.r_[np.diff(peaks) / float(data.sampling_rate), np.nan]
    med_rr = float(np.nanmedian(np.r_[rr_prev, rr_next])) if len(peaks) > 1 else np.nan
    beat_rows = []
    prob_by_peak = {int(pk): float(pr) for pk, pr in zip(screened_peaks.tolist(), probs.tolist())}
    for i, peak in enumerate(peaks[:max_beats]):
        prev_rr = float(rr_prev[i]) if np.isfinite(rr_prev[i]) else None
        next_rr = float(rr_next[i]) if np.isfinite(rr_next[i]) else None
        local_prob = prob_by_peak.get(int(peak))
        label = "normal_like"
        reasons = []
        if local_prob is not None and local_prob >= threshold:
            label = "abnormal_beat_model_positive"
            reasons.append("beat_cnn_probability_high")
        if prev_rr is not None and np.isfinite(med_rr) and prev_rr < 0.75 * med_rr and next_rr is not None and next_rr > 1.15 * med_rr:
            label = "premature_beat_proxy"
            reasons.append("short_long_rr_pattern")
        elif prev_rr is not None and np.isfinite(med_rr) and prev_rr > 1.4 * med_rr:
            label = "pause_or_escape_beat_proxy"
            reasons.append("long_preceding_rr")
        beat_rows.append({"beat_index": int(i), "r_peak_index": int(peak), "time_s": float(peak / data.sampling_rate), "label": label, "abnormal_probability": local_prob, "prev_rr_s": prev_rr, "next_rr_s": next_rr, "reasons": reasons})
    subtype = (beat_details or {}).get("subtype_details") or {}
    return {
        "tool": "ECG_classify_beats",
        "num_beats": int(len(peaks)),
        "num_returned_beats": int(len(beat_rows)),
        "beat_rows": beat_rows,
        "beat_model_details": beat_details,
        "subtype_summary": subtype,
        "model_source": model_source,
        "model_name": model_name,
        "decision_threshold": threshold,
        "confidence": 0.72 if beat_details and not beat_details.get("error") else 0.42,
        "method": "beat_cnn_probability_plus_rr_pattern_labels",
        "disclaimer": "Beat labels are research/proxy labels; validated beat-level diagnosis requires expert annotations and lead-aware review.",
    }


def ECG_classify_rhythm_segment(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    arr = ECG_screen_arrhythmia(signal_path, sampling_rate, column)
    rhythm = arr.get("predicted_rhythm")
    probs = arr.get("rhythm_probabilities")
    return {
        "tool": "ECG_classify_rhythm_segment",
        "predicted_rhythm": rhythm,
        "rhythm_probabilities": probs,
        "arrhythmia_risk": arr.get("arrhythmia_risk"),
        "arrhythmia_score": arr.get("arrhythmia_score"),
        "arrhythmia_flags": arr.get("arrhythmia_flags", []),
        "model_source": arr.get("rhythm_model_source") or arr.get("model_source"),
        "source_arrhythmia_result": arr,
        "confidence": arr.get("confidence", 0.0),
        "method": "mitdb_plus_afdb_partial_rhythm_feature_classifier_with_arrhythmia_screen_fallback",
        "disclaimer": "Segment rhythm classification is a screening output, not a diagnostic 12-lead interpretation.",
    }


def ECG_detect_afib(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    arr = ECG_screen_arrhythmia(signal_path, sampling_rate, column)
    probs = arr.get("rhythm_probabilities") or {}
    rhythm = str(arr.get("predicted_rhythm") or "").lower()
    af_prob = None
    for key, val in probs.items() if isinstance(probs, dict) else []:
        if str(key).lower() in {"af", "afib", "atrial_fibrillation"}:
            af_prob = float(val)
            break
    rr_cv = arr.get("rr_cv")
    ectopy = arr.get("ectopy_proxy_fraction")
    heuristic = 0.0
    if rr_cv is not None:
        heuristic += min(0.55, max(0.0, (float(rr_cv) - 0.08) * 3.2))
    if ectopy is not None:
        heuristic += min(0.25, float(ectopy) * 1.5)
    if "af" in rhythm:
        heuristic = max(heuristic, 0.65)
    binary_prob = arr.get("af_binary_probability")
    binary_threshold = arr.get("af_binary_decision_threshold") or 0.63
    if binary_prob is not None:
        af_score = float(binary_prob)
        decision_threshold = float(binary_threshold)
        method = "mitdb_plus_afdb_partial_binary_af_feature_classifier"
        model_source = arr.get("af_binary_model_source")
        cv_metrics = arr.get("af_binary_cv_metrics")
    else:
        af_score = af_prob if af_prob is not None else min(1.0, heuristic)
        decision_threshold = 0.55
        method = "mitdb_plus_afdb_partial_rhythm_classifier_af_probability_or_rr_irregularity_proxy"
        model_source = arr.get("rhythm_model_source")
        cv_metrics = arr.get("rhythm_cv_metrics")
    risk = "afib_likely" if af_score >= decision_threshold else "afib_possible" if af_score >= 0.55 * decision_threshold else "afib_unlikely"
    return {
        "tool": "ECG_detect_afib",
        "afib_probability": float(af_score),
        "decision_threshold": decision_threshold,
        "afib_risk": risk,
        "predicted_rhythm": arr.get("predicted_rhythm"),
        "rhythm_probabilities": probs,
        "rhythm_af_probability": af_prob,
        "binary_af_probability": binary_prob,
        "rr_cv": rr_cv,
        "ectopy_proxy_fraction": ectopy,
        "model_source": model_source,
        "cv_metrics": cv_metrics,
        "confidence": min(0.86, max(0.32, abs(float(af_score) - decision_threshold) * 1.15 + 0.42)),
        "source_arrhythmia_result": arr,
        "method": method,
        "disclaimer": "AFib screening only; confirm with clinician-reviewed ECG and appropriate rhythm labels.",
    }


def ECG_analyze_qt_interval(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    morph = ECG_measure_morphology_intervals(signal_path, sampling_rate, column)
    dl = ECG_delineate_waves_dl(signal_path, sampling_rate, column)
    qtc = morph.get("qtc_interval_ms")
    qtc_f = morph.get("qtc_fridericia_ms")
    flags = []
    if qtc is not None and qtc > 500:
        flags.append("marked_qtc_prolongation_proxy")
    elif qtc is not None and qtc > 470:
        flags.append("qtc_prolongation_proxy")
    if qtc is not None and qtc < 340:
        flags.append("short_qtc_proxy")
    risk = "elevated" if flags else "low"
    dl_quality_flags = list(dl.get("delineation_quality_flags", [])) if isinstance(dl, dict) else []
    reliability_flags = list(morph.get("interval_quality_flags", [])) + ["dl_" + f for f in dl_quality_flags]
    return {"tool": "ECG_analyze_qt_interval", "qt_interval_ms": morph.get("qt_interval_ms"), "qtc_interval_ms": qtc, "qtc_fridericia_ms": qtc_f, "qt_flags": flags, "qt_risk": risk, "reliability_flags": reliability_flags, "dl_event_validation_metrics": dl.get("event_validation_metrics") if isinstance(dl, dict) else None, "morphology_result": morph, "dl_delineation_result": dl, "confidence": min(0.74, max(morph.get("confidence", 0.0), 0.45 if not dl.get("error") else 0.0)), "method": "dwt_interval_screening_with_qtdb_unet_delineation_evidence", "disclaimer": "QT/QTc screening requires calibrated ECG, stable rhythm, lead review, and medication/clinical context."}


def ECG_screen_conduction_block(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    peak_result = ECG_detect_r_peaks(signal_path, sampling_rate, column)
    peaks = np.asarray(peak_result.get("r_peak_indices", []), dtype=int)
    morph = ECG_measure_morphology_intervals(signal_path, sampling_rate, column)
    dl = ECG_delineate_waves_dl(signal_path, sampling_rate, column)
    pr = morph.get("pr_interval_ms")
    qrs = morph.get("qrs_duration_ms")
    cd_probability, cd_threshold, cd_model_source, cd_cv_metrics = _predict_ptbxl_superclass_model(ECG_PTBXL_CD_MODEL_PATH, data.values, data.sampling_rate, peaks if len(peaks) else None)
    flags = []
    if pr is not None and pr > 300:
        flags.append("marked_pr_prolongation_av_block_proxy")
    elif pr is not None and pr > 220:
        flags.append("first_degree_av_block_proxy")
    if qrs is not None and qrs >= 120:
        flags.append("bundle_branch_block_or_intraventricular_conduction_delay_proxy")
    elif qrs is not None and qrs >= 110:
        flags.append("borderline_qrs_widening_proxy")
    if cd_probability is not None and cd_probability >= (cd_threshold or 0.58):
        flags.append("ptbxl_cd_model_positive")
    risk = "elevated" if flags else "low"
    dl_quality_flags = list(dl.get("delineation_quality_flags", [])) if isinstance(dl, dict) else []
    reliability_flags = list(morph.get("interval_quality_flags", [])) + ["dl_" + f for f in dl_quality_flags]
    if cd_probability is None:
        reliability_flags.append("ptbxl_cd_model_unavailable")
    confidence = min(0.78, max(morph.get("confidence", 0.0), 0.45 if not dl.get("error") else 0.0))
    if cd_probability is not None:
        confidence = max(confidence, min(0.80, abs(float(cd_probability) - float(cd_threshold or 0.58)) * 0.9 + 0.45))
    return {"tool": "ECG_screen_conduction_block", "pr_interval_ms": pr, "qrs_duration_ms": qrs, "conduction_disturbance_probability": cd_probability, "decision_threshold": cd_threshold, "ptbxl_cd_model_source": cd_model_source, "ptbxl_cd_cv_metrics": cd_cv_metrics, "conduction_flags": flags, "conduction_risk": risk, "reliability_flags": reliability_flags, "dl_event_validation_metrics": dl.get("event_validation_metrics") if isinstance(dl, dict) else None, "morphology_result": morph, "dl_delineation_result": dl, "confidence": confidence, "method": "ptbxl_cd_feature_classifier_plus_pr_qrs_interval_screening", "disclaimer": "Conduction block type classification needs lead morphology and expert review; this is interval-based screening."}



def _st_feature_row(values: np.ndarray, sampling_rate: float, peaks: np.ndarray | None = None) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    fs = float(sampling_rate)
    if len(values) < max(32, int(fs * 3)) or fs <= 0:
        return {}
    try:
        b, a = scipy_signal.butter(2, 0.5 / (fs / 2), btype='highpass')
        xf = scipy_signal.filtfilt(b, a, values).astype(float)
    except Exception:
        xf = values - np.nanmedian(values)
    if peaks is None or len(peaks) < 3:
        peaks, _ = neurokit_nabian2018_peaks(xf, fs, low_hz=None, high_hz=None, fallback_threshold_scale=0.6)
    peaks = np.asarray(peaks, dtype=int)
    peaks = peaks[(peaks > int(0.25 * fs)) & (peaks < len(xf) - int(0.35 * fs))]
    rr = np.diff(peaks) / fs if len(peaks) > 1 else np.asarray([])
    st_vals = []
    st60 = []
    st_slopes = []
    t_vals = []
    qrs_amp = []
    for peak in peaks:
        base0 = max(0, peak - int(0.08 * fs))
        base1 = max(base0 + 1, peak - int(0.02 * fs))
        baseline = float(np.median(xf[base0:base1])) if base1 > base0 else float(xf[peak])
        j60 = peak + int(0.06 * fs)
        j80 = peak + int(0.08 * fs)
        j120 = peak + int(0.12 * fs)
        t0 = peak + int(0.12 * fs)
        t1 = min(len(xf), peak + int(0.36 * fs))
        if j120 < len(xf):
            v60 = float(xf[j60] - baseline)
            v80 = float(xf[j80] - baseline)
            v120 = float(xf[j120] - baseline)
            st60.append(v60)
            st_vals.append(v80)
            st_slopes.append((v120 - v60) / 0.06)
        if t1 > t0:
            seg = xf[t0:t1] - baseline
            t_vals.append(float(seg[np.argmax(np.abs(seg))]))
        q0 = max(0, peak - int(0.04 * fs))
        q1 = min(len(xf), peak + int(0.04 * fs))
        if q1 > q0:
            qrs_amp.append(float(np.max(xf[q0:q1]) - np.min(xf[q0:q1])))
    st = np.asarray(st_vals, dtype=float)
    st60 = np.asarray(st60, dtype=float)
    slope = np.asarray(st_slopes, dtype=float)
    tv = np.asarray(t_vals, dtype=float)
    qa = np.asarray(qrs_amp, dtype=float)
    dyn = float(np.percentile(xf, 95) - np.percentile(xf, 5)) if len(xf) else 0.0
    qrs_med = float(np.median(qa)) if len(qa) else dyn
    scale = max(qrs_med, dyn, 1e-8)
    def stat(prefix: str, arr: np.ndarray) -> dict[str, float]:
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return {prefix + '_median': 0.0, prefix + '_mean_abs': 0.0, prefix + '_p95_abs': 0.0, prefix + '_std': 0.0, prefix + '_iqr': 0.0, prefix + '_pos_frac': 0.0, prefix + '_neg_frac': 0.0}
        return {
            prefix + '_median': float(np.median(arr)),
            prefix + '_mean_abs': float(np.mean(np.abs(arr))),
            prefix + '_p95_abs': float(np.percentile(np.abs(arr), 95)),
            prefix + '_std': float(np.std(arr)),
            prefix + '_iqr': float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            prefix + '_pos_frac': float(np.mean(arr > 0)),
            prefix + '_neg_frac': float(np.mean(arr < 0)),
        }
    feats = {
        'duration_s': len(xf) / fs,
        'num_beats': float(len(peaks)),
        'hr_bpm': float(60.0 / np.median(rr)) if len(rr) else 0.0,
        'rr_cv': float(np.std(rr) / np.mean(rr)) if len(rr) and np.mean(rr) > 0 else 0.0,
        'signal_dynamic_range': dyn,
        'qrs_amp_median': qrs_med,
        'st_abs_over_qrs': float(np.median(np.abs(st)) / scale) if len(st) else 0.0,
        'st_p95_abs_over_qrs': float(np.percentile(np.abs(st), 95) / scale) if len(st) else 0.0,
        'st_elevation_fraction_0p1mv': float(np.mean(st > 0.1)) if len(st) else 0.0,
        'st_depression_fraction_0p1mv': float(np.mean(st < -0.1)) if len(st) else 0.0,
    }
    for key, val in stat('st80', st).items():
        feats[key] = val
    for key, val in stat('st60', st60).items():
        feats[key] = val
    for key, val in stat('st_slope', slope).items():
        feats[key] = val
    for key, val in stat('t_amp', tv).items():
        feats[key] = val
    return {key: float(np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)) for key, val in feats.items()}



def _qrs_width_proxy_features(values: np.ndarray, sampling_rate: float, peaks: np.ndarray | None = None) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    fs = float(sampling_rate)
    if fs <= 0 or len(values) < int(3 * fs):
        return {}
    try:
        b, a = scipy_signal.butter(2, [0.5 / (fs / 2), min(35.0 / (fs / 2), 0.99)], btype='bandpass')
        xf = scipy_signal.filtfilt(b, a, values)
    except Exception:
        xf = values - np.nanmedian(values)
    if peaks is None or len(peaks) < 3:
        peaks, _ = neurokit_nabian2018_peaks(xf, fs, low_hz=None, high_hz=None, fallback_threshold_scale=0.6)
    peaks = np.asarray(peaks, dtype=int)
    peaks = peaks[(peaks > int(0.18 * fs)) & (peaks < len(xf) - int(0.22 * fs))]
    widths = []
    slopes = []
    amps = []
    for peak in peaks:
        lo = peak - int(0.16 * fs)
        hi = peak + int(0.16 * fs)
        if lo < 0 or hi > len(xf):
            continue
        seg = xf[lo:hi]
        if len(seg) < 8:
            continue
        edge = max(1, int(0.04 * fs))
        baseline = float(np.median(np.r_[seg[:edge], seg[-edge:]]))
        centered = seg - baseline
        amp = float(np.max(centered) - np.min(centered))
        if amp <= 1e-8:
            continue
        above = np.flatnonzero(np.abs(centered) >= 0.35 * np.max(np.abs(centered)))
        if len(above):
            widths.append(float((above[-1] - above[0] + 1) * 1000.0 / fs))
        diff = np.diff(seg)
        slopes.append(float(np.percentile(np.abs(diff), 95)))
        amps.append(amp)
    rr = np.diff(peaks) / fs if len(peaks) > 1 else np.asarray([])
    def stat(prefix: str, arr: list[float]) -> dict[str, float]:
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return {prefix + '_median': 0.0, prefix + '_iqr': 0.0, prefix + '_p90': 0.0}
        return {prefix + '_median': float(np.median(arr)), prefix + '_iqr': float(np.percentile(arr, 75) - np.percentile(arr, 25)), prefix + '_p90': float(np.percentile(arr, 90))}
    out = {'num_beats': float(len(peaks)), 'rr_cv': float(np.std(rr) / np.mean(rr)) if len(rr) and np.mean(rr) > 0 else 0.0}
    out.update(stat('qrs_width35_ms', widths))
    out.update(stat('qrs_slope', slopes))
    out.update(stat('qrs_amp', amps))
    return {k: float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for k, v in out.items()}


def _ptbxl_superclass_feature_row(values: np.ndarray, sampling_rate: float, peaks: np.ndarray | None = None) -> dict[str, float]:
    feats = _st_feature_row(values, sampling_rate, peaks)
    feats.update(_qrs_width_proxy_features(values, sampling_rate, peaks))
    return feats


def _predict_ptbxl_superclass_model(model_path: Path, values: np.ndarray, sampling_rate: float, peaks: np.ndarray | None = None) -> tuple[float | None, float | None, str | None, dict | None]:
    if not model_path.exists():
        return None, None, None, None
    try:
        cache_key = ('ecg_ptbxl_superclass_classifier', str(model_path), float(model_path.stat().st_mtime))
        bundle = _TORCH_MODEL_CACHE.get(cache_key)
        if bundle is None:
            bundle = joblib.load(model_path)
            _TORCH_MODEL_CACHE[cache_key] = bundle
        names = list(bundle.get('feature_names', []))
        feats = _ptbxl_superclass_feature_row(values, sampling_rate, peaks)
        x = np.asarray([[float(feats.get(name, 0.0)) for name in names]], dtype=float)
        model = bundle['model']
        proba = model.predict_proba(x)[0]
        classes = list(getattr(model, 'classes_', [0, 1]))
        score = float(proba[classes.index(1)]) if 1 in classes else float(np.max(proba))
        return score, float(bundle.get('threshold', 0.5)), str(model_path), bundle.get('cv_metrics')
    except Exception as exc:
        return None, None, str(model_path), {'error': f'ptbxl_model_error:{type(exc).__name__}:{str(exc)[:100]}'}


def _predict_st_feature_model(model_path: Path, values: np.ndarray, sampling_rate: float, peaks: np.ndarray | None = None) -> tuple[float | None, float | None, str | None, dict | None]:
    if not model_path.exists():
        return None, None, None, None
    try:
        cache_key = ('ecg_st_feature_classifier', str(model_path), float(model_path.stat().st_mtime))
        cached = _TORCH_MODEL_CACHE.get(cache_key)
        if cached is None:
            bundle = joblib.load(model_path)
            _TORCH_MODEL_CACHE[cache_key] = bundle
            cached = bundle
        bundle = cached
        names = list(bundle.get('feature_names', []))
        feats = _st_feature_row(values, sampling_rate, peaks)
        x = np.asarray([[float(feats.get(name, 0.0)) for name in names]], dtype=float)
        model = bundle['model']
        proba = model.predict_proba(x)[0]
        classes = list(getattr(model, 'classes_', [0, 1]))
        score = float(proba[classes.index(1)]) if 1 in classes else float(np.max(proba))
        threshold = float(bundle.get('threshold', 0.59))
        metrics = bundle.get('cv_metrics')
        if isinstance(metrics, dict) and bundle.get('episode_oof_metrics'):
            metrics = dict(metrics)
            metrics['episode_oof_metrics'] = bundle.get('episode_oof_metrics')
            metrics['episode_oof_scoring'] = bundle.get('episode_oof_scoring')
            metrics['episode_oof_report_path'] = bundle.get('episode_oof_report_path')
        return score, threshold, str(model_path), metrics
    except Exception as exc:
        return None, None, str(model_path), {'error': f'st_model_error:{type(exc).__name__}:{str(exc)[:100]}'}

def ECG_screen_ischemia_st(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    peak_result = ECG_detect_r_peaks(signal_path, sampling_rate, column)
    peaks = np.asarray(peak_result.get("r_peak_indices", []), dtype=int)
    morph = ECG_measure_morphology_intervals(signal_path, sampling_rate, column)
    dl = ECG_delineate_waves_dl(signal_path, sampling_rate, column)
    st_norm = morph.get("st_deviation_normalized")
    st_raw = morph.get("st_deviation_proxy")
    st_probability, st_threshold, st_model_source, st_cv_metrics = _predict_st_feature_model(ECG_ST_MODEL_PATH, data.values, data.sampling_rate, peaks if len(peaks) else None)
    sttc_probability, sttc_threshold, sttc_model_source, sttc_cv_metrics = _predict_ptbxl_superclass_model(ECG_PTBXL_STTC_MODEL_PATH, data.values, data.sampling_rate, peaks if len(peaks) else None)
    flags = []
    if st_norm is not None and st_norm >= 0.18:
        flags.append("st_elevation_proxy")
    if st_norm is not None and st_norm <= -0.18:
        flags.append("st_depression_proxy")
    if st_probability is not None and st_probability >= (st_threshold or 0.59):
        flags.append("edb_st_episode_model_positive")
    if sttc_probability is not None and sttc_probability >= (sttc_threshold or 0.39):
        flags.append("ptbxl_sttc_model_positive")
    edb_positive = st_probability is not None and st_probability >= (st_threshold or 0.59)
    sttc_positive = sttc_probability is not None and sttc_probability >= (sttc_threshold or 0.39)
    if st_probability is not None:
        if edb_positive:
            risk = "ischemia_st_abnormality_possible"
        elif sttc_positive:
            risk = "st_t_abnormality_possible_low_ischemia_specificity"
        else:
            risk = "low_st_abnormality_evidence"
        method = "edb12_st_feature_classifier_plus_ptbxl_sttc_classifier_and_j_point_proxy"
        confidence = min(0.78, max(0.34, abs(float(st_probability) - float(st_threshold or 0.59)) * 0.9 + 0.46))
        if sttc_probability is not None:
            confidence = max(confidence, min(0.76, abs(float(sttc_probability) - float(sttc_threshold or 0.39)) * 0.75 + 0.43))
    else:
        risk = "st_t_abnormality_possible_low_ischemia_specificity" if sttc_positive else "ischemia_st_abnormality_possible" if flags else "low_st_abnormality_evidence"
        method = "ptbxl_sttc_classifier_plus_j_point_st_proxy_with_qtdb_unet_delineation_evidence"
        confidence = min(0.66, max(morph.get("confidence", 0.0), 0.42 if not dl.get("error") else 0.0))
    return {"tool": "ECG_screen_ischemia_st", "st_deviation_proxy": st_raw, "st_deviation_normalized": st_norm, "st_abnormal_probability": st_probability, "decision_threshold": st_threshold, "st_model_source": st_model_source, "st_model_cv_metrics": st_cv_metrics, "sttc_abnormal_probability": sttc_probability, "sttc_decision_threshold": sttc_threshold, "ptbxl_sttc_model_source": sttc_model_source, "ptbxl_sttc_cv_metrics": sttc_cv_metrics, "ischemia_st_flags": flags, "ischemia_st_risk": risk, "morphology_result": morph, "dl_delineation_result": dl, "confidence": confidence, "method": method, "benchmark_direction": "European ST-T and Long-Term ST databases are the right labeled benchmarks for true ST/ischemia evaluation.", "disclaimer": "Single-lead normalized ST/ischemia screening is not diagnostic for ischemia or MI; use calibrated multi-lead ECG and clinical context."}


def _load_12lead_ecg_array(signal_path: str) -> tuple[np.ndarray | None, str | None]:
    path = Path(signal_path)
    try:
        if path.suffix.lower() == '.npy':
            arr = np.load(path)
        elif path.suffix.lower() == '.npz':
            data = np.load(path)
            key = data.files[0]
            arr = data[key]
        elif path.suffix.lower() in {'.hea', '.dat'} or (path.with_suffix('.hea').exists() and not path.is_file()):
            try:
                import wfdb
            except Exception as exc:
                return None, f'wfdb_unavailable:{type(exc).__name__}'
            record_base = path.with_suffix('') if path.suffix.lower() in {'.hea', '.dat'} else path
            rec = wfdb.rdrecord(str(record_base))
            arr = np.asarray(rec.p_signal, dtype=np.float32)
        else:
            try:
                arr = np.genfromtxt(path, delimiter=',', names=True, dtype=float)
                if getattr(arr, 'dtype', None) is not None and arr.dtype.names:
                    cols = []
                    for name in arr.dtype.names:
                        col = np.asarray(arr[name], dtype=np.float32)
                        if col.ndim == 1 and np.isfinite(col).any():
                            cols.append(col)
                    arr = np.stack(cols, axis=1) if cols else np.empty((0, 0), dtype=np.float32)
                else:
                    arr = np.asarray(arr, dtype=np.float32)
            except Exception:
                arr = np.loadtxt(path, delimiter=',', dtype=np.float32)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            return None, 'expected_12lead_input_but_found_single_vector'
        if arr.ndim > 2:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            return None, f'expected_2d_array_got_shape_{tuple(arr.shape)}'
        if arr.shape[0] == 12 and arr.shape[1] >= 12:
            leads = arr
        elif arr.shape[1] >= 12:
            leads = arr[:, :12].T
        else:
            return None, f'expected_at_least_12_leads_got_shape_{tuple(arr.shape)}'
        chans = [_robust_resample_ecg(leads[i], 1000) for i in range(12)]
        return np.stack(chans, axis=0).astype(np.float32), None
    except Exception as exc:
        return None, f'load_12lead_error:{type(exc).__name__}:{str(exc)[:120]}'


def _load_ptbxl_12lead_model(target: str):
    if torch is None or _ECGTwelveLeadResNet is None:
        return None, None, f'torch_unavailable_for_{target}'
    model_path = ECG_PTBXL_12LEAD_MODEL_DIR / f'ecg_ptbxl_{target}_12lead_resnet.pt'
    if not model_path.exists():
        return None, None, f'missing_model:{model_path}'
    try:
        cache_key = ('ecg_ptbxl_12lead_resnet', target, str(model_path), float(model_path.stat().st_mtime))
        cached = _TORCH_MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached[0], cached[1], None
        bundle = torch.load(model_path, map_location='cpu', weights_only=False)
        model = _ECGTwelveLeadResNet()
        model.load_state_dict(bundle['model_state_dict'])
        model.eval()
        _TORCH_MODEL_CACHE[cache_key] = (model, bundle)
        return model, bundle, None
    except Exception as exc:
        return None, None, f'model_load_error:{type(exc).__name__}:{str(exc)[:120]}'


def ECG_classify_12lead_ptbxl_superclasses(signal_path: str, sampling_rate: float | None = None, column: str | None = None) -> dict:
    """Classify PTB-XL diagnostic superclasses from 12-lead ECG using full PTB-XL ResNet models."""
    leads, load_error = _load_12lead_ecg_array(signal_path)
    if load_error or leads is None:
        return {"tool": "ECG_classify_12lead_ptbxl_superclasses", "error": load_error or "failed_to_load_12lead_ecg", "input_path": signal_path, "requires": "12-lead ECG CSV/NPY/NPZ/WFDB record with at least 12 leads"}
    probabilities = {}
    thresholds = {}
    predictions = {}
    model_paths = {}
    metrics = {}
    errors = {}
    x = torch.tensor(leads[None, :, :], dtype=torch.float32) if torch is not None else None
    for target in ECG_PTBXL_12LEAD_TARGETS:
        model, bundle, err = _load_ptbxl_12lead_model(target)
        if err or model is None or bundle is None or x is None:
            errors[target] = err or 'model_unavailable'
            continue
        with torch.no_grad():
            prob = float(torch.sigmoid(model(x)).detach().cpu().numpy()[0])
        thr = float(bundle.get('threshold', 0.5))
        probabilities[target.upper()] = prob
        thresholds[target.upper()] = thr
        predictions[target.upper()] = bool(prob >= thr)
        model_paths[target.upper()] = str(ECG_PTBXL_12LEAD_MODEL_DIR / f'ecg_ptbxl_{target}_12lead_resnet.pt')
        cv = bundle.get('cv_metrics') or {}
        metrics[target.upper()] = {k: cv.get(k) for k in ['average_precision', 'roc_auc', 'f1', 'precision', 'recall', 'accuracy', 'eval_records', 'eval_folds'] if k in cv}
    positive = [name for name, flag in predictions.items() if flag]
    return {
        "tool": "ECG_classify_12lead_ptbxl_superclasses",
        "input_path": signal_path,
        "classes": [t.upper() for t in ECG_PTBXL_12LEAD_TARGETS],
        "probabilities": probabilities,
        "decision_thresholds": thresholds,
        "predicted_positive_classes": positive,
        "predictions": predictions,
        "model_paths": model_paths,
        "cv_metrics": metrics,
        "errors": errors,
        "training_report": str(ECG_PTBXL_12LEAD_REPORT_PATH) if ECG_PTBXL_12LEAD_REPORT_PATH.exists() else None,
        "method": "ptbxl_full_12lead_resnet_superclass_binary_models_fold10_thresholds",
        "confidence": float(max(probabilities.values())) if probabilities else 0.0,
        "disclaimer": "PTB-XL superclass output is research-use screening/classification support only; it is not a clinical diagnosis and requires 12-lead ECG quality review.",
    }


def ECG_assess_stress_fatigue_hrv(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    hrv = ECG_compute_hrv(signal_path, sampling_rate, column)
    if hrv.get("error"):
        return {"tool": "ECG_assess_stress_fatigue_hrv", "error": hrv.get("error"), "hrv_result": hrv, "confidence": 0.1}
    rmssd = hrv.get("rmssd_ms")
    sdnn = hrv.get("sdnn_ms")
    lf_hf = hrv.get("lf_hf_ratio")
    mean_hr = hrv.get("mean_heart_rate_bpm") or hrv.get("median_heart_rate_bpm") or hrv.get("mean_hr_bpm") or hrv.get("median_hr_bpm")
    flags = []
    score = 0.0
    if mean_hr is not None and mean_hr > 95:
        flags.append("elevated_hr_stress_proxy"); score += 0.25
    if rmssd is not None and rmssd < 20:
        flags.append("low_rmssd_recovery_fatigue_proxy"); score += 0.30
    if sdnn is not None and sdnn < 30:
        flags.append("low_sdnn_autonomic_strain_proxy"); score += 0.25
    if lf_hf is not None and lf_hf > 3.0:
        flags.append("high_lf_hf_sympathetic_balance_proxy"); score += 0.20
    level = "high_strain_or_fatigue_proxy" if score >= 0.55 else "moderate_strain_proxy" if score >= 0.30 else "low_strain_proxy"
    return {"tool": "ECG_assess_stress_fatigue_hrv", "stress_fatigue_score": float(min(1.0, score)), "stress_fatigue_level": level, "stress_fatigue_flags": flags, "mean_hr_bpm": mean_hr, "rmssd_ms": rmssd, "sdnn_ms": sdnn, "lf_hf_ratio": lf_hf, "hrv_result": hrv, "confidence": min(0.72, hrv.get("confidence", 0.0)), "method": "hrv_autonomic_strain_recovery_proxy", "disclaimer": "Stress/fatigue inference from ECG HRV is nonspecific and should be contextualized with sleep, activity, EDA, symptoms, and baseline calibration."}


class _ECGDelineationConvBlock(nn.Module if nn is not None else object):
    def __init__(self, c_in: int, c_out: int):
        if nn is None:
            return
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, 9, padding=4), nn.BatchNorm1d(c_out), nn.ReLU(),
            nn.Conv1d(c_out, c_out, 9, padding=4), nn.BatchNorm1d(c_out), nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)


class _ECGTinyUNet1D(nn.Module if nn is not None else object):
    def __init__(self, classes: int = 4, base: int = 8):
        if nn is None:
            return
        super().__init__()
        self.e1 = _ECGDelineationConvBlock(1, base)
        self.e2 = _ECGDelineationConvBlock(base, base * 2)
        self.e3 = _ECGDelineationConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool1d(2)
        self.mid = _ECGDelineationConvBlock(base * 4, base * 8)
        self.u3 = nn.ConvTranspose1d(base * 8, base * 4, 2, stride=2)
        self.d3 = _ECGDelineationConvBlock(base * 8, base * 4)
        self.u2 = nn.ConvTranspose1d(base * 4, base * 2, 2, stride=2)
        self.d2 = _ECGDelineationConvBlock(base * 4, base * 2)
        self.u1 = nn.ConvTranspose1d(base * 2, base, 2, stride=2)
        self.d1 = _ECGDelineationConvBlock(base * 2, base)
        self.head = nn.Conv1d(base, classes, 1)
    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        z = self.mid(self.pool(e3))
        z = self.u3(z)
        if z.shape[-1] != e3.shape[-1]:
            z = torch.nn.functional.pad(z, (0, e3.shape[-1] - z.shape[-1]))
        z = self.d3(torch.cat([z, e3], 1))
        z = self.u2(z)
        if z.shape[-1] != e2.shape[-1]:
            z = torch.nn.functional.pad(z, (0, e2.shape[-1] - z.shape[-1]))
        z = self.d2(torch.cat([z, e2], 1))
        z = self.u1(z)
        if z.shape[-1] != e1.shape[-1]:
            z = torch.nn.functional.pad(z, (0, e1.shape[-1] - z.shape[-1]))
        z = self.d1(torch.cat([z, e1], 1))
        return self.head(z)


def _load_ecg_delineation_model():
    if torch is None or nn is None or not ECG_DELINEATION_MODEL_PATH.exists():
        return None
    key = str(ECG_DELINEATION_MODEL_PATH)
    if key in _TORCH_MODEL_CACHE:
        return _TORCH_MODEL_CACHE[key]
    checkpoint = torch.load(ECG_DELINEATION_MODEL_PATH, map_location='cpu', weights_only=False)
    base = int(checkpoint.get('base', 8)) if isinstance(checkpoint, dict) else 8
    model = _ECGTinyUNet1D(classes=4, base=base)
    state = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.eval()
    metrics = None
    metrics_path = ECG_DELINEATION_MODEL_PATH.with_name('ecg_delineation_qtdb_cached90_event_eval.json')
    if metrics_path.exists():
        try:
            import json
            metrics = json.loads(metrics_path.read_text())
        except Exception:
            metrics = None
    _TORCH_MODEL_CACHE[key] = (model, checkpoint if isinstance(checkpoint, dict) else {}, metrics)
    return _TORCH_MODEL_CACHE[key]


def _segments_from_mask(mask: np.ndarray, sampling_rate: float) -> dict:
    out = {}
    names = {1: 'p_wave', 2: 'qrs_complex', 3: 't_wave'}
    duration_limits_ms = {'p_wave': (20.0, 220.0), 'qrs_complex': (24.0, 220.0), 't_wave': (40.0, 520.0)}
    merge_gap = max(2, int(round(0.08 * sampling_rate)))
    for cls, name in names.items():
        binary = np.asarray(mask) == cls
        starts = np.flatnonzero(binary & ~np.r_[False, binary[:-1]])
        stops = np.flatnonzero(binary & ~np.r_[binary[1:], False]) + 1
        raw = [(int(s), int(e)) for s, e in zip(starts, stops) if e > s]
        merged = []
        for start, stop in raw:
            if merged and start <= merged[-1][1] + merge_gap:
                merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
            else:
                merged.append((start, stop))
        lo_ms, hi_ms = duration_limits_ms[name]
        lo = int(round(lo_ms * sampling_rate / 1000.0))
        hi = int(round(hi_ms * sampling_rate / 1000.0))
        segs = []
        for start, stop in merged:
            if not (lo <= stop - start <= hi):
                continue
            segs.append({'start_sample': int(start), 'stop_sample': int(stop), 'start_s': float(start / sampling_rate), 'stop_s': float(stop / sampling_rate), 'duration_ms': float((stop - start) * 1000.0 / sampling_rate)})
        out[name + '_segments'] = segs[:500]
        out[name + '_count'] = int(len(segs))
    return out


def ECG_delineate_waves_dl(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    if not (200 <= data.sampling_rate <= 300):
        return {'tool': 'ECG_delineate_waves_dl', 'error': 'QTDB delineation model currently expects ECG near 250 Hz', 'sampling_rate': data.sampling_rate, 'confidence': 0.1}
    loaded = _load_ecg_delineation_model()
    if loaded is None:
        return {'tool': 'ECG_delineate_waves_dl', 'error': 'delineation model weights unavailable', 'model_path': str(ECG_DELINEATION_MODEL_PATH), 'confidence': 0.0}
    if len(loaded) == 3:
        model, checkpoint, event_eval = loaded
    else:
        model, checkpoint = loaded
        event_eval = None
    x = data.values.astype(np.float32)
    med = float(np.nanmedian(x))
    iqr = float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25)) + 1e-6
    x_norm = np.clip((x - med) / iqr, -8, 8).astype(np.float32)
    window = 1024
    stride = 512
    votes = np.zeros((4, len(x_norm)), dtype=np.float32)
    counts = np.zeros(len(x_norm), dtype=np.float32)
    starts = list(range(0, max(1, len(x_norm) - window + 1), stride))
    if starts and starts[-1] + window < len(x_norm):
        starts.append(len(x_norm) - window)
    if not starts and len(x_norm) >= window:
        starts = [0]
    with torch.no_grad():
        for start in starts:
            seg = x_norm[start:start + window]
            if len(seg) < window:
                continue
            logits = model(torch.tensor(seg[None, None, :], dtype=torch.float32))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            votes[:, start:start + window] += probs
            counts[start:start + window] += 1.0
    valid = counts > 0
    if not np.any(valid):
        return {'tool': 'ECG_delineate_waves_dl', 'error': 'signal too short for 1024-sample delineation window', 'confidence': 0.1}
    votes[:, valid] /= counts[valid][None, :]
    pred = np.argmax(votes, axis=0).astype(np.uint8)
    pred[~valid] = 0
    segments = _segments_from_mask(pred, data.sampling_rate)
    wave_fraction = float(np.mean(pred > 0))
    confidence = 0.45
    if segments.get('qrs_complex_count', 0) >= 3:
        confidence = 0.58
    event_metrics = None
    delineation_quality_flags = []
    if isinstance(event_eval, dict):
        event_metrics = event_eval.get('results')
        t_f1 = (((event_metrics or {}).get('t') or {}).get('f1')) if isinstance(event_metrics, dict) else None
        macro_f1 = (event_metrics or {}).get('macro_event_f1') if isinstance(event_metrics, dict) else None
        if t_f1 is not None and t_f1 < 0.20:
            delineation_quality_flags.append('weak_t_wave_event_validation')
        if macro_f1 is not None and macro_f1 < 0.50:
            delineation_quality_flags.append('low_macro_event_validation')
    return {'tool': 'ECG_delineate_waves_dl', 'model_source': str(ECG_DELINEATION_MODEL_PATH), 'training_source': 'QTDB q1c/q2c cached 90-record manual delineation subset', 'class_labels': {'0': 'background', '1': 'p_wave', '2': 'qrs_complex', '3': 't_wave'}, **segments, 'wave_pixel_fraction': wave_fraction, 'event_validation_metrics': event_metrics, 'delineation_quality_flags': delineation_quality_flags, 'confidence': confidence, 'method': 'qtdb_cached90_unet_p_qrs_t_segmentation_experimental', 'disclaimer': 'Experimental QTDB model; event-level validation is weak for T waves, so use as optional morphology evidence only until stronger LUDB/full-QTDB validation.'}
