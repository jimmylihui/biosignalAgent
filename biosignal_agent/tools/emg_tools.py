from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal

from .common import bandpass_filter, load_csv_signal, signal_quality_summary


def EMG_assess_quality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    return {"tool": "EMG_assess_quality", "source": data.source, **signal_quality_summary(data.values)}


def EMG_summarize_activation(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    if len(data.values) == 0:
        return {"tool": "EMG_summarize_activation", "error": "empty signal", "confidence": 0.0}
    high = min(150.0, data.sampling_rate * 0.45)
    filtered = bandpass_filter(data.values, data.sampling_rate, low_hz=20.0, high_hz=high, order=3) if high > 25 else data.values
    rms = float(np.sqrt(np.nanmean(filtered ** 2)))
    mav = float(np.nanmean(np.abs(filtered)))
    return {"tool": "EMG_summarize_activation", "rms": rms, "mean_absolute_value": mav, "num_samples": int(len(filtered)), "confidence": 0.65, "method": "bandpass_rms_summary"}



def EMG_estimate_fatigue(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    if len(data.values) < max(8, int(data.sampling_rate)):
        return {"tool": "EMG_estimate_fatigue", "error": "signal too short", "confidence": 0.0}
    high = min(250.0, data.sampling_rate * 0.45)
    filtered = bandpass_filter(data.values, data.sampling_rate, low_hz=20.0, high_hz=high, order=3) if high > 25 else data.values
    freqs, psd = scipy_signal.welch(filtered, fs=data.sampling_rate, nperseg=min(len(filtered), int(data.sampling_rate * 2)))
    mask = (freqs >= 20) & (freqs <= high)
    if not np.any(mask):
        return {"tool": "EMG_estimate_fatigue", "error": "insufficient EMG bandwidth", "confidence": 0.1}
    f = freqs[mask]
    pxx = psd[mask]
    cumulative = np.cumsum(pxx)
    median_frequency = float(f[np.searchsorted(cumulative, cumulative[-1] / 2.0)]) if cumulative[-1] > 0 else None
    rms = float(np.sqrt(np.nanmean(filtered ** 2)))
    fatigue_proxy = "possible_fatigue_proxy" if median_frequency is not None and median_frequency < 60 else "no_fatigue_proxy"
    return {
        "tool": "EMG_estimate_fatigue",
        "median_frequency_hz": median_frequency,
        "rms": rms,
        "fatigue_proxy": fatigue_proxy,
        "confidence": 0.55,
        "method": "emg_median_frequency_screening",
        "disclaimer": "Screening heuristic only; muscle fatigue needs task protocol, normalization, and repeated contractions.",
    }



def EMG_detect_bursts(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    if len(values) < max(8, int(data.sampling_rate)):
        return {"tool": "EMG_detect_bursts", "error": "signal too short", "confidence": 0.0}
    high = min(250.0, data.sampling_rate * 0.45)
    filtered = bandpass_filter(values, data.sampling_rate, low_hz=20.0, high_hz=high, order=3) if high > 25 else values
    window = max(1, int(0.05 * data.sampling_rate))
    envelope = np.sqrt(np.convolve(filtered ** 2, np.ones(window) / window, mode="same"))
    threshold = max(float(np.nanmedian(envelope) + 3.0 * np.nanstd(envelope)), 1e-8)
    active = envelope > threshold
    min_len = max(1, int(0.05 * data.sampling_rate))
    bursts = []
    start = None
    for idx, flag in enumerate(active):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            if idx - start >= min_len:
                bursts.append((start, idx))
            start = None
    if start is not None and len(active) - start >= min_len:
        bursts.append((start, len(active)))
    return {
        "tool": "EMG_detect_bursts",
        "burst_count": int(len(bursts)),
        "burst_intervals_s": [[float(a / data.sampling_rate), float(b / data.sampling_rate)] for a, b in bursts[:20]],
        "activation_threshold": threshold,
        "confidence": 0.55,
        "method": "emg_rms_envelope_burst_detection",
        "disclaimer": "Burst/onset proxy only; task-specific EMG onset detection requires protocol and labeled contractions.",
    }


def _emg_read_multichannel(signal_path: str, columns: list[str] | None = None) -> np.ndarray:
    import pandas as pd

    try:
        frame = pd.read_csv(signal_path, sep=None, engine="python")
    except Exception:
        frame = pd.read_csv(signal_path, sep=r"\s+", header=None, engine="python")
    if frame.shape[1] == 1:
        frame = pd.read_csv(signal_path, sep=r"\s+", header=None, engine="python")
    if columns:
        values = frame[columns].to_numpy(dtype=float)
    else:
        numeric = frame.select_dtypes(include=[np.number]).copy()
        drop = [name for name in numeric.columns if str(name).lower() in {"time", "class", "label", "target"}]
        numeric = numeric.drop(columns=drop, errors="ignore")
        values = numeric.iloc[:, :8].to_numpy(dtype=float)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("expected at least one numeric EMG channel")
    if values.shape[1] < 8:
        pad = np.zeros((values.shape[0], 8 - values.shape[1]), dtype=float)
        values = np.concatenate([values, pad], axis=1)
    return np.nan_to_num(values[:, :8])


def _emg_gesture_features(window: np.ndarray) -> list[float]:
    features: list[float] = []
    for channel in range(window.shape[1]):
        x = window[:, channel].astype(float) - float(np.median(window[:, channel]))
        dx = np.diff(x)
        dd = np.diff(dx)
        threshold = max(float(np.std(x)) * 0.02, 1e-7)
        features.extend([
            float(np.sqrt(np.mean(x * x))),
            float(np.mean(np.abs(x))),
            float(np.sum(np.abs(dx))),
            float(np.var(x)),
            float(np.mean(np.abs(dx))),
            float(np.mean(np.abs(x) > threshold)),
            float(np.sum((x[:-1] * x[1:] < 0) & (np.abs(x[:-1] - x[1:]) > threshold)) / max(1, len(x))),
            float(np.sum((dx[:-1] * dx[1:] < 0) & (np.abs(dd) > threshold)) / max(1, len(dd))),
        ])
    rms = np.sqrt(np.mean(window * window, axis=0)) + 1e-12
    features.extend([float(rms.max() / rms.min()), float(rms.std() / rms.mean()), float(np.argmax(rms))])
    return features


def _emg_median_frequency(values: np.ndarray, sampling_rate: float) -> float:
    freqs, psd = scipy_signal.welch(values, fs=sampling_rate, nperseg=min(len(values), int(sampling_rate * 2)))
    mask = (freqs >= 10) & (freqs <= min(95, sampling_rate * 0.45))
    if not np.any(mask):
        return 0.0
    freqs = freqs[mask]
    psd = psd[mask]
    cumulative = np.cumsum(psd)
    if cumulative[-1] <= 0:
        return 0.0
    return float(freqs[np.searchsorted(cumulative, cumulative[-1] / 2.0)])


def _emg_fatigue_ml_features(window: np.ndarray, sampling_rate: float) -> list[float]:
    features: list[float] = []
    for channel in range(window.shape[1]):
        x = window[:, channel].astype(float) - float(np.median(window[:, channel]))
        dx = np.diff(x)
        variance = float(np.var(x))
        centered = x - float(np.mean(x))
        std = float(np.sqrt(variance + 1e-12))
        skew = float(np.mean((centered / std) ** 3))
        kurtosis = float(np.mean((centered / std) ** 4) - 3.0)
        features.extend([
            float(np.sqrt(np.mean(x * x))),
            float(np.mean(np.abs(x))),
            variance,
            float(np.sum(np.abs(dx))),
            float(np.mean(np.abs(dx))),
            _emg_median_frequency(x, sampling_rate),
            float(np.mean(np.abs(x) > std * 0.5)),
            skew,
            kurtosis,
        ])
    return features


def _load_joblib_model(path: str):
    import joblib
    return joblib.load(path)



def _emg_read_single_channel(signal_path: str, column: str | None = None) -> np.ndarray:
    import pandas as pd

    try:
        frame = pd.read_csv(signal_path, sep=None, engine="python")
    except Exception:
        frame = pd.read_csv(signal_path, sep=r"\s+", header=None, engine="python")
    if frame.shape[1] == 1:
        try:
            frame = pd.read_csv(signal_path, sep=r"\s+", header=None, engine="python")
        except Exception:
            pass
    if column is not None and column in frame.columns:
        values = frame[column].to_numpy(dtype=float)
    else:
        numeric = frame.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            raise ValueError("expected at least one numeric EMG channel")
        values = numeric.iloc[:, 0].to_numpy(dtype=float)
    return np.nan_to_num(values[np.isfinite(values)])


def _emg_neuromuscular_features(values: np.ndarray, sampling_rate: float) -> list[float]:
    x = np.asarray(values, dtype=float)
    x = np.nan_to_num(x - float(np.nanmedian(x)))
    absx = np.abs(x)
    dx = np.diff(x, prepend=x[0])
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    waveform_length = float(np.sum(np.abs(np.diff(x))))
    zero_crossing_rate = float(np.mean((x[:-1] * x[1:]) < 0)) if len(x) > 1 else 0.0
    slope_sign_changes = float(np.mean(np.diff(np.sign(np.diff(x))) != 0)) if len(x) > 2 else 0.0
    freqs, psd = scipy_signal.welch(x, fs=sampling_rate, nperseg=min(len(x), 1024), noverlap=min(len(x) // 2, 512))
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    total_power = float(trap(psd, freqs) + 1e-12)
    cumulative = np.cumsum(psd)
    median_frequency = float(freqs[np.searchsorted(cumulative, cumulative[-1] / 2.0)]) if len(freqs) and cumulative[-1] > 0 else 0.0
    mean_frequency = float(trap(freqs * psd, freqs) / total_power) if total_power > 0 else 0.0
    centered = x - float(np.mean(x))
    std = float(np.std(x) + 1e-12)
    skew = float(np.mean((centered / std) ** 3))
    kurtosis = float(np.mean((centered / std) ** 4) - 3.0)
    band_ratios = []
    for low, high in [(20, 60), (60, 150), (150, 300), (300, 450), (450, 900), (900, 1600)]:
        mask = (freqs >= low) & (freqs < min(high, sampling_rate * 0.5))
        band_ratios.append(float(trap(psd[mask], freqs[mask]) / total_power) if np.any(mask) else 0.0)
    envelope = np.abs(scipy_signal.hilbert(x)) if len(x) > 8 else absx
    envelope_median = float(np.median(envelope) + 1e-12)
    return [
        rms,
        float(np.mean(absx)),
        float(np.var(x)),
        waveform_length,
        float(np.mean(np.abs(dx))),
        float(np.percentile(absx, 50)),
        float(np.percentile(absx, 75)),
        float(np.percentile(absx, 90)),
        float(np.percentile(absx, 95)),
        skew,
        kurtosis,
        zero_crossing_rate,
        slope_sign_changes,
        median_frequency,
        mean_frequency,
        *band_ratios,
        float(np.std(envelope) / (np.mean(envelope) + 1e-12)),
        float(np.percentile(envelope, 95) / envelope_median),
        float(np.mean(envelope > np.percentile(envelope, 90))),
    ]


def EMG_classify_gesture(signal_path: str, sampling_rate: float = 200.0, columns: list[str] | None = None, window_sec: float = 0.25, step_sec: float = 0.25) -> dict:
    model_path = "/data1/jiahui/biosignal-agent/outputs/emg_uci_gesture_6class_feature_ensemble.joblib"
    try:
        bundle = _load_joblib_model(model_path)
        model = bundle["model"]
        classes = [int(v) for v in bundle["classes"]]
        values = _emg_read_multichannel(signal_path, columns)
    except Exception as exc:
        return {"tool": "EMG_classify_gesture", "error": str(exc), "confidence": 0.0}
    win = max(8, int(window_sec * sampling_rate))
    step = max(1, int(step_sec * sampling_rate))
    if len(values) < win:
        return {"tool": "EMG_classify_gesture", "error": "signal too short", "confidence": 0.0}
    X = np.asarray([_emg_gesture_features(values[start:start + win]) for start in range(0, len(values) - win + 1, step)], dtype=float)
    probabilities = model.predict_proba(X)
    mean_prob = probabilities.mean(axis=0)
    best_idx = int(np.argmax(mean_prob))
    class_names = {1: "rest", 2: "fist", 3: "wrist_flexion", 4: "wrist_extension", 5: "radial_deviation", 6: "ulnar_deviation"}
    predicted_class = classes[best_idx]
    return {
        "tool": "EMG_classify_gesture",
        "predicted_class": int(predicted_class),
        "predicted_gesture": class_names.get(predicted_class, str(predicted_class)),
        "class_probabilities": {class_names.get(c, str(c)): float(mean_prob[i]) for i, c in enumerate(classes)},
        "num_windows": int(len(X)),
        "confidence": float(mean_prob[best_idx]),
        "method": "uci_gesture_6class_hudgins_feature_ensemble",
        "model_path": model_path,
    }


def EMG_estimate_fatigue_ml(signal_path: str, sampling_rate: float = 200.0, columns: list[str] | None = None, window_sec: float = 4.0, step_sec: float = 2.0) -> dict:
    model_path = "/data1/jiahui/biosignal-agent/outputs/emg_fatigue_feature_ensemble.joblib"
    try:
        bundle = _load_joblib_model(model_path)
        model = bundle["model"]
        values = _emg_read_multichannel(signal_path, columns)
    except Exception as exc:
        return {"tool": "EMG_estimate_fatigue_ml", "error": str(exc), "confidence": 0.0}
    win = max(8, int(window_sec * sampling_rate))
    step = max(1, int(step_sec * sampling_rate))
    if len(values) < win:
        return {"tool": "EMG_estimate_fatigue_ml", "error": "signal too short", "confidence": 0.0}
    X = np.asarray([_emg_fatigue_ml_features(values[start:start + win], sampling_rate) for start in range(0, len(values) - win + 1, step)], dtype=float)
    fatigue_prob = float(model.predict_proba(X)[:, 1].mean())
    return {
        "tool": "EMG_estimate_fatigue_ml",
        "fatigue_probability": fatigue_prob,
        "fatigue_state": "likely_fatigued" if fatigue_prob >= 0.5 else "likely_non_fatigued",
        "num_windows": int(len(X)),
        "confidence": float(abs(fatigue_prob - 0.5) * 2.0),
        "method": "emg_fatigue_feature_ensemble_early_vs_late_protocol_proxy",
        "model_path": model_path,
        "disclaimer": "Model was trained on early-vs-late fatigue protocol labels; clinical fatigue interpretation requires task normalization and validation.",
    }


def _emg_physical_action_features(window: np.ndarray) -> list[float]:
    features: list[float] = []
    for channel in range(window.shape[1]):
        x = window[:, channel].astype(float) - float(np.median(window[:, channel]))
        dx = np.diff(x)
        dd = np.diff(dx)
        threshold = max(float(np.std(x)) * 0.02, 1e-7)
        features.extend([
            float(np.sqrt(np.mean(x * x))),
            float(np.mean(np.abs(x))),
            float(np.var(x)),
            float(np.sum(np.abs(dx))),
            float(np.mean(np.abs(dx))),
            float(np.percentile(np.abs(x), 75)),
            float(np.percentile(np.abs(x), 95)),
            float(np.sum((x[:-1] * x[1:] < 0) & (np.abs(x[:-1] - x[1:]) > threshold)) / max(1, len(x))),
            float(np.sum((dx[:-1] * dx[1:] < 0) & (np.abs(dd) > threshold)) / max(1, len(dd))),
        ])
    rms = np.sqrt(np.mean(window * window, axis=0)) + 1e-12
    features.extend([float(rms.max() / rms.min()), float(rms.std() / rms.mean()), float(np.argmax(rms))])
    return features


def EMG_classify_physical_action(signal_path: str, sampling_rate: float = 1000.0, columns: list[str] | None = None, window_samples: int = 250, step_samples: int = 125) -> dict:
    model_path = "/data1/jiahui/biosignal-agent/outputs/emg_physical_action_aggressive_binary_feature_ensemble.joblib"
    try:
        bundle = _load_joblib_model(model_path)
        model = bundle["model"]
        values = _emg_read_multichannel(signal_path, columns)
    except Exception as exc:
        return {"tool": "EMG_classify_physical_action", "error": str(exc), "confidence": 0.0}
    win = max(8, int(window_samples))
    step = max(1, int(step_samples))
    if len(values) < win:
        return {"tool": "EMG_classify_physical_action", "error": "signal too short", "confidence": 0.0}
    X = np.asarray([_emg_physical_action_features(values[start:start + win]) for start in range(0, len(values) - win + 1, step)], dtype=float)
    aggressive_prob = float(model.predict_proba(X)[:, 1].mean())
    return {
        "tool": "EMG_classify_physical_action",
        "action_category": "aggressive_or_high_intensity" if aggressive_prob >= 0.5 else "normal_activity",
        "aggressive_probability": aggressive_prob,
        "num_windows": int(len(X)),
        "confidence": float(abs(aggressive_prob - 0.5) * 2.0),
        "method": "uci_physical_action_binary_hudgins_feature_ensemble",
        "model_path": model_path,
        "disclaimer": "This is a dataset-specific normal/aggressive physical-action screen, not a safety or violence detector for deployment without validation.",
    }


def EMG_classify_action(signal_path: str, sampling_rate: float = 1000.0, columns: list[str] | None = None, window_samples: int = 250, step_samples: int = 125) -> dict:
    model_path = "/data1/jiahui/biosignal-agent/outputs/emg_physical_action_20class_feature_ensemble.joblib"
    try:
        bundle = _load_joblib_model(model_path)
        model = bundle["model"]
        label_encoder = bundle["label_encoder"]
        values = _emg_read_multichannel(signal_path, columns)
    except Exception as exc:
        return {"tool": "EMG_classify_action", "error": str(exc), "confidence": 0.0}
    win = max(8, int(window_samples))
    step = max(1, int(step_samples))
    if len(values) < win:
        return {"tool": "EMG_classify_action", "error": "signal too short", "confidence": 0.0}
    features = np.asarray([_emg_physical_action_features(values[start:start + win]) for start in range(0, len(values) - win + 1, step)], dtype=float)
    probabilities = model.predict_proba(features).mean(axis=0)
    labels = [str(label) for label in label_encoder.classes_]
    order = np.argsort(probabilities)[::-1]
    best_idx = int(order[0])
    return {
        "tool": "EMG_classify_action",
        "predicted_action": labels[best_idx],
        "action_probabilities": {labels[int(idx)]: float(probabilities[int(idx)]) for idx in order[:10]},
        "top5_actions": [labels[int(idx)] for idx in order[:5]],
        "num_windows": int(len(features)),
        "confidence": float(probabilities[best_idx]),
        "method": "uci_physical_action_20class_hudgins_feature_ensemble",
        "model_path": model_path,
        "disclaimer": "20-class UCI Physical Action model reaches subject-held-out accuracy 0.269/macro-F1 0.266 on four subjects; use as a coarse research baseline, not a robust activity recognizer.",
    }


def EMG_predict_movement_intent(signal_path: str, sampling_rate: float = 200.0, columns: list[str] | None = None, window_sec: float = 0.25, step_sec: float = 0.25) -> dict:
    model_path = "/data1/jiahui/biosignal-agent/outputs/emg_uci_intention_preonset_feature_ensemble.joblib"
    try:
        bundle = _load_joblib_model(model_path)
        model = bundle["model"]
        classes = [int(v) for v in bundle["classes"]]
        values = _emg_read_multichannel(signal_path, columns)
    except Exception as exc:
        return {"tool": "EMG_predict_movement_intent", "error": str(exc), "confidence": 0.0}
    win = max(8, int(window_sec * sampling_rate))
    step = max(1, int(step_sec * sampling_rate))
    if len(values) < win:
        return {"tool": "EMG_predict_movement_intent", "error": "signal too short", "confidence": 0.0}
    X = np.asarray([_emg_gesture_features(values[start:start + win]) for start in range(0, len(values) - win + 1, step)], dtype=float)
    probabilities = model.predict_proba(X)
    mean_prob = probabilities.mean(axis=0)
    best_idx = int(np.argmax(mean_prob))
    class_names = {2: "intend_fist", 3: "intend_wrist_flexion", 4: "intend_wrist_extension", 5: "intend_radial_deviation", 6: "intend_ulnar_deviation"}
    predicted_class = classes[best_idx]
    return {
        "tool": "EMG_predict_movement_intent",
        "predicted_intent_class": int(predicted_class),
        "predicted_intent": class_names.get(predicted_class, str(predicted_class)),
        "intent_probabilities": {class_names.get(c, str(c)): float(mean_prob[i]) for i, c in enumerate(classes)},
        "num_windows": int(len(X)),
        "confidence": float(mean_prob[best_idx]),
        "method": "uci_preonset_intention_hudgins_feature_ensemble",
        "model_path": model_path,
        "disclaimer": "Intent labels are inferred from gesture onset timing; real-time intent use requires causal validation and calibration.",
    }



def EMG_screen_neuromuscular_abnormality(signal_path: str, sampling_rate: float = 4000.0, column: str | None = None, window_sec: float = 0.5, step_sec: float = 0.25) -> dict:
    model_path = "/data1/jiahui/biosignal-agent/outputs/emgdb_neuromuscular_condition_feature_ensemble.joblib"
    try:
        bundle = _load_joblib_model(model_path)
        model = bundle["model"]
        label_encoder = bundle["label_encoder"]
        values = _emg_read_single_channel(signal_path, column)
    except Exception as exc:
        return {"tool": "EMG_screen_neuromuscular_abnormality", "error": str(exc), "confidence": 0.0}
    win = max(16, int(window_sec * sampling_rate))
    step = max(1, int(step_sec * sampling_rate))
    if len(values) < win:
        return {"tool": "EMG_screen_neuromuscular_abnormality", "error": "signal too short", "confidence": 0.0}
    features = np.asarray([_emg_neuromuscular_features(values[start:start + win], sampling_rate) for start in range(0, len(values) - win + 1, step)], dtype=float)
    probabilities = model.predict_proba(features).mean(axis=0)
    labels = list(label_encoder.classes_)
    best_idx = int(np.argmax(probabilities))
    probability_map = {str(label): float(probabilities[idx]) for idx, label in enumerate(labels)}
    abnormal_probability = float(sum(probability_map.get(label, 0.0) for label in ("myopathy", "neuropathy")))
    return {
        "tool": "EMG_screen_neuromuscular_abnormality",
        "predicted_condition": str(labels[best_idx]),
        "condition_probabilities": probability_map,
        "abnormal_probability": abnormal_probability,
        "num_windows": int(len(features)),
        "confidence": float(probabilities[best_idx]),
        "method": "emgdb_healthy_myopathy_neuropathy_feature_ensemble_smoke",
        "model_path": model_path,
        "disclaimer": "Tiny EMGDB smoke model trained from one processed file per class with window-level CV; research screen only, not a diagnosis or deployable neuromuscular disease detector.",
    }



def _emg_read_lower_limb(signal_path: str, columns: list[str] | None = None) -> np.ndarray:
    import pandas as pd

    rows: list[list[float]] = []
    try:
        for line in Path(signal_path).read_text(errors="ignore").splitlines():
            parts = line.strip().replace(",", ".").split()
            if len(parts) >= 5:
                try:
                    rows.append([float(value) for value in parts[:5]])
                except ValueError:
                    pass
    except Exception:
        rows = []
    if rows:
        return np.nan_to_num(np.asarray(rows, dtype=float))
    try:
        frame = pd.read_csv(signal_path, sep=None, engine="python")
    except Exception:
        frame = pd.read_csv(signal_path, sep=r"\s+", header=None, engine="python")
    if columns:
        values = frame[columns].to_numpy(dtype=float)
    else:
        numeric = frame.select_dtypes(include=[np.number])
        values = numeric.iloc[:, :5].to_numpy(dtype=float)
    if values.ndim != 2 or values.shape[1] < 4:
        raise ValueError("expected at least four lower-limb EMG channels")
    if values.shape[1] == 4:
        values = np.concatenate([values, np.zeros((values.shape[0], 1), dtype=float)], axis=1)
    return np.nan_to_num(values[:, :5])


def _emg_lower_limb_channel_features(values: np.ndarray, sampling_rate: float) -> list[float]:
    x = np.asarray(values, dtype=float)
    x = np.nan_to_num(x - float(np.median(x)))
    absx = np.abs(x)
    dx = np.diff(x)
    dd = np.diff(dx)
    std = float(np.std(x) + 1e-12)
    threshold = max(std * 0.02, 1e-8)
    freqs, psd = scipy_signal.welch(x, fs=sampling_rate, nperseg=min(len(x), 1024), noverlap=min(len(x) // 2, 512))
    mask = (freqs >= 20) & (freqs <= min(450, sampling_rate * 0.45))
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    if np.any(mask):
        band_freqs = freqs[mask]
        band_psd = psd[mask]
        cumulative = np.cumsum(band_psd)
        median_frequency = float(band_freqs[np.searchsorted(cumulative, cumulative[-1] / 2.0)]) if cumulative[-1] > 0 else 0.0
        mean_frequency = float(trap(band_freqs * band_psd, band_freqs) / (trap(band_psd, band_freqs) + 1e-12))
    else:
        median_frequency = 0.0
        mean_frequency = 0.0
    centered = x - float(np.mean(x))
    skew = float(np.mean((centered / std) ** 3)) if std > 1e-10 else 0.0
    kurtosis = float(np.mean((centered / std) ** 4) - 3.0) if std > 1e-10 else 0.0
    return [
        float(np.sqrt(np.mean(x * x) + 1e-12)),
        float(np.mean(absx)),
        float(np.var(x)),
        float(np.sum(np.abs(dx))),
        float(np.mean(np.abs(dx))) if len(dx) else 0.0,
        float(np.percentile(absx, 50)),
        float(np.percentile(absx, 75)),
        float(np.percentile(absx, 95)),
        float(np.sum((x[:-1] * x[1:] < 0) & (np.abs(x[:-1] - x[1:]) > threshold)) / max(1, len(x) - 1)) if len(x) > 1 else 0.0,
        float(np.sum((dx[:-1] * dx[1:] < 0) & (np.abs(dd) > threshold)) / max(1, len(dd))) if len(dd) else 0.0,
        skew,
        kurtosis,
        median_frequency,
        mean_frequency,
    ]


def _emg_lower_limb_features(window: np.ndarray, sampling_rate: float) -> list[float]:
    emg = window[:, :4]
    flexion = window[:, 4]
    features: list[float] = []
    for channel in range(4):
        features.extend(_emg_lower_limb_channel_features(emg[:, channel], sampling_rate))
    centered = emg - np.median(emg, axis=0)
    rms = np.sqrt(np.mean(centered * centered, axis=0) + 1e-12)
    dfx = np.diff(flexion)
    features.extend([
        float(rms.max() / rms.min()),
        float(rms.std() / rms.mean()),
        float(np.argmax(rms)),
        float(np.mean(flexion)),
        float(np.std(flexion)),
        float(np.ptp(flexion)),
        float(np.percentile(flexion, 5)),
        float(np.percentile(flexion, 95)),
        float(np.mean(np.abs(dfx))) if len(dfx) else 0.0,
    ])
    return features


def _emg_lower_limb_predict(signal_path: str, model_path: str, tool_name: str, sampling_rate: float, columns: list[str] | None, window_sec: float, step_sec: float) -> dict:
    try:
        bundle = _load_joblib_model(model_path)
        model = bundle["model"]
        label_encoder = bundle["label_encoder"]
        values = _emg_read_lower_limb(signal_path, columns)
    except Exception as exc:
        return {"tool": tool_name, "error": str(exc), "confidence": 0.0}
    win = max(16, int(window_sec * sampling_rate))
    step = max(1, int(step_sec * sampling_rate))
    if len(values) < win:
        return {"tool": tool_name, "error": "signal too short", "confidence": 0.0}
    features = np.asarray([_emg_lower_limb_features(values[start:start + win], sampling_rate) for start in range(0, len(values) - win + 1, step)], dtype=float)
    probabilities = model.predict_proba(features).mean(axis=0)
    labels = list(label_encoder.classes_)
    best_idx = int(np.argmax(probabilities))
    return {
        "predicted_label": str(labels[best_idx]),
        "label_probabilities": {str(label): float(probabilities[idx]) for idx, label in enumerate(labels)},
        "num_windows": int(len(features)),
        "confidence": float(probabilities[best_idx]),
        "model_path": model_path,
    }


def EMG_classify_lower_limb_exercise(signal_path: str, sampling_rate: float = 1000.0, columns: list[str] | None = None, window_sec: float = 1.0, step_sec: float = 0.5) -> dict:
    model_path = "/data1/jiahui/biosignal-agent/outputs/emg_lower_limb_exercise_feature_ensemble.joblib"
    result = _emg_lower_limb_predict(signal_path, model_path, "EMG_classify_lower_limb_exercise", sampling_rate, columns, window_sec, step_sec)
    if "error" in result:
        return result
    return {
        "tool": "EMG_classify_lower_limb_exercise",
        "predicted_exercise": result["predicted_label"],
        "exercise_probabilities": result["label_probabilities"],
        "num_windows": result["num_windows"],
        "confidence": result["confidence"],
        "method": "uci_lower_limb_subject_heldout_feature_ensemble",
        "model_path": model_path,
        "disclaimer": "Model was trained on UCI lower-limb EMG/goniometry from 22 male subjects; validate before using for new rehab protocols or devices.",
    }


def EMG_screen_knee_rehab_status(signal_path: str, sampling_rate: float = 1000.0, columns: list[str] | None = None, window_sec: float = 1.0, step_sec: float = 0.5) -> dict:
    model_path = "/data1/jiahui/biosignal-agent/outputs/emg_lower_limb_knee_status_feature_ensemble.joblib"
    result = _emg_lower_limb_predict(signal_path, model_path, "EMG_screen_knee_rehab_status", sampling_rate, columns, window_sec, step_sec)
    if "error" in result:
        return result
    abnormal_probability = float(result["label_probabilities"].get("abnormal_knee", 0.0))
    return {
        "tool": "EMG_screen_knee_rehab_status",
        "predicted_status": result["predicted_label"],
        "status_probabilities": result["label_probabilities"],
        "abnormal_knee_probability": abnormal_probability,
        "num_windows": result["num_windows"],
        "confidence": result["confidence"],
        "method": "uci_lower_limb_knee_status_subject_heldout_feature_ensemble",
        "model_path": model_path,
        "disclaimer": "Research rehab screen only. It was trained on a small UCI lower-limb cohort and must not be used as a knee diagnosis without protocol/device validation.",
    }


def EMG_analyze_gait_activation(signal_path: str, sampling_rate: float = 1000.0, columns: list[str] | None = None, window_sec: float = 0.2) -> dict:
    try:
        values = _emg_read_lower_limb(signal_path, columns)
    except Exception as exc:
        return {"tool": "EMG_analyze_gait_activation", "error": str(exc), "confidence": 0.0}
    if len(values) < max(16, int(window_sec * sampling_rate)):
        return {"tool": "EMG_analyze_gait_activation", "error": "signal too short", "confidence": 0.0}
    emg = values[:, :4]
    flexion = values[:, 4]
    muscle_names = ["rectus_femoris", "biceps_femoris", "vastus_medialis", "semitendinosus"]
    window = max(1, int(window_sec * sampling_rate))
    kernel = np.ones(window) / window
    channel_summary = {}
    activation_matrix = []
    for idx, name in enumerate(muscle_names):
        x = emg[:, idx] - float(np.median(emg[:, idx]))
        envelope = np.sqrt(np.convolve(x * x, kernel, mode="same"))
        threshold = float(np.median(envelope) + 2.0 * np.std(envelope))
        active = envelope > threshold
        activation_matrix.append(active)
        channel_summary[name] = {
            "rms": float(np.sqrt(np.mean(x * x) + 1e-12)),
            "mean_absolute_value": float(np.mean(np.abs(x))),
            "activation_duty_fraction": float(np.mean(active)),
            "activation_threshold": threshold,
        }
    activation_matrix_np = np.vstack(activation_matrix)
    coactivation_fraction = float(np.mean(np.sum(activation_matrix_np, axis=0) >= 2))
    quad_rms = channel_summary["rectus_femoris"]["rms"] + channel_summary["vastus_medialis"]["rms"]
    ham_rms = channel_summary["biceps_femoris"]["rms"] + channel_summary["semitendinosus"]["rms"]
    flexion_range = float(np.ptp(flexion))
    return {
        "tool": "EMG_analyze_gait_activation",
        "muscle_activation": channel_summary,
        "coactivation_fraction": coactivation_fraction,
        "quadriceps_to_hamstrings_rms_ratio": float(quad_rms / (ham_rms + 1e-12)),
        "knee_flexion_range_deg": flexion_range,
        "num_samples": int(len(values)),
        "confidence": 0.65,
        "method": "lower_limb_emg_rms_envelope_activation_summary",
        "disclaimer": "Protocol-aware gait interpretation needs gait-event labels, side/channel mapping, normalization, and subject calibration.",
    }



_GEDS_PHASE_COLUMNS = [
    "EMG_taR", "EMG_taL", "ACCx_taR", "ACCy_taR", "ACCz_taR", "GYRx_taR", "GYRy_taR", "GYRz_taR",
    "ACCx_taL", "ACCy_taL", "ACCz_taL", "GYRx_taL", "GYRy_taL", "GYRz_taL", "FSR_hsR", "FSR_toR", "FSR_hsL", "FSR_toL",
]


def _emg_read_geds_frame(signal_path: str, columns: list[str] | None = None):
    import pandas as pd

    frame = pd.read_csv(signal_path, sep="\t")
    numeric = frame.select_dtypes(include=[np.number]).copy()
    if columns:
        missing = [column for column in columns if column not in numeric.columns]
        if missing:
            raise ValueError(f"missing GEDS columns: {missing}")
        return numeric[columns]
    return numeric


def _emg_geds_column_features(values: np.ndarray) -> list[float]:
    x = np.nan_to_num(np.asarray(values, dtype=float))
    if len(x) == 0:
        return [0.0] * 9
    dx = np.diff(x)
    return [
        float(np.mean(x)),
        float(np.std(x)),
        float(np.min(x)),
        float(np.max(x)),
        float(np.percentile(x, 25)),
        float(np.percentile(x, 50)),
        float(np.percentile(x, 75)),
        float(np.sqrt(np.mean(x * x) + 1e-12)),
        float(np.mean(np.abs(dx))) if len(dx) else 0.0,
    ]


def _emg_geds_speed_features(window) -> list[float]:
    features: list[float] = []
    for column in [name for name in window.columns if name != "Time"]:
        features.extend(_emg_geds_column_features(window[column].to_numpy()))
    return features


def _emg_geds_phase_features(window) -> list[float]:
    values = window[_GEDS_PHASE_COLUMNS].to_numpy(dtype=float)
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    min_values = np.min(values, axis=0)
    max_values = np.max(values, axis=0)
    rms = np.sqrt(np.mean(values * values, axis=0) + 1e-12)
    mad = np.mean(np.abs(np.diff(values, axis=0)), axis=0)
    return np.concatenate([mean, std, min_values, max_values, rms, mad]).astype(float).tolist()


def EMG_classify_gait_speed(signal_path: str, sampling_rate: float = 1000.0, columns: list[str] | None = None, window_sec: float = 2.0, step_sec: float = 1.0) -> dict:
    model_path = "/data1/jiahui/biosignal-agent/outputs/emg_geds_walking_speed_feature_ensemble.joblib"
    try:
        bundle = _load_joblib_model(model_path)
        model = bundle["model"]
        label_encoder = bundle["label_encoder"]
        frame = _emg_read_geds_frame(signal_path, columns)
    except Exception as exc:
        return {"tool": "EMG_classify_gait_speed", "error": str(exc), "confidence": 0.0}
    win = max(16, int(window_sec * sampling_rate))
    step = max(1, int(step_sec * sampling_rate))
    if len(frame) < win:
        return {"tool": "EMG_classify_gait_speed", "error": "signal too short", "confidence": 0.0}
    features = np.asarray([_emg_geds_speed_features(frame.iloc[start:start + win]) for start in range(0, len(frame) - win + 1, step)], dtype=float)
    probabilities = model.predict_proba(features).mean(axis=0)
    labels = list(label_encoder.classes_)
    best_idx = int(np.argmax(probabilities))
    return {
        "tool": "EMG_classify_gait_speed",
        "predicted_speed": str(labels[best_idx]),
        "speed_probabilities": {str(label): float(probabilities[idx]) for idx, label in enumerate(labels)},
        "num_windows": int(len(features)),
        "confidence": float(probabilities[best_idx]),
        "method": "geds_walking_speed_subject_heldout_feature_ensemble",
        "model_path": model_path,
        "disclaimer": "Trained on the full GEDS S00-S22 414-trial subject-held-out benchmark with EMG/IMU/FSR wearable channels; validate on target sensors before deployment.",
    }


def EMG_estimate_gait_phase(signal_path: str, sampling_rate: float = 1000.0, window_sec: float = 0.5, step_sec: float = 0.5) -> dict:
    model_path = "/data1/jiahui/biosignal-agent/outputs/emg_geds_right_gait_phase_feature_ensemble.joblib"
    try:
        bundle = _load_joblib_model(model_path)
        model = bundle["model"]
        label_encoder = bundle["label_encoder"]
        frame = _emg_read_geds_frame(signal_path, _GEDS_PHASE_COLUMNS)
    except Exception as exc:
        return {"tool": "EMG_estimate_gait_phase", "error": str(exc), "confidence": 0.0}
    win = max(16, int(window_sec * sampling_rate))
    step = max(1, int(step_sec * sampling_rate))
    if len(frame) < win:
        return {"tool": "EMG_estimate_gait_phase", "error": "signal too short", "confidence": 0.0}
    features = np.asarray([_emg_geds_phase_features(frame.iloc[start:start + win]) for start in range(0, len(frame) - win + 1, step)], dtype=float)
    probabilities = model.predict_proba(features)
    mean_prob = probabilities.mean(axis=0)
    labels = list(label_encoder.classes_)
    best_idx = int(np.argmax(mean_prob))
    window_predictions = np.argmax(probabilities, axis=1)
    phase_fraction = {str(label): float(np.mean(window_predictions == idx)) for idx, label in enumerate(labels)}
    return {
        "tool": "EMG_estimate_gait_phase",
        "dominant_phase": str(labels[best_idx]),
        "phase_probabilities": {str(label): float(mean_prob[idx]) for idx, label in enumerate(labels)},
        "phase_fraction_by_window": phase_fraction,
        "num_windows": int(len(features)),
        "confidence": float(mean_prob[best_idx]),
        "method": "geds_right_stance_swing_subject_heldout_feature_ensemble",
        "model_path": model_path,
        "disclaimer": "Right-foot stance/swing model trained from GEDS event labels on the full S00-S22 414-trial subject-held-out benchmark; gait-event timing should be validated for new devices and placements.",
    }



def _emg_read_ninapro_signal(signal_path: str, columns: list[str] | None = None) -> np.ndarray:
    if signal_path.lower().endswith(".mat"):
        from scipy.io import loadmat

        mat = loadmat(signal_path)
        if "emg" not in mat:
            raise ValueError("NinaPro-style .mat file must contain an 'emg' array")
        values = np.asarray(mat["emg"], dtype=float)
    else:
        import pandas as pd

        try:
            frame = pd.read_csv(signal_path, sep=None, engine="python")
        except Exception:
            frame = pd.read_csv(signal_path, sep=r"\s+", header=None, engine="python")
        if frame.shape[1] == 1:
            frame = pd.read_csv(signal_path, sep=r"\s+", header=None, engine="python")
        if columns:
            values = frame[columns].to_numpy(dtype=float)
        else:
            numeric = frame.select_dtypes(include=[np.number]).copy()
            drop = [name for name in numeric.columns if str(name).lower() in {"time", "class", "label", "target", "stimulus", "restimulus", "repetition", "rerepetition"}]
            values = numeric.drop(columns=drop, errors="ignore").iloc[:, :10].to_numpy(dtype=float)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("expected at least one numeric sEMG channel")
    if values.shape[1] < 10:
        values = np.concatenate([values, np.zeros((values.shape[0], 10 - values.shape[1]), dtype=float)], axis=1)
    return np.nan_to_num(values[:, :10])


def _emg_ninapro_features(window: np.ndarray) -> list[float]:
    x = np.nan_to_num(window.astype(float))
    dx = np.diff(x, axis=0)
    threshold = np.maximum(np.std(x, axis=0) * 0.02, 1e-7)
    rms = np.sqrt(np.mean(x * x, axis=0) + 1e-12)
    mav = np.mean(np.abs(x), axis=0)
    waveform_length = np.sum(np.abs(dx), axis=0)
    variance = np.var(x, axis=0)
    mean_abs_diff = np.mean(np.abs(dx), axis=0) if len(dx) else np.zeros(x.shape[1])
    duty = np.mean(np.abs(x) > threshold, axis=0)
    zero_crossing = np.mean((x[:-1] * x[1:] < 0) & (np.abs(x[:-1] - x[1:]) > threshold), axis=0) if len(x) > 1 else np.zeros(x.shape[1])
    if len(dx) > 1:
        dd = np.diff(dx, axis=0)
        slope_changes = np.mean((dx[:-1] * dx[1:] < 0) & (np.abs(dd) > threshold), axis=0)
    else:
        slope_changes = np.zeros(x.shape[1])
    return np.concatenate([
        rms,
        mav,
        waveform_length,
        variance,
        mean_abs_diff,
        duty,
        zero_crossing,
        slope_changes,
        [rms.max() / (rms.min() + 1e-12), rms.std() / (rms.mean() + 1e-12), float(np.argmax(rms))],
    ]).astype(float).tolist()


def _emg_ninapro_augmented_features(window: np.ndarray) -> list[float]:
    base = np.asarray(_emg_ninapro_features(window), dtype=float)
    blocks = []
    for start in range(0, 80, 10):
        block = base[start:start + 10]
        denom = float(np.mean(np.abs(block)) + 1e-6)
        blocks.extend((block / denom).tolist())
        blocks.extend(np.log1p(np.abs(block)).tolist())
    rms_ratio = base[:10] / (float(np.sum(base[:10])) + 1e-6)
    return np.concatenate([base, rms_ratio, np.asarray(blocks, dtype=float)]).astype(float).tolist()


def _emg_ninapro_cnn_probabilities(values: np.ndarray, sampling_rate: float, window_sec: float, step_sec: float):
    import torch
    from torch import nn

    class _SEBlock(nn.Module):
        def __init__(self, channels: int, reduction: int = 8):
            super().__init__()
            hidden = max(4, channels // reduction)
            self.fc = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, channels), nn.Sigmoid())

        def forward(self, x):
            return x * self.fc(x).unsqueeze(-1)

    class _ConvBlock(nn.Module):
        def __init__(self, cin: int, cout: int, kernel: int, dilation: int = 1):
            super().__init__()
            pad = (kernel // 2) * dilation
            self.net = nn.Sequential(nn.Conv1d(cin, cout, kernel, padding=pad, dilation=dilation), nn.BatchNorm1d(cout), nn.ReLU(), nn.Dropout(0.15), _SEBlock(cout))

        def forward(self, x):
            return self.net(x)

    class _MultiStreamCNN(nn.Module):
        def __init__(self, in_channels: int, n_classes: int, width: int = 64):
            super().__init__()
            self.s3 = nn.Sequential(_ConvBlock(in_channels, width, 3), _ConvBlock(width, width, 3, 2))
            self.s5 = nn.Sequential(_ConvBlock(in_channels, width, 5), _ConvBlock(width, width, 5, 2))
            self.s9 = nn.Sequential(_ConvBlock(in_channels, width, 9), _ConvBlock(width, width, 9, 2))
            self.mix = nn.Sequential(nn.Conv1d(width * 3, width * 2, 1), nn.BatchNorm1d(width * 2), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.25), nn.Linear(width * 2, n_classes))

        def forward(self, x):
            return self.mix(torch.cat([self.s3(x), self.s5(x), self.s9(x)], dim=1))

    model_path = "/data1/jiahui/biosignal-agent/outputs/emg_ninapro_db1_52class_multistream_cnn_calibrated.pt"
    checkpoint = torch.load(model_path, map_location="cpu")
    labels = [int(label) for label in checkpoint["labels"]]
    model = _MultiStreamCNN(int(checkpoint["in_channels"]), len(labels))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    values = np.nan_to_num(values.astype(float))
    values = (values - values.mean(axis=0, keepdims=True)) / (values.std(axis=0, keepdims=True) + 1e-6)
    win = max(16, int(window_sec * sampling_rate))
    step = max(1, int(step_sec * sampling_rate))
    if len(values) < win:
        raise ValueError("signal too short")
    windows = np.asarray([values[start:start + win].T for start in range(0, len(values) - win + 1, step)], dtype=np.float32)
    probs = []
    with torch.no_grad():
        for start in range(0, len(windows), 512):
            xb = torch.from_numpy(windows[start:start + 512])
            probs.append(torch.softmax(model(xb), dim=1).numpy())
    return np.vstack(probs).mean(axis=0), labels, model_path, len(windows)


def EMG_classify_prosthetic_gesture(signal_path: str, sampling_rate: float = 100.0, columns: list[str] | None = None, window_sec: float = 0.2, step_sec: float = 0.1, backend: str = "feature") -> dict:
    try:
        values = _emg_read_ninapro_signal(signal_path, columns)
        if backend == "multistream_cnn":
            cnn_window_sec = max(window_sec, 1.0)
            cnn_step_sec = max(step_sec, 0.25)
            probabilities, labels, model_path, num_windows = _emg_ninapro_cnn_probabilities(values, sampling_rate, cnn_window_sec, cnn_step_sec)
            method = "ninapro_db1_52class_multistream_cnn_trial_voting"
            disclaimer = "NinaPro DB1 multi-stream CNN calibrated trial-voting reaches top-1 0.732/top-5 0.937; strict subject-held-out trial top-1 is 0.274. Use with user-specific calibration."
        elif backend == "feature":
            model_path = "/data1/jiahui/biosignal-agent/outputs/emg_ninapro_db1_52_gesture_augmented_extratrees.joblib"
            bundle = _load_joblib_model(model_path)
            model = bundle["model"]
            label_encoder = bundle["label_encoder"]
            win = max(8, int(window_sec * sampling_rate))
            step = max(1, int(step_sec * sampling_rate))
            if len(values) < win:
                return {"tool": "EMG_classify_prosthetic_gesture", "error": "signal too short", "confidence": 0.0}
            features = np.asarray([_emg_ninapro_augmented_features(values[start:start + win]) for start in range(0, len(values) - win + 1, step)], dtype=float)
            probabilities = model.predict_proba(features).mean(axis=0)
            labels = [int(label) for label in label_encoder.classes_]
            num_windows = int(len(features))
            method = "ninapro_db1_52class_augmented_extratrees_feature_ensemble"
            disclaimer = "NinaPro DB1 augmented calibrated-user repetition CV reaches top-1 0.585/top-5 0.821, but strict subject-held-out top-1 remains 0.107; deploy with user-specific calibration."
        else:
            return {"tool": "EMG_classify_prosthetic_gesture", "error": "backend must be 'feature' or 'multistream_cnn'", "confidence": 0.0}
    except Exception as exc:
        return {"tool": "EMG_classify_prosthetic_gesture", "error": str(exc), "confidence": 0.0}
    order = np.argsort(probabilities)[::-1]
    best_idx = int(order[0])
    return {
        "tool": "EMG_classify_prosthetic_gesture",
        "backend": backend,
        "predicted_gesture_id": labels[best_idx],
        "gesture_probabilities": {str(labels[int(idx)]): float(probabilities[int(idx)]) for idx in order[:10]},
        "top5_gesture_ids": [labels[int(idx)] for idx in order[:5]],
        "num_windows": int(num_windows),
        "confidence": float(probabilities[best_idx]),
        "method": method,
        "model_path": model_path,
        "disclaimer": disclaimer,
    }
