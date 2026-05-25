from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew

from .common import load_csv_signal, signal_quality_summary


ACC_ACTIVITY_TRIAXIAL_MODEL_PATH = Path("/data1/jiahui/biosignal-agent/outputs/acc_activity/acc_uci_har_triaxial_activity_ensemble.joblib")
ACC_ACTIVITY_MAG_MODEL_PATH = Path("/data1/jiahui/biosignal-agent/outputs/acc_activity/acc_uci_har_raw_magnitude_activity_ensemble.joblib")
ACC_ACTIVITY_TRIAXIAL_METRICS = {
    "benchmark": "UCI HAR raw total_acc x/y/z, official train/test subject split",
    "accuracy": 0.8218527315914489,
    "balanced_accuracy": 0.8208466845520906,
    "macro_f1": 0.8197333757310723,
    "weighted_f1": 0.8180408412216634,
    "macro_auroc_ovr": 0.9751833014399319,
    "coarse_active_rest_accuracy": 1.0,
}
ACC_ACTIVITY_MAG_METRICS = {
    "benchmark": "UCI HAR raw total_acc magnitude only, official train/test subject split",
    "accuracy": 0.6287750254496097,
    "balanced_accuracy": 0.6349762132414681,
    "macro_f1": 0.6282605782395362,
    "macro_auroc_ovr": 0.9179925237594752,
    "coarse_active_rest_accuracy": 1.0,
}
ACC_ACTIVE_LABELS = {"walking", "walking_upstairs", "walking_downstairs"}
ACC_FALL_MODEL_PATH = Path("/data1/jiahui/biosignal-agent/outputs/acc_fall/acc_unimib_fall_ensemble.joblib")
ACC_FALL_METRICS = {
    "benchmark": "UniMiB SHAR processed CSV mirror, fall-vs-ADL, subject-independent GroupKFold",
    "windows": 1196,
    "accuracy": 0.9824414715719063,
    "balanced_accuracy": 0.9803425491218918,
    "macro_f1": 0.9817487775474212,
    "fall_f1": 0.9853044086773968,
    "fall_recall": 0.9915492957746479,
    "specificity": 0.9691358024691358,
    "auroc": 0.9978902219903785,
}



def _read_acc_frame(signal_path: str, column: str | None = None) -> np.ndarray:
    df = pd.read_csv(signal_path)
    numeric = df.apply(pd.to_numeric, errors="coerce").select_dtypes(include=[np.number])
    if column and column in numeric:
        return numeric[[column]].to_numpy(dtype=float)
    if numeric.shape[1] >= 3:
        lower = {c.lower(): c for c in numeric.columns}
        keys = []
        for names in [("x", "acc_x", "accel_x", "total_acc_x"), ("y", "acc_y", "accel_y", "total_acc_y"), ("z", "acc_z", "accel_z", "total_acc_z")]:
            found = next((lower[n] for n in names if n in lower), None)
            keys.append(found)
        if all(keys):
            return numeric[keys].to_numpy(dtype=float)
        return numeric.iloc[:, :3].to_numpy(dtype=float)
    return numeric.iloc[:, [0]].to_numpy(dtype=float)


def _clean_acc_frame(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.size == 0:
        return arr.reshape(0, 1)
    out = arr.copy()
    for col in range(out.shape[1]):
        x = out[:, col]
        finite = np.isfinite(x)
        fill = float(np.nanmedian(x[finite])) if finite.any() else 0.0
        out[:, col] = np.where(finite, x, fill)
    return out


def _acc_magnitude(frame: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = _clean_acc_frame(frame)
    if arr.shape[1] >= 3:
        return np.linalg.norm(arr[:, :3], axis=1), True
    return arr[:, 0], False


def _gravity_scale(mag: np.ndarray) -> float:
    baseline = float(np.nanmedian(np.abs(mag))) if len(mag) else 1.0
    if 5.0 <= baseline <= 15.0:
        return 9.80665
    if 0.5 <= baseline <= 1.5:
        return 1.0
    return max(baseline, 1e-6)


def _angle_degrees(a: np.ndarray, b: np.ndarray) -> float | None:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return None
    cosang = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def _acc_one_axis_features(values: np.ndarray, sampling_rate: float) -> list[float]:
    x = np.asarray(values, dtype=float).ravel()
    finite = np.isfinite(x)
    if not finite.any():
        return [0.0] * 25
    x = np.where(finite, x, float(np.nanmedian(x[finite])))
    x = x - np.nanmean(x)
    n = len(x)
    fs = float(sampling_rate)
    freqs, psd = scipy_signal.welch(x, fs=fs, nperseg=min(n, max(8, int(fs * 2))))
    def band(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.trapezoid(psd[mask], freqs[mask])) if np.any(mask) else 0.0
    bands = {"low": (0.3, 1.0), "step": (1.0, 3.0), "mid": (3.0, 8.0), "high": (8.0, min(20.0, fs * 0.45))}
    bp = {name: band(*rng) for name, rng in bands.items()}
    total = sum(bp.values()) + 1e-12
    rel = {name: val / total for name, val in bp.items()}
    dx = np.diff(x, prepend=x[0]) * fs
    q = np.percentile(x, [5, 25, 50, 75, 95])
    dominant = float(freqs[np.argmax(psd)]) if len(psd) else 0.0
    feats = [float(np.mean(x)), float(np.std(x)), float(np.min(x)), float(np.max(x)), float(np.ptp(x)), *[float(v) for v in q], float(skew(x)), float(kurtosis(x)), float(np.mean(np.abs(x))), float(np.sqrt(np.mean(x * x))), float(np.mean(dx)), float(np.std(dx)), float(np.percentile(np.abs(dx), 95)), dominant, *[float(bp[k]) for k in sorted(bp)], *[float(rel[k]) for k in sorted(rel)], float(rel["step"] / (rel["low"] + 1e-12))]
    return [0.0 if not np.isfinite(v) else float(v) for v in feats]


def _acc_activity_features(frame: np.ndarray, sampling_rate: float) -> tuple[list[float], bool]:
    arr = np.asarray(frame, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    triaxial = arr.shape[1] >= 3
    if triaxial:
        axes = [arr[:, 0], arr[:, 1], arr[:, 2]]
        mag = np.linalg.norm(arr[:, :3], axis=1)
    else:
        mag = arr[:, 0]
        axes = [mag, mag, mag]
    feats = []
    for axis in axes + [mag]:
        feats.extend(_acc_one_axis_features(axis, sampling_rate))
    if triaxial and len(arr) > 1:
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            corr = np.corrcoef(axes[i], axes[j])[0, 1]
            feats.append(0.0 if not np.isfinite(corr) else float(corr))
    elif triaxial:
        feats.extend([0.0, 0.0, 0.0])
    return feats, triaxial


def ACC_assess_quality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    return {"tool": "ACC_assess_quality", "source": data.source, **signal_quality_summary(data.values)}


def ACC_summarize_activity(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    frame = _read_acc_frame(signal_path, column)
    mag, triaxial = _acc_magnitude(frame)
    if len(mag) == 0:
        return {"tool": "ACC_summarize_activity", "error": "empty signal", "confidence": 0.0}
    g_scale = _gravity_scale(mag)
    mag_g = mag / g_scale
    centered = mag_g - np.nanmedian(mag_g)
    activity = float(np.nanstd(centered))
    dynamic_range = float(np.nanpercentile(mag_g, 95) - np.nanpercentile(mag_g, 5))
    jerk = np.diff(mag_g, prepend=mag_g[0]) * float(sampling_rate)
    jerk_p95 = float(np.nanpercentile(np.abs(jerk), 95)) if len(jerk) else 0.0
    label = "high" if activity > 0.35 or jerk_p95 > 3.0 else "moderate" if activity > 0.08 or jerk_p95 > 0.8 else "low"
    return {
        "tool": "ACC_summarize_activity",
        "activity_std": activity,
        "dynamic_range_g": dynamic_range,
        "jerk_p95_g_per_s": jerk_p95,
        "activity_level": label,
        "input_type": "triaxial" if triaxial else "single_axis_or_magnitude",
        "confidence": 0.72 if triaxial else 0.58,
        "method": "magnitude_activity_summary",
    }


def ACC_extract_actigraphy_features(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    frame = _read_acc_frame(signal_path, column)
    mag, triaxial = _acc_magnitude(frame)
    if len(mag) < max(4, int(float(sampling_rate) * 5)):
        return {"tool": "ACC_extract_actigraphy_features", "error": "signal too short", "confidence": 0.0}
    fs = float(sampling_rate)
    g_scale = _gravity_scale(mag)
    mag_g = mag / g_scale
    enmo = np.maximum(mag_g - 1.0, 0.0) if triaxial else np.abs(mag_g - np.nanmedian(mag_g))
    epoch = max(1, int(fs * 60.0))
    n_epochs = max(1, int(np.ceil(len(enmo) / epoch)))
    epoch_means = np.asarray([float(np.nanmean(enmo[i * epoch : min(len(enmo), (i + 1) * epoch)])) for i in range(n_epochs)])
    restful = epoch_means < 0.03
    sedentary = epoch_means < 0.05
    light = (epoch_means >= 0.05) & (epoch_means < 0.12)
    moderate_vigorous = epoch_means >= 0.12
    anglez = None
    if triaxial:
        arr = _clean_acc_frame(frame)[:, :3]
        denom = np.linalg.norm(arr, axis=1) + 1e-12
        anglez = np.degrees(np.arcsin(np.clip(arr[:, 2] / denom, -1.0, 1.0)))
    return {
        "tool": "ACC_extract_actigraphy_features",
        "duration_min": float(len(mag) / fs / 60.0),
        "mean_enmo_g": float(np.nanmean(enmo)),
        "median_enmo_g": float(np.nanmedian(enmo)),
        "p95_enmo_g": float(np.nanpercentile(enmo, 95)),
        "restful_fraction": float(np.mean(restful)),
        "sedentary_fraction": float(np.mean(sedentary)),
        "light_activity_fraction": float(np.mean(light)),
        "moderate_vigorous_fraction": float(np.mean(moderate_vigorous)),
        "anglez_mean_deg": None if anglez is None else float(np.nanmean(anglez)),
        "anglez_std_deg": None if anglez is None else float(np.nanstd(anglez)),
        "input_type": "triaxial" if triaxial else "single_axis_or_magnitude",
        "confidence": 0.68 if triaxial else 0.5,
        "method": "enmo_epoch_actigraphy_features",
    }


def ACC_estimate_sleep_wake(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    features = ACC_extract_actigraphy_features(signal_path, sampling_rate, column)
    if features.get("error"):
        summary = ACC_summarize_activity(signal_path, sampling_rate, column)
        activity = float(summary.get("activity_std", 0.0) or 0.0)
        hint = "wake_or_active" if activity > 0.08 else "sleep_or_rest_short_window"
        return {
            "tool": "ACC_estimate_sleep_wake",
            "sleep_wake_hint": hint,
            "activity_std": activity,
            "fallback_result": summary,
            "error": features.get("error"),
            "confidence": 0.25,
            "method": "short_window_activity_fallback",
            "disclaimer": "Window is too short for actigraphy sleep/wake scoring; use longer PSG-calibrated actigraphy when possible.",
        }
    restful = float(features.get("restful_fraction", 0.0))
    mean_enmo = float(features.get("mean_enmo_g", 0.0))
    duration_min = float(features.get("duration_min", 0.0))
    if restful >= 0.85 and mean_enmo < 0.03:
        hint = "sleep_or_sustained_rest"
        conf = 0.65 if duration_min >= 10 else 0.5
    elif restful <= 0.45 or mean_enmo > 0.08:
        hint = "wake_or_active"
        conf = 0.68 if duration_min >= 5 else 0.52
    else:
        hint = "uncertain_rest_wake"
        conf = 0.42
    return {
        "tool": "ACC_estimate_sleep_wake",
        "sleep_wake_hint": hint,
        "restful_fraction": restful,
        "mean_enmo_g": mean_enmo,
        "duration_min": duration_min,
        "actigraphy_features": features,
        "confidence": conf,
        "method": "enmo_actigraphy_sleep_wake_proxy",
        "disclaimer": "Actigraphy proxy only; validated sleep/wake scoring needs PSG-labeled actigraphy such as MESA/SHHS and device-specific calibration.",
    }


def ACC_detect_activity_bouts(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    frame = _read_acc_frame(signal_path, column)
    mag, triaxial = _acc_magnitude(frame)
    if len(mag) < max(4, int(float(sampling_rate) * 5)):
        return {"tool": "ACC_detect_activity_bouts", "error": "signal too short", "confidence": 0.0}
    fs = float(sampling_rate)
    g_scale = _gravity_scale(mag)
    values = mag / g_scale
    window = max(1, int(fs * 2.0))
    kernel = np.ones(window) / window
    centered = values - np.nanmedian(values)
    activity = np.sqrt(np.convolve(centered ** 2, kernel, mode="same"))
    threshold = max(float(np.nanmedian(activity) + 2.0 * np.nanstd(activity)), 0.06)
    active = activity > threshold
    events = []
    start = None
    for idx, flag in enumerate(active):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            if idx - start >= max(1, int(fs)):
                events.append((start, idx))
            start = None
    if start is not None and len(active) - start >= max(1, int(fs)):
        events.append((start, len(active)))
    duration_min = len(values) / fs / 60.0
    return {
        "tool": "ACC_detect_activity_bouts",
        "activity_bout_count": int(len(events)),
        "activity_bout_rate_per_min": float(len(events) / duration_min) if duration_min > 0 else None,
        "activity_threshold_g": threshold,
        "input_type": "triaxial" if triaxial else "single_axis_or_magnitude",
        "event_intervals_s": [[float(a / fs), float(b / fs)] for a, b in events[:20]],
        "confidence": 0.68 if triaxial else 0.52,
        "method": "accelerometer_magnitude_rms_bout_detection",
        "disclaimer": "Activity bout proxy only; validated activity classification needs labeled posture/activity data.",
    }


def ACC_detect_fall_proxy(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    frame = _read_acc_frame(signal_path, column)
    arr = _clean_acc_frame(frame)
    mag, triaxial = _acc_magnitude(arr)
    fs = float(sampling_rate)
    if len(mag) < max(4, int(fs * 3)):
        return {"tool": "ACC_detect_fall_proxy", "error": "signal too short", "confidence": 0.0}
    g_scale = _gravity_scale(mag)
    mag_g = mag / g_scale
    dynamic = mag_g - np.nanmedian(mag_g)
    jerk = np.diff(mag_g, prepend=mag_g[0]) * fs
    adaptive = max(float(np.nanpercentile(np.abs(dynamic), 99) * 0.85), float(np.nanstd(dynamic) * 3.5), 0.45)
    impact_mask = (np.abs(dynamic) >= adaptive) | (mag_g >= 2.2) | (np.abs(jerk) >= max(8.0, float(np.nanpercentile(np.abs(jerk), 99) * 0.85)))
    impact_indices = np.flatnonzero(impact_mask)
    refractory = max(1, int(fs))
    events = []
    last = -refractory
    for idx in impact_indices:
        if idx - last >= refractory:
            post = dynamic[idx : min(len(dynamic), idx + int(1.5 * fs))]
            post_inactive = bool(len(post) >= max(3, int(0.4 * fs)) and float(np.nanstd(post[-max(3, int(0.4 * fs)):])) < 0.12)
            angle_change = None
            posture_change = False
            if triaxial and len(arr) >= int(fs):
                pre = arr[max(0, idx - int(0.75 * fs)) : max(1, idx - int(0.25 * fs)), :3]
                post_arr = arr[min(len(arr), idx + int(0.5 * fs)) : min(len(arr), idx + int(1.5 * fs)), :3]
                if len(pre) and len(post_arr):
                    angle_change = _angle_degrees(np.nanmedian(pre, axis=0), np.nanmedian(post_arr, axis=0))
                    posture_change = bool(angle_change is not None and angle_change >= 35.0)
            events.append({
                "index": int(idx),
                "time_s": float(idx / fs),
                "peak_magnitude_g": float(mag_g[idx]),
                "dynamic_g": float(dynamic[idx]),
                "jerk_g_per_s": float(jerk[idx]),
                "post_impact_inactivity": post_inactive,
                "posture_change_degrees": angle_change,
                "posture_change": posture_change,
            })
            last = idx
    fall_like = [e for e in events if e["post_impact_inactivity"] or e["posture_change"]]
    risk = "fall_pattern_detected" if fall_like else "possible_fall_or_impact_proxy" if events else "no_fall_proxy"
    confidence = 0.7 if fall_like and triaxial else 0.48 if events else 0.55
    return {
        "tool": "ACC_detect_fall_proxy",
        "impact_event_count": int(len(events)),
        "fall_like_event_count": int(len(fall_like)),
        "impact_events": events[:20],
        "impact_threshold_dynamic_g": adaptive,
        "fall_risk": risk,
        "input_type": "triaxial" if triaxial else "single_axis_or_magnitude",
        "confidence": confidence,
        "method": "triaxial_impact_posture_inactivity_fall_proxy",
        "disclaimer": "Fall proxy only; simulated fall datasets overestimate real-world performance. Prefer ACC_detect_fall_ml when a matching device/window is available.",
    }


def ACC_detect_fall_ml(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    frame = _read_acc_frame(signal_path, column)
    if len(frame) < max(16, int(float(sampling_rate) * 1.0)):
        proxy = ACC_detect_fall_proxy(signal_path, sampling_rate, column)
        return {"tool": "ACC_detect_fall_ml", "error": "signal too short for fall classifier", "fallback_result": proxy, "confidence": 0.2}
    if not ACC_FALL_MODEL_PATH.exists():
        proxy = ACC_detect_fall_proxy(signal_path, sampling_rate, column)
        return {"tool": "ACC_detect_fall_ml", "error": f"trained model not found: {ACC_FALL_MODEL_PATH}", "fallback_result": proxy, "confidence": 0.25}
    feats, triaxial = _acc_activity_features(frame, sampling_rate)
    bundle = joblib.load(ACC_FALL_MODEL_PATH)
    model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
    X = np.asarray([feats], dtype=float)
    classes = list(getattr(model, "classes_", []))
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        probabilities = {str(cls): float(prob) for cls, prob in zip(classes, probs)}
        label = str(classes[int(np.argmax(probs))])
    else:
        label = str(model.predict(X)[0])
        probabilities = {label: 1.0}
    proxy = ACC_detect_fall_proxy(signal_path, sampling_rate, column)
    return {
        "tool": "ACC_detect_fall_ml",
        "fall_label": label,
        "fall_probability": float(probabilities.get("fall", 0.0)),
        "label_probabilities": probabilities,
        "proxy_result": proxy,
        "input_type": "triaxial" if triaxial else "single_axis_or_magnitude",
        "model_source": str(ACC_FALL_MODEL_PATH),
        "model_metrics": ACC_FALL_METRICS,
        "confidence": float(min(0.92, max(0.3, max(probabilities.values()) if probabilities else 0.3))),
        "method": "unimib_shar_triaxial_fall_adl_feature_ensemble",
        "disclaimer": "Research fall classifier trained on simulated smartphone-pocket UniMiB SHAR windows; real-world falls, elderly users, and different device placement require external validation/calibration.",
    }



def ACC_classify_activity_ml(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    frame = _read_acc_frame(signal_path, column)
    if len(frame) < max(16, int(float(sampling_rate) * 1.0)):
        summary = ACC_summarize_activity(signal_path, sampling_rate, column)
        return {"tool": "ACC_classify_activity_ml", "error": "signal too short for UCI-HAR activity classifier", "fallback_result": summary, "confidence": 0.2}
    feats, triaxial = _acc_activity_features(frame, sampling_rate)
    model_path = ACC_ACTIVITY_TRIAXIAL_MODEL_PATH if triaxial and ACC_ACTIVITY_TRIAXIAL_MODEL_PATH.exists() else ACC_ACTIVITY_MAG_MODEL_PATH
    metrics = ACC_ACTIVITY_TRIAXIAL_METRICS if model_path == ACC_ACTIVITY_TRIAXIAL_MODEL_PATH else ACC_ACTIVITY_MAG_METRICS
    if not model_path.exists():
        summary = ACC_summarize_activity(signal_path, sampling_rate, column)
        return {"tool": "ACC_classify_activity_ml", "error": f"trained model not found: {model_path}", "fallback_result": summary, "confidence": 0.2}
    bundle = joblib.load(model_path)
    model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
    X = np.asarray([feats], dtype=float)
    classes = list(getattr(model, "classes_", []))
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        probabilities = {str(cls): float(prob) for cls, prob in zip(classes, probs)}
        label = str(classes[int(np.argmax(probs))])
    else:
        label = str(model.predict(X)[0])
        probabilities = {label: 1.0}
    coarse = "active" if label in ACC_ACTIVE_LABELS else "rest"
    return {
        "tool": "ACC_classify_activity_ml",
        "activity_label": label,
        "activity_probabilities": probabilities,
        "coarse_activity_label": coarse,
        "input_type": "triaxial" if triaxial else "single_axis_or_magnitude",
        "model_source": str(model_path),
        "model_metrics": metrics,
        "confidence": float(min(0.9, max(0.25, max(probabilities.values()) if probabilities else 0.25))),
        "method": "uci_har_raw_acc_feature_ensemble_activity_classification",
        "disclaimer": "Research activity classifier trained on UCI HAR smartphone accelerometer data; device placement/orientation and population can shift performance.",
    }
