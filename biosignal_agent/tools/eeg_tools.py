from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew

from .common import load_csv_signal, signal_quality_summary


EEG_SLEEP_STAGE_MODEL_PATH = Path("/data1/jiahui/biosignal-agent/outputs/eeg_sleep/eeg_ucddb_coarse_sleep_stage_feature_ensemble.joblib")
EEG_SLEEP_STAGE_CV_METRICS = {
    "benchmark": "UCDDB ucddb002 single-channel C3A2 EEG, 30 s windows, 3 coarse classes, 5-fold stratified window CV",
    "accuracy": 0.7414829659318637,
    "balanced_accuracy": 0.7298620208968373,
    "macro_f1": 0.7307323127604476,
    "weighted_f1": 0.7406007965344764,
    "cohen_kappa": 0.6058071657228414,
    "macro_auroc_ovr": 0.8968277517916734,
    "caveat": "single UCDDB record window CV; not subject-independent sleep staging",
}
EEG_SEIZURE_MODEL_PATH = Path("/data1/jiahui/biosignal-agent/outputs/eeg_seizure/eeg_chbmit_chb01_seizure_feature_ensemble.joblib")
EEG_SEIZURE_CV_METRICS = {
    "benchmark": "CHB-MIT chb01 FP1-F7, 10 s windows, seizure vs non-seizure, 3 EDF-file grouped CV",
    "accuracy": 0.9473684210526315,
    "balanced_accuracy": 0.9361111111111111,
    "macro_f1": 0.9231460674157304,
    "weighted_f1": 0.9481253696037848,
    "auroc": 0.9587962962962963,
    "num_windows": 114,
    "label_counts": {"seizure": 24, "non_seizure": 90},
    "caveat": "small chb01 subset; not full CHB-MIT or clinical seizure detection",
}


def _clean_eeg(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float).ravel()
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=float)
    med = float(np.nanmedian(x[finite]))
    x = np.where(finite, x, med)
    lo, hi = np.percentile(x, [0.5, 99.5])
    if hi > lo:
        x = np.clip(x, lo, hi)
    return x - np.nanmedian(x)


def _eeg_sleep_feature_vector(values: np.ndarray, sampling_rate: float) -> list[float]:
    x = _clean_eeg(values)
    n = len(x)
    if n == 0:
        return [0.0] * 64
    fs = float(sampling_rate)
    freqs, psd = scipy_signal.welch(x, fs=fs, nperseg=min(n, int(fs * 4)))
    bands = {
        "delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
        "sigma": (12, 16), "beta": (13, 30), "gamma": (30, min(45, fs * 0.45)),
        "slow": (0.5, 1.5), "spindle": (11, 16),
    }
    def band(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.trapezoid(psd[mask], freqs[mask])) if np.any(mask) else 0.0
    total = band(0.5, min(45, fs * 0.45)) + 1e-12
    bp = {name: band(*rng) for name, rng in bands.items()}
    rel = {name: val / total for name, val in bp.items()}
    dx = np.diff(x, prepend=x[0]) * fs
    zcr = float(np.mean(np.diff(np.signbit(x)).astype(float))) if len(x) > 1 else 0.0
    hjorth_activity = float(np.var(x))
    hjorth_mobility = float(np.sqrt(np.var(dx) / (np.var(x) + 1e-12)))
    ddx = np.diff(dx, prepend=dx[0]) * fs
    hjorth_complexity = float(np.sqrt(np.var(ddx) / (np.var(dx) + 1e-12)) / (hjorth_mobility + 1e-12))
    q = np.percentile(x, [1, 5, 25, 50, 75, 95, 99])
    p = psd[(freqs >= 0.5) & (freqs <= min(45, fs * 0.45))]
    p = p / (p.sum() + 1e-12)
    spectral_entropy = float(-np.sum(p * np.log2(p + 1e-12)) / np.log2(len(p) + 1e-12)) if len(p) else 0.0
    ratios = [
        rel["delta"] / (rel["theta"] + 1e-12), rel["theta"] / (rel["alpha"] + 1e-12),
        rel["alpha"] / (rel["delta"] + 1e-12), rel["sigma"] / (rel["delta"] + 1e-12),
        (rel["delta"] + rel["theta"]) / (rel["alpha"] + rel["beta"] + 1e-12),
    ]
    feats = [
        float(np.mean(x)), float(np.std(x)), float(np.min(x)), float(np.max(x)), float(np.ptp(x)),
        *[float(v) for v in q], float(skew(x)), float(kurtosis(x)),
        float(np.mean(np.abs(x))), float(np.sqrt(np.mean(x * x))), zcr,
        hjorth_activity, hjorth_mobility, hjorth_complexity, spectral_entropy,
        *[float(bp[k]) for k in sorted(bp)], *[float(rel[k]) for k in sorted(rel)], *ratios,
        float(np.mean(dx)), float(np.std(dx)), float(np.percentile(np.abs(dx), 95)),
    ]
    return [0.0 if not np.isfinite(v) else float(v) for v in feats]



def _eeg_seizure_feature_vector(values: np.ndarray, sampling_rate: float) -> list[float]:
    x = _clean_eeg(values)
    n = len(x)
    if n == 0:
        return [0.0] * 50
    fs = float(sampling_rate)
    freqs, psd = scipy_signal.welch(x, fs=fs, nperseg=min(n, int(fs * 2)))
    def band(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.trapezoid(psd[mask], freqs[mask])) if np.any(mask) else 0.0
    bands = {
        "delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30),
        "gamma": (30, min(80, fs * 0.45)), "hfo_proxy": (80, min(120, fs * 0.45)),
    }
    bp = {name: band(*rng) for name, rng in bands.items()}
    total = sum(bp.values()) + 1e-12
    rel = {name: val / total for name, val in bp.items()}
    dx = np.diff(x, prepend=x[0]) * fs
    env = np.abs(scipy_signal.hilbert(x)) if len(x) > 8 else np.abs(x)
    robust = float(np.nanmedian(np.abs(x)) * 1.4826 + 1e-8)
    spike = np.abs(x) > 6.0 * robust
    spike_edges = np.flatnonzero(np.diff(spike.astype(int), prepend=0) == 1)
    line_length = float(np.sum(np.abs(np.diff(x))) / (n + 1e-12))
    hjorth_activity = float(np.var(x))
    hjorth_mobility = float(np.sqrt(np.var(dx) / (np.var(x) + 1e-12)))
    ddx = np.diff(dx, prepend=dx[0]) * fs
    hjorth_complexity = float(np.sqrt(np.var(ddx) / (np.var(dx) + 1e-12)) / (hjorth_mobility + 1e-12))
    p = psd[(freqs >= 0.5) & (freqs <= min(80, fs * 0.45))]
    p = p / (p.sum() + 1e-12)
    spectral_entropy = float(-np.sum(p * np.log2(p + 1e-12)) / np.log2(len(p) + 1e-12)) if len(p) else 0.0
    q = np.percentile(x, [1, 5, 25, 50, 75, 95, 99])
    eq = np.percentile(env, [50, 75, 90, 95, 99])
    feats = [
        float(np.mean(x)), float(np.std(x)), float(np.min(x)), float(np.max(x)), float(np.ptp(x)),
        *[float(v) for v in q], float(skew(x)), float(kurtosis(x)), float(np.mean(np.abs(x))),
        float(np.sqrt(np.mean(x * x))), line_length, hjorth_activity, hjorth_mobility, hjorth_complexity,
        spectral_entropy, float(np.mean(env)), float(np.std(env)), *[float(v) for v in eq],
        float(len(spike_edges)), float(len(spike_edges) / (n / fs / 60.0 + 1e-12)),
        float(np.mean(dx)), float(np.std(dx)), float(np.percentile(np.abs(dx), 95)),
        *[float(bp[k]) for k in sorted(bp)], *[float(rel[k]) for k in sorted(rel)],
        float(rel["gamma"] / (rel["alpha"] + rel["theta"] + 1e-12)),
        float(rel["beta"] / (rel["delta"] + 1e-12)),
    ]
    return [0.0 if not np.isfinite(v) else float(v) for v in feats]

def EEG_assess_quality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    return {"tool": "EEG_assess_quality", "source": data.source, **signal_quality_summary(data.values)}


def EEG_compute_bandpower(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    if len(data.values) < data.sampling_rate:
        return {"tool": "EEG_compute_bandpower", "error": "signal too short", "confidence": 0.0}
    freqs, psd = scipy_signal.welch(data.values, fs=data.sampling_rate, nperseg=min(len(data.values), int(data.sampling_rate * 4)))
    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, min(45, data.sampling_rate * 0.45))}
    powers = {}
    total = float(np.trapezoid(psd[(freqs >= 0.5) & (freqs <= min(45, data.sampling_rate * 0.45))], freqs[(freqs >= 0.5) & (freqs <= min(45, data.sampling_rate * 0.45))]))
    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        powers[f"{name}_power"] = float(np.trapezoid(psd[mask], freqs[mask])) if np.any(mask) else 0.0
    powers["total_power"] = total
    powers["confidence"] = 0.65
    powers["method"] = "welch_bandpower"
    powers["tool"] = "EEG_compute_bandpower"
    return powers



def EEG_estimate_sleep_stage_features(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    bandpower = EEG_compute_bandpower(signal_path, sampling_rate, column)
    if bandpower.get("error"):
        return {"tool": "EEG_estimate_sleep_stage_features", "error": bandpower["error"], "confidence": 0.0}
    total = float(bandpower.get("total_power") or 0.0)
    if total <= 0:
        return {"tool": "EEG_estimate_sleep_stage_features", "error": "zero EEG power", "confidence": 0.1}
    delta_ratio = float(bandpower.get("delta_power", 0.0) / total)
    theta_ratio = float(bandpower.get("theta_power", 0.0) / total)
    alpha_ratio = float(bandpower.get("alpha_power", 0.0) / total)
    beta_ratio = float(bandpower.get("beta_power", 0.0) / total)
    if delta_ratio > 0.45:
        stage_hint = "n3_like_slow_wave"
    elif theta_ratio > 0.30 and alpha_ratio < 0.20:
        stage_hint = "n1_n2_like"
    elif alpha_ratio > 0.25 or beta_ratio > 0.25:
        stage_hint = "wake_rem_like"
    else:
        stage_hint = "uncertain"
    return {
        "tool": "EEG_estimate_sleep_stage_features",
        "delta_ratio": delta_ratio,
        "theta_ratio": theta_ratio,
        "alpha_ratio": alpha_ratio,
        "beta_ratio": beta_ratio,
        "sleep_stage_hint": stage_hint,
        "confidence": 0.5,
        "method": "single_channel_bandpower_rules",
        "disclaimer": "Feature heuristic only; sleep staging requires labeled epochs and usually EEG/EOG/EMG context.",
    }



def EEG_screen_seizure_like_activity(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    if len(values) < max(8, int(data.sampling_rate * 2)):
        return {"tool": "EEG_screen_seizure_like_activity", "error": "signal too short", "confidence": 0.0}
    centered = values - np.nanmedian(values)
    robust_scale = float(np.nanmedian(np.abs(centered)) * 1.4826 + 1e-8)
    spike_candidates = np.abs(centered) > robust_scale * 6.0
    spike_edges = np.flatnonzero(np.diff(spike_candidates.astype(int), prepend=0) == 1)
    duration_min = len(values) / float(data.sampling_rate) / 60.0 if data.sampling_rate else 0.0
    spike_rate = float(len(spike_edges) / duration_min) if duration_min > 0 else None
    bandpower = EEG_compute_bandpower(signal_path, sampling_rate, column)
    total = float(bandpower.get("total_power") or 0.0)
    fast_power = float(bandpower.get("beta_power", 0.0) + bandpower.get("gamma_power", 0.0))
    fast_power_ratio = float(fast_power / total) if total > 0 else 0.0
    risk = "possible_seizure_like_activity_proxy" if (spike_rate is not None and spike_rate > 12) or fast_power_ratio > 0.45 else "no_seizure_like_activity_proxy"
    return {
        "tool": "EEG_screen_seizure_like_activity",
        "spike_count": int(len(spike_edges)),
        "spike_rate_per_min": spike_rate,
        "fast_power_ratio": fast_power_ratio,
        "seizure_like_risk": risk,
        "confidence": 0.5,
        "method": "eeg_robust_spike_fast_power_screening",
        "disclaimer": "Research heuristic only; seizure detection requires validated EEG montages, artifacts checks, and clinical labels.",
    }



def EEG_estimate_drowsiness(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    bandpower = EEG_compute_bandpower(signal_path, sampling_rate, column)
    if bandpower.get("error"):
        return {"tool": "EEG_estimate_drowsiness", "error": bandpower["error"], "confidence": 0.0}
    total = float(bandpower.get("total_power") or 0.0)
    if total <= 0:
        return {"tool": "EEG_estimate_drowsiness", "error": "zero EEG power", "confidence": 0.1}
    theta_alpha_ratio = float(bandpower.get("theta_power", 0.0) / (bandpower.get("alpha_power", 0.0) + 1e-12))
    slow_power_ratio = float((bandpower.get("theta_power", 0.0) + bandpower.get("delta_power", 0.0)) / total)
    drowsiness_hint = "possible_drowsiness_proxy" if theta_alpha_ratio > 1.2 or slow_power_ratio > 0.55 else "low_drowsiness_proxy"
    return {
        "tool": "EEG_estimate_drowsiness",
        "theta_alpha_ratio": theta_alpha_ratio,
        "slow_power_ratio": slow_power_ratio,
        "drowsiness_hint": drowsiness_hint,
        "confidence": 0.5,
        "method": "eeg_theta_alpha_slow_power_proxy",
        "disclaimer": "Drowsiness proxy only; vigilance assessment requires labeled task context and artifact handling.",
    }


def EEG_detect_artifact_proxy(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    if len(values) < max(8, int(data.sampling_rate)):
        return {"tool": "EEG_detect_artifact_proxy", "error": "signal too short", "confidence": 0.0}
    centered = values - np.nanmedian(values)
    robust_scale = float(np.nanmedian(np.abs(centered)) * 1.4826 + 1e-8)
    extreme_fraction = float(np.mean(np.abs(centered) > robust_scale * 8.0))
    freqs, psd = scipy_signal.welch(centered, fs=data.sampling_rate, nperseg=min(len(centered), int(data.sampling_rate * 4)))
    high_mask = (freqs >= 30) & (freqs <= min(45, data.sampling_rate * 0.45))
    total = float(np.trapezoid(psd, freqs)) if len(freqs) else 0.0
    high_ratio = float(np.trapezoid(psd[high_mask], freqs[high_mask]) / (total + 1e-12)) if np.any(high_mask) else 0.0
    flags = []
    if extreme_fraction > 0.01:
        flags.append("large_amplitude_artifact_proxy")
    if high_ratio > 0.35:
        flags.append("muscle_high_frequency_artifact_proxy")
    artifact_level = "high" if len(flags) >= 2 else "moderate" if flags else "low"
    return {
        "tool": "EEG_detect_artifact_proxy",
        "eeg_artifact_level": artifact_level,
        "eeg_artifact_flags": flags,
        "extreme_amplitude_fraction": extreme_fraction,
        "high_frequency_power_ratio": high_ratio,
        "confidence": 0.55,
        "method": "eeg_extreme_amplitude_high_frequency_proxy",
        "disclaimer": "EEG artifact proxy only; robust EEG QC requires channel montage and labeled artifacts.",
    }



def EEG_classify_sleep_stage_ml(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    if len(values) < max(20, int(data.sampling_rate * 20)):
        fallback = EEG_estimate_sleep_stage_features(signal_path, sampling_rate, column)
        return {
            "tool": "EEG_classify_sleep_stage_ml",
            "error": "signal too short for 30 s EEG sleep-stage classifier; used bandpower fallback evidence only",
            "fallback_result": fallback,
            "confidence": 0.2,
            "method": "ucddb_eeg_sleep_stage_feature_ensemble_short_signal_fallback",
            "disclaimer": "Sleep staging requires PSG context and subject-independent validation; this is a research coarse-stage classifier.",
        }
    if not EEG_SLEEP_STAGE_MODEL_PATH.exists():
        fallback = EEG_estimate_sleep_stage_features(signal_path, sampling_rate, column)
        return {
            "tool": "EEG_classify_sleep_stage_ml",
            "error": f"trained model not found: {EEG_SLEEP_STAGE_MODEL_PATH}",
            "fallback_result": fallback,
            "confidence": 0.2,
            "method": "missing_ucddb_sleep_stage_model_fallback",
        }
    bundle = joblib.load(EEG_SLEEP_STAGE_MODEL_PATH)
    model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
    X = np.asarray([_eeg_sleep_feature_vector(values, data.sampling_rate)], dtype=float)
    classes = list(getattr(model, "classes_", []))
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        probabilities = {str(cls): float(prob) for cls, prob in zip(classes, probs)}
        stage = str(classes[int(np.argmax(probs))])
    else:
        stage = str(model.predict(X)[0])
        probabilities = {stage: 1.0}
    return {
        "tool": "EEG_classify_sleep_stage_ml",
        "predicted_sleep_stage": stage,
        "stage_probabilities": probabilities,
        "model_source": str(EEG_SLEEP_STAGE_MODEL_PATH),
        "model_cv_metrics": EEG_SLEEP_STAGE_CV_METRICS,
        "confidence": float(min(0.8, max(0.25, max(probabilities.values()) if probabilities else 0.25))),
        "method": "ucddb_single_channel_eeg_feature_ensemble_coarse_sleep_stage",
        "disclaimer": "Coarse sleep stage classifier trained on one UCDDB record with window-level CV; not AASM-equivalent, not subject-independent, and not diagnostic.",
    }



def EEG_screen_seizure_ml(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    if len(values) < max(5, int(data.sampling_rate * 5)):
        fallback = EEG_screen_seizure_like_activity(signal_path, sampling_rate, column)
        return {
            "tool": "EEG_screen_seizure_ml",
            "error": "signal too short for 10 s CHB-MIT seizure classifier; used heuristic fallback evidence only",
            "fallback_result": fallback,
            "confidence": 0.2,
            "method": "chbmit_eeg_seizure_feature_ensemble_short_signal_fallback",
        }
    if not EEG_SEIZURE_MODEL_PATH.exists():
        fallback = EEG_screen_seizure_like_activity(signal_path, sampling_rate, column)
        return {
            "tool": "EEG_screen_seizure_ml",
            "error": f"trained model not found: {EEG_SEIZURE_MODEL_PATH}",
            "fallback_result": fallback,
            "confidence": 0.2,
            "method": "missing_chbmit_seizure_model_fallback",
        }
    bundle = joblib.load(EEG_SEIZURE_MODEL_PATH)
    model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
    X = np.asarray([_eeg_seizure_feature_vector(values, data.sampling_rate)], dtype=float)
    classes = list(getattr(model, "classes_", []))
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        probabilities = {str(cls): float(prob) for cls, prob in zip(classes, probs)}
        seizure_probability = probabilities.get("seizure", float(np.max(probs)))
    else:
        pred = str(model.predict(X)[0])
        seizure_probability = 1.0 if pred == "seizure" else 0.0
        probabilities = {pred: 1.0}
    risk = "seizure_likely" if seizure_probability >= 0.5 else "non_seizure_likely"
    return {
        "tool": "EEG_screen_seizure_ml",
        "seizure_probability": float(seizure_probability),
        "seizure_risk": risk,
        "class_probabilities": probabilities,
        "model_source": str(EEG_SEIZURE_MODEL_PATH),
        "model_cv_metrics": EEG_SEIZURE_CV_METRICS,
        "confidence": float(min(0.85, max(0.25, abs(seizure_probability - 0.5) * 1.2 + 0.35))),
        "method": "chbmit_chb01_single_channel_eeg_feature_ensemble_seizure_screen",
        "disclaimer": "Research seizure-screening model trained on a small CHB-MIT chb01 subset; not full-montage, not full-dataset validated, not diagnostic, and not a clinical alarm.",
    }
