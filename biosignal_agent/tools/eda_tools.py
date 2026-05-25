from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew

from .common import load_csv_signal, signal_quality_summary


EDA_STRESS_MODEL_PATH = Path("/data1/jiahui/biosignal-agent/outputs/eda_wesad/eda_wesad_binary_feature_ensemble.joblib")
EDA_STRESS_CV_METRICS = {
    "benchmark": "WESAD wrist EDA, 60 s windows, 10 s stride, subject-grouped 5-fold CV",
    "accuracy": 0.791268758526603,
    "balanced_accuracy": 0.7802591012923117,
    "macro_f1": 0.7778739485312613,
    "weighted_f1": 0.7920873302485405,
    "auroc": 0.8695682776632961,
}
EDA_AFFECT_MODEL_PATH = Path("/data1/jiahui/biosignal-agent/outputs/eda_wesad/eda_wesad_three_class_feature_ensemble.joblib")
EDA_AFFECT_CNN_PATH = Path("/data1/jiahui/biosignal-agent/outputs/eda_wesad/eda_wesad_three_class_raw_cnn.pt")
EDA_AFFECT_CV_METRICS = {
    "benchmark": "WESAD wrist EDA baseline/stress/amusement, 60 s windows, 10 s stride, subject-grouped 5-fold CV",
    "feature_accuracy": 0.5901593514117977,
    "feature_balanced_accuracy": 0.49610163619002545,
    "feature_macro_f1": 0.48443881368138636,
    "feature_macro_auroc_ovr": 0.6877457441971826,
    "raw_cnn_accuracy": 0.601341906625664,
    "raw_cnn_balanced_accuracy": 0.568003894853204,
    "raw_cnn_macro_f1": 0.5682094896848934,
    "raw_cnn_macro_auroc_ovr": 0.7594973454133324,
}


def _clean_eda(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float).ravel()
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=float)
    med = float(np.nanmedian(x[finite]))
    x = np.where(finite, x, med)
    lo, hi = np.percentile(x, [1, 99])
    if hi > lo:
        x = np.clip(x, lo, hi)
    return x


def _eda_feature_vector(values: np.ndarray, sampling_rate: float) -> list[float]:
    x = _clean_eda(values)
    n = len(x)
    if n == 0:
        return [0.0] * 44
    fs = float(sampling_rate)
    t = np.arange(n) / max(fs, 1e-6)
    if n >= int(fs * 5):
        kernel = max(5, int(fs * 15) | 1)
        if kernel >= n:
            kernel = max(3, (n // 2) * 2 - 1)
        tonic = scipy_signal.medfilt(x, kernel_size=kernel) if kernel >= 3 else np.full_like(x, np.median(x))
    else:
        tonic = np.full_like(x, np.median(x))
    phasic = x - tonic
    dx = np.diff(x, prepend=x[0]) * fs
    dph = np.diff(phasic, prepend=phasic[0]) * fs
    duration_min = max(n / fs / 60.0, 1e-9)
    prominence = max(float(np.nanstd(phasic)) * 0.5, 0.01)
    peaks, props = scipy_signal.find_peaks(phasic, distance=max(1, int(fs)), prominence=prominence)
    rises = np.diff(peaks) / fs if len(peaks) > 1 else np.array([])
    slope = np.polyfit(t, x, 1)[0] if n > 3 else 0.0
    tonic_slope = np.polyfit(t, tonic, 1)[0] if n > 3 else 0.0
    if n >= 16:
        freqs, pxx = scipy_signal.welch(x - np.mean(x), fs=fs, nperseg=min(n, 128))
    else:
        freqs, pxx = np.array([0.0]), np.array([0.0])
    def band(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.trapezoid(pxx[mask], freqs[mask])) if mask.any() else 0.0
    q = np.percentile(x, [5, 25, 50, 75, 95])
    tq = np.percentile(tonic, [5, 50, 95])
    pq = np.percentile(phasic, [5, 50, 95])
    amp = props.get("prominences", np.array([], dtype=float))
    feats = [
        float(np.mean(x)), float(np.std(x)), float(np.min(x)), float(np.max(x)), float(np.ptp(x)),
        *[float(v) for v in q], float(skew(x)), float(kurtosis(x)), float(slope),
        float(np.mean(tonic)), float(np.std(tonic)), *[float(v) for v in tq], float(tonic_slope),
        float(np.mean(phasic)), float(np.std(phasic)), float(np.max(phasic)), float(np.min(phasic)), *[float(v) for v in pq],
        float(np.mean(dx)), float(np.std(dx)), float(np.percentile(np.abs(dx), 95)),
        float(np.mean(dph)), float(np.std(dph)), float(np.percentile(np.abs(dph), 95)),
        float(len(peaks)), float(len(peaks) / duration_min), float(np.mean(amp) if len(amp) else 0.0),
        float(np.max(amp) if len(amp) else 0.0), float(np.std(amp) if len(amp) else 0.0),
        float(np.mean(rises) if len(rises) else 0.0), float(np.std(rises) if len(rises) else 0.0),
        band(0.00, 0.045), band(0.045, 0.15), band(0.15, 0.40), band(0.40, 1.0),
        float(np.mean(np.abs(phasic)) / (abs(np.mean(x)) + 1e-6)),
        float(np.sum(dx > np.percentile(dx, 90)) / max(1, n)),
    ]
    return [0.0 if not np.isfinite(v) else float(v) for v in feats]


def _eda_windows(values: np.ndarray, sampling_rate: float, window_seconds: float = 60.0, step_seconds: float = 10.0) -> list[np.ndarray]:
    nwin = max(1, int(round(window_seconds * sampling_rate)))
    nstep = max(1, int(round(step_seconds * sampling_rate)))
    x = _clean_eda(values)
    if len(x) < nwin:
        return [x]
    return [x[start:start + nwin] for start in range(0, len(x) - nwin + 1, nstep)]


def EDA_assess_quality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    return {"tool": "EDA_assess_quality", "source": data.source, **signal_quality_summary(data.values)}


def EDA_summarize(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    if len(values) == 0:
        return {"tool": "EDA_summarize", "error": "empty signal", "confidence": 0.0}
    smoothed = scipy_signal.medfilt(values, kernel_size=max(3, int(data.sampling_rate) // 2 * 2 + 1)) if len(values) > 5 else values
    phasic = values - smoothed
    return {"tool": "EDA_summarize", "mean_level": float(np.nanmean(values)), "tonic_median": float(np.nanmedian(smoothed)), "phasic_std": float(np.nanstd(phasic)), "num_samples": int(len(values)), "confidence": 0.65, "method": "median_filter_tonic_phasic_summary"}



def EDA_detect_arousal_events(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    if len(values) < max(5, int(data.sampling_rate * 5)):
        return {"tool": "EDA_detect_arousal_events", "error": "signal too short", "confidence": 0.0}
    kernel_size = max(3, int(data.sampling_rate) // 2 * 2 + 1)
    smoothed = scipy_signal.medfilt(values, kernel_size=kernel_size) if len(values) > kernel_size else values
    phasic = values - smoothed
    distance = max(1, int(1.0 * data.sampling_rate))
    prominence = max(float(np.nanstd(phasic)) * 0.75, 1e-8)
    peaks, _ = scipy_signal.find_peaks(phasic, distance=distance, prominence=prominence)
    duration_min = len(values) / float(data.sampling_rate) / 60.0 if data.sampling_rate else 0.0
    rate = float(len(peaks) / duration_min) if duration_min > 0 else None
    arousal_level = "high_arousal_proxy" if rate is not None and rate > 6 else "low_arousal_proxy"
    return {
        "tool": "EDA_detect_arousal_events",
        "arousal_event_indices": peaks.tolist(),
        "arousal_event_count": int(len(peaks)),
        "arousal_rate_per_min": rate,
        "arousal_level": arousal_level,
        "confidence": 0.6,
        "method": "eda_phasic_peak_screening",
        "disclaimer": "Screening heuristic only; EDA arousal events are not equivalent to emotion or stress labels.",
    }


def EDA_screen_stress_proxy(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    summary = EDA_summarize(signal_path, sampling_rate, column)
    events = EDA_detect_arousal_events(signal_path, sampling_rate, column)
    if summary.get("error"):
        return {"tool": "EDA_screen_stress_proxy", "error": summary["error"], "confidence": 0.1}
    phasic_std = float(summary.get("phasic_std") or 0.0)
    mean_level = abs(float(summary.get("mean_level") or 0.0))
    normalized_phasic = float(phasic_std / (mean_level + 1e-8))
    arousal_rate = events.get("arousal_rate_per_min") if not events.get("error") else None
    score = 0
    flags = []
    if arousal_rate is not None and arousal_rate > 6.0:
        score += 1
        flags.append("high_scr_rate")
    if normalized_phasic > 0.08:
        score += 1
        flags.append("high_phasic_variability")
    if float(summary.get("tonic_median") or 0.0) > float(summary.get("mean_level") or 0.0) * 1.2:
        score += 1
        flags.append("elevated_tonic_level_proxy")
    stress_level = "elevated_stress_arousal_proxy" if score >= 2 else "low_stress_arousal_proxy"
    return {
        "tool": "EDA_screen_stress_proxy",
        "stress_arousal_score": score,
        "stress_arousal_level": stress_level,
        "stress_arousal_flags": flags,
        "arousal_rate_per_min": arousal_rate,
        "normalized_phasic_std": normalized_phasic,
        "mean_level": summary.get("mean_level"),
        "tonic_median": summary.get("tonic_median"),
        "confidence": 0.55,
        "method": "eda_tonic_phasic_arousal_score_baseline",
        "disclaimer": "Stress/arousal proxy only; validated stress classification requires labeled protocol data and multimodal context.",
    }



def EDA_screen_stress_ml(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    if len(values) < max(20, int(data.sampling_rate * 20)):
        proxy = EDA_screen_stress_proxy(signal_path, sampling_rate, column)
        return {
            "tool": "EDA_screen_stress_ml",
            "error": "signal too short for WESAD-style 60 s stress classifier; used proxy evidence only",
            "proxy_result": proxy,
            "confidence": 0.25,
            "method": "wesad_eda_feature_ensemble_fallback_proxy",
            "disclaimer": "EDA stress classification is protocol- and subject-dependent; not a diagnosis or lie detector.",
        }
    if not EDA_STRESS_MODEL_PATH.exists():
        proxy = EDA_screen_stress_proxy(signal_path, sampling_rate, column)
        return {
            "tool": "EDA_screen_stress_ml",
            "error": f"trained model not found: {EDA_STRESS_MODEL_PATH}",
            "proxy_result": proxy,
            "confidence": 0.25,
            "method": "missing_wesad_model_fallback_proxy",
        }
    bundle = joblib.load(EDA_STRESS_MODEL_PATH)
    model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
    windows = _eda_windows(values, data.sampling_rate)
    X = np.asarray([_eda_feature_vector(win, data.sampling_rate) for win in windows], dtype=float)
    classes = list(getattr(model, "classes_", []))
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        stress_index = classes.index("stress") if "stress" in classes else int(np.argmax(np.mean(probs, axis=0)))
        stress_probs = probs[:, stress_index]
    else:
        pred = model.predict(X)
        stress_probs = np.asarray([1.0 if p == "stress" else 0.0 for p in pred], dtype=float)
    probability = float(np.mean(stress_probs))
    high_fraction = float(np.mean(stress_probs >= 0.5))
    level = "stress_likely" if probability >= 0.5 else "non_stress_likely"
    return {
        "tool": "EDA_screen_stress_ml",
        "stress_probability": probability,
        "stress_window_fraction": high_fraction,
        "stress_level": level,
        "num_windows": int(len(windows)),
        "model_source": str(EDA_STRESS_MODEL_PATH),
        "model_cv_metrics": EDA_STRESS_CV_METRICS,
        "confidence": float(min(0.9, max(0.35, abs(probability - 0.5) * 1.2 + 0.35))),
        "method": "wesad_eda_feature_ensemble_subject_grouped_cv",
        "disclaimer": "Trained on WESAD wrist EDA stress protocol. Use as research stress/arousal screening only; not diagnostic and not a standalone lie detector.",
    }



def EDA_extract_tonic_phasic_features(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    if len(values) < max(5, int(data.sampling_rate * 5)):
        return {"tool": "EDA_extract_tonic_phasic_features", "error": "signal too short", "confidence": 0.0}
    x = _clean_eda(values)
    features = _eda_feature_vector(x, data.sampling_rate)
    n = len(x)
    fs = float(data.sampling_rate)
    kernel = max(5, int(fs * 15) | 1)
    if kernel >= n:
        kernel = max(3, (n // 2) * 2 - 1)
    tonic = scipy_signal.medfilt(x, kernel_size=kernel) if kernel >= 3 else np.full_like(x, np.median(x))
    phasic = x - tonic
    prominence = max(float(np.nanstd(phasic)) * 0.5, 0.01)
    peaks, props = scipy_signal.find_peaks(phasic, distance=max(1, int(fs)), prominence=prominence)
    duration_min = len(x) / fs / 60.0
    return {
        "tool": "EDA_extract_tonic_phasic_features",
        "mean_level": float(np.mean(x)),
        "tonic_mean": float(np.mean(tonic)),
        "tonic_std": float(np.std(tonic)),
        "phasic_mean": float(np.mean(phasic)),
        "phasic_std": float(np.std(phasic)),
        "scr_count": int(len(peaks)),
        "scr_rate_per_min": float(len(peaks) / max(duration_min, 1e-9)),
        "scr_mean_prominence": float(np.mean(props.get("prominences", [0.0])) if len(peaks) else 0.0),
        "scr_max_prominence": float(np.max(props.get("prominences", [0.0])) if len(peaks) else 0.0),
        "dynamic_range": float(np.ptp(x)),
        "slope_per_second": float(np.polyfit(np.arange(n) / fs, x, 1)[0]) if n > 3 else 0.0,
        "feature_vector_length": int(len(features)),
        "confidence": 0.75,
        "method": "median_tonic_phasic_scr_feature_extraction_wesad_compatible",
        "disclaimer": "EDA tonic/phasic features reflect autonomic arousal and sensor/contact effects; interpretation needs protocol context.",
    }


def EDA_classify_affective_state_ml(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    if len(values) < max(20, int(data.sampling_rate * 20)):
        return {
            "tool": "EDA_classify_affective_state_ml",
            "error": "signal too short for WESAD-style 60 s baseline/stress/amusement classifier",
            "confidence": 0.15,
            "method": "wesad_three_class_feature_ensemble_unavailable_short_signal",
        }
    windows = _eda_windows(values, data.sampling_rate)
    if EDA_AFFECT_CNN_PATH.exists():
        try:
            import torch
            from torch import nn

            class _EdaCnn(nn.Module):
                def __init__(self, n_classes: int):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Conv1d(1, 32, 9, padding=4), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                        nn.Conv1d(32, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                        nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
                        nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.25), nn.Linear(128, n_classes),
                    )

                def forward(self, x):
                    return self.net(x)

            payload = torch.load(EDA_AFFECT_CNN_PATH, map_location="cpu")
            labels = list(payload.get("labels", ["amusement", "baseline", "stress"]))
            model = _EdaCnn(len(labels))
            model.load_state_dict(payload["state_dict"])
            model.eval()
            raw = []
            for win in windows:
                z = _clean_eda(win)
                z = (z - np.mean(z)) / (np.std(z) + 1e-6)
                raw.append(z.astype(np.float32))
            with torch.no_grad():
                logits = model(torch.tensor(np.asarray(raw)[:, None, :], dtype=torch.float32))
                mean_probs = torch.softmax(logits, dim=1).numpy().mean(axis=0)
            probabilities = {str(cls): float(prob) for cls, prob in zip(labels, mean_probs)}
            state = str(labels[int(np.argmax(mean_probs))])
            return {
                "tool": "EDA_classify_affective_state_ml",
                "predicted_state": state,
                "state_probabilities": probabilities,
                "num_windows": int(len(windows)),
                "model_source": str(EDA_AFFECT_CNN_PATH),
                "model_cv_metrics": EDA_AFFECT_CV_METRICS,
                "confidence": float(min(0.8, max(0.25, max(probabilities.values()) if probabilities else 0.25))),
                "method": "wesad_baseline_stress_amusement_raw_eda_cnn_subject_grouped_cv",
                "disclaimer": "This is WESAD protocol-state classification (baseline/stress/amusement), not general emotion, anxiety, pain, deception, or clinical diagnosis.",
            }
        except Exception as exc:
            cnn_error = str(exc)
    else:
        cnn_error = f"trained CNN model not found: {EDA_AFFECT_CNN_PATH}"
    if not EDA_AFFECT_MODEL_PATH.exists():
        return {
            "tool": "EDA_classify_affective_state_ml",
            "error": f"{cnn_error}; fallback model not found: {EDA_AFFECT_MODEL_PATH}",
            "confidence": 0.15,
            "method": "missing_wesad_three_class_models",
        }
    bundle = joblib.load(EDA_AFFECT_MODEL_PATH)
    model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
    X = np.asarray([_eda_feature_vector(win, data.sampling_rate) for win in windows], dtype=float)
    probs = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    classes = list(getattr(model, "classes_", []))
    if probs is not None and classes:
        mean_probs = np.mean(probs, axis=0)
        probabilities = {str(cls): float(prob) for cls, prob in zip(classes, mean_probs)}
        state = str(classes[int(np.argmax(mean_probs))])
    else:
        preds = model.predict(X)
        counts = {str(k): float(v / len(preds)) for k, v in zip(*np.unique(preds, return_counts=True))}
        probabilities = counts
        state = max(counts, key=counts.get)
    return {
        "tool": "EDA_classify_affective_state_ml",
        "predicted_state": state,
        "state_probabilities": probabilities,
        "num_windows": int(len(windows)),
        "model_source": str(EDA_AFFECT_MODEL_PATH),
        "model_cv_metrics": EDA_AFFECT_CV_METRICS,
        "cnn_error": cnn_error,
        "confidence": float(min(0.75, max(0.25, max(probabilities.values()) if probabilities else 0.25))),
        "method": "wesad_baseline_stress_amusement_feature_ensemble_fallback_subject_grouped_cv",
        "disclaimer": "This is WESAD protocol-state classification (baseline/stress/amusement), not general emotion, anxiety, pain, deception, or clinical diagnosis.",
    }


def EDA_route_task_recommendation(task: str, signal_path: str | None = None, sampling_rate: float | None = None, column: str | None = None) -> dict:
    task_l = task.lower()
    mapping = {
        "stress": ("EDA_screen_stress_ml", "Use WESAD-trained EDA stress classifier, with tonic/phasic features as supporting evidence."),
        "arousal": ("EDA_extract_tonic_phasic_features", "Use SCR rate/prominence and tonic/phasic features; WESAD affective-state classifier can be used only for baseline/stress/amusement protocol states."),
        "emotion": ("EDA_classify_affective_state_ml", "Use only as WESAD baseline/stress/amusement state classifier; use DEAP/AMIGOS-like multimodal data for general emotion."),
        "anxiety": ("EDA_screen_stress_ml", "Treat as stress/arousal support only; not a clinical anxiety classifier."),
        "cognitive": ("EDA_extract_tonic_phasic_features", "EDA can support workload studies but needs task labels and preferably HR/ACC/EEG/context."),
        "attention": ("EDA_extract_tonic_phasic_features", "EDA is auxiliary for cognitive load/attention; do not infer attention from EDA alone."),
        "pain": ("EDA_extract_tonic_phasic_features", "EDA can support pain-response studies with stimulus labels; no standalone pain tool is exposed."),
        "sleep": ("EDA_extract_tonic_phasic_features", "EDA can support autonomic arousal/awakening detection; PSG/ACC remain primary."),
        "lie": ("none", "Do not use EDA alone for lie detection; reliability and ethics are insufficient."),
        "deception": ("none", "Do not use EDA alone for deception detection; reliability and ethics are insufficient."),
        "seizure": ("EDA_extract_tonic_phasic_features", "EDA should be auxiliary with ACC/PPG/video-EEG labels for seizure alarms."),
        "ux": ("EDA_extract_tonic_phasic_features", "Use arousal features tied to stimulus timestamps; not valence by itself."),
        "exercise": ("EDA_extract_tonic_phasic_features", "EDA is heavily confounded by sweat/motion/temperature; use as auxiliary with ACC/PPG."),
        "recovery": ("EDA_extract_tonic_phasic_features", "Use tonic trend/SCR suppression as auxiliary recovery evidence with HRV/ACC."),
    }
    selected = ("EDA_extract_tonic_phasic_features", "Default EDA feature extraction; task-specific interpretation requires labels and context.")
    for key, value in mapping.items():
        if key in task_l:
            selected = value
            break
    result = {
        "tool": "EDA_route_task_recommendation",
        "requested_task": task,
        "recommended_tool": selected[0],
        "recommendation": selected[1],
        "confidence": 0.7 if selected[0] != "none" else 0.95,
        "method": "eda_task_safety_and_benchmark_router",
    }
    if signal_path and sampling_rate and selected[0] == "EDA_screen_stress_ml":
        result["stress_result"] = EDA_screen_stress_ml(signal_path, sampling_rate, column)
    elif signal_path and sampling_rate and selected[0] == "EDA_classify_affective_state_ml":
        result["affective_state_result"] = EDA_classify_affective_state_ml(signal_path, sampling_rate, column)
    elif signal_path and sampling_rate and selected[0] == "EDA_extract_tonic_phasic_features":
        result["feature_result"] = EDA_extract_tonic_phasic_features(signal_path, sampling_rate, column)
    return result
