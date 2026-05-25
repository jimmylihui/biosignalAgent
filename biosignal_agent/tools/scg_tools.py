from __future__ import annotations

import numpy as np
import pandas as pd
from functools import lru_cache
from pathlib import Path
import joblib
from scipy import signal as scipy_signal

from .common import bpm_from_peaks, interval_regularity, load_csv_signal, signal_quality_summary
from .peak_detectors import neurokit_nabian2018_peaks



_DEFAULT_SCG_AO_MODEL_PATH = Path("/data1/jiahui/biosignal-agent/outputs/scg_vhd_ao_ecg_anchor_cnn.pt")
SCG_MECHANICAL_CLASSIFIER_PATH = Path("/data1/jiahui/biosignal-agent/outputs/scg_vhd_mechanical_subtype_classifier.joblib")
SCG_RHC_HF_MODEL_PATH = Path("/data1/jiahui/biosignal-agent/outputs/scg_rhc_hf_feature_ensemble.joblib")


class _SCGBeatAOModel:
    def __init__(self, torch_module, base: int = 32):
        nn = torch_module.nn
        self.model = nn.Sequential(
            nn.Conv1d(1, base, 9, padding=4), nn.BatchNorm1d(base), nn.SiLU(),
            nn.Conv1d(base, base, 9, padding=4), nn.BatchNorm1d(base), nn.SiLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(base, base * 2, 7, padding=3), nn.BatchNorm1d(base * 2), nn.SiLU(),
            nn.Conv1d(base * 2, base * 2, 7, padding=3), nn.BatchNorm1d(base * 2), nn.SiLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(base * 2, base * 4, 5, padding=2), nn.BatchNorm1d(base * 4), nn.SiLU(),
            nn.Conv1d(base * 4, base * 4, 5, padding=2), nn.BatchNorm1d(base * 4), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(base * 4, base * 2, 5, padding=2), nn.BatchNorm1d(base * 2), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(base * 2, base, 5, padding=2), nn.BatchNorm1d(base), nn.SiLU(),
            nn.Conv1d(base, 1, 1),
        )

    def __call__(self, x):
        y = self.model(x)
        return y[..., : x.shape[-1]]

    def eval(self):
        self.model.eval()
        return self

    def load_state_dict(self, state_dict):
        stripped = {k[4:] if k.startswith("net.") else k: v for k, v in state_dict.items()}
        self.model.load_state_dict(stripped)


@lru_cache(maxsize=2)
def _load_ecg_anchor_ao_model(model_path: str = str(_DEFAULT_SCG_AO_MODEL_PATH)):
    path = Path(model_path)
    if not path.exists():
        return None
    try:
        import torch

        payload = torch.load(path, map_location="cpu")
        wrapper = _SCGBeatAOModel(torch)
        wrapper.load_state_dict(payload["model_state_dict"])
        wrapper.eval()
        return {
            "torch": torch,
            "model": wrapper,
            "target_fs": float(payload.get("target_fs", 256.0)),
            "pre_s": float(payload.get("pre_s", 0.05)),
            "post_s": float(payload.get("post_s", 0.35625)),
            "path": str(path),
        }
    except Exception:
        return None


def _predict_ao_with_ecg_anchor_model(scg_values: np.ndarray, scg_fs: float, ecg_values: np.ndarray, ecg_fs: float) -> tuple[np.ndarray, dict]:
    bundle = _load_ecg_anchor_ao_model()
    if bundle is None:
        return np.asarray([], dtype=int), {"ml_backend_available": False}
    torch = bundle["torch"]
    target_fs = bundle["target_fs"]
    pre_s = bundle["pre_s"]
    post_s = bundle["post_s"]
    scg = _safe_bandpass(scg_values, scg_fs, 0.8, 35.0).astype(np.float32)
    if scg_fs != target_fs:
        scg = scipy_signal.resample(scg, int(round(len(scg) * target_fs / scg_fs))).astype(np.float32)
    ecg_r = _detect_ecg_r_peaks(ecg_values, ecg_fs)
    if ecg_fs != target_fs:
        ecg_r_model_fs = np.asarray(np.round(ecg_r * target_fs / ecg_fs), dtype=int)
    else:
        ecg_r_model_fs = np.asarray(ecg_r, dtype=int)
    if len(ecg_r_model_fs) < 2:
        return np.asarray([], dtype=int), {"ml_backend_available": True, "reason": "too_few_ecg_r_peaks"}

    pre = int(round(pre_s * target_fs))
    n = int(round((pre_s + post_s) * target_fs))
    lo = int(round((pre_s + 0.04) * target_fs))
    hi = min(n, int(round((pre_s + 0.32) * target_fs)))
    preds = []
    scores = []
    with torch.no_grad():
        for r_peak in ecg_r_model_fs:
            start = int(r_peak) - pre
            end = start + n
            if start < 0 or end > len(scg) or hi <= lo:
                continue
            seg = torch.from_numpy(scg[start:end][None, None, :].astype(np.float32))
            prob = torch.sigmoid(bundle["model"](seg)).cpu().numpy()[0, 0]
            local = lo + int(np.argmax(prob[lo:hi]))
            preds.append(start + local)
            scores.append(float(prob[local]))
    if scg_fs != target_fs:
        preds = np.asarray(np.round(np.asarray(preds) * scg_fs / target_fs), dtype=int)
    else:
        preds = np.asarray(preds, dtype=int)
    return preds, {
        "ml_backend_available": True,
        "ml_model_path": bundle["path"],
        "ml_target_fs": target_fs,
        "ml_mean_peak_probability": float(np.mean(scores)) if scores else None,
        "ecg_r_count": int(len(ecg_r)),
    }

def _bandpower(values: np.ndarray, sampling_rate: float, low_hz: float, high_hz: float) -> float:
    if len(values) < max(8, sampling_rate * 2):
        return 0.0
    high = min(high_hz, sampling_rate * 0.45)
    if high <= low_hz:
        return 0.0
    centered = values - np.nanmedian(values)
    freqs, psd = scipy_signal.welch(centered, fs=sampling_rate, nperseg=min(len(centered), int(sampling_rate * 8)))
    mask = (freqs >= low_hz) & (freqs <= high)
    return float(np.trapezoid(psd[mask], freqs[mask])) if np.any(mask) else 0.0


def _flat_window_fraction(values: np.ndarray, sampling_rate: float) -> float:
    window = max(8, int(round(sampling_rate)))
    if len(values) < window:
        return 0.0
    global_std = float(np.nanstd(values)) + 1e-12
    fractions = []
    for start in range(0, len(values) - window + 1, window):
        fractions.append(float(np.nanstd(values[start:start + window]) < 0.05 * global_std))
    return float(np.mean(fractions)) if fractions else 0.0


def SCG_assess_quality(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    summary = signal_quality_summary(values)
    if len(values) == 0:
        return {"tool": "SCG_assess_quality", "source": data.source, **summary}

    dynamic_range = float(np.nanpercentile(values, 95) - np.nanpercentile(values, 5))
    total_power = _bandpower(values, data.sampling_rate, 0.08, min(45.0, data.sampling_rate * 0.45)) + 1e-12
    respiration_motion_power = _bandpower(values, data.sampling_rate, 0.08, 0.7)
    cardiac_power = _bandpower(values, data.sampling_rate, 0.8, 35.0)
    high_frequency_power = _bandpower(values, data.sampling_rate, 35.0, min(80.0, data.sampling_rate * 0.45))
    motion_power_ratio = float(respiration_motion_power / total_power)
    cardiac_power_ratio = float(cardiac_power / total_power)
    high_frequency_power_ratio = float(high_frequency_power / total_power)
    flat_window_fraction = _flat_window_fraction(values, data.sampling_rate)

    reasons = []
    if summary.get("quality") == "bad":
        reasons.append("generic_signal_quality_failure")
    if dynamic_range < 0.1:
        reasons.append("very_low_dynamic_range")
    if flat_window_fraction > 0.10:
        reasons.append("flat_or_dropout_windows")
    if motion_power_ratio > 0.55:
        reasons.append("dominant_low_frequency_motion")
    if high_frequency_power_ratio > 0.30:
        reasons.append("dominant_high_frequency_noise")
    if cardiac_power_ratio < 0.25:
        reasons.append("weak_scg_cardiac_band")

    if reasons:
        quality = "bad" if len(reasons) >= 1 else "moderate"
        confidence = 0.25 if quality == "bad" else 0.55
    elif cardiac_power_ratio < 0.45 or motion_power_ratio > 0.35 or high_frequency_power_ratio > 0.18:
        quality = "moderate"
        confidence = 0.6
    else:
        quality = "good"
        confidence = 0.82

    return {
        "tool": "SCG_assess_quality",
        "source": data.source,
        "quality": quality,
        "confidence": confidence,
        "reasons": reasons,
        "dynamic_range": dynamic_range,
        "flat_window_fraction": flat_window_fraction,
        "motion_power_ratio": motion_power_ratio,
        "cardiac_power_ratio": cardiac_power_ratio,
        "high_frequency_power_ratio": high_frequency_power_ratio,
        "generic_quality": summary,
    }


def SCG_detect_j_peaks(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    peaks, details = neurokit_nabian2018_peaks(data.values, data.sampling_rate, low_hz=0.8, high_hz=20.0, fallback_threshold_scale=0.30)
    heart_rate = bpm_from_peaks(peaks, data.sampling_rate)
    regularity = interval_regularity(peaks, data.sampling_rate)
    base_confidence = 0.62 if details["method"] == "nabian2018" else 0.45
    confidence = min(base_confidence, regularity["regularity_confidence"])
    if heart_rate is None or not 35 <= heart_rate <= 220:
        confidence = 0.25
    return {"tool": "SCG_detect_j_peaks", "j_peak_indices": peaks.tolist(), "num_peaks": int(len(peaks)), "heart_rate_bpm": heart_rate, "confidence": confidence, **regularity, **details}



def _safe_bandpass(values: np.ndarray, sampling_rate: float, low_hz: float, high_hz: float) -> np.ndarray:
    if len(values) < max(16, int(sampling_rate)):
        return values - np.nanmedian(values) if len(values) else values
    high = min(high_hz, sampling_rate * 0.45)
    if high <= low_hz:
        return values - np.nanmedian(values)
    centered = values - np.nanmedian(values)
    sos = scipy_signal.butter(3, [low_hz / (0.5 * sampling_rate), high / (0.5 * sampling_rate)], btype="bandpass", output="sos")
    return scipy_signal.sosfiltfilt(sos, centered)


def _local_abs_extrema(values: np.ndarray, center: int, sampling_rate: float, start_s: float, end_s: float) -> int | None:
    lo = max(0, int(center + round(start_s * sampling_rate)))
    hi = min(len(values), int(center + round(end_s * sampling_rate)))
    if hi <= lo:
        return None
    return int(lo + np.argmax(np.abs(values[lo:hi])))


def _ecg_peaks_plausible(peaks: np.ndarray, sampling_rate: float, n_samples: int) -> bool:
    if len(peaks) < 2:
        return False
    interval_hr = bpm_from_peaks(peaks, sampling_rate)
    duration_min = max(n_samples / sampling_rate / 60.0, 1e-6)
    count_hr = len(peaks) / duration_min
    return bool(interval_hr is not None and 35 <= interval_hr <= 220 and 35 <= count_hr <= 220)


def _detect_ecg_r_peaks(values: np.ndarray, sampling_rate: float) -> np.ndarray:
    if len(values) < sampling_rate * 3:
        return np.asarray([], dtype=int)
    try:
        import neurokit2 as nk

        cleaned = nk.ecg_clean(values, sampling_rate=sampling_rate, method="pantompkins1985")
        _, info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate, method="pantompkins1985", correct_artifacts=True)
        peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
        if _ecg_peaks_plausible(peaks, sampling_rate, len(values)):
            return peaks
    except Exception:
        pass
    try:
        peaks, _ = neurokit_nabian2018_peaks(values, sampling_rate, low_hz=None, high_hz=None, fallback_threshold_scale=0.6)
        if _ecg_peaks_plausible(peaks, sampling_rate, len(values)):
            return peaks.astype(int)
    except Exception:
        pass
    filtered = _safe_bandpass(values, sampling_rate, 5.0, 20.0)
    envelope = np.abs(filtered)
    height = float(np.nanpercentile(envelope, 95))
    distance = max(1, int(round(0.36 * sampling_rate)))
    peaks, _ = scipy_signal.find_peaks(envelope, height=height, distance=distance)
    return peaks.astype(int)


def _pair_previous_events(reference: np.ndarray, targets: np.ndarray, sampling_rate: float, min_delay_s: float, max_delay_s: float) -> list[tuple[int, int]]:
    pairs = []
    reference = np.asarray(reference, dtype=int)
    for target in np.asarray(targets, dtype=int):
        candidates = reference[(reference <= target - int(round(min_delay_s * sampling_rate))) & (reference >= target - int(round(max_delay_s * sampling_rate)))]
        if len(candidates):
            pairs.append((int(candidates[-1]), int(target)))
    return pairs


def _median_iqr_ms(intervals_s: list[float]) -> dict:
    intervals = np.asarray([x for x in intervals_s if np.isfinite(x) and x > 0], dtype=float)
    if len(intervals) == 0:
        return {"median_ms": None, "iqr_ms": None, "n": 0}
    return {
        "median_ms": float(np.median(intervals) * 1000.0),
        "iqr_ms": float((np.percentile(intervals, 75) - np.percentile(intervals, 25)) * 1000.0) if len(intervals) > 1 else 0.0,
        "n": int(len(intervals)),
    }


def SCG_detect_fiducial_points(
    signal_path: str,
    sampling_rate: float,
    column: str | None = None,
    ecg_path: str | None = None,
    ecg_sampling_rate: float | None = None,
    ecg_column: str | None = None,
) -> dict:
    """Detect approximate SCG fiducial points.

    AO is approximated from the existing SCG J-peak detector. AC, MC, and MO are
    local mechanical extrema in physiological windows around AO. If ECG is
    provided, ECG R peaks are paired to AO for R-to-AO timing; this is not a full
    Q-onset PEP measurement.
    """
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    if len(values) < data.sampling_rate * 3:
        return {"tool": "SCG_detect_fiducial_points", "error": "signal too short", "confidence": 0.0}

    filtered = _safe_bandpass(values, data.sampling_rate, 0.8, 35.0)
    ml_details = None
    ao = np.asarray([], dtype=int)
    peak_details = {}
    if ecg_path:
        ecg_data_for_ml = load_csv_signal(ecg_path, ecg_sampling_rate or data.sampling_rate, ecg_column)
        ml_ao, ml_details = _predict_ao_with_ecg_anchor_model(values, data.sampling_rate, ecg_data_for_ml.values, ecg_data_for_ml.sampling_rate)
        if len(ml_ao) >= 2:
            ao = np.asarray(ml_ao, dtype=int)
            peak_details = {"detector_backend": "ecg_anchor_cnn", **ml_details}
    if len(ao) < 2:
        peaks, peak_details = neurokit_nabian2018_peaks(values, data.sampling_rate, low_hz=0.8, high_hz=20.0, fallback_threshold_scale=0.30)
        if ml_details is not None:
            peak_details = {**peak_details, "ml_backend": ml_details}
        ao = np.asarray(peaks, dtype=int)
    if len(ao) < 2:
        return {"tool": "SCG_detect_fiducial_points", "ao_indices": ao.tolist(), "confidence": 0.2, "method": "heuristic_scg_fiducials", **peak_details}

    cycle_s = float(np.median(np.diff(ao)) / data.sampling_rate)
    ac, mc, mo = [], [], []
    for i, aortic_open in enumerate(ao):
        ac_end = min(0.45, max(0.22, 0.70 * cycle_s))
        ac_i = _local_abs_extrema(filtered, int(aortic_open), data.sampling_rate, 0.18, ac_end)
        mc_i = _local_abs_extrema(filtered, int(aortic_open), data.sampling_rate, -0.14, -0.02)
        if ac_i is not None:
            next_ao = int(ao[i + 1]) if i + 1 < len(ao) else len(filtered) - 1
            mo_end = min(0.30, max(0.10, (next_ao - ac_i) / data.sampling_rate - 0.03))
            mo_i = _local_abs_extrema(filtered, ac_i, data.sampling_rate, 0.06, mo_end)
        else:
            mo_i = None
        ac.append(ac_i)
        mc.append(mc_i)
        mo.append(mo_i)

    ecg_r = []
    r_to_ao_ms = []
    if ecg_path:
        ecg_data = load_csv_signal(ecg_path, ecg_sampling_rate or data.sampling_rate, ecg_column)
        ecg_r = _detect_ecg_r_peaks(ecg_data.values, ecg_data.sampling_rate).tolist()
        if ecg_data.sampling_rate != data.sampling_rate:
            ecg_r_scg_fs = np.asarray(np.round(np.asarray(ecg_r) * data.sampling_rate / ecg_data.sampling_rate), dtype=int)
        else:
            ecg_r_scg_fs = np.asarray(ecg_r, dtype=int)
        pairs = _pair_previous_events(ecg_r_scg_fs, ao, data.sampling_rate, min_delay_s=0.04, max_delay_s=0.30)
        r_to_ao_ms = [float((a - r) / data.sampling_rate * 1000.0) for r, a in pairs]

    regularity = interval_regularity(ao, data.sampling_rate)
    if peak_details.get("detector_backend") == "ecg_anchor_cnn":
        confidence = min(0.78, max(0.62, regularity["regularity_confidence"]))
    else:
        confidence = min(0.58, regularity["regularity_confidence"])
    if ecg_path and len(r_to_ao_ms) >= max(3, int(0.5 * len(ao))):
        confidence = min(0.82 if peak_details.get("detector_backend") == "ecg_anchor_cnn" else 0.70, confidence + 0.12)
    return {
        "tool": "SCG_detect_fiducial_points",
        "source": data.source,
        "ao_indices": ao.tolist(),
        "ac_indices": [int(x) if x is not None else None for x in ac],
        "mc_indices": [int(x) if x is not None else None for x in mc],
        "mo_indices": [int(x) if x is not None else None for x in mo],
        "ecg_r_indices": ecg_r,
        "r_to_ao_ms": r_to_ao_ms,
        "num_beats": int(len(ao)),
        "heart_rate_bpm": bpm_from_peaks(ao, data.sampling_rate),
        "confidence": confidence,
        "method": "ecg_anchor_cnn_fiducial_windows" if peak_details.get("detector_backend") == "ecg_anchor_cnn" else "heuristic_scg_fiducial_windows",
        **regularity,
        **peak_details,
        "disclaimer": "ECG-assisted CNN AO detector is used when ECG is supplied and the trained model is available; otherwise this falls back to heuristic SCG fiducial windows. ECG input gives R-to-AO timing, not Q-onset PEP.",
    }


def SCG_compute_cardiac_time_intervals(
    signal_path: str,
    sampling_rate: float,
    column: str | None = None,
    ecg_path: str | None = None,
    ecg_sampling_rate: float | None = None,
    ecg_column: str | None = None,
) -> dict:
    fid = SCG_detect_fiducial_points(signal_path, sampling_rate, column, ecg_path, ecg_sampling_rate, ecg_column)
    if fid.get("error"):
        return {"tool": "SCG_compute_cardiac_time_intervals", **fid}
    fs = float(sampling_rate)
    ao = fid.get("ao_indices", [])
    ac = fid.get("ac_indices", [])
    mc = fid.get("mc_indices", [])
    mo = fid.get("mo_indices", [])

    lvet, ivct, ivrt = [], [], []
    for a, c, m, o in zip(ao, ac, mc, mo):
        if a is not None and c is not None and c > a:
            lvet.append((c - a) / fs)
        if m is not None and a is not None and a > m:
            ivct.append((a - m) / fs)
        if c is not None and o is not None and o > c:
            ivrt.append((o - c) / fs)
    r_to_ao = [x / 1000.0 for x in fid.get("r_to_ao_ms", [])]
    pep_like = _median_iqr_ms(r_to_ao)
    lvet_summary = _median_iqr_ms(lvet)
    ivct_summary = _median_iqr_ms(ivct)
    ivrt_summary = _median_iqr_ms(ivrt)
    ratio = None
    if pep_like["median_ms"] is not None and lvet_summary["median_ms"] not in (None, 0):
        ratio = float(pep_like["median_ms"] / lvet_summary["median_ms"])
    confidence = fid.get("confidence", 0.3)
    if not ecg_path:
        confidence = min(confidence, 0.45)
    return {
        "tool": "SCG_compute_cardiac_time_intervals",
        "heart_rate_bpm": fid.get("heart_rate_bpm"),
        "r_to_ao_ms": pep_like,
        "lvet_ms": lvet_summary,
        "ivct_ms": ivct_summary,
        "ivrt_ms": ivrt_summary,
        "r_to_ao_over_lvet": ratio,
        "num_beats": fid.get("num_beats"),
        "confidence": confidence,
        "method": "fiducial_window_cardiac_time_intervals",
        "fiducial_summary": {"ao_count": len(ao), "ac_count": sum(x is not None for x in ac), "mc_count": sum(x is not None for x in mc), "mo_count": sum(x is not None for x in mo)},
        "disclaimer": "Research-use timing proxy. True PEP requires ECG Q-onset and validated AO labels; this tool reports R-to-AO when ECG is supplied.",
    }


def SCG_estimate_contractility_proxy(
    signal_path: str,
    sampling_rate: float,
    column: str | None = None,
    ecg_path: str | None = None,
    ecg_sampling_rate: float | None = None,
    ecg_column: str | None = None,
) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    intervals = SCG_compute_cardiac_time_intervals(signal_path, sampling_rate, column, ecg_path, ecg_sampling_rate, ecg_column)
    if intervals.get("error"):
        return {"tool": "SCG_estimate_contractility_proxy", **intervals}
    fid = SCG_detect_fiducial_points(signal_path, sampling_rate, column, ecg_path, ecg_sampling_rate, ecg_column)
    filtered = _safe_bandpass(data.values, data.sampling_rate, 5.0, 35.0)
    ao = np.asarray(fid.get("ao_indices", []), dtype=int)
    systolic_energy = []
    ao_amplitude = []
    for i, a in enumerate(ao):
        end = int(fid.get("ac_indices", [None] * len(ao))[i] or min(len(filtered), a + int(0.30 * data.sampling_rate)))
        if end > a:
            systolic_energy.append(float(np.mean(filtered[a:end] ** 2)))
            ao_amplitude.append(float(abs(filtered[a])))
    energy = float(np.median(systolic_energy)) if systolic_energy else None
    amplitude = float(np.median(ao_amplitude)) if ao_amplitude else None
    ratio = intervals.get("r_to_ao_over_lvet")
    r_to_ao_ms = intervals.get("r_to_ao_ms", {}).get("median_ms")
    label = "indeterminate"
    score = None
    drivers = []
    if ratio is not None:
        score = float(np.clip(1.0 - (ratio - 0.25) / 0.35, 0.0, 1.0))
        if ratio > 0.55:
            label = "prolonged_electromechanical_timing_proxy"
            drivers.append("high_r_to_ao_over_lvet")
        elif ratio < 0.35:
            label = "short_electromechanical_timing_proxy"
            drivers.append("low_r_to_ao_over_lvet")
        else:
            label = "intermediate_electromechanical_timing_proxy"
    elif energy is not None:
        score = float(np.clip(np.log1p(energy) / 3.0, 0.0, 1.0))
        label = "mechanical_energy_proxy_only"
        drivers.append("scg_systolic_energy_without_ecg")
    confidence = intervals.get("confidence", 0.35)
    if ratio is None:
        confidence = min(confidence, 0.40)
    return {
        "tool": "SCG_estimate_contractility_proxy",
        "contractility_proxy_label": label,
        "contractility_proxy_score": score,
        "drivers": drivers,
        "r_to_ao_ms": r_to_ao_ms,
        "lvet_ms": intervals.get("lvet_ms", {}).get("median_ms"),
        "r_to_ao_over_lvet": ratio,
        "median_systolic_energy": energy,
        "median_ao_amplitude": amplitude,
        "confidence": confidence,
        "method": "scg_time_interval_and_energy_proxy",
        "disclaimer": "Not an ejection fraction, cardiac output, or diagnosis. Interpret as an SCG timing/energy proxy only with validation.",
    }


def SCG_assess_sensor_placement(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    quality = SCG_assess_quality(signal_path, sampling_rate, column)
    peaks = SCG_detect_j_peaks(signal_path, sampling_rate, column)
    reasons = list(quality.get("reasons", []))
    hr = peaks.get("heart_rate_bpm")
    if hr is None or not 35 <= hr <= 220:
        reasons.append("implausible_or_missing_cardiac_rhythm")
    if peaks.get("interval_cv") is not None and peaks["interval_cv"] > 0.35:
        reasons.append("unstable_peak_intervals")
    cardiac_ratio = quality.get("cardiac_power_ratio")
    if cardiac_ratio is not None and cardiac_ratio < 0.35:
        reasons.append("weak_sternal_cardiac_component")
    if quality.get("motion_power_ratio", 0.0) > 0.45:
        reasons.append("likely_motion_or_loose_sensor")
    if reasons:
        placement = "poor" if len(reasons) >= 2 or quality.get("quality") == "bad" else "questionable"
    else:
        placement = "acceptable"
    confidence = 0.75 if placement == "acceptable" else 0.55 if placement == "questionable" else 0.35
    return {
        "tool": "SCG_assess_sensor_placement",
        "placement_quality": placement,
        "heart_rate_bpm": hr,
        "num_peaks": peaks.get("num_peaks"),
        "reasons": sorted(set(reasons)),
        "confidence": confidence,
        "quality_summary": quality,
        "peak_summary": {k: peaks.get(k) for k in ["method", "interval_cv", "regularity_confidence", "confidence"]},
        "method": "scg_quality_peak_morphology_placement_proxy",
        "disclaimer": "Placement proxy inferred from signal morphology; it does not identify the physical sensor location directly.",
    }



def _ratio(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator is None or not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return 0.0
    return float(numerator / denominator)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _scg_basic_feature_map(values: np.ndarray, sampling_rate: float) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {}
    centered = values - np.nanmedian(values)
    abs_centered = np.abs(centered)
    total_power = _bandpower(values, sampling_rate, 0.08, min(80.0, sampling_rate * 0.45)) + 1e-12
    cardiac_power = _bandpower(values, sampling_rate, 0.8, 35.0)
    low_motion_power = _bandpower(values, sampling_rate, 0.08, 0.7)
    systolic_power = _bandpower(values, sampling_rate, 5.0, 25.0)
    high_power = _bandpower(values, sampling_rate, 35.0, min(80.0, sampling_rate * 0.45))
    filtered = _safe_bandpass(values, sampling_rate, 0.8, 35.0)
    freqs, psd = scipy_signal.welch(filtered, fs=sampling_rate, nperseg=min(len(filtered), int(sampling_rate * 8)))
    psd_sum = float(np.sum(psd)) + 1e-12
    centroid = float(np.sum(freqs * psd) / psd_sum) if len(freqs) else 0.0
    return {
        "mean_abs": float(np.mean(abs_centered)),
        "std": float(np.nanstd(centered)),
        "iqr": float(np.percentile(centered, 75) - np.percentile(centered, 25)),
        "p95_abs": float(np.percentile(abs_centered, 95)),
        "p99_abs": float(np.percentile(abs_centered, 99)),
        "crest_factor": _ratio(float(np.max(abs_centered)), float(np.sqrt(np.mean(centered ** 2))) + 1e-12),
        "cardiac_power_ratio": float(cardiac_power / total_power),
        "low_motion_power_ratio": float(low_motion_power / total_power),
        "systolic_power_ratio": float(systolic_power / total_power),
        "high_power_ratio": float(high_power / total_power),
        "spectral_centroid_hz": centroid,
    }


def _scg_mechanical_feature_map(
    signal_path: str,
    sampling_rate: float,
    column: str | None = None,
    ecg_path: str | None = None,
    ecg_sampling_rate: float | None = None,
    ecg_column: str | None = None,
) -> dict[str, float]:
    data = load_csv_signal(signal_path, sampling_rate, column)
    quality = SCG_assess_quality(signal_path, sampling_rate, column)
    peaks = SCG_detect_j_peaks(signal_path, sampling_rate, column)
    intervals = SCG_compute_cardiac_time_intervals(signal_path, sampling_rate, column, ecg_path, ecg_sampling_rate, ecg_column)
    features = _scg_basic_feature_map(data.values, data.sampling_rate)
    features.update({
        "quality_confidence": _safe_float(quality.get("confidence")),
        "quality_is_good": 1.0 if quality.get("quality") == "good" else 0.0,
        "quality_is_moderate": 1.0 if quality.get("quality") == "moderate" else 0.0,
        "peak_hr_bpm": _safe_float(peaks.get("heart_rate_bpm")),
        "peak_interval_cv": _safe_float(peaks.get("interval_cv")),
        "peak_regularity_confidence": _safe_float(peaks.get("regularity_confidence")),
        "num_peaks_per_min": _safe_float(peaks.get("num_peaks")) / max(len(data.values) / data.sampling_rate / 60.0, 1e-6),
        "r_to_ao_ms": _safe_float(intervals.get("r_to_ao_ms", {}).get("median_ms")),
        "r_to_ao_iqr_ms": _safe_float(intervals.get("r_to_ao_ms", {}).get("iqr_ms")),
        "lvet_ms": _safe_float(intervals.get("lvet_ms", {}).get("median_ms")),
        "lvet_iqr_ms": _safe_float(intervals.get("lvet_ms", {}).get("iqr_ms")),
        "ivct_ms": _safe_float(intervals.get("ivct_ms", {}).get("median_ms")),
        "ivrt_ms": _safe_float(intervals.get("ivrt_ms", {}).get("median_ms")),
        "r_to_ao_over_lvet": _safe_float(intervals.get("r_to_ao_over_lvet")),
        "interval_confidence": _safe_float(intervals.get("confidence")),
    })
    return features


def _scg_feature_vector(feature_map: dict[str, float], feature_names: list[str]) -> np.ndarray:
    return np.asarray([_safe_float(feature_map.get(name)) for name in feature_names], dtype=float)


@lru_cache(maxsize=1)
def _load_scg_mechanical_classifier(model_path: str = str(SCG_MECHANICAL_CLASSIFIER_PATH)):
    path = Path(model_path)
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None

@lru_cache(maxsize=1)
def _load_scg_rhc_hf_model(model_path: str = str(SCG_RHC_HF_MODEL_PATH)):
    path = Path(model_path)
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def _preprocess_scg_rhc_axis(values: np.ndarray, sampling_rate: float, target_fs: float = 100.0) -> np.ndarray:
    x = _safe_bandpass(np.asarray(values, dtype=float), sampling_rate, 0.8, 35.0)
    if sampling_rate != target_fs:
        x = scipy_signal.resample(x, int(round(len(x) * target_fs / sampling_rate)))
    return x.astype(float)


def _scg_rhc_axis_features(values: np.ndarray, sampling_rate: float, prefix: str) -> dict[str, float]:
    centered = np.asarray(values, dtype=float)
    centered = centered[np.isfinite(centered)]
    if len(centered) == 0:
        centered = np.zeros(1, dtype=float)
    centered = centered - np.nanmedian(centered)
    freqs, psd = scipy_signal.welch(centered, fs=sampling_rate, nperseg=min(len(centered), int(max(sampling_rate * 8, 16))))
    total = float(np.trapezoid(psd, freqs) + 1e-12) if len(freqs) else 1e-12

    def band_ratio(lo: float, hi: float) -> float:
        if not len(freqs):
            return 0.0
        mask = (freqs >= lo) & (freqs <= hi)
        if not np.any(mask):
            return 0.0
        return float(np.trapezoid(psd[mask], freqs[mask]) / total)

    abs_centered = np.abs(centered)
    return {
        f"{prefix}_std": float(np.nanstd(centered)),
        f"{prefix}_iqr": float(np.nanpercentile(centered, 75) - np.nanpercentile(centered, 25)),
        f"{prefix}_p95_abs": float(np.nanpercentile(abs_centered, 95)),
        f"{prefix}_cardiac_power_ratio": band_ratio(0.8, 20.0),
        f"{prefix}_resp_power_ratio": band_ratio(0.08, 0.7),
        f"{prefix}_hf_noise_ratio": band_ratio(35.0, min(80.0, sampling_rate * 0.45)),
        f"{prefix}_systolic_band_ratio": band_ratio(5.0, 30.0),
    }


def _load_scg_multiaxis_frame(signal_path: str, lat_column: str | None, hf_column: str | None, dv_column: str | None) -> tuple[np.ndarray, list[str], bool]:
    frame = pd.read_csv(signal_path)
    numeric = list(frame.select_dtypes("number").columns)
    if not numeric:
        raise ValueError("SCG CSV contains no numeric columns")
    if lat_column is None:
        lat_column = numeric[0]
    if hf_column is None:
        hf_column = numeric[1] if len(numeric) > 1 else lat_column
    if dv_column is None:
        dv_column = numeric[2] if len(numeric) > 2 else lat_column
    cols = [lat_column, hf_column, dv_column]
    arr = frame[cols].to_numpy(dtype=float)
    good = np.isfinite(arr).all(axis=1)
    arr = arr[good]
    return arr, cols, len(set(cols)) == 1


def _scg_rhc_window_features(scg_xyz: np.ndarray, sampling_rate: float, start: int, stop: int, ecg_values: np.ndarray | None, feature_names: list[str]) -> dict[str, float]:
    seg = scg_xyz[start:stop]
    mag = np.linalg.norm(seg, axis=1)
    out = {"start_s": float(start / sampling_rate)}
    out.update(_scg_rhc_axis_features(mag, sampling_rate, "scg_mag"))
    out.update(_scg_rhc_axis_features(seg[:, 0], sampling_rate, "scg_lat"))
    out.update(_scg_rhc_axis_features(seg[:, 1], sampling_rate, "scg_hf"))
    out.update(_scg_rhc_axis_features(seg[:, 2], sampling_rate, "scg_dv"))
    if ecg_values is not None and len(ecg_values) >= stop:
        peaks = _detect_ecg_r_peaks(ecg_values[start:stop], sampling_rate)
        rr = np.diff(peaks) / sampling_rate if len(peaks) > 1 else np.asarray([])
        out["ecg_hr_bpm"] = float(60.0 / np.median(rr)) if len(rr) else 0.0
        out["ecg_rr_cv"] = float(np.std(rr) / (np.mean(rr) + 1e-12)) if len(rr) > 1 else 0.0
        out["ecg_peak_count"] = float(len(peaks))
    else:
        out["ecg_hr_bpm"] = 0.0
        out["ecg_rr_cv"] = 0.0
        out["ecg_peak_count"] = 0.0
    return {name: _safe_float(out.get(name)) for name in feature_names}


def SCG_screen_heart_failure_hemodynamics(
    signal_path: str,
    sampling_rate: float,
    lat_column: str | None = None,
    hf_column: str | None = None,
    dv_column: str | None = None,
    ecg_path: str | None = None,
    ecg_column: str | None = None,
    window_seconds: float = 20.0,
    stride_seconds: float = 20.0,
) -> dict:
    model_bundle = _load_scg_rhc_hf_model()
    if model_bundle is None:
        return {"tool": "SCG_screen_heart_failure_hemodynamics", "error": "scg_rhc_hf_model_not_available", "model_path": str(SCG_RHC_HF_MODEL_PATH), "confidence": 0.0}
    scg_xyz, used_columns, single_axis = _load_scg_multiaxis_frame(signal_path, lat_column, hf_column, dv_column)
    target_fs = 100.0
    scg_axes = [_preprocess_scg_rhc_axis(scg_xyz[:, i], sampling_rate, target_fs) for i in range(3)]
    min_len = min(map(len, scg_axes))
    scg_xyz = np.stack([axis[:min_len] for axis in scg_axes], axis=1)
    ecg_values = None
    if ecg_path:
        ecg_frame = pd.read_csv(ecg_path)
        if ecg_column is None:
            ecg_column = "signal" if "signal" in ecg_frame.columns else ecg_frame.select_dtypes("number").columns[0]
        ecg_values = ecg_frame[ecg_column].to_numpy(dtype=float)
        if sampling_rate != target_fs:
            ecg_values = scipy_signal.resample(ecg_values, int(round(len(ecg_values) * target_fs / sampling_rate)))
        n = min(len(ecg_values), len(scg_xyz))
        ecg_values = ecg_values[:n]
        scg_xyz = scg_xyz[:n]
    sampling_rate = target_fs
    win = max(int(round(window_seconds * sampling_rate)), 8)
    stride = max(int(round(stride_seconds * sampling_rate)), 1)
    if len(scg_xyz) < win:
        win = len(scg_xyz)
    starts = list(range(0, max(len(scg_xyz) - win + 1, 1), stride))[:256]
    feature_names = list(model_bundle.get("feature_names", []))
    rows = [_scg_rhc_window_features(scg_xyz, sampling_rate, start, start + win, ecg_values, feature_names) for start in starts if start + win <= len(scg_xyz)]
    if not rows:
        return {"tool": "SCG_screen_heart_failure_hemodynamics", "error": "too_short_signal", "confidence": 0.0}
    x = np.asarray([[row[name] for name in feature_names] for row in rows], dtype=float)
    probabilities = {}
    decision_thresholds = model_bundle.get("decision_thresholds", {})
    for target, model in model_bundle.get("models", {}).items():
        prob = model.predict_proba(x)[:, 1]
        if model_bundle.get("deployment_flip_probability", {}).get(target):
            prob = 1.0 - prob
        threshold = float(decision_thresholds.get(target, 0.5))
        probabilities[target] = {
            "mean_probability": float(np.mean(prob)),
            "max_probability": float(np.max(prob)),
            "decision_threshold": threshold,
            "window_positive_fraction": float(np.mean(prob >= threshold)),
            "window_positive_fraction_at_0p5": float(np.mean(prob >= 0.5)),
        }
    reports = model_bundle.get("reports", {})
    unavailable_targets = {
        name: report.get("reason", "not saved for deployment")
        for name, report in reports.items()
        if name not in probabilities
    }
    primary_target = "elevated_pcwp" if "elevated_pcwp" in probabilities else "elevated_pam" if "elevated_pam" in probabilities else None
    primary = probabilities.get(primary_target, {"mean_probability": 0.0})

    def clinical_threshold(name: str, item: dict) -> float:
        threshold = float(item.get("decision_threshold", 0.5))
        if name in {"elevated_pcwp", "decompensated_physiology"}:
            threshold = max(threshold, 0.45)
        elif name == "elevated_pam":
            threshold = max(threshold, 0.20)
        return threshold

    def target_positive(name: str) -> bool:
        item = probabilities.get(name)
        if not item:
            return False
        threshold = clinical_threshold(name, item)
        if name in {"elevated_pcwp", "decompensated_physiology"}:
            return item["mean_probability"] >= threshold
        return item["mean_probability"] >= threshold or (item.get("window_positive_fraction", 0.0) >= 0.5 and item["mean_probability"] >= 0.8 * threshold)

    pcwp = probabilities.get("elevated_pcwp")
    decomp = probabilities.get("decompensated_physiology")
    pam_pos = target_positive("elevated_pam")
    pcwp_pos = target_positive("elevated_pcwp")
    decomp_pos = target_positive("decompensated_physiology")
    pcwp_support = pcwp or decomp
    pcwp_support_mean = max([x["mean_probability"] for x in [pcwp, decomp] if x], default=0.0)
    pcwp_threshold = clinical_threshold("elevated_pcwp", pcwp_support) if pcwp_support else 0.45
    pcwp_borderline = bool(pcwp_support and pcwp_support_mean >= max(0.43, 0.9 * pcwp_threshold))
    if pcwp_pos or decomp_pos or (pcwp_borderline and pam_pos):
        risk = "elevated_proxy"
    elif pcwp_borderline or pam_pos:
        risk = "borderline_proxy"
    else:
        risk = "low_proxy"
    confidence = 0.45 if single_axis else 0.65
    if len(rows) >= 3:
        confidence += 0.1
    return {
        "tool": "SCG_screen_heart_failure_hemodynamics",
        "hemodynamic_congestion_risk": risk,
        "target_probabilities": probabilities,
        "primary_target": primary_target,
        "unavailable_targets": unavailable_targets,
        "model_path": str(SCG_RHC_HF_MODEL_PATH),
        "model_note": model_bundle.get("note"),
        "used_columns": used_columns,
        "single_axis_fallback": single_axis,
        "num_windows": len(rows),
        "window_seconds": window_seconds,
        "confidence": min(confidence, 0.75),
        "method": "SCG-RHC feature ensemble for elevated PCWP/PAM and decompensated physiology screening",
        "disclaimer": "Research screening proxy only. Public SCG-RHC training data are still small in this workspace; use ECG/clinical context and invasive/echo standards for diagnosis.",
    }

def SCG_screen_mechanical_abnormality(
    signal_path: str,
    sampling_rate: float,
    column: str | None = None,
    ecg_path: str | None = None,
    ecg_sampling_rate: float | None = None,
    ecg_column: str | None = None,
) -> dict:
    quality = SCG_assess_quality(signal_path, sampling_rate, column)
    peaks = SCG_detect_j_peaks(signal_path, sampling_rate, column)
    intervals = SCG_compute_cardiac_time_intervals(signal_path, sampling_rate, column, ecg_path, ecg_sampling_rate, ecg_column)
    flags = []
    if quality.get("quality") == "bad":
        flags.append("unreliable_signal_quality")
    hr = peaks.get("heart_rate_bpm")
    if hr is None or hr < 40 or hr > 180:
        flags.append("implausible_mechanical_rate")
    if peaks.get("interval_cv") is not None and peaks["interval_cv"] > 0.30:
        flags.append("irregular_mechanical_intervals")
    if quality.get("cardiac_power_ratio", 1.0) < 0.30:
        flags.append("weak_cardiac_mechanical_component")
    ratio = intervals.get("r_to_ao_over_lvet")
    if ratio is not None and ratio > 0.55:
        flags.append("prolonged_r_to_ao_over_lvet_proxy")
    lvet = intervals.get("lvet_ms", {}).get("median_ms")
    if lvet is not None and (lvet < 140 or lvet > 450):
        flags.append("out_of_range_lvet_proxy")
    rule_risk = "low"
    if "unreliable_signal_quality" in flags:
        rule_risk = "indeterminate"
    elif len(flags) >= 2:
        rule_risk = "elevated_proxy"
    elif len(flags) == 1:
        rule_risk = "borderline_proxy"

    learned = None
    model_bundle = _load_scg_mechanical_classifier()
    if model_bundle is not None:
        feature_map = _scg_mechanical_feature_map(signal_path, sampling_rate, column, ecg_path, ecg_sampling_rate, ecg_column)
        feature_names = list(model_bundle.get("feature_names", []))
        labels = list(model_bundle.get("labels", []))
        if feature_names and labels:
            x = _scg_feature_vector(feature_map, feature_names)[None, :]
            probabilities = {}
            for label, model in zip(labels, model_bundle.get("models", [])):
                if hasattr(model, "predict_proba"):
                    probabilities[label] = float(model.predict_proba(x)[0, 1])
                else:
                    probabilities[label] = float(model.predict(x)[0])
            max_probability = max(probabilities.values(), default=0.0)
            learned_risk = "elevated_proxy" if max_probability >= 0.65 else "borderline_proxy" if max_probability >= 0.35 else "low"
            learned = {
                "risk": learned_risk,
                "max_probability": max_probability,
                "subtype_probabilities": probabilities,
                "model_path": str(SCG_MECHANICAL_CLASSIFIER_PATH),
                "model_note": model_bundle.get("note"),
            }

    risk = rule_risk
    confidence = min(quality.get("confidence", 0.5), peaks.get("confidence", 0.5), intervals.get("confidence", 0.45))
    if learned is not None:
        confidence = min(0.75, max(confidence, 0.45))
    return {
        "tool": "SCG_screen_mechanical_abnormality",
        "mechanical_abnormality_risk": risk,
        "rule_based_risk": rule_risk,
        "learned_mechanical_classifier": learned,
        "flags": flags,
        "heart_rate_bpm": hr,
        "interval_cv": peaks.get("interval_cv"),
        "lvet_ms": lvet,
        "r_to_ao_over_lvet": ratio,
        "confidence": confidence,
        "method": "rule_proxy_with_vhd_supervised_subtype_probabilities" if learned is not None else "rule_based_scg_mechanical_abnormality_screen",
        "disclaimer": "Screening proxy only. The learned subtype probabilities are trained on the VHD dataset for moderate-or-greater valve labels and are not a diagnosis; they do not override the conservative rule-based risk.",
    }


def _respiration_band_rate(values: np.ndarray, sampling_rate: float, low_hz: float = 0.08, high_hz: float = 0.7) -> dict:
    high = min(high_hz, sampling_rate * 0.45)
    if len(values) < sampling_rate * 20 or high <= low_hz:
        return {"respiratory_rate_bpm": None, "respiration_power_ratio": 0.0}
    centered = values - np.nanmedian(values)
    sos = scipy_signal.butter(3, [low_hz / (0.5 * sampling_rate), high / (0.5 * sampling_rate)], btype="bandpass", output="sos")
    filtered = scipy_signal.sosfiltfilt(sos, centered)
    freqs, psd = scipy_signal.welch(filtered, fs=sampling_rate, nperseg=min(len(filtered), int(sampling_rate * 32)))
    mask = (freqs >= low_hz) & (freqs <= high)
    if not len(freqs) or not np.any(mask):
        return {"respiratory_rate_bpm": None, "respiration_power_ratio": 0.0}
    respiratory_rate = float(freqs[mask][np.argmax(psd[mask])] * 60.0)
    respiration_power_ratio = float(np.trapezoid(psd[mask], freqs[mask]) / (np.trapezoid(psd, freqs) + 1e-12))
    return {"respiratory_rate_bpm": respiratory_rate, "respiration_power_ratio": respiration_power_ratio}


def _scg_envelope_respiration_candidate(values: np.ndarray, sampling_rate: float, cardiac_low_hz: float, cardiac_high_hz: float) -> dict:
    high = min(cardiac_high_hz, sampling_rate * 0.45)
    if high <= cardiac_low_hz:
        return {"respiratory_rate_bpm": None, "respiration_power_ratio": 0.0}
    centered = values - np.nanmedian(values)
    sos = scipy_signal.butter(3, [cardiac_low_hz / (0.5 * sampling_rate), high / (0.5 * sampling_rate)], btype="bandpass", output="sos")
    cardiac = scipy_signal.sosfiltfilt(sos, centered)
    envelope = np.abs(scipy_signal.hilbert(cardiac))
    return _respiration_band_rate(envelope, sampling_rate)


def SCG_estimate_respiration(signal_path: str, sampling_rate: float, column: str | None = None) -> dict:
    data = load_csv_signal(signal_path, sampling_rate, column)
    values = data.values
    if len(values) < data.sampling_rate * 20:
        return {"tool": "SCG_estimate_respiration", "error": "signal too short", "confidence": 0.0}
    high = min(0.7, data.sampling_rate * 0.45)
    if high <= 0.08:
        return {"tool": "SCG_estimate_respiration", "error": "sampling rate too low", "confidence": 0.1}

    candidates = []
    direct = _respiration_band_rate(values, data.sampling_rate)
    candidates.append({"method": "direct_low_frequency", **direct})
    for low_hz, high_hz, name in [(0.8, 20.0, "cardiac_envelope_0p8_20hz"), (5.0, 35.0, "cardiac_envelope_5_35hz"), (10.0, 35.0, "cardiac_envelope_10_35hz")]:
        cand = _scg_envelope_respiration_candidate(values, data.sampling_rate, low_hz, high_hz)
        candidates.append({"method": name, **cand})

    valid = [c for c in candidates if c["respiratory_rate_bpm"] is not None and 4 <= c["respiratory_rate_bpm"] <= 45]
    best = max(valid, key=lambda c: c["respiration_power_ratio"], default=None)
    respiratory_rate = best["respiratory_rate_bpm"] if best else None
    respiration_power_ratio = best["respiration_power_ratio"] if best else 0.0
    if respiratory_rate is None:
        confidence = 0.25
    elif 5 <= respiratory_rate <= 40:
        confidence = float(min(0.75, max(0.45, 0.35 + respiration_power_ratio)))
    else:
        confidence = 0.35
    return {
        "tool": "SCG_estimate_respiration",
        "respiratory_rate_bpm": respiratory_rate,
        "respiration_power_ratio": respiration_power_ratio,
        "confidence": confidence,
        "method": best["method"] if best else "mechanical_signal_respiration_bandpower_proxy",
        "candidates": candidates,
        "disclaimer": "Mechanical-signal respiration proxy only; validate against respiratory reference signals.",
    }
