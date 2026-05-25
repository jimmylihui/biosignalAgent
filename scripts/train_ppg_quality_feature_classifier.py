from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import neurokit2 as nk
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.ppg_tools import _ppg_artifact_metrics, _ppg_quality_feature_vector  # noqa: E402
from biosignal_agent.tools.peak_detectors import ppg_multiscale_systolic_peaks  # noqa: E402
from scripts.evaluate_ppg_peak_detectors import estimate_best_lag_match  # noqa: E402

FEATURE_NAMES = [
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


def raw_record_paths(raw_dir: Path) -> list[Path]:
    return sorted(raw_dir.rglob("*_data.csv"))


def ecg_rpeaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    _, info = nk.ecg_peaks(ecg, sampling_rate=fs, method="nabian2018", correct_artifacts=True)
    return np.asarray(info.get("ECG_R_Peaks", []), dtype=int)


def natural_label(ppg: np.ndarray, ecg: np.ndarray, fs: float) -> tuple[str | None, dict[str, Any]]:
    try:
        rpeaks = ecg_rpeaks(ecg, fs)
        ppeaks, _ = ppg_multiscale_systolic_peaks(ppg, fs)
        match = estimate_best_lag_match(rpeaks, ppeaks, fs)
    except Exception as exc:
        return None, {"label_error": str(exc)}
    ratio = len(ppeaks) / max(1, len(rpeaks))
    artifact = _ppg_artifact_metrics(ppg, fs)
    label = None
    if (
        match["f1"] >= 0.85
        and match["sensitivity"] >= 0.80
        and match["ppv"] >= 0.80
        and 0.75 <= ratio <= 1.35
        and artifact["artifact_score"] < 0.45
    ):
        label = "good"
    elif match["f1"] < 0.65 or match["sensitivity"] < 0.65 or match["ppv"] < 0.65 or ratio < 0.55 or ratio > 1.65 or artifact["artifact_score"] >= 0.55:
        label = "poor"
    return label, {
        "ecg_peaks": int(len(rpeaks)),
        "ppg_peaks": int(len(ppeaks)),
        "peak_count_ratio": float(ratio),
        "lag_corrected_match": match,
        **artifact,
    }


def augment_poor(ppg: np.ndarray, fs: float, rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
    finite = ppg[np.isfinite(ppg)]
    if len(finite) == 0:
        return []
    scale = float(np.nanstd(finite)) + 1e-8
    t = np.arange(len(ppg)) / fs
    augmented = []
    noisy = ppg + rng.normal(0, 0.45 * scale, size=len(ppg))
    augmented.append(("hf_noise", noisy))
    drift = ppg + np.sin(2 * np.pi * rng.uniform(0.04, 0.12) * t) * rng.uniform(0.8, 1.5) * scale
    augmented.append(("baseline_drift", drift))
    clipped = np.clip(ppg, np.nanpercentile(ppg, 15), np.nanpercentile(ppg, 85))
    augmented.append(("clipping", clipped))
    dropout = ppg.copy()
    for _ in range(2):
        width = int(rng.uniform(1.0, 4.0) * fs)
        if width >= len(dropout):
            continue
        start = int(rng.integers(0, len(dropout) - width))
        dropout[start:start + width] = np.nanmedian(dropout)
    augmented.append(("dropout", dropout))
    return augmented


def add_sample(rows: list[dict[str, Any]], signal: np.ndarray, fs: float, label: str, record: str, group: str, source: str, details: dict[str, Any]) -> None:
    rows.append({
        "x": _ppg_quality_feature_vector(signal, fs),
        "y": label,
        "record": record,
        "group": group,
        "source": source,
        "details": details,
    })


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    rows = []
    for path in raw_record_paths(Path(args.raw_dir)):
        frame = pd.read_csv(path)
        if not {"PPG", "ECG"}.issubset(frame.columns):
            continue
        fs = 1.0 / float(np.median(np.diff(frame["Time"].to_numpy(float)))) if "Time" in frame.columns else 125.0
        ppg = frame["PPG"].to_numpy(float)
        ecg = frame["ECG"].to_numpy(float)
        record = path.stem.replace("_data", "")
        win = int(args.window_s * fs)
        step = int(args.step_s * fs)
        for start in range(0, max(0, len(ppg) - win + 1), step):
            ppg_win = ppg[start:start + win]
            ecg_win = ecg[start:start + win]
            label, details = natural_label(ppg_win, ecg_win, fs)
            if label is None:
                continue
            window_record = f"{record}_{start / fs:.1f}s"
            add_sample(rows, ppg_win, fs, label, window_record, record, "natural", details)
            if args.augment and label == "good":
                for aug_name, aug_signal in augment_poor(ppg_win, fs, rng):
                    aug_details = {"augmentation": aug_name, "parent_label": label}
                    aug_details.update(_ppg_artifact_metrics(aug_signal, fs))
                    add_sample(rows, aug_signal, fs, "poor", f"{window_record}_{aug_name}", record, f"augmented_{aug_name}", aug_details)
    return rows


def metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    labels = ["poor", "good"]
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "labels": labels,
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def models(seed: int) -> dict[str, Any]:
    return {
        "logistic_l2_balanced": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.7, random_state=seed)),
        "random_forest_balanced": RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=8, class_weight="balanced", random_state=seed),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    rows = build_rows(args)
    if len(rows) < 30 or len(set(row["y"] for row in rows)) < 2:
        raise RuntimeError("Not enough labeled PPG quality windows.")
    x = np.asarray([row["x"] for row in rows], dtype=float)
    y = np.asarray([row["y"] for row in rows])
    groups = np.asarray([row["group"] for row in rows])
    n_splits = min(args.folds, len(set(groups)))
    cv = GroupKFold(n_splits=n_splits)
    reports = {}
    best_name, best_score, best_model = None, -1.0, None
    for name, model in models(args.seed).items():
        pred = cross_val_predict(model, x, y, groups=groups, cv=cv)
        report = metrics(list(y), list(pred))
        reports[name] = report
        score = report["per_class"]["good"]["f1"] + report["per_class"]["poor"]["f1"] + 0.1 * report["macro_f1"]
        if score > best_score:
            best_name, best_score, best_model = name, score, model
    best_model.fit(x, y)
    payload = {
        "model": best_model,
        "model_name": best_name,
        "feature_names": FEATURE_NAMES,
        "cv_metrics": reports[best_name],
        "all_cv_metrics": reports,
        "training_windows": int(len(y)),
        "label_counts": dict(Counter(y)),
        "source_counts": dict(Counter(row["source"] for row in rows)),
        "reference": "Pseudo labels from ECG-derived lag-corrected PPG peak match plus artifact thresholds; augmented poor windows simulate common wearable/ICU artifacts.",
        "window_s": args.window_s,
        "step_s": args.step_s,
    }
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.model_out)
    report = {key: value for key, value in payload.items() if key != "model"}
    report["sample_preview"] = [{k: row[k] for k in ["record", "y", "source", "details"]} for row in rows[:80]]
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a feature-based PPG signal-quality classifier from ECG-derived pseudo labels and artifact augmentation.")
    parser.add_argument("--raw-dir", default="/data1/jiahui/biosignal-agent/datasets/raw/mimic_perform_af")
    parser.add_argument("--model-out", default="/data1/jiahui/biosignal-agent/outputs/ppg_signal_quality_feature_classifier.joblib")
    parser.add_argument("--report-out", default="/data1/jiahui/biosignal-agent/outputs/ppg_signal_quality_feature_classifier_report.json")
    parser.add_argument("--window-s", type=float, default=30.0)
    parser.add_argument("--step-s", type=float, default=30.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--augment", action="store_true", default=True)
    args = parser.parse_args()
    report = train(args)
    print(json.dumps({
        "model_name": report["model_name"],
        "training_windows": report["training_windows"],
        "label_counts": report["label_counts"],
        "source_counts": report["source_counts"],
        "cv_metrics": report["cv_metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()
