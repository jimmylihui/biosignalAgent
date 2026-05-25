from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.ppg_tools import (  # noqa: E402
    PPG_detect_peaks,
    _ppg_artifact_metrics,
    _ppg_irregularity_feature_vector,
    _pulse_interval_features,
)
from biosignal_agent.tools.common import load_csv_signal  # noqa: E402


def row_features(record: dict[str, Any]) -> tuple[list[float], dict[str, Any]]:
    data = load_csv_signal(record["path"], float(record["sampling_rate"]), None)
    peak_result = PPG_detect_peaks(record["path"], float(record["sampling_rate"]), None)
    peaks = np.asarray(peak_result.get("peak_indices", []), dtype=int)
    interval_features = _pulse_interval_features(peaks, data.sampling_rate)
    artifact_metrics = _ppg_artifact_metrics(data.values, data.sampling_rate)
    vector = _ppg_irregularity_feature_vector(interval_features, artifact_metrics, peak_result.get("heart_rate_bpm"))
    detail = {
        "record": record["record"],
        "label": record["label"],
        "heart_rate_bpm": peak_result.get("heart_rate_bpm"),
        **interval_features,
        **artifact_metrics,
    }
    return vector, detail


def metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    labels = ["non_af", "af"]
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
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
        "confusion_matrix": cm.tolist(),
    }


def build_models(seed: int) -> dict[str, Any]:
    return {
        "logistic_l2_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=0.8, random_state=seed),
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=300,
            max_depth=3,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
        ),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(Path(args.manifest).read_text())
    records = manifest.get("records", [])
    xs, ys, details = [], [], []
    for record in records:
        try:
            vector, detail = row_features(record)
        except Exception as exc:
            print(f"skip {record.get('record')}: {exc}", file=sys.stderr)
            continue
        xs.append(vector)
        ys.append(record["label"])
        details.append(detail)
    if len(set(ys)) < 2 or len(xs) < 6:
        raise RuntimeError("Need at least two classes and six usable records for PPG classifier training.")
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys)
    cv = LeaveOneOut() if args.cv == "loo" else StratifiedKFold(n_splits=min(args.folds, min(Counter(ys).values())), shuffle=True, random_state=args.seed)
    reports = {}
    best_name, best_score = None, -1.0
    best_model = None
    for name, model in build_models(args.seed).items():
        pred = cross_val_predict(model, x, y, cv=cv)
        report = metrics(list(y), list(pred))
        reports[name] = report
        score = report["per_class"]["af"]["f1"] + 0.25 * report["macro_f1"]
        if score > best_score:
            best_name, best_score, best_model = name, score, model
    best_model.fit(x, y)
    payload = {
        "model": best_model,
        "model_name": best_name,
        "feature_names": [
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
            "heart_rate_bpm",
            "num_valid_intervals",
            "artifact_score",
            "baseline_wander_ratio",
            "high_frequency_noise_ratio",
        ],
        "cv_metrics": reports[best_name],
        "all_cv_metrics": reports,
        "training_records": len(y),
        "label_counts": dict(Counter(ys)),
        "manifest": str(args.manifest),
    }
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.model_out)
    report = {key: value for key, value in payload.items() if key != "model"}
    report["record_features"] = details
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an artifact-aware PPG AF/non-AF interval-feature classifier.")
    parser.add_argument("--manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/ppg_af_manifest.json")
    parser.add_argument("--model-out", default="/data1/jiahui/biosignal-agent/outputs/ppg_pulse_irregularity_feature_classifier.joblib")
    parser.add_argument("--report-out", default="/data1/jiahui/biosignal-agent/outputs/ppg_pulse_irregularity_feature_classifier_report.json")
    parser.add_argument("--cv", choices=["loo", "stratified"], default="loo")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    report = train(args)
    print(json.dumps({
        "model_name": report["model_name"],
        "training_records": report["training_records"],
        "label_counts": report["label_counts"],
        "cv_metrics": report["cv_metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()
