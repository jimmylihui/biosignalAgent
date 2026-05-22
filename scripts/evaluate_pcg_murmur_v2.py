from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.pcg_tools import PCG_extract_murmur_features, PCG_screen_murmur_proxy

FEATURE_NAMES = [
    "low_band_power",
    "mid_band_power",
    "high_band_power",
    "very_high_band_power",
    "mid_band_ratio",
    "high_band_ratio",
    "spectral_centroid_hz",
    "spectral_entropy",
    "zero_crossing_rate",
    "envelope_std",
    "envelope_p90_median_ratio",
    "envelope_p95_median_ratio",
    "envelope_p99_median_ratio",
    "continuous_fraction_60",
    "continuous_fraction_75",
    "num_sounds",
    "heart_rate_bpm",
    "sound_interval_cv",
]


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    return {
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def choose_cv(y: np.ndarray):
    counts = Counter(y.tolist())
    min_class = min(counts.values()) if counts else 0
    if min_class >= 3:
        n_splits = min(5, min_class)
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=13), f"stratified_{n_splits}_fold"
    return LeaveOneOut(), "leave_one_out"


def evaluate(manifest_path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text())
    rows = []
    x_rows = []
    y = []
    for record in manifest.get("records", []):
        if record.get("label") not in {"normal", "abnormal"}:
            continue
        features = PCG_extract_murmur_features(record["path"], float(record["sampling_rate"]), column=None)
        proxy = PCG_screen_murmur_proxy(record["path"], float(record["sampling_rate"]), column=None)
        vector = [features.get(name) for name in FEATURE_NAMES]
        vector = [np.nan if value is None else value for value in vector]
        x_rows.append(vector)
        truth = 1 if record["label"] == "abnormal" else 0
        y.append(truth)
        rows.append({
            "record": record["record"],
            "truth": record["label"],
            "proxy_prediction": "abnormal" if proxy.get("murmur_risk") == "possible_murmur_proxy" else "normal",
            "murmur_proxy_score": proxy.get("murmur_proxy_score"),
            **{name: features.get(name) for name in FEATURE_NAMES},
            "feature_error": features.get("error"),
        })
    x = np.asarray(x_rows, dtype=float)
    y_arr = np.asarray(y, dtype=int)
    if len(set(y)) < 2 or len(y) < 4:
        raise ValueError("Need at least two classes and four records for PCG murmur v2 evaluation.")
    cv, cv_name = choose_cv(y_arr)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", solver="liblinear", random_state=13)),
    ])
    proba = cross_val_predict(model, x, y_arr, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    for row, prob, pred_value in zip(rows, proba.tolist(), pred.tolist()):
        row["prediction_probability_abnormal"] = float(prob)
        row["prediction"] = "abnormal" if pred_value else "normal"
    truth_labels = ["abnormal" if item else "normal" for item in y]
    pred_labels = [row["prediction"] for row in rows]
    return {
        "manifest": str(manifest_path),
        "num_records": len(rows),
        "feature_names": FEATURE_NAMES,
        "model": "median_imputer_standard_scaler_balanced_logistic_regression",
        "cv": cv_name,
        "truth_counts": dict(Counter(truth_labels)),
        "prediction_counts": dict(Counter(pred_labels)),
        "metrics": binary_metrics(y, pred.tolist()),
        "rows": rows,
    }


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a feature-based PCG murmur/normal-abnormal baseline.")
    parser.add_argument("--manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/pcg_murmur_manifest.json")
    parser.add_argument("--out-json", default="/data1/jiahui/biosignal-agent/outputs/pcg_murmur_v2_eval.json")
    parser.add_argument("--out-csv", default="/data1/jiahui/biosignal-agent/outputs/pcg_murmur_v2_eval.csv")
    args = parser.parse_args()
    report = evaluate(args.manifest)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    write_csv(report["rows"], args.out_csv)
    print(json.dumps({key: report[key] for key in ["num_records", "cv", "truth_counts", "prediction_counts", "metrics"]}, indent=2))


if __name__ == "__main__":
    main()
