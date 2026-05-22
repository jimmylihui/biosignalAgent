from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.acc_tools import ACC_detect_activity_bouts, ACC_summarize_activity
from scripts.benchmark_metrics import binary_metrics, multiclass_metrics


def load_features(path: str) -> list[float]:
    return json.loads(Path(path).read_text())["features"]


def choose_cv(y: list[str]):
    counts = Counter(y)
    n_splits = min(5, min(counts.values())) if counts else 0
    if n_splits < 2:
        raise ValueError("Need at least two examples per activity class for cross-validation.")
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=13), f"stratified_{n_splits}_fold"


def evaluate(manifest_path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text())
    records = manifest.get("records", [])
    x = np.asarray([load_features(row["feature_path"]) for row in records], dtype=float)
    y = [row["activity_label"] for row in records]
    coarse_y = [row["coarse_activity_label"] for row in records]
    cv, cv_name = choose_cv(y)
    model = RandomForestClassifier(n_estimators=100, random_state=13, class_weight="balanced")
    pred = cross_val_predict(model, x, y, cv=cv)
    coarse_pred = ["active" if label in {"walking", "walking_upstairs", "walking_downstairs"} else "rest" for label in pred]
    rows = []
    for record, y_true, y_pred, coarse_true, coarse_value in zip(records, y, pred.tolist(), coarse_y, coarse_pred):
        summary = ACC_summarize_activity(record["path"], float(record["sampling_rate"]), column=None)
        bouts = ACC_detect_activity_bouts(record["path"], float(record["sampling_rate"]), column=None)
        rows.append({
            "record": record["record"],
            "truth": y_true,
            "prediction": y_pred,
            "truth_coarse": coarse_true,
            "prediction_coarse": coarse_value,
            "activity_std": summary.get("activity_std"),
            "activity_level": summary.get("activity_level"),
            "activity_bout_count": bouts.get("activity_bout_count"),
            "tool_error": summary.get("error") or bouts.get("error"),
        })
    coarse_rows = [{"truth": row["truth_coarse"], "prediction": row["prediction_coarse"]} for row in rows]
    return {
        "manifest": str(manifest_path),
        "num_windows": len(rows),
        "model": "random_forest_on_uci_har_561_features",
        "cv": cv_name,
        "truth_counts": dict(Counter(y)),
        "prediction_counts": dict(Counter(pred.tolist())),
        "coarse_truth_counts": dict(Counter(coarse_y)),
        "coarse_prediction_counts": dict(Counter(coarse_pred)),
        "metrics": {
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        },
        "multiclass_metrics": multiclass_metrics(rows),
        "coarse_binary_metrics": binary_metrics(coarse_rows, positive="active"),
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
    parser = argparse.ArgumentParser(description="Evaluate UCI-HAR activity baselines.")
    parser.add_argument("--manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/acc_activity_manifest.json")
    parser.add_argument("--out-json", default="/data1/jiahui/biosignal-agent/outputs/acc_activity_eval.json")
    parser.add_argument("--out-csv", default="/data1/jiahui/biosignal-agent/outputs/acc_activity_eval.csv")
    args = parser.parse_args()
    report = evaluate(args.manifest)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    write_csv(report["rows"], args.out_csv)
    print(json.dumps({key: report[key] for key in ["num_windows", "cv", "truth_counts", "prediction_counts", "metrics", "coarse_binary_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
