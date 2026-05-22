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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.acc_tools import ACC_detect_fall_proxy, ACC_summarize_activity
from scripts.benchmark_metrics import binary_metrics

FEATURES = ["mean", "std", "max", "min", "range", "energy"]


def load_feature_vector(path: str) -> list[float]:
    data = json.loads(Path(path).read_text())["features"]
    return [float(data.get(name, np.nan)) for name in FEATURES]


def evaluate(manifest_path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text())
    records = manifest.get("records", [])
    x = np.asarray([load_feature_vector(row["feature_path"]) for row in records], dtype=float)
    y = [row["label"] for row in records]
    min_class = min(Counter(y).values()) if y else 0
    if min_class < 2:
        raise ValueError("Need at least two fall and ADL windows for cross-validation.")
    cv = StratifiedKFold(n_splits=min(5, min_class), shuffle=True, random_state=13)
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("classifier", LogisticRegression(class_weight="balanced", random_state=13))])
    pred = cross_val_predict(model, x, y, cv=cv)
    rows = []
    for record, truth, prediction in zip(records, y, pred.tolist()):
        fall = ACC_detect_fall_proxy(record["path"], float(record["sampling_rate"]), column=None)
        summary = ACC_summarize_activity(record["path"], float(record["sampling_rate"]), column=None)
        proxy_prediction = "fall" if fall.get("fall_risk") == "possible_fall_or_impact_proxy" else "adl"
        rows.append({
            "record": record["record"],
            "truth": truth,
            "prediction": prediction,
            "proxy_prediction": proxy_prediction,
            "fall_risk": fall.get("fall_risk"),
            "impact_event_count": fall.get("impact_event_count"),
            "activity_std": summary.get("activity_std"),
            "activity_level": summary.get("activity_level"),
            "tool_error": fall.get("error") or summary.get("error"),
        })
    return {
        "manifest": str(manifest_path),
        "num_windows": len(rows),
        "model": "logistic_regression_on_acc_magnitude_features",
        "truth_counts": dict(Counter(y)),
        "prediction_counts": dict(Counter(pred.tolist())),
        "metrics": binary_metrics(rows, positive="fall"),
        "proxy_metrics": binary_metrics([{ "truth": row["truth"], "prediction": row["proxy_prediction"] } for row in rows], positive="fall"),
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
    parser = argparse.ArgumentParser(description="Evaluate ACC fall-vs-ADL baselines.")
    parser.add_argument("--manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/acc_fall_manifest.json")
    parser.add_argument("--out-json", default="/data1/jiahui/biosignal-agent/outputs/acc_fall_eval.json")
    parser.add_argument("--out-csv", default="/data1/jiahui/biosignal-agent/outputs/acc_fall_eval.csv")
    args = parser.parse_args()
    report = evaluate(args.manifest)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    write_csv(report["rows"], args.out_csv)
    print(json.dumps({key: report[key] for key in ["num_windows", "truth_counts", "prediction_counts", "metrics", "proxy_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
