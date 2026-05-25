from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.acc_tools import ACC_detect_fall_proxy, ACC_summarize_activity, _acc_activity_features, _read_acc_frame
from scripts.benchmark_metrics import binary_metrics


def load_feature_vector(path: str, sampling_rate: float) -> list[float]:
    frame = _read_acc_frame(path)
    feats, _ = _acc_activity_features(frame, sampling_rate)
    return feats


def make_model() -> Pipeline:
    forest = RandomForestClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=2, class_weight="balanced_subsample", random_state=13, n_jobs=-1)
    extra = ExtraTreesClassifier(n_estimators=700, max_features="sqrt", min_samples_leaf=1, class_weight="balanced", random_state=17, n_jobs=-1)
    voter = VotingClassifier(estimators=[("rf", forest), ("et", extra)], voting="soft", weights=[1, 2])
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("classifier", voter)])


def choose_cv(y: list[str], groups: list[str | None]):
    usable_groups = [g for g in groups if g]
    if len(set(usable_groups)) >= 3 and len(usable_groups) == len(groups):
        counts = Counter(groups)
        n_splits = min(5, len(counts))
        return GroupKFold(n_splits=n_splits), groups, f"subject_independent_group_kfold_{n_splits}"
    min_class = min(Counter(y).values()) if y else 0
    n_splits = min(5, min_class)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=13), None, f"stratified_kfold_{n_splits}"


def evaluate(manifest_path: str | Path, model_out: str | Path | None = None) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text())
    records = manifest.get("records", [])
    x = np.asarray([load_feature_vector(row["path"], float(row["sampling_rate"])) for row in records], dtype=float)
    y = [row["label"] for row in records]
    groups = [row.get("subject_id") for row in records]
    min_class = min(Counter(y).values()) if y else 0
    if min_class < 2:
        raise ValueError("Need at least two fall and ADL windows for cross-validation.")
    model = make_model()
    cv, cv_groups, cv_name = choose_cv(y, groups)
    pred = cross_val_predict(model, x, y, cv=cv, groups=cv_groups, n_jobs=None)
    prob = None
    try:
        proba = cross_val_predict(model, x, y, cv=cv, groups=cv_groups, method="predict_proba", n_jobs=None)
        classes = sorted(set(y))
        fall_idx = classes.index("fall") if "fall" in classes else 1
        prob = proba[:, fall_idx]
    except Exception:
        prob = None

    rows = []
    for i, (record, truth, prediction) in enumerate(zip(records, y, pred.tolist())):
        fall = ACC_detect_fall_proxy(record["path"], float(record["sampling_rate"]), column=None)
        summary = ACC_summarize_activity(record["path"], float(record["sampling_rate"]), column=None)
        proxy_prediction = "fall" if fall.get("fall_risk") in {"fall_pattern_detected", "possible_fall_or_impact_proxy"} else "adl"
        rows.append({
            "record": record["record"],
            "subject_id": record.get("subject_id"),
            "activity_name": record.get("activity_name"),
            "truth": truth,
            "prediction": prediction,
            "fall_probability": None if prob is None else float(prob[i]),
            "proxy_prediction": proxy_prediction,
            "fall_risk": fall.get("fall_risk"),
            "impact_event_count": fall.get("impact_event_count"),
            "activity_std": summary.get("activity_std"),
            "activity_level": summary.get("activity_level"),
            "tool_error": fall.get("error") or summary.get("error"),
        })

    metrics = binary_metrics(rows, positive="fall")
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y, pred))
    metrics["macro_f1"] = float(f1_score(y, pred, average="macro"))
    if prob is not None:
        metrics["auroc"] = float(roc_auc_score([1 if yy == "fall" else 0 for yy in y], prob))

    final_model = make_model()
    final_model.fit(x, y)
    if model_out:
        model_out = Path(model_out)
        model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": final_model, "features": "acc_activity_features_v1", "sampling_rate_hz": 50.0, "metrics": metrics, "cv": cv_name, "source_manifest": str(manifest_path)}, model_out)

    return {
        "manifest": str(manifest_path),
        "num_windows": len(rows),
        "model": "rf_extratrees_soft_voting_on_triaxial_acc_features",
        "cv": cv_name,
        "truth_counts": dict(Counter(y)),
        "prediction_counts": dict(Counter(pred.tolist())),
        "metrics": metrics,
        "proxy_metrics": binary_metrics([{ "truth": row["truth"], "prediction": row["proxy_prediction"] } for row in rows], positive="fall"),
        "model_out": str(model_out) if model_out else None,
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
    parser.add_argument("--out-json", default="/data1/jiahui/biosignal-agent/outputs/acc_fall/acc_fall_eval.json")
    parser.add_argument("--out-csv", default="/data1/jiahui/biosignal-agent/outputs/acc_fall/acc_fall_eval.csv")
    parser.add_argument("--model-out", default="/data1/jiahui/biosignal-agent/outputs/acc_fall/acc_unimib_fall_ensemble.joblib")
    args = parser.parse_args()
    report = evaluate(args.manifest, args.model_out)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    write_csv(report["rows"], args.out_csv)
    print(json.dumps({key: report[key] for key in ["num_windows", "cv", "truth_counts", "prediction_counts", "metrics", "proxy_metrics", "model_out"]}, indent=2))


if __name__ == "__main__":
    main()
