#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_resp_spo2_ucddb_event_model import load, resp_features, spo2_features

LABELS = ["normal", "respiratory_event"]


def metric_block(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None) -> dict[str, Any]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "labels": LABELS,
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        pos = LABELS.index("respiratory_event")
        binary = (np.asarray(y_true) == "respiratory_event").astype(int)
        out["auroc"] = float(roc_auc_score(binary, y_prob[:, pos]))
        out["average_precision"] = float(average_precision_score(binary, y_prob[:, pos]))
    return out


def model_pipeline(seed: int = 211) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ensemble", VotingClassifier([
            ("extra", ExtraTreesClassifier(n_estimators=450, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1)),
            ("rf", RandomForestClassifier(n_estimators=350, min_samples_leaf=2, class_weight="balanced_subsample", random_state=seed + 1, n_jobs=-1)),
            ("hgb", HistGradientBoostingClassifier(max_iter=180, learning_rate=0.05, l2_regularization=0.05, random_state=seed + 2)),
        ], voting="soft")),
    ])


def load_matrix(manifest_path: Path, limit: int | None = None) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    manifest = json.loads(manifest_path.read_text())
    records = manifest.get("records", [])[:limit]
    x_resp, x_spo2, y, groups = [], [], [], []
    for idx, rec in enumerate(records):
        if idx and idx % 250 == 0:
            print(f"processed {idx}/{len(records)} windows", flush=True)
        rf = resp_features(load(rec["resp_path"]), float(rec["resp_sampling_rate"]))
        sf = spo2_features(load(rec["spo2_path"]), float(rec["spo2_sampling_rate"]))
        x_resp.append(rf)
        x_spo2.append(sf)
        y.append(rec["respiratory_event_label"])
        groups.append(rec["record"])
    xr = np.asarray(x_resp, dtype=np.float32)
    xs = np.asarray(x_spo2, dtype=np.float32)
    return {"resp_only": xr, "spo2_only": xs, "resp_plus_spo2": np.concatenate([xr, xs], axis=1)}, np.asarray(y), np.asarray(groups)


def evaluate_model(name: str, x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    all_true, all_pred, all_prob, folds = [], [], [], []
    for fold, (tr, te) in enumerate(gkf.split(x, y, groups), 1):
        model = clone(model_pipeline(seed=211 + fold * 10))
        model.fit(x[tr], y[tr])
        pred = model.predict(x[te])
        prob = model.predict_proba(x[te])
        all_true.extend(y[te].tolist())
        all_pred.extend(pred.tolist())
        all_prob.extend(prob.tolist())
        folds.append({"fold": fold, "test_records": sorted(set(groups[te].tolist())), **metric_block(y[te], pred, prob)})
    return {
        "name": name,
        "overall": metric_block(np.asarray(all_true), np.asarray(all_pred), np.asarray(all_prob, dtype=float)),
        "folds": folds,
    }


def markdown_table(report: dict[str, Any]) -> str:
    rows = []
    for name in ["resp_only", "spo2_only", "resp_plus_spo2"]:
        m = report["models"][name]["overall"]
        rows.append([name, m.get("balanced_accuracy"), m.get("macro_f1"), m.get("auroc"), m.get("average_precision"), m.get("accuracy")])
    def fmt(v: Any) -> str:
        return "" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
    lines = [
        "# Multimodal vs Unimodal Public Benchmark Test",
        "",
        f"Dataset: {report['dataset']}",
        f"Task: {report['task']}",
        f"Validation: {report['validation']}",
        f"Windows: {report['num_windows']}; records: {report['num_records']}; labels: `{report['label_counts']}`",
        "",
        "| model | balanced accuracy | macro F1 | AUROC | average precision | accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| " + " | ".join([str(row[0]), *[fmt(v) for v in row[1:]]]) + " |")
    best = max(rows, key=lambda r: (r[2] or 0.0, r[3] or 0.0))
    lines.extend(["", f"Best by macro F1: `{best[0]}`.", "", report["interpretation"]])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare unimodal and multimodal models on public biosignal benchmarks.")
    ap.add_argument("--manifest", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/psg_sleep_manifest.json"))
    ap.add_argument("--out-json", type=Path, default=Path("/data1/jiahui/biosignal-agent/outputs/multimodal_public_benchmark_comparison.json"))
    ap.add_argument("--out-md", type=Path, default=Path("/data1/jiahui/biosignal-agent/outputs/paper_tables/multimodal_public_benchmark_comparison.md"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    x_by_model, y, groups = load_matrix(args.manifest, args.limit)
    reports = {name: evaluate_model(name, x, y, groups) for name, x in x_by_model.items()}
    fusion = reports["resp_plus_spo2"]["overall"]
    best_uni = max([reports["resp_only"]["overall"], reports["spo2_only"]["overall"]], key=lambda m: m.get("macro_f1", 0.0))
    delta = float(fusion.get("macro_f1", 0.0) - best_uni.get("macro_f1", 0.0))
    interpretation = (
        f"On this run, RESP+SpO2 fusion changed macro F1 by {delta:+.4f} relative to the best unimodal baseline. "
        "Positive delta supports multimodal benefit; negative delta means this simple fusion did not beat the strongest single modality under this split."
    )
    report = {
        "dataset": "UCDDB PSG windows from processed psg_sleep_manifest",
        "task": "respiratory_event vs normal",
        "validation": "record-level GroupKFold; records are held out, windows from the same record do not cross folds",
        "num_windows": int(len(y)),
        "num_records": int(len(np.unique(groups))),
        "label_counts": dict(Counter(y.tolist())),
        "models": reports,
        "interpretation": interpretation,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(markdown_table(report))
    print(json.dumps({
        "num_windows": report["num_windows"],
        "num_records": report["num_records"],
        "label_counts": report["label_counts"],
        "resp_only": reports["resp_only"]["overall"],
        "spo2_only": reports["spo2_only"]["overall"],
        "resp_plus_spo2": reports["resp_plus_spo2"]["overall"],
        "out_json": str(args.out_json),
        "out_md": str(args.out_md),
    }, indent=2))


if __name__ == "__main__":
    main()
