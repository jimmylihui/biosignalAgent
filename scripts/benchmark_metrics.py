from __future__ import annotations

from collections import Counter
from typing import Any


def binary_metrics(rows: list[dict[str, Any]], truth_key: str = "truth", pred_key: str = "prediction", positive: str = "positive") -> dict[str, Any]:
    tp = sum(1 for row in rows if row[truth_key] == positive and row[pred_key] == positive)
    tn = sum(1 for row in rows if row[truth_key] != positive and row[pred_key] != positive)
    fp = sum(1 for row in rows if row[truth_key] != positive and row[pred_key] == positive)
    fn = sum(1 for row in rows if row[truth_key] == positive and row[pred_key] != positive)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(rows) if rows else 0.0
    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
    }


def multiclass_metrics(rows: list[dict[str, Any]], truth_key: str = "truth", pred_key: str = "prediction") -> dict[str, Any]:
    labels = sorted({row[truth_key] for row in rows} | {row[pred_key] for row in rows})
    per_label = {}
    f1s = []
    for label in labels:
        metrics = binary_metrics(rows, truth_key, pred_key, label)
        support = sum(1 for row in rows if row[truth_key] == label)
        per_label[label] = {"support": support, **metrics}
        if support:
            f1s.append(metrics["f1"])
    accuracy = sum(1 for row in rows if row[truth_key] == row[pred_key]) / len(rows) if rows else 0.0
    return {"accuracy": accuracy, "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0, "per_label": per_label}


def counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(row.get(key) for row in rows))
