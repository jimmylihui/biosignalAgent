from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.common import load_csv_signal
from biosignal_agent.tools.ecg_tools import ECG_APNEA_RR_EDR_MODEL_PATH, ECG_screen_sleep_apnea, _predict_apnea_rr_edr_model

warnings.filterwarnings("ignore", category=RuntimeWarning)


def confusion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(1 for row in rows if row["truth"] == "apnea" and row["prediction"] == "apnea")
    tn = sum(1 for row in rows if row["truth"] == "normal" and row["prediction"] == "normal")
    fp = sum(1 for row in rows if row["truth"] == "normal" and row["prediction"] == "apnea")
    fn = sum(1 for row in rows if row["truth"] == "apnea" and row["prediction"] == "normal")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(rows) if rows else 0.0
    return {"true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn, "precision": precision, "recall_sensitivity": recall, "specificity": specificity, "f1": f1, "accuracy": accuracy}


def _fast_rr_edr_result(record: dict[str, Any]) -> dict[str, Any]:
    data = load_csv_signal(record["path"], float(record["sampling_rate"]), None)
    probability, model_name, model_source, threshold, cv_metrics = _predict_apnea_rr_edr_model(ECG_APNEA_RR_EDR_MODEL_PATH, data.values, data.sampling_rate)
    if probability is None:
        return {"apnea_risk": "low", "error": model_name or "rr_edr_model_unavailable"}
    threshold = float(threshold if threshold is not None else 0.5)
    return {
        "apnea_risk": "elevated" if probability >= threshold else "low",
        "apnea_probability": float(probability),
        "decision_threshold": threshold,
        "model_name": model_name,
        "model_source": model_source,
        "deep_cv_metrics": cv_metrics,
        "method": "fast_ecg_rr_edr_cnn_apnea_screening",
    }


def evaluate(manifest_path: str | Path, fast_rr_edr: bool = False) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text())
    rows = []
    for index, record in enumerate(manifest.get("records", []), start=1):
        if index % 250 == 0:
            print(f"processed {index}", flush=True)
        result = _fast_rr_edr_result(record) if fast_rr_edr else ECG_screen_sleep_apnea(record["path"], float(record["sampling_rate"]), column=None)
        prediction = "apnea" if result.get("apnea_risk") == "elevated" else "normal"
        rows.append({
            "record": record["record"],
            "minute": record["minute"],
            "truth": record["label"],
            "prediction": prediction,
            "apnea_risk": result.get("apnea_risk"),
            "apnea_probability": result.get("apnea_probability"),
            "decision_threshold": result.get("decision_threshold"),
            "method": result.get("method"),
            "model_name": result.get("model_name"),
            "apnea_proxy_score": result.get("apnea_proxy_score"),
            "apnea_proxy_flags": result.get("apnea_proxy_flags", []),
            "heart_rate_bpm": result.get("heart_rate_bpm"),
            "sdnn_ms": result.get("sdnn_ms"),
            "rmssd_ms": result.get("rmssd_ms"),
            "rr_cv": result.get("rr_cv"),
            "confidence": result.get("confidence"),
            "error": result.get("error"),
        })
    return {
        "manifest": str(manifest_path),
        "num_windows": len(rows),
        "truth_counts": dict(Counter(row["truth"] for row in rows)),
        "prediction_counts": dict(Counter(row["prediction"] for row in rows)),
        "metrics": confusion(rows),
        "rows": rows,
    }


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["record", "minute", "truth", "prediction", "apnea_risk", "apnea_probability", "decision_threshold", "method", "model_name", "apnea_proxy_score", "apnea_proxy_flags", "heart_rate_bpm", "sdnn_ms", "rmssd_ms", "rr_cv", "confidence", "error"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key)) if isinstance(row.get(key), list) else row.get(key) for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ECG-only apnea screening proxy on Apnea-ECG minute labels.")
    parser.add_argument("--manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/apnea_ecg_manifest.json")
    parser.add_argument("--out-json", default="/data1/jiahui/biosignal-agent/outputs/apnea_ecg_eval.json")
    parser.add_argument("--out-csv", default="/data1/jiahui/biosignal-agent/outputs/apnea_ecg_eval.csv")
    parser.add_argument("--fast-rr-edr", action="store_true", help="Evaluate the primary RR/EDR apnea model without full tool evidence/R-peak reporting.")
    args = parser.parse_args()
    report = evaluate(args.manifest, fast_rr_edr=args.fast_rr_edr)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    write_csv(report["rows"], args.out_csv)
    print(json.dumps({key: report[key] for key in ["num_windows", "truth_counts", "prediction_counts", "metrics"]}, indent=2))


if __name__ == "__main__":
    main()
