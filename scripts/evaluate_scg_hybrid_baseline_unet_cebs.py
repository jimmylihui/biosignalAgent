from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_scg_heartbeat_unet_cebs import (  # noqa: E402
    UNet1D,
    add_hr_metrics,
    complete_records,
    load_record,
    match_peaks,
    predict_record,
)
from biosignal_agent.tools.scg_tools import SCG_detect_j_peaks  # noqa: E402


def aggregate(per: list[dict[str, Any]]) -> dict[str, Any]:
    total = {"tp": 0, "fp": 0, "fn": 0}
    maes, count_hr, interval_hr = [], [], []
    for m in per:
        total["tp"] += m["tp"]
        total["fp"] += m["fp"]
        total["fn"] += m["fn"]
        if m.get("mae_ms") is not None:
            maes.append(m["mae_ms"])
        if m.get("count_hr_abs_error_bpm") is not None:
            count_hr.append(m["count_hr_abs_error_bpm"])
        if m.get("interval_hr_abs_error_bpm") is not None:
            interval_hr.append(m["interval_hr_abs_error_bpm"])
    sens = total["tp"] / (total["tp"] + total["fn"]) if total["tp"] + total["fn"] else 0.0
    ppv = total["tp"] / (total["tp"] + total["fp"]) if total["tp"] + total["fp"] else 0.0
    f1 = 2 * sens * ppv / (sens + ppv) if sens + ppv else 0.0
    return {
        **total,
        "sensitivity": sens,
        "ppv": ppv,
        "f1": f1,
        "mae_ms": float(np.mean(maes)) if maes else None,
        "count_hr_mae_bpm": float(np.mean(count_hr)) if count_hr else None,
        "interval_hr_mae_bpm": float(np.mean(interval_hr)) if interval_hr else None,
        "per_record": per,
    }


def cache_records(model: UNet1D, raw_dir: Path, records: list[str], args: argparse.Namespace, dev: torch.device) -> dict[str, Any]:
    cache = {}
    with tempfile.TemporaryDirectory() as td:
        for rec in records:
            ref, _, prob, n_samples = predict_record(model, raw_dir, rec, args, dev)
            x, _ = load_record(raw_dir, rec, args.target_fs)
            path = Path(td) / f"{rec}.csv"
            pd.DataFrame({"signal": x}).to_csv(path, index=False)
            baseline = SCG_detect_j_peaks(str(path), args.target_fs)
            cache[rec] = {
                "ref": ref,
                "prob": prob,
                "x": x,
                "n_samples": n_samples,
                "baseline_peaks": np.asarray(baseline.get("j_peak_indices", []), dtype=int),
            }
    return cache


def evaluate_hybrid(cache: dict[str, Any], args: argparse.Namespace, window_s: float, scorer: str, threshold: float | None) -> dict[str, Any]:
    per = []
    window = int(round(window_s * args.target_fs))
    for rec, c in cache.items():
        pred = []
        env = np.abs(c["x"])
        prob = c["prob"]
        for peak in c["baseline_peaks"]:
            lo = max(0, int(peak) - window)
            hi = min(len(prob), int(peak) + window + 1)
            if hi <= lo:
                continue
            score = prob[lo:hi] if scorer == "prob" else env[lo:hi]
            refined = lo + int(np.argmax(score))
            if threshold is not None and prob[refined] < threshold:
                refined = int(peak)
            pred.append(refined)
        pred_arr = np.asarray(pred, dtype=int)
        metrics = match_peaks(c["ref"], pred_arr, args.target_fs, args.tolerance_s)
        add_hr_metrics(metrics, c["ref"], pred_arr, args.target_fs, c["n_samples"])
        metrics.update(record=rec, ref_peaks=int(len(c["ref"])), pred_peaks=int(len(pred_arr)))
        per.append(metrics)
    return aggregate(per)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="/data1/jiahui/biosignal-agent/datasets/raw/cebsdb")
    ap.add_argument("--model", default="/data1/jiahui/biosignal-agent/outputs/scg_cebs_unet_heartbeat_b001_b020_hrselect.pt")
    ap.add_argument("--report", default="/data1/jiahui/biosignal-agent/outputs/scg_cebs_hybrid_baseline_unet_b001_b020_report.json")
    ap.add_argument("--val-count", type=int, default=2)
    ap.add_argument("--test-count", type=int, default=2)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    payload = torch.load(args.model, map_location="cpu")
    args.target_fs = float(payload["target_fs"])
    args.seconds = float(payload["seconds"])
    args.threshold = float(payload["threshold"])
    args.min_distance_s = float(payload["min_distance_s"])
    args.tolerance_s = float(payload["tolerance_s"])

    model = UNet1D()
    model.load_state_dict(payload["model_state_dict"])
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(dev).eval()

    raw_dir = Path(args.raw_dir)
    records = complete_records(raw_dir)
    holdout = args.val_count + args.test_count
    val_records = records[-holdout:-args.test_count]
    test_records = records[-args.test_count:]

    val_cache = cache_records(model, raw_dir, val_records, args, dev)
    test_cache = cache_records(model, raw_dir, test_records, args, dev)
    rows = []
    for window_s in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.25]:
        for scorer in ["prob", "env"]:
            for threshold in [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
                result = evaluate_hybrid(val_cache, args, window_s, scorer, threshold)
                rows.append({"window_s": window_s, "scorer": scorer, "threshold": threshold, "result": result})
    rows.sort(key=lambda row: (row["result"]["f1"], -row["result"]["interval_hr_mae_bpm"], -row["result"]["count_hr_mae_bpm"]), reverse=True)
    best = rows[0]
    test = evaluate_hybrid(test_cache, args, best["window_s"], best["scorer"], best["threshold"])
    report = {
        "records": records,
        "val_records": val_records,
        "test_records": test_records,
        "selected_on_validation": {
            "window_s": best["window_s"],
            "scorer": best["scorer"],
            "threshold": best["threshold"],
            "validation": {k: v for k, v in best["result"].items() if k != "per_record"},
        },
        "test_hybrid": test,
        "note": "Baseline peaks define beat count/HR; local U-Net probability or SCG envelope only adjusts peak position within a short window.",
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ["val_records", "test_records", "selected_on_validation"]}, indent=2))
    print(json.dumps({k: test[k] for k in ["sensitivity", "ppv", "f1", "mae_ms", "count_hr_mae_bpm", "interval_hr_mae_bpm"]}, indent=2))


if __name__ == "__main__":
    main()
