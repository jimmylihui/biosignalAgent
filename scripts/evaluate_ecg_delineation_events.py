#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_ecg_delineation_unet_qtdb import TinyUNet1D

CLASS_TO_NAME = {1: "p", 2: "qrs", 3: "t"}


def segments_from_mask(mask: np.ndarray, cls: int, min_len: int = 2) -> list[tuple[int, int]]:
    b = np.asarray(mask) == cls
    starts = np.flatnonzero(b & ~np.r_[False, b[:-1]])
    stops = np.flatnonzero(b & ~np.r_[b[1:], False]) + 1
    return [(int(s), int(e)) for s, e in zip(starts, stops) if e - s >= min_len]


def filter_segments(segments: list[tuple[int, int]], fs: float, wave: str) -> list[tuple[int, int]]:
    limits_ms = {"p": (20, 220), "qrs": (24, 220), "t": (40, 520)}
    lo, hi = limits_ms.get(wave, (10, 1000))
    lo_samp = int(round(lo * fs / 1000.0))
    hi_samp = int(round(hi * fs / 1000.0))
    return [(s, e) for s, e in segments if lo_samp <= e - s <= hi_samp]


def merge_segments(segments: list[tuple[int, int]], gap: int = 8) -> list[tuple[int, int]]:
    if not segments:
        return []
    segments = sorted((int(s), int(e)) for s, e in segments if e > s)
    merged = [segments[0]]
    for s, e in segments[1:]:
        ps, pe = merged[-1]
        if s <= pe + gap:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def point_match(pred_points: list[int], truth_points: list[int], tol: int) -> dict:
    used = set()
    for p in sorted(pred_points):
        best = None
        best_err = None
        for j, t in enumerate(truth_points):
            if j in used:
                continue
            err = abs(int(p) - int(t))
            if err <= tol and (best is None or err < best_err):
                best = j
                best_err = err
        if best is not None:
            used.add(best)
    tp = len(used)
    fp = len(pred_points) - tp
    fn = len(truth_points) - tp
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": float(precision), "recall": float(recall), "f1": float(f1)}


def greedy_match(pred: list[tuple[int, int]], truth: list[tuple[int, int]], tol: int) -> dict:
    used = set()
    onset_err = []
    offset_err = []
    iou_vals = []
    for ps, pe in pred:
        best = None
        best_score = None
        for j, (ts, te) in enumerate(truth):
            if j in used:
                continue
            if abs(ps - ts) <= tol and abs(pe - te) <= tol:
                score = abs(ps - ts) + abs(pe - te)
                if best is None or score < best_score:
                    best = j
                    best_score = score
        if best is not None:
            used.add(best)
            ts, te = truth[best]
            onset_err.append(ps - ts)
            offset_err.append(pe - te)
            inter = max(0, min(pe, te) - max(ps, ts))
            union = max(pe, te) - min(ps, ts)
            iou_vals.append(inter / union if union > 0 else 0.0)
    tp = len(used)
    fp = len(pred) - tp
    fn = len(truth) - tp
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "median_abs_onset_error_samples": float(np.median(np.abs(onset_err))) if onset_err else None,
        "median_abs_offset_error_samples": float(np.median(np.abs(offset_err))) if offset_err else None,
        "mean_iou_matched": float(np.mean(iou_vals)) if iou_vals else None,
    }


def load_model(model_path: Path):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    base = int(ckpt.get("base", 12))
    model = TinyUNet1D(classes=4, base=base)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def main():
    ap = argparse.ArgumentParser(description="Event-level onset/offset scoring for QTDB ECG delineation model.")
    ap.add_argument("--manifest", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/qtdb_delineation_cached_manual_manifest.json"))
    ap.add_argument("--model-path", type=Path, default=Path("/data1/jiahui/biosignal-agent/outputs/ecg_delineation_qtdb_cached90_unet.pt"))
    ap.add_argument("--out-json", type=Path, default=Path("/data1/jiahui/biosignal-agent/outputs/ecg_delineation_qtdb_cached90_event_eval.json"))
    ap.add_argument("--tolerance-ms", type=float, default=80.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    payload = json.loads(args.manifest.read_text())
    rows = payload["rows"]
    if args.limit:
        rows = rows[: args.limit]
    model, ckpt = load_model(args.model_path)
    val_records = set(ckpt.get("val_records") or [])
    if val_records:
        rows = [r for r in rows if r["record"] in val_records]
    per_class = {name: {"pred": [], "truth": []} for name in CLASS_TO_NAME.values()}
    fs_values = []
    with torch.no_grad():
        for r in rows:
            arr = np.load(r["path"])
            x = arr["signal"].astype(np.float32)
            y = arr["mask"].astype(np.int64)
            fs = float(arr["fs"])
            fs_values.append(fs)
            logits = model(torch.tensor(x[None, None, :], dtype=torch.float32))
            pred = logits.argmax(1).cpu().numpy()[0]
            for cls, name in CLASS_TO_NAME.items():
                per_class[name]["pred"].extend([(r["record"], int(r["start_sample"] + s), int(r["start_sample"] + e)) for s, e in segments_from_mask(pred, cls, max(2, int(0.012 * fs)))])
                per_class[name]["truth"].extend([(r["record"], int(r["start_sample"] + s), int(r["start_sample"] + e)) for s, e in segments_from_mask(y, cls, max(2, int(0.012 * fs)))])
    fs = float(np.median(fs_values)) if fs_values else 250.0
    tol = int(round(args.tolerance_ms * fs / 1000.0))
    results = {}
    for name, vals in per_class.items():
        pred_by = {}
        truth_by = {}
        for rec, s, e in vals["pred"]:
            pred_by.setdefault(rec, []).append((s, e))
        for rec, s, e in vals["truth"]:
            truth_by.setdefault(rec, []).append((s, e))
        agg = {"tp": 0, "fp": 0, "fn": 0, "onset_tp": 0, "onset_fp": 0, "onset_fn": 0, "offset_tp": 0, "offset_fp": 0, "offset_fn": 0}
        for rec in sorted(set(pred_by) | set(truth_by)):
            pred_segments = filter_segments(merge_segments(pred_by.get(rec, []), gap=max(2, tol)), fs, name)
            truth_segments = filter_segments(merge_segments(truth_by.get(rec, []), gap=max(2, tol)), fs, name)
            m = greedy_match(pred_segments, truth_segments, tol)
            onset = point_match([s for s, _ in pred_segments], [s for s, _ in truth_segments], tol)
            offset = point_match([e for _, e in pred_segments], [e for _, e in truth_segments], tol)
            agg["tp"] += m["tp"]; agg["fp"] += m["fp"]; agg["fn"] += m["fn"]
            agg["onset_tp"] += onset["tp"]; agg["onset_fp"] += onset["fp"]; agg["onset_fn"] += onset["fn"]
            agg["offset_tp"] += offset["tp"]; agg["offset_fp"] += offset["fp"]; agg["offset_fn"] += offset["fn"]
        precision = agg["tp"] / (agg["tp"] + agg["fp"] + 1e-12)
        recall = agg["tp"] / (agg["tp"] + agg["fn"] + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        onset_precision = agg["onset_tp"] / (agg["onset_tp"] + agg["onset_fp"] + 1e-12)
        onset_recall = agg["onset_tp"] / (agg["onset_tp"] + agg["onset_fn"] + 1e-12)
        onset_f1 = 2 * onset_precision * onset_recall / (onset_precision + onset_recall + 1e-12)
        offset_precision = agg["offset_tp"] / (agg["offset_tp"] + agg["offset_fp"] + 1e-12)
        offset_recall = agg["offset_tp"] / (agg["offset_tp"] + agg["offset_fn"] + 1e-12)
        offset_f1 = 2 * offset_precision * offset_recall / (offset_precision + offset_recall + 1e-12)
        results[name] = {"tp": agg["tp"], "fp": agg["fp"], "fn": agg["fn"], "precision": float(precision), "recall": float(recall), "f1": float(f1), "onset_f1": float(onset_f1), "offset_f1": float(offset_f1)}
    results["macro_event_f1"] = float(np.mean([results[k]["f1"] for k in CLASS_TO_NAME.values()]))
    results["macro_onset_f1"] = float(np.mean([results[k]["onset_f1"] for k in CLASS_TO_NAME.values()]))
    results["macro_offset_f1"] = float(np.mean([results[k]["offset_f1"] for k in CLASS_TO_NAME.values()]))
    out = {"model_path": str(args.model_path), "manifest": str(args.manifest), "val_records": sorted(val_records), "num_windows": len(rows), "tolerance_ms": args.tolerance_ms, "results": results}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
