from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.abp_tools import _abp_pulse_summary

PHYSIONET_DIR = "challenge-2009/1.0.0"
RAW_DIR = Path("/data1/jiahui/biosignal-agent/datasets/raw/challenge_2009")
OUT_DIR = Path("/data1/jiahui/biosignal-agent/outputs/abp_challenge2009")
CACHE_DIR = OUT_DIR / "segment_cache"


def parse_answers(path: Path) -> dict[str, str]:
    answers: dict[str, str] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            answers[parts[0]] = parts[1]
    return answers


def read_tail_abp(record_name: str, pn_subdir: str, tail_minutes: float) -> tuple[np.ndarray, float, dict[str, Any]]:
    header = wfdb.rdheader(record_name, pn_dir=pn_subdir)
    fs = float(header.fs)
    target = int(tail_minutes * 60.0 * fs)
    chunks: list[np.ndarray] = []
    meta_segments = []
    remaining = target

    seg_names = list(getattr(header, "seg_name", []) or [])
    seg_lens = list(getattr(header, "seg_len", []) or [])
    if seg_names and seg_lens:
        for seg_name, seg_len in reversed(list(zip(seg_names, seg_lens))):
            seg_len = int(seg_len or 0)
            if remaining <= 0:
                break
            if seg_len <= 0 or str(seg_name).endswith("_layout"):
                continue
            take = min(remaining, seg_len)
            sampfrom = max(0, seg_len - take)
            try:
                seg_header = wfdb.rdheader(str(seg_name), pn_dir=pn_subdir)
                names = [str(x).strip(",") for x in (seg_header.sig_name or [])]
                if "ABP" not in names:
                    chunk = np.full(take, np.nan, dtype=float)
                    err = "ABP absent in segment"
                else:
                    rec = wfdb.rdrecord(str(seg_name), pn_dir=pn_subdir, sampfrom=sampfrom, channels=[names.index("ABP")])
                    chunk = np.asarray(rec.p_signal[:, 0], dtype=float) if rec.p_signal is not None else np.full(take, np.nan, dtype=float)
                    err = None
                chunks.append(chunk)
                meta_segments.append({"segment": str(seg_name), "take_samples": int(take), "sampfrom": int(sampfrom), "signals": names, "error": err})
                remaining -= take
            except Exception as exc:
                chunks.append(np.full(take, np.nan, dtype=float))
                meta_segments.append({"segment": str(seg_name), "take_samples": int(take), "sampfrom": int(sampfrom), "error": f"wfdb segment read failed: {exc}"})
                remaining -= take
        values = np.concatenate(list(reversed(chunks))) if chunks else np.array([], dtype=float)
        return values, fs, {"segments_read": list(reversed(meta_segments)), "target_samples": int(target), "duration_read_s": float(len(values) / fs) if fs else 0.0, "sig_len": int(header.sig_len or 0)}

    sig_len = int(header.sig_len or 0)
    sampfrom = max(0, sig_len - target) if sig_len > 0 else 0
    try:
        rec = wfdb.rdrecord(record_name, pn_dir=pn_subdir, sampfrom=sampfrom)
    except Exception as exc:
        return np.array([], dtype=float), fs, {"error": f"wfdb read failed: {exc}", "sampfrom": sampfrom, "sig_len": sig_len}
    names = [str(x).strip(",") for x in (rec.sig_name or [])]
    if "ABP" not in names:
        return np.array([], dtype=float), fs, {"error": "ABP channel not found", "signals": names, "sampfrom": sampfrom, "sig_len": sig_len}
    x = np.asarray(rec.p_signal[:, names.index("ABP")], dtype=float) if rec.p_signal is not None else np.array([], dtype=float)
    return x, fs, {"signals": names, "sampfrom": sampfrom, "sig_len": sig_len, "duration_read_s": float(len(x) / fs) if fs else 0.0}


def segment_features(base_id: str, suffix: str, event: int, tail_minutes: float) -> dict[str, Any]:
    record_name = f"{base_id}{suffix}"
    cache_path = CACHE_DIR / f"event{event}_{record_name}_tail{int(tail_minutes * 60)}s.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass
    subdir = f"{PHYSIONET_DIR}/test-set-{'a' if event == 1 else 'b'}/{record_name}"
    values, fs, meta = read_tail_abp(record_name, subdir, tail_minutes)
    out: dict[str, Any] = {"segment": record_name, "fs": fs, **meta}
    finite = values[np.isfinite(values)]
    if len(finite) < max(100, int(fs * 60)):
        out.update({"error": meta.get("error", "not enough finite ABP samples"), "score": None})
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(out, indent=2))
        return out
    summary = _abp_pulse_summary(values, fs)
    if summary.get("error"):
        out.update({"error": summary["error"], "score": None})
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(out, indent=2))
        return out
    maps = np.asarray(summary.get("beat_map_values", []), dtype=float)
    sys = np.asarray(summary.get("beat_systolic_values", []), dtype=float)
    dia = np.asarray(summary.get("beat_diastolic_values", []), dtype=float)
    pp = np.asarray(summary.get("beat_pulse_pressure_values", []), dtype=float)
    hypo = (maps < 65.0) | (sys < 90.0)
    official_low = maps <= 60.0
    severe = maps < 55.0
    narrow = pp < 25.0
    low_map_p05 = float(np.nanpercentile(maps, 5)) if len(maps) else np.nan
    median_map = float(np.nanmedian(maps)) if len(maps) else np.nan
    minute_idx = np.floor(np.asarray(summary.get("pulse_indices", []), dtype=float) / fs / 60.0).astype(int)
    n_minutes = int(np.floor(len(values) / fs / 60.0)) if fs else 0
    minute_maps = []
    for minute in range(n_minutes):
        vals = maps[minute_idx == minute]
        minute_maps.append(float(np.nanmean(vals)) if len(vals) else np.nan)
    minute_maps_arr = np.asarray(minute_maps, dtype=float)
    valid_minutes = np.isfinite(minute_maps_arr) & (minute_maps_arr > 10.0)
    low_minutes = valid_minutes & (minute_maps_arr <= 60.0)
    low_minute_fraction = float(np.mean(low_minutes[valid_minutes])) if np.any(valid_minutes) else 0.0
    recent_slope = 0.0
    if int(np.sum(valid_minutes)) >= 3:
        yy = minute_maps_arr[valid_minutes]
        xx = np.arange(len(yy), dtype=float)
        recent_slope = float(np.polyfit(xx, yy, 1)[0])
    # Challenge heuristic: combine official minute-MAP low burden with recent low MAP depth and instability.
    score = 0.0
    score += 2.2 * low_minute_fraction
    score += max(0.0, (70.0 - low_map_p05) / 25.0)
    score += 1.0 * float(np.mean(hypo))
    score += 1.4 * float(np.mean(severe))
    score += 0.35 * float(np.mean(narrow))
    if recent_slope < -1.0:
        score += min(0.4, abs(recent_slope) / 10.0)
    if summary.get("heart_rate_bpm") and summary["heart_rate_bpm"] > 110:
        score += 0.15
    out.update({
        "num_pulses": summary.get("num_pulses"),
        "heart_rate_bpm": summary.get("heart_rate_bpm"),
        "median_systolic": summary.get("median_systolic_value"),
        "median_diastolic": summary.get("approx_diastolic_value"),
        "median_map": median_map,
        "map_p05": low_map_p05,
        "median_pulse_pressure": summary.get("median_pulse_pressure"),
        "hypotensive_beat_fraction": float(np.mean(hypo)),
        "official_low_map_beat_fraction": float(np.mean(official_low)),
        "low_map_minute_fraction": low_minute_fraction,
        "minute_map_slope_mmHg_per_min": recent_slope,
        "valid_minute_count": int(np.sum(valid_minutes)),
        "severe_hypotensive_beat_fraction": float(np.mean(severe)),
        "narrow_pulse_pressure_fraction": float(np.mean(narrow)),
        "artifact_rejected_fraction": summary.get("artifact_rejected_fraction"),
        "score": float(score),
    })
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out, indent=2))
    return out


def evaluate_event(event: int, tail_minutes: float, top_h: int | None = None, suffixes: list[str] | None = None) -> dict[str, Any]:
    answer_file = RAW_DIR / ("event-1-answers.txt" if event == 1 else "event-2-answers.txt")
    answers = parse_answers(answer_file)
    if top_h is None:
        top_h = 5 if event == 1 else 13
    suffixes = suffixes or ["a", "b", "c"]
    rows = []
    for base_id, truth in sorted(answers.items()):
        segs = [segment_features(base_id, suffix, event, tail_minutes) for suffix in suffixes]
        valid = [seg for seg in segs if seg.get("score") is not None]
        best = max(valid, key=lambda r: float(r.get("score", -1))) if valid else {"score": -1.0, "error": "no valid ABP segment"}
        rows.append({"record": base_id, "truth": truth, "score": float(best.get("score", -1.0)), "best_segment": best.get("segment"), "best_features": best, "segments": segs})
    ranked = sorted(rows, key=lambda r: r["score"], reverse=True)
    predicted_h = {row["record"] for row in ranked[:top_h]}
    for row in rows:
        row["prediction"] = "H" if row["record"] in predicted_h else "C"
    correct = sum(1 for row in rows if row["truth"] == row["prediction"])
    tp = sum(1 for row in rows if row["truth"] == "H" and row["prediction"] == "H")
    tn = sum(1 for row in rows if row["truth"] == "C" and row["prediction"] == "C")
    fp = sum(1 for row in rows if row["truth"] == "C" and row["prediction"] == "H")
    fn = sum(1 for row in rows if row["truth"] == "H" and row["prediction"] == "C")
    return {
        "event": event,
        "tail_minutes": tail_minutes,
        "top_h": top_h,
        "suffixes": suffixes,
        "num_records": len(rows),
        "score": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "predicted_h_records": sorted(predicted_h),
        "truth_h_records": sorted([r for r, y in answers.items() if y == "H"]),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ABP event-burden heuristic on PhysioNet/CinC Challenge 2009.")
    parser.add_argument("--event", type=int, choices=[1, 2], default=1)
    parser.add_argument("--tail-minutes", type=float, default=30.0)
    parser.add_argument("--top-h", type=int, default=None)
    parser.add_argument("--suffixes", default="a,b,c", help="Comma-separated segment suffixes to use, e.g. c or a,b,c.")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()
    suffixes = [x.strip() for x in args.suffixes.split(",") if x.strip()]
    report = evaluate_event(args.event, args.tail_minutes, args.top_h, suffixes)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out_json or OUT_DIR / f"event{args.event}_tail{int(args.tail_minutes)}m_eval.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ["event", "tail_minutes", "top_h", "suffixes", "num_records", "score", "accuracy", "confusion", "predicted_h_records", "truth_h_records"]}, indent=2))


if __name__ == "__main__":
    main()
