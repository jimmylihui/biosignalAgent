from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.ppg_tools import PPG_estimate_respiration_modulation  # noqa: E402


def reference_resp_rate(resp: np.ndarray, fs: float) -> dict[str, Any]:
    values = resp[np.isfinite(resp)]
    if len(values) < fs * 20:
        return {"respiratory_rate_bpm": None, "confidence": 0.0}
    values = values - np.nanmedian(values)
    high = min(0.7, fs * 0.45)
    sos = scipy_signal.butter(3, [0.08 / (0.5 * fs), high / (0.5 * fs)], btype="bandpass", output="sos")
    filtered = scipy_signal.sosfiltfilt(sos, values)
    freqs, psd = scipy_signal.welch(filtered, fs=fs, nperseg=min(len(filtered), int(fs * 32)))
    mask = (freqs >= 0.08) & (freqs <= high)
    if not np.any(mask):
        return {"respiratory_rate_bpm": None, "confidence": 0.0}
    band_freqs = freqs[mask]
    band_psd = psd[mask]
    peak_idx = int(np.argmax(band_psd))
    peak_power = float(band_psd[peak_idx])
    total_power = float(np.trapezoid(band_psd, band_freqs)) + 1e-12
    return {
        "respiratory_rate_bpm": float(band_freqs[peak_idx] * 60.0),
        "reference_peak_power_ratio": float(peak_power / (np.mean(band_psd) + 1e-12)),
        "reference_band_power": total_power,
        "confidence": 0.85,
    }


def legacy_envelope_resp_rate(ppg: np.ndarray, fs: float) -> dict[str, Any]:
    if len(ppg) < fs * 20:
        return {"respiratory_rate_bpm": None, "method": "legacy_ppg_envelope_respiration_bandpower_proxy"}
    envelope = np.abs(scipy_signal.hilbert(ppg - np.nanmedian(ppg)))
    high = min(0.7, fs * 0.45)
    sos = scipy_signal.butter(3, [0.08 / (0.5 * fs), high / (0.5 * fs)], btype="bandpass", output="sos")
    band = scipy_signal.sosfiltfilt(sos, envelope)
    freqs, psd = scipy_signal.welch(band, fs=fs, nperseg=min(len(band), int(fs * 16)))
    mask = (freqs >= 0.08) & (freqs <= high)
    rate = float(freqs[mask][np.argmax(psd[mask])] * 60.0) if np.any(mask) else None
    return {"respiratory_rate_bpm": rate, "method": "legacy_ppg_envelope_respiration_bandpower_proxy"}


def _find_channel(sig_names: list[str], candidates: list[str]) -> int | None:
    cleaned = [name.strip().strip(',').lower() for name in sig_names]
    for candidate in candidates:
        if candidate.lower() in cleaned:
            return cleaned.index(candidate.lower())
    for i, name in enumerate(cleaned):
        if any(candidate.lower() in name for candidate in candidates):
            return i
    return None


def iter_processed_records(data_dir: Path, fs: float) -> list[dict[str, Any]]:
    records = []
    for ppg_path in sorted(data_dir.glob("bidmc_*_ppg_60s.csv")):
        record = ppg_path.name.replace("_ppg_60s.csv", "")
        resp_path = data_dir / f"{record}_resp_60s.csv"
        if resp_path.exists():
            records.append({"record": record, "ppg_path": ppg_path, "resp": pd.read_csv(resp_path).select_dtypes("number").iloc[:, 0].to_numpy(float), "fs": fs})
    return records


def iter_wfdb_records(raw_dir: Path, out_dir: Path) -> list[dict[str, Any]]:
    records = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for hea in sorted(raw_dir.glob("bidmc*.hea")):
        rec_name = hea.stem
        try:
            rec = wfdb.rdrecord(str(raw_dir / rec_name))
        except Exception as exc:
            print(f"skip {rec_name}: {exc}", file=sys.stderr)
            continue
        resp_idx = _find_channel(rec.sig_name, ["RESP"])
        ppg_idx = _find_channel(rec.sig_name, ["PLETH", "PPG"])
        if resp_idx is None or ppg_idx is None:
            continue
        ppg = rec.p_signal[:, ppg_idx].astype(float)
        resp = rec.p_signal[:, resp_idx].astype(float)
        ppg_path = out_dir / f"{rec_name}_ppg_full.csv"
        if not ppg_path.exists():
            pd.DataFrame({"signal": ppg}).to_csv(ppg_path, index=False)
        records.append({"record": rec_name, "ppg_path": ppg_path, "ppg": ppg, "resp": resp, "fs": float(rec.fs)})
    return records


def evaluate(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for item in records:
        ref = reference_resp_rate(item["resp"], item["fs"])
        ppg = item.get("ppg")
        if ppg is None:
            ppg = pd.read_csv(item["ppg_path"]).select_dtypes("number").iloc[:, 0].to_numpy(float)
        legacy = legacy_envelope_resp_rate(ppg, item["fs"])
        pred = PPG_estimate_respiration_modulation(str(item["ppg_path"]), item["fs"])
        truth = ref.get("respiratory_rate_bpm")
        estimate = pred.get("respiratory_rate_bpm")
        legacy_estimate = legacy.get("respiratory_rate_bpm")
        rows.append({
            "record": item["record"],
            "duration_s": float(len(item["resp"]) / item["fs"]),
            "truth_resp_rate_bpm": truth,
            "pred_resp_rate_bpm": estimate,
            "legacy_resp_rate_bpm": legacy_estimate,
            "abs_error_bpm": abs(estimate - truth) if estimate is not None and truth is not None else None,
            "legacy_abs_error_bpm": abs(legacy_estimate - truth) if legacy_estimate is not None and truth is not None else None,
            "method": pred.get("method"),
            "respiration_source": pred.get("respiration_source"),
            "supporting_source": pred.get("supporting_source"),
            "harmonic_promoted_from_bpm": pred.get("harmonic_promoted_from_bpm"),
            "confidence": pred.get("confidence"),
            "heart_rate_bpm": pred.get("heart_rate_bpm"),
            "ppg_quality": pred.get("ppg_quality"),
            "respiratory_modulation_index": pred.get("respiratory_modulation_index"),
            "reference": ref,
            "tool_error": pred.get("error"),
        })
    errors = [row["abs_error_bpm"] for row in rows if row["abs_error_bpm"] is not None]
    legacy_errors = [row["legacy_abs_error_bpm"] for row in rows if row["legacy_abs_error_bpm"] is not None]
    return {
        "num_records": len(rows),
        "mae_bpm": float(np.mean(errors)) if errors else None,
        "median_abs_error_bpm": float(np.median(errors)) if errors else None,
        "legacy_mae_bpm": float(np.mean(legacy_errors)) if legacy_errors else None,
        "legacy_median_abs_error_bpm": float(np.median(legacy_errors)) if legacy_errors else None,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPG-derived respiration against BIDMC resp waveform reference.")
    parser.add_argument("--data-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/real_world"))
    parser.add_argument("--raw-wfdb-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/raw/bidmc"))
    parser.add_argument("--processed-out-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed/bidmc_full"))
    parser.add_argument("--source", choices=["processed_60s", "wfdb_full"], default="wfdb_full")
    parser.add_argument("--sampling-rate", type=float, default=125.0)
    parser.add_argument("--out-json", type=Path, default=Path("/data1/jiahui/biosignal-agent/outputs/ppg_respiration_bidmc_eval.json"))
    args = parser.parse_args()
    records = iter_wfdb_records(args.raw_wfdb_dir, args.processed_out_dir) if args.source == "wfdb_full" else iter_processed_records(args.data_dir, args.sampling_rate)
    report = evaluate(records)
    report["source"] = args.source
    report["reference"] = "BIDMC RESP waveform spectral respiratory-rate reference; legacy baseline is previous Hilbert-envelope-only PPG method."
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ["source", "num_records", "legacy_mae_bpm", "legacy_median_abs_error_bpm", "mae_bpm", "median_abs_error_bpm"]}, indent=2))
    for row in report["rows"][:20]:
        print(row["record"], "truth", row["truth_resp_rate_bpm"], "legacy", row["legacy_resp_rate_bpm"], "pred", row["pred_resp_rate_bpm"], "err", row["abs_error_bpm"], "src", row.get("respiration_source"))


if __name__ == "__main__":
    main()
