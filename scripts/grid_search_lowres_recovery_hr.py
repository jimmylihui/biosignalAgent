from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.digitize_tools import _crop_rgb_image, _signal_from_mask, pixel_feature_matrix
from scripts.evaluate_lowres_recovery_digitization import RESAMPLE_METHODS, load_records
from scripts.evaluate_waveform_digitization import corrcoef, detect_peaks, heart_rate, load_signal, nrmse, peak_f1


def upscale_image_array(path: str, scale: int, method: str) -> np.ndarray:
    image = Image.open(path).convert('RGB')
    width, height = image.size
    up = image.resize((width * scale, height * scale), resample=RESAMPLE_METHODS[method])
    return np.asarray(up, dtype=np.uint8)


def digitize_cached(image_arr: np.ndarray, record: dict[str, Any], model, threshold: float, trace_method: str, out_csv: Path) -> dict[str, Any]:
    h, w = image_arr.shape[:2]
    left = int(record['crop_left']); right = w - int(record['crop_right'])
    top = int(record['crop_top']); bottom = h - int(record['crop_bottom'])
    crop = image_arr[top:bottom, left:right]
    features = pixel_feature_matrix(crop)
    probs = model.predict_proba(features)[:, list(model.classes_).index(1)]
    mask = probs.reshape(crop.shape[:2]) >= float(threshold)
    result = _signal_from_mask(
        mask,
        float(record['sampling_rate']),
        str(out_csv),
        record['image_path'],
        float(record['value_min']),
        float(record['value_max']),
        1,
        'Signal_digitize_waveform_image_ml',
        'cached_grid_search',
        model_source='cached',
        confidence_scale=max(0.3, float(np.mean(probs[mask.ravel()])) if np.any(mask) else 0.0),
        trace_method=trace_method,
    )
    result['mask_pixel_fraction'] = float(np.mean(mask)) if mask.size else 0.0
    return result


def evaluate(record: dict[str, Any], high: dict[str, Any], digitized: dict[str, Any]) -> dict[str, Any]:
    row = {'record': record['record'], 'modality': record['modality'], 'variant': record.get('variant'), 'digitizer_error': digitized.get('error')}
    if digitized.get('error'):
        return row
    ref = load_signal(high['reference_path'])
    pred = load_signal(digitized['out_csv'])
    n = min(len(ref), len(pred)); ref = ref[:n]; pred = pred[:n]
    fs = float(high['sampling_rate'])
    ref_peaks = detect_peaks(ref, fs); pred_peaks = detect_peaks(pred, fs)
    pm = peak_f1(ref_peaks, pred_peaks, tolerance=max(5, int(round(fs * 0.02))))
    ref_hr = heart_rate(ref_peaks, fs); pred_hr = heart_rate(pred_peaks, fs)
    row.update({
        'waveform_correlation': corrcoef(ref, pred),
        'nrmse': nrmse(ref, pred),
        'peak_f1': pm['f1'],
        'reference_hr_bpm': ref_hr,
        'digitized_hr_bpm': pred_hr,
        'hr_abs_error_bpm': abs(ref_hr - pred_hr) if ref_hr is not None and pred_hr is not None else None,
        'digitized_csv': digitized['out_csv'],
    })
    return row


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return float(np.mean(vals)) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser(description='Fast SCG low-res recovery grid search with cached pixel model.')
    ap.add_argument('--modality', default='scg')
    ap.add_argument('--lowres-manifest', default='/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark_manifest.json')
    ap.add_argument('--highres-manifest', default='/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark_highres_aligned_manifest.json')
    ap.add_argument('--model-path', default='/data1/jiahui/biosignal-agent/outputs/waveform_digitization_pixel_model_highres.joblib')
    ap.add_argument('--out-dir', default='/data1/jiahui/biosignal-agent/outputs/scg_fast_grid')
    ap.add_argument('--out-json', default='/data1/jiahui/biosignal-agent/outputs/scg_fast_grid_summary.json')
    args = ap.parse_args()
    low = load_records(args.lowres_manifest); high = load_records(args.highres_manifest)
    ids = [rid for rid, rec in low.items() if rid in high and rec.get('modality') == args.modality]
    bundle = joblib.load(args.model_path); model = bundle['model']
    upscales = ['nearest','bilinear','bicubic','lanczos']
    traces = ['median','full','path','momentum','lazy','fragmented']
    thresholds = [0.2,0.35,0.5,0.65,0.8,0.9]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summaries=[]
    for up in upscales:
        image_cache = {rid: upscale_image_array(low[rid]['image_path'], 4, up) for rid in ids}
        for trace in traces:
            for th in thresholds:
                rows=[]
                for rid in ids:
                    rec = {**high[rid], 'image_path': low[rid]['image_path'], 'crop_left': int(low[rid]['crop_left'])*4, 'crop_right': int(low[rid]['crop_right'])*4, 'crop_top': int(low[rid]['crop_top'])*4, 'crop_bottom': int(low[rid]['crop_bottom'])*4}
                    out_csv = out_dir / f'{rid}_{up}_{trace}_t{th}.csv'
                    dig = digitize_cached(image_cache[rid], rec, model, th, trace, out_csv)
                    rows.append(evaluate(low[rid], high[rid], dig))
                ok=[r for r in rows if not r.get('digitizer_error')]
                summaries.append({
                    'upscale': up, 'trace': trace, 'threshold': th, 'num_records': len(rows), 'num_ok': len(ok),
                    'mean_hr_abs_error_bpm': mean(ok,'hr_abs_error_bpm'), 'mean_peak_f1': mean(ok,'peak_f1'),
                    'mean_waveform_correlation': mean(ok,'waveform_correlation'), 'mean_nrmse': mean(ok,'nrmse'), 'rows': rows,
                })
    summaries.sort(key=lambda r: (r['mean_hr_abs_error_bpm'] if r['mean_hr_abs_error_bpm'] is not None else 1e9, -(r['mean_peak_f1'] or 0)))
    Path(args.out_json).write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries[:20], indent=2))

if __name__ == '__main__':
    main()
