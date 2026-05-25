from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.agent.planning_agent import PlanningBioSignalAgent
from biosignal_agent.tools.digitize_tools import Signal_digitize_waveform_image_ml
from biosignal_agent.tools.image_cnn_tools import Signal_classify_modality_from_image_cnn
from biosignal_agent.tools.image_modality_tools import Signal_classify_modality_from_image
from biosignal_agent.tools.modality_tools import Signal_classify_modality

OUT = Path('/data1/jiahui/biosignal-agent/outputs/full_image_ocr_pipeline_demo')
OUT.mkdir(parents=True, exist_ok=True)


def load_values(path: str | Path) -> np.ndarray:
    df = pd.read_csv(path)
    col = 'signal' if 'signal' in df.columns else df.select_dtypes('number').columns[-1]
    return df[col].to_numpy(dtype=float)


def render_demo_image() -> dict[str, Any]:
    ref = Path('/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark_more_10s/references/ecg_00_bidmc01_clean_reference.csv')
    values = load_values(ref)
    fs = 152.0
    t = np.arange(len(values)) / fs
    # Use round y ticks so OCR has large, clean labels.
    ymin = float(np.floor(np.nanmin(values) * 10) / 10)
    ymax = float(np.ceil(np.nanmax(values) * 10) / 10)
    yticks = np.linspace(ymin, ymax, 5)
    image_path = OUT / 'demo_ecg_axis_labeled.png'
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=160)
    ax.plot(t, values, color='black', linewidth=1.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Time (s)', fontsize=13)
    ax.set_ylabel('ECG (mV)', fontsize=13)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_yticks(yticks)
    ax.tick_params(labelsize=12)
    ax.grid(True, color='#d0d0d0', linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.1)
    fig.tight_layout()
    fig.savefig(image_path)
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    dpi = fig.dpi
    # Pixel bbox in saved image coordinates; y origin converted from bottom to top.
    fig_w, fig_h = fig.get_size_inches() * dpi
    plot_bbox = {
        'left': int(round(bbox.x0 * dpi)),
        'right': int(round(bbox.x1 * dpi)),
        'top': int(round(fig_h - bbox.y1 * dpi)),
        'bottom': int(round(fig_h - bbox.y0 * dpi)),
    }
    plt.close(fig)
    return {
        'image_path': str(image_path),
        'reference_path': str(ref),
        'truth_sampling_rate': fs,
        'truth_duration_s': 10.0,
        'truth_value_min': ymin,
        'truth_value_max': ymax,
        'render_plot_bbox': plot_bbox,
    }


def numbers(text: str) -> list[float]:
    vals = []
    for tok in re.findall(r'[-+]?\d+(?:\.\d+)?', text):
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return vals


def rapidocr_boxes(path: Path):
    from rapidocr_onnxruntime import RapidOCR
    res, _ = RapidOCR()(str(path))
    return res or []


def make_preprocessed_crop(img: Image.Image, box: tuple[int, int, int, int], scale: int = 4) -> Path:
    crop = img.crop(box).convert('L')
    crop = ImageOps.autocontrast(crop)
    crop = crop.filter(ImageFilter.SHARPEN)
    crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.LANCZOS)
    out = OUT / f'ocr_crop_{len(list(OUT.glob("ocr_crop_*.png")))}.png'
    crop.convert('RGB').save(out)
    return out


def estimate_plot_bbox(image_path: str) -> dict[str, int]:
    # For this demo-style plot, detect the axes rectangle from long dark lines.
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    edges = cv2.Canny(img, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=int(min(w, h) * 0.30), maxLineGap=8)
    xs, ys = [], []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            if dx > w * 0.45 and dy < 5:
                ys.append((y1 + y2) / 2)
            if dy > h * 0.35 and dx < 5:
                xs.append((x1 + x2) / 2)
    if len(xs) >= 2 and len(ys) >= 2:
        return {'left': int(min(xs)), 'right': int(max(xs)), 'top': int(min(ys)), 'bottom': int(max(ys))}
    # Conservative fallback for matplotlib-like tight_layout plot.
    return {'left': int(w * 0.10), 'right': int(w * 0.98), 'top': int(h * 0.07), 'bottom': int(h * 0.78)}


def ocr_x_axis(image_path: str, plot_bbox: dict[str, int]) -> dict[str, Any]:
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    box = (max(0, plot_bbox['left'] - 30), max(0, plot_bbox['bottom'] - 15), min(w, plot_bbox['right'] + 30), h)
    crop_path = make_preprocessed_crop(img, box, scale=4)
    crop_h = Image.open(crop_path).height
    picked = []
    for box_pts, text, conf in rapidocr_boxes(crop_path):
        vals = numbers(str(text))
        if not vals:
            continue
        cy = float(np.mean([p[1] for p in box_pts]))
        cx = float(np.mean([p[0] for p in box_pts]))
        # Bottom numeric row: tick labels, not y-axis or title text.
        if cy >= crop_h * 0.30:
            picked.append((cx, str(text), float(conf), vals))
    picked.sort(key=lambda x: x[0])
    tick_values = [v for _, _, _, vals in picked for v in vals]
    unique = sorted(set(tick_values))
    duration = float(unique[-1] - unique[0]) if len(unique) >= 4 else None
    return {
        'crop_path': str(crop_path),
        'ocr_text': ' '.join(t for _, t, _, _ in picked),
        'tick_values': tick_values,
        'duration_s': duration,
        'status': 'ok' if duration is not None else 'insufficient_x_ticks',
    }


def ocr_y_axis(image_path: str, plot_bbox: dict[str, int]) -> dict[str, Any]:
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    box = (0, max(0, plot_bbox['top'] - 25), min(w, plot_bbox['left'] + 35), min(h, plot_bbox['bottom'] + 25))
    crop_path = make_preprocessed_crop(img, box, scale=4)
    picked = []
    for box_pts, text, conf in rapidocr_boxes(crop_path):
        vals = numbers(str(text))
        if not vals:
            continue
        cy = float(np.mean([p[1] for p in box_pts]))
        for v in vals:
            picked.append((cy, float(v), str(text), float(conf)))
    picked.sort(key=lambda x: x[0])
    values = [v for _, v, _, _ in picked]
    unique = sorted(set(values))
    if len(unique) >= 2:
        value_min, value_max = float(unique[0]), float(unique[-1])
        status = 'ok'
    else:
        value_min = value_max = None
        status = 'insufficient_y_ticks'
    return {
        'crop_path': str(crop_path),
        'ocr_text': ' '.join(t for _, _, t, _ in picked),
        'tick_values': values,
        'value_min': value_min,
        'value_max': value_max,
        'status': status,
    }


def compact_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    keep = ['tool', 'heart_rate_bpm', 'num_peaks', 'mean_rr_ms', 'sdnn_ms', 'rmssd_ms', 'quality', 'prediction', 'confidence', 'method', 'error']
    return {k: result.get(k) for k in keep if k in result}


def main() -> None:
    question = 'This is a user-uploaded signal image. Identify the signal type, recover the waveform, estimate heart rate and HRV, and screen for arrhythmia-like rhythm irregularity.'
    demo = render_demo_image()
    image_path = demo['image_path']
    img = Image.open(image_path)
    plot_bbox = estimate_plot_bbox(image_path)
    x_axis = ocr_x_axis(image_path, plot_bbox)
    y_axis = ocr_y_axis(image_path, plot_bbox)
    duration_s = x_axis['duration_s']
    if duration_s is None:
        report = {
            'input': {'image_path': image_path, 'question': question},
            'axis_ocr': {'plot_bbox': plot_bbox, 'x_axis': x_axis, 'y_axis': y_axis},
            'pipeline_ok': False,
            'failure_stage': 'missing_x_axis_scale',
            'error': 'x-axis duration could not be recovered; stopping before digitization and signal-tool execution.',
            'render_truth_for_demo_only': demo,
        }
        out_json = OUT / 'demo_full_pipeline_report.json'
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps({'out_json': str(out_json), 'error': report['error']}, indent=2, ensure_ascii=False))
        return
    sampling_rate = (plot_bbox['right'] - plot_bbox['left'] + 1) / duration_s
    value_min = y_axis['value_min'] if y_axis['value_min'] is not None else None
    value_max = y_axis['value_max'] if y_axis['value_max'] is not None else None

    crop_left = plot_bbox['left'] + 2
    crop_right = img.width - plot_bbox['right'] + 2
    crop_top = plot_bbox['top'] + 2
    crop_bottom = img.height - plot_bbox['bottom'] + 2

    feature_classifier = Signal_classify_modality_from_image(image_path, crop_left=crop_left, crop_right=crop_right, crop_top=crop_top, crop_bottom=crop_bottom)
    cnn_classifier = Signal_classify_modality_from_image_cnn(image_path, crop_left=crop_left, crop_right=crop_right, crop_top=crop_top, crop_bottom=crop_bottom)
    modality = str(cnn_classifier.get('predicted_modality') or feature_classifier.get('predicted_modality') or 'ecg').lower()

    out_csv = OUT / 'demo_digitized_signal.csv'
    digitized = Signal_digitize_waveform_image_ml(
        image_path=image_path,
        sampling_rate=sampling_rate,
        out_csv=str(out_csv),
        value_min=value_min,
        value_max=value_max,
        crop_left=crop_left,
        crop_right=crop_right,
        crop_top=crop_top,
        crop_bottom=crop_bottom,
        model_path='/data1/jiahui/biosignal-agent/outputs/waveform_digitization_pixel_model_highres.joblib',
        probability_threshold=0.5,
        smooth_window=3,
        trace_method='median',
    )

    signal_classifier = Signal_classify_modality(str(out_csv), sampling_rate=sampling_rate, column='signal') if not digitized.get('error') else {'error': digitized.get('error')}
    agent = PlanningBioSignalAgent()
    agent_report = agent.run(question, str(out_csv), sampling_rate=sampling_rate, column='signal', fallback_modality=modality) if not digitized.get('error') else {'error': digitized.get('error')}

    report = {
        'input': {'image_path': image_path, 'question': question},
        'image_modality_classifier_cnn': cnn_classifier,
        'image_modality_classifier_feature': feature_classifier,
        'selected_modality': modality,
        'axis_ocr': {'plot_bbox': plot_bbox, 'x_axis': x_axis, 'y_axis': y_axis},
        'calibration_used': {
            'duration_s': duration_s,
            'sampling_rate_hz': sampling_rate,
            'value_min': value_min,
            'value_max': value_max,
            'crop': {'left': crop_left, 'right': crop_right, 'top': crop_top, 'bottom': crop_bottom},
        },
        'digitization': digitized,
        'post_digitization_modality_classifier': signal_classifier,
        'agent_report': agent_report,
        'compact_report': {
            'modality': modality,
            'sampling_rate_hz': round(float(sampling_rate), 3),
            'axis_ocr_status': {'x': x_axis['status'], 'y': y_axis['status']},
            'digitized_csv': str(out_csv),
            'tools': [compact_tool_result(call['result']) for call in agent_report.get('tool_calls', [])] if isinstance(agent_report, dict) else [],
            'findings': agent_report.get('findings', []) if isinstance(agent_report, dict) else [],
            'disclaimer': agent_report.get('disclaimer') if isinstance(agent_report, dict) else None,
        },
        'render_truth_for_demo_only': demo,
    }
    out_json = OUT / 'demo_full_pipeline_report.json'
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({'out_json': str(out_json), 'compact_report': report['compact_report']}, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
