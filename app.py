from __future__ import annotations

import base64
import json
import os
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from biosignal_agent.agent.planning_agent import PlanningBioSignalAgent
from biosignal_agent.agent.schema_loader import load_tool_schemas
from biosignal_agent.tools.digitize_tools import (
    Signal_digitize_waveform_image,
    Signal_digitize_waveform_image_ml,
    Signal_estimate_image_scale,
    _crop_rgb_image,
)
from biosignal_agent.tools.digitize_unet_tools import Signal_digitize_waveform_image_unet, UNET_MODEL_PATH, build_waveform_segmentation_model, select_waveform_mask_area
from biosignal_agent.tools.image_modality_tools import Signal_classify_modality_from_image
from biosignal_agent.tools.modality_tools import Signal_classify_modality

DISCLAIMER = "Prototype output for research use only; not a clinical diagnosis."
MODALITIES = ["auto", "ecg", "ppg", "bcg", "scg", "resp", "spo2", "abp", "pcg", "acc", "eda", "eeg", "emg"]
DEFAULT_CSV_QUESTION = "Analyze this biosignal, choose suitable tools, estimate core rates/features, and produce a concise research-use report."
DEFAULT_IMAGE_QUESTION = "Classify this waveform image, digitize the trace, then analyze the recovered signal with suitable tools."


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _read_signal(csv_path: str, column: str | None = None) -> tuple[np.ndarray, str]:
    frame = pd.read_csv(csv_path)
    if column and column in frame.columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        return series.dropna().to_numpy(dtype=float), column
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        converted = frame.apply(pd.to_numeric, errors="coerce")
        numeric = converted.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError("No numeric signal column was found in the uploaded CSV.")
    if "signal" in numeric.columns:
        return numeric["signal"].dropna().to_numpy(dtype=float), "signal"
    name = str(numeric.columns[-1]) if "time_s" in numeric.columns and len(numeric.columns) > 1 else str(numeric.columns[0])
    return numeric[name].dropna().to_numpy(dtype=float), name


def _plot_signal(values: np.ndarray, sampling_rate: float | None, title: str):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    if values.size:
        max_points = min(values.size, 5000)
        shown = values[:max_points]
        if sampling_rate and sampling_rate > 0:
            x = np.arange(max_points) / float(sampling_rate)
            ax.set_xlabel("Time (s)")
        else:
            x = np.arange(max_points)
            ax.set_xlabel("Sample")
        ax.plot(x, shown, linewidth=1.2)
    ax.set_title(title)
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig




def _trajectory_md(steps: list[dict[str, Any]]) -> str:
    lines = ["## Agent trajectory", ""]
    for idx, step in enumerate(steps, 1):
        title = step.get("title", f"Step {idx}")
        status = step.get("status", "ok")
        lines.append(f"### {idx}. {title}")
        lines.append(f"**Status:** `{status}`")
        detail = step.get("detail")
        if detail:
            lines.append("")
            lines.append(str(detail))
        items = step.get("items") or []
        if items:
            lines.append("")
            for item in items:
                lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines)


def _compact_tool_result(result: dict[str, Any], max_items: int = 10) -> dict[str, Any]:
    keep = {}
    for key, value in result.items():
        if key in {"r_peak_indices", "peak_indices", "indices", "signal", "samples", "probabilities_per_sample"}:
            if isinstance(value, list):
                keep[key] = {"count": len(value), "preview": value[:8]}
            else:
                keep[key] = "omitted_for_display"
        elif isinstance(value, (str, int, float, bool)) or value is None:
            keep[key] = value
        elif isinstance(value, list):
            keep[key] = value[:max_items]
        elif isinstance(value, dict):
            keep[key] = {str(k): v for k, v in list(value.items())[:max_items]}
    return keep


def _extract_optional_ocr_text(image_path: str) -> dict[str, Any]:
    try:
        import pytesseract
        from PIL import Image, ImageOps

        image = Image.open(image_path).convert("L")
        # Upscale and increase contrast a little; this helps small plot titles.
        image = ImageOps.autocontrast(image.resize((image.width * 2, image.height * 2)))
        text = pytesseract.image_to_string(image)
        cleaned = " ".join(text.split())
        return {"available": True, "text": cleaned[:800]}
    except Exception as exc:
        return {"available": False, "text": "", "error": str(exc)}


def _modality_from_text_hint(text: str, filename: str = "") -> tuple[str | None, str]:
    hay = f"{text} {filename}".lower()
    rules = [
        ("ecg", ["ecg", "ekg", "qrs", "r-peak", "r peak", "filtered ecg", "original ecg"]),
        ("ppg", ["ppg", "pleth", "photopleth"]),
        ("pcg", ["pcg", "phonocardi", "heart sound", "murmur"]),
        ("resp", ["resp", "respiration", "breathing", "airflow"]),
        ("spo2", ["spo2", "saturation", "oximetry"]),
        ("abp", ["abp", "arterial", "blood pressure"]),
        ("eda", ["eda", "gsr", "skin conductance"]),
        ("eeg", ["eeg", "electroencephal"]),
        ("emg", ["emg", "electromy"]),
        ("scg", ["scg", "seismocardi"]),
        ("bcg", ["bcg", "ballistocardi"]),
        ("acc", ["accelerometer", "acceleration", "actigraphy"]),
    ]
    for modality, needles in rules:
        for needle in needles:
            if needle in hay:
                return modality, needle
    return None, ""


def _blue_trace_mask(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(float)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    # Matplotlib default blue is roughly (31, 119, 180); allow anti-aliased variants.
    blueish = (b > 85) & (g > 55) & (r < 130) & (b > r + 25) & (g > r + 15)
    saturated = (np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])) > 35
    not_gray_grid = np.abs(r - g) + np.abs(g - b) > 35
    return blueish & saturated & not_gray_grid


def _select_trace_component(mask: np.ndarray, panel_policy: str = "auto") -> tuple[np.ndarray, dict[str, Any]]:
    from scipy import ndimage

    labels, num = ndimage.label(mask)
    if num <= 0:
        return mask, {"num_components": 0, "selected": None}
    components = []
    h, w = mask.shape
    for label_id in range(1, num + 1):
        ys, xs = np.nonzero(labels == label_id)
        if len(xs) < 20:
            continue
        x_span = int(xs.max() - xs.min() + 1)
        y_span = int(ys.max() - ys.min() + 1)
        if x_span < max(20, int(w * 0.08)):
            continue
        components.append({
            "label": label_id,
            "pixels": int(len(xs)),
            "x_span": x_span,
            "y_span": y_span,
            "x_min": int(xs.min()),
            "x_max": int(xs.max()),
            "y_min": int(ys.min()),
            "y_max": int(ys.max()),
            "y_center": float(np.mean(ys)),
        })
    if not components:
        return mask, {"num_components": int(num), "selected": None}
    # If there are two ECG panels, bottom panel usually corresponds to filtered signal.
    if panel_policy == "top":
        chosen = min(components, key=lambda c: c["y_center"])
    elif panel_policy == "bottom":
        chosen = max(components, key=lambda c: c["y_center"])
    else:
        # Prefer bottom among similarly wide components; otherwise choose widest/most signal-like.
        wide = sorted(components, key=lambda c: c["x_span"], reverse=True)[:3]
        chosen = max(wide, key=lambda c: (c["y_center"], c["x_span"], c["pixels"]))
    selected = labels == int(chosen["label"])
    return selected, {"num_components": int(num), "candidates": components[:8], "selected": chosen, "panel_policy": panel_policy}


def _digitize_color_trace_image(
    image_path: str,
    sampling_rate: float | None,
    out_csv: str,
    value_min: float | None = None,
    value_max: float | None = None,
    panel_policy: str = "auto",
) -> dict[str, Any]:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image)
    mask = _blue_trace_mask(rgb)
    raw_fraction = float(mask.mean()) if mask.size else 0.0
    selected, component_info = _select_trace_component(mask, panel_policy=panel_policy)
    if not np.any(selected):
        return {"tool": "Signal_digitize_plot_image_color_trace", "error": "no blue/color trace component detected", "confidence": 0.0, "raw_mask_fraction": raw_fraction, "component_info": component_info}
    ys, xs = np.nonzero(selected)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    crop = selected[y_min:y_max + 1, x_min:x_max + 1]
    height, width = crop.shape
    y_values = np.full(width, np.nan, dtype=float)
    for x in range(width):
        rows = np.flatnonzero(crop[:, x])
        if len(rows):
            y_values[x] = float(np.median(rows))
    finite = np.isfinite(y_values)
    coverage = float(finite.mean()) if len(y_values) else 0.0
    if finite.sum() < max(8, int(width * 0.05)):
        return {"tool": "Signal_digitize_plot_image_color_trace", "error": "too few trace columns after color segmentation", "confidence": 0.0, "pixel_coverage": coverage, "component_info": component_info}
    x_idx = np.arange(width)
    y_values[~finite] = np.interp(x_idx[~finite], x_idx[finite], y_values[finite])
    # Invert image coordinates. Use robust component bounds rather than full figure bounds.
    normalized = 1.0 - 2.0 * (y_values / max(1, height - 1))
    if value_min is not None and value_max is not None and float(value_max) != float(value_min):
        values = float(value_min) + (normalized + 1.0) / 2.0 * (float(value_max) - float(value_min))
        scale = "calibrated_from_user_y_bounds"
    else:
        values = normalized
        scale = "normalized_within_selected_trace_panel"
    frame = pd.DataFrame({"signal": values})
    if sampling_rate is not None and sampling_rate > 0:
        frame.insert(0, "time_s", np.arange(len(values), dtype=float) / float(sampling_rate))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)
    return {
        "tool": "Signal_digitize_plot_image_color_trace",
        "out_csv": str(out_csv),
        "num_points": int(len(values)),
        "sampling_rate": float(sampling_rate) if sampling_rate else None,
        "pixel_coverage": coverage,
        "raw_mask_fraction": raw_fraction,
        "selected_bbox": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
        "component_info": component_info,
        "value_min": float(np.nanmin(values)) if len(values) else None,
        "value_max": float(np.nanmax(values)) if len(values) else None,
        "scale": scale,
        "confidence": min(0.97, max(0.35, coverage)),
        "method": "color_trace_component_panel_digitizer",
        "disclaimer": "Color-trace plot digitizer; verify selected panel, axes, and calibration before clinical use.",
    }

def _short_report(result: dict[str, Any]) -> str:
    lines = ["## BioSignalAgent Report", "", f"**Modality:** `{result.get('modality', 'unknown')}`", ""]
    plan = result.get("plan") or []
    if plan:
        lines.append("**Selected tools:** " + ", ".join(f"`{tool}`" for tool in plan))
        lines.append("")
    findings = result.get("findings") or []
    if findings:
        lines.append("**Findings**")
        for finding in findings[:12]:
            lines.append(f"- {finding}")
    else:
        lines.append("No concise findings were produced. Inspect the JSON tool outputs below.")
    lines.append("")
    lines.append(f"**Safety note:** {DISCLAIMER}")
    return "\n".join(lines)


def run_csv_demo(csv_file: str | None, question: str, sampling_rate: float, modality_hint: str, column: str | None):
    steps: list[dict[str, Any]] = []
    if not csv_file:
        return "Upload a CSV file first.", "", {}, None
    if not sampling_rate or sampling_rate <= 0:
        return "Sampling rate must be a positive number.", "", {}, None
    question = (question or DEFAULT_CSV_QUESTION).strip()
    fallback = None if modality_hint == "auto" else modality_hint
    try:
        values, used_column = _read_signal(csv_file, column or None)
        steps.append({"title": "Input parsing", "items": [f"CSV column: `{used_column}`", f"Samples: `{len(values)}`", f"Sampling rate: `{sampling_rate}` Hz"]})
        classifier = Signal_classify_modality(csv_file, float(sampling_rate), column=used_column)
        if fallback is None:
            fallback = classifier.get("predicted_modality")
        steps.append({
            "title": "Modality routing",
            "items": [
                f"Classifier prediction: `{classifier.get('predicted_modality')}` confidence `{classifier.get('confidence')}`",
                f"Final route: `{fallback}`",
            ],
        })
        planner = PlanningBioSignalAgent()
        plan = planner.plan(question, fallback)
        steps.append({"title": "Tool planning", "items": [f"Selected `{len(plan)}` tools: " + ", ".join(f"`{tool}`" for tool in plan)]})
        result = planner.run(question, csv_file, float(sampling_rate), used_column, fallback)
        result["input_column"] = used_column
        result["modality_classifier"] = classifier
        result["disclaimer"] = DISCLAIMER
        tool_items = []
        for call in result.get("tool_calls", [])[:8]:
            compact = _compact_tool_result(call.get("result", {}))
            tool_items.append(f"`{call.get('tool')}` -> keys {list(compact)[:8]}")
        steps.append({"title": "Tool execution", "items": tool_items or ["No tools executed."]})
        steps.append({"title": "Grounded report", "detail": "Report is assembled from local tool outputs and includes a research-use limitation."})
        return _trajectory_md(steps), _short_report(result), _jsonable(result), _plot_signal(values, sampling_rate, "Uploaded signal")
    except Exception as exc:
        steps.append({"title": "Failure", "status": "error", "detail": f"{type(exc).__name__}: {exc}"})
        return _trajectory_md(steps), f"Error: {type(exc).__name__}: {exc}", {"error": str(exc), "stage": "csv_demo"}, None


def _digitize_with_fallback(image_path: str, sampling_rate: float | None, out_csv: str, value_min: float | None, value_max: float | None, trace_method: str):
    unet = Signal_digitize_waveform_image_unet(
        image_path=image_path,
        sampling_rate=sampling_rate,
        out_csv=out_csv,
        value_min=value_min,
        value_max=value_max,
        probability_threshold=0.65,
        trace_method=trace_method,
        smooth_window=3,
    )
    if not unet.get("error") and float(unet.get("pixel_coverage") or 0.0) >= 0.15:
        unet["fallback_priority"] = "segmentation_unet_first"
        return unet
    result = Signal_digitize_waveform_image_ml(
        image_path=image_path,
        sampling_rate=sampling_rate,
        out_csv=out_csv,
        value_min=value_min,
        value_max=value_max,
        trace_method=trace_method,
        smooth_window=3,
    )
    if result.get("error"):
        fallback = Signal_digitize_waveform_image(
            image_path=image_path,
            sampling_rate=sampling_rate,
            out_csv=out_csv,
            value_min=value_min,
            value_max=value_max,
            trace_method=trace_method,
            smooth_window=3,
        )
        fallback["fallback_from_unet_error"] = unet.get("error")
        fallback["fallback_from_ml_error"] = result.get("error")
        return fallback
    result["fallback_from_unet_error"] = unet.get("error")
    result["fallback_from_unet_pixel_coverage"] = unet.get("pixel_coverage")
    return result


def run_image_demo(image_file: str | None, question: str, sampling_rate: float | None, modality_hint: str, value_min: float | None, value_max: float | None, trace_method: str):
    steps: list[dict[str, Any]] = []
    if not image_file:
        return "Upload an image first.", "", {}, None, None
    question = (question or DEFAULT_IMAGE_QUESTION).strip()
    fallback = None if modality_hint == "auto" else modality_hint
    try:
        work_dir = Path(tempfile.mkdtemp(prefix="biosignalagent_demo_"))
        out_csv = work_dir / "digitized_signal.csv"
        ocr = _extract_optional_ocr_text(image_file)
        hint_modality, hint_token = _modality_from_text_hint(ocr.get("text", ""), Path(image_file).name)
        image_classifier = Signal_classify_modality_from_image(image_file)
        routed_by = "user_hint" if fallback else "image_classifier"
        if fallback is None and hint_modality:
            fallback = hint_modality
            routed_by = f"text_hint:{hint_token}"
        if fallback is None:
            fallback = image_classifier.get("predicted_modality")
        steps.append({
            "title": "Input and text understanding",
            "items": [
                f"OCR available: `{ocr.get('available')}`",
                f"OCR/title text preview: `{ocr.get('text', '')[:160]}`",
                f"Text hint modality: `{hint_modality}` via `{hint_token}`" if hint_modality else "No text modality hint found.",
            ],
        })
        steps.append({
            "title": "Modality routing",
            "items": [
                f"Image classifier prediction: `{image_classifier.get('predicted_modality')}` confidence `{image_classifier.get('confidence')}`",
                f"Final route: `{fallback}` ({routed_by})",
            ],
        })
        scale = Signal_estimate_image_scale(image_file, duration_s=None, use_ocr=True)
        sr = float(sampling_rate) if sampling_rate and sampling_rate > 0 else scale.get("sampling_rate")
        steps.append({
            "title": "Scale/OCR extraction",
            "items": [
                f"Sampling rate used: `{sr or 'default_for_tools_100Hz'}`",
                f"Scale confidence: `{scale.get('confidence')}`",
                f"OCR status: `{scale.get('ocr_status')}`",
            ],
        })
        color_digitized = _digitize_color_trace_image(
            image_file,
            sr,
            str(out_csv),
            value_min=value_min,
            value_max=value_max,
            panel_policy="auto",
        )
        if color_digitized.get("error"):
            digitized = _digitize_with_fallback(image_file, sr, str(out_csv), value_min, value_max, trace_method)
            digitizer_route = "fallback_unet_or_ml_or_dark_trace"
            if digitized.get("fallback_from_ml_error"):
                digitizer_route += " after ML model unavailable"
        else:
            digitized = color_digitized
            digitizer_route = "color_trace_panel_digitizer"
        steps.append({
            "title": "Panel and trace digitization",
            "items": [
                f"Digitizer: `{digitizer_route}`",
                f"Points: `{digitized.get('num_points')}`",
                f"Pixel coverage: `{digitized.get('pixel_coverage')}`",
                f"Selected box: `{digitized.get('selected_bbox')}`",
                f"Failure: `{digitized.get('error')}`" if digitized.get("error") else "Digitization completed.",
            ],
        })
        if digitized.get("error"):
            payload = {"ocr": ocr, "image_classifier": image_classifier, "scale": scale, "digitization": digitized, "disclaimer": DISCLAIMER}
            return _trajectory_md(steps), "Digitization failed. Inspect JSON details.", _jsonable(payload), None, None
        values, used_column = _read_signal(str(out_csv), "signal")
        planner = PlanningBioSignalAgent()
        plan = planner.plan(question, fallback)
        steps.append({"title": "Tool planning", "items": [f"Selected `{len(plan)}` tools for `{fallback}`: " + ", ".join(f"`{tool}`" for tool in plan)]})
        report = planner.run(question, str(out_csv), float(sr or 100.0), used_column, fallback)
        tool_items = []
        for call in report.get("tool_calls", [])[:8]:
            compact = _compact_tool_result(call.get("result", {}))
            tool_items.append(f"`{call.get('tool')}` -> keys {list(compact)[:8]}")
        steps.append({"title": "Tool execution", "items": tool_items or ["No tools executed."]})
        steps.append({"title": "Grounded report", "detail": "Report is grounded in the digitized signal and local tool outputs. Verify image calibration before use."})
        payload = {
            "image_path": image_file,
            "ocr": ocr,
            "image_classifier": image_classifier,
            "scale": scale,
            "digitization": digitized,
            "signal_report": report,
            "disclaimer": DISCLAIMER,
        }
        return _trajectory_md(steps), _short_report(report), _jsonable(payload), _plot_signal(values, sr, "Digitized waveform"), str(out_csv)
    except Exception as exc:
        steps.append({"title": "Failure", "status": "error", "detail": f"{type(exc).__name__}: {exc}"})
        return _trajectory_md(steps), f"Error: {type(exc).__name__}: {exc}", {"error": str(exc), "stage": "image_demo"}, None, None




def _file_path_from_gradio(file_obj: Any) -> str | None:
    if file_obj is None:
        return None
    if isinstance(file_obj, (str, Path)):
        return str(file_obj)
    if isinstance(file_obj, dict):
        for key in ("path", "name", "orig_name"):
            value = file_obj.get(key)
            if value:
                return str(value)
    value = getattr(file_obj, "name", None) or getattr(file_obj, "path", None)
    return str(value) if value else None


def _is_image_path(path: str) -> bool:
    return Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _is_csv_path(path: str) -> bool:
    return Path(path).suffix.lower() in {".csv", ".txt", ".tsv"}



def _pil_to_data_uri(image, fmt: str = "PNG", max_width: int = 780) -> str:
    from PIL import Image

    image = image.convert("RGB")
    if image.width > max_width:
        ratio = max_width / float(image.width)
        image = image.resize((max_width, max(1, int(image.height * ratio))), Image.BILINEAR)
    buf = BytesIO()
    save_kwargs = {"quality": 86} if fmt.upper() in {"JPEG", "JPG"} else {}
    image.save(buf, format=fmt, **save_kwargs)
    mime = "jpeg" if fmt.upper() in {"JPEG", "JPG"} else fmt.lower()
    return f"data:image/{mime};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _image_preview_card(title: str, image_path: str, caption: str = "") -> str:
    try:
        from PIL import Image

        uri = _pil_to_data_uri(Image.open(image_path), fmt="JPEG")
        return _visual_card(title, uri, caption)
    except Exception as exc:
        return f"<div><b>{title}</b><br><span style='color:#999'>preview unavailable: {exc}</span></div>"


def _visual_card(title: str, data_uri: str, caption: str = "") -> str:
    caption_html = f"<div style='color:#777;font-size:0.9em;margin-top:4px'>{caption}</div>" if caption else ""
    return (
        "<div style='border:1px solid #e3e3e3;border-radius:8px;padding:10px;margin:10px 0;background:#fff'>"
        f"<div style='font-weight:600;margin-bottom:6px'>{title}</div>"
        f"<img src='{data_uri}' style='max-width:100%;height:auto;border-radius:6px;border:1px solid #eee'/>"
        f"{caption_html}</div>"
    )


def _segmentation_overlay_card(image_path: str, probability_threshold: float = 0.65) -> str:
    try:
        import torch
        from PIL import Image

        checkpoint = torch.load(UNET_MODEL_PATH, map_location="cpu", weights_only=False)
        input_height, input_width = checkpoint.get("input_size", [160, 384])
        rgb, _crop, _size = _crop_rgb_image(image_path, 0, 0, 0, 0)
        height, width = rgb.shape[:2]
        resized = Image.fromarray(rgb).resize((int(input_width), int(input_height)), Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
        model = build_waveform_segmentation_model(checkpoint.get("model_type") or checkpoint.get("backbone"))
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
        mask_small = prob >= float(probability_threshold)
        mask = Image.fromarray((mask_small.astype(np.uint8) * 255), mode="L").resize((width, height), Image.NEAREST)
        raw_mask = np.asarray(mask, dtype=np.uint8) > 0
        selected_mask, area_info = select_waveform_mask_area(raw_mask, panel_policy="bottom", pad=max(3, int(height * 0.01)))
        bbox = (area_info.get("selected") or {})
        base = Image.fromarray(rgb).convert("RGBA")
        area_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        if bbox:
            from PIL import ImageDraw

            draw = ImageDraw.Draw(area_layer)
            xy = [int(bbox["x_min"]), int(bbox["y_min"]), int(bbox["x_max"]), int(bbox["y_max"])]
            draw.rectangle(xy, fill=(255, 193, 7, 45), outline=(255, 152, 0, 220), width=3)
        red = Image.new("RGBA", (width, height), (255, 35, 35, 0))
        alpha = selected_mask.astype(np.uint8) * 165
        red.putalpha(Image.fromarray(alpha, mode="L"))
        overlay = Image.alpha_composite(Image.alpha_composite(base, area_layer), red).convert("RGB")
        mask_fraction = float(raw_mask.mean()) if width and height else 0.0
        selected_fraction = float(selected_mask.mean()) if width and height else 0.0
        area_fraction = bbox.get("area_fraction") if bbox else None
        caption = (
            f"Amber area = selected mask region/panel used for digitization; red pixels = curve mask inside that area. "
            f"model={Path(UNET_MODEL_PATH).name}, threshold={probability_threshold}, "
            f"mask_fraction={mask_fraction:.4f}, selected_mask_fraction={selected_fraction:.4f}"
            + (f", area_fraction={float(area_fraction):.4f}." if area_fraction is not None else ".")
        )
        return _visual_card("Segmentation mask area", _pil_to_data_uri(overlay, fmt="JPEG"), caption)
    except Exception as exc:
        return f"<div style='border:1px solid #eee;border-radius:8px;padding:10px;margin:10px 0'><b>Segmentation overlay</b><br><span style='color:#999'>Unavailable: {exc}</span></div>"


def _signal_preview_card(csv_path: str | None, sampling_rate: float | None, title: str = "Digitized waveform") -> str:
    if not csv_path:
        return ""
    try:
        values, _column = _read_signal(csv_path, "signal")
        fig = _plot_signal(values, sampling_rate, title)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        return _visual_card(title, uri, f"Preview of the recovered one-dimensional signal from `{Path(csv_path).name}`.")
    except Exception as exc:
        return f"<div><b>{title}</b><br><span style='color:#999'>preview unavailable: {exc}</span></div>"


def _image_pipeline_visuals(payload: dict[str, Any]) -> list[str]:
    digitization = payload.get("digitization") or {}
    image_path = digitization.get("image_path") or payload.get("image_path")
    out_csv = digitization.get("out_csv")
    sampling_rate = digitization.get("sampling_rate")
    visuals: list[str] = []
    if image_path:
        visuals.append(_image_preview_card("Uploaded image", image_path, "Original input seen by the image pipeline."))
        visuals.append(_segmentation_overlay_card(image_path, probability_threshold=0.65))
    if out_csv:
        visuals.append(_signal_preview_card(out_csv, sampling_rate, "Digitized waveform preview"))
    return visuals


def _tool_call_card(name: str, args: dict[str, Any] | None = None, result: dict[str, Any] | None = None, icon: str = "🛠️") -> str:
    args = args or {}
    result = result or {}
    compact = _compact_tool_result(result, max_items=6) if result else {}
    arg_text = " ".join(f"{key}={json.dumps(_jsonable(value))}" for key, value in list(args.items())[:5])
    result_bits = []
    for key, value in list(compact.items())[:5]:
        if isinstance(value, dict) and "count" in value:
            result_bits.append(f"{key}: {value.get('count')}")
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result_bits.append(f"{key}: {value}")
    result_text = " | ".join(result_bits)
    if result_text:
        result_text = f"<br><span style='color:#777'>↳ {result_text}</span>"
    return (
        "<div style='border:1px solid #e0e0e0;border-radius:7px;padding:10px 12px;"
        "margin:8px 0;background:#fbfbfb'>"
        f"<span style='opacity:.75'>⌄</span> {icon} <b>{name}</b> "
        f"<span style='color:#999'>{arg_text}</span>{result_text}</div>"
    )


def _tool_cards_from_payload(kind: str, payload: dict[str, Any]) -> list[str]:
    cards: list[str] = []
    if kind == "waveform image":
        ocr = payload.get("ocr") or {}
        cards.append(_tool_call_card("Signal_read_image_text_ocr", {"image": "uploaded"}, {"available": ocr.get("available"), "text": (ocr.get("text") or "")[:120]}, "🧾"))
        cards.append(_tool_call_card("Signal_classify_modality_from_image_cnn", {"image": "uploaded"}, payload.get("image_classifier") or {}, "🧭"))
        cards.append(_tool_call_card("Signal_estimate_image_scale", {"use_ocr": True}, payload.get("scale") or {}, "📏"))
        cards.append(_tool_call_card("Signal_digitize_waveform_image", {"trace": "visible waveform"}, payload.get("digitization") or {}, "🔧"))
        report = payload.get("signal_report") or {}
    else:
        report = payload or {}
        classifier = payload.get("modality_classifier") if isinstance(payload, dict) else None
        if classifier:
            cards.append(_tool_call_card("Signal_classify_modality", {"input": "csv"}, classifier, "🧭"))
    for call in (report.get("tool_calls") or [])[:6]:
        cards.append(_tool_call_card(str(call.get("tool") or "tool"), {}, call.get("result") or {}, "🛠️"))
    return cards


def _human_answer_from_pipeline(kind: str, question: str, report: str, payload: dict[str, Any]) -> str:
    signal_report = payload.get("signal_report") if kind == "waveform image" else payload
    signal_report = signal_report or {}
    modality = signal_report.get("modality") or "unknown"
    plan = signal_report.get("plan") or []
    findings = signal_report.get("findings") or []
    cards = _tool_cards_from_payload(kind, payload if isinstance(payload, dict) else {})
    lines = [
        "I’ll break this down the way I would in an agent run: first identify what the input is, then call the smallest set of tools needed, and only then give the answer.",
        "",
    ]
    if kind == "waveform image":
        ocr = payload.get("ocr") or {}
        image_classifier = payload.get("image_classifier") or {}
        final_modality = modality
        lines.extend([
            f"The image looks like a `{final_modality}` workflow target. I also checked text/OCR hints because plot screenshots can confuse the image classifier.",
            "",
        ])
        if image_classifier.get("predicted_modality") and image_classifier.get("predicted_modality") != final_modality:
            lines.extend([
                f"One important detail: the image CNN leaned toward `{image_classifier.get('predicted_modality')}`, but the OCR/title context pointed to `{final_modality}`, so I used the text-grounded route for this case.",
                "",
            ])
        if ocr.get("text"):
            lines.extend([f"OCR picked up: `{str(ocr.get('text'))[:180]}`", ""])
        visuals = _image_pipeline_visuals(payload)
        if visuals:
            lines.append("Here are the visual checkpoints I used:")
            lines.extend(visuals)
            lines.append("")
    else:
        lines.extend([f"I read the uploaded CSV and routed it as `{modality}` before selecting tools.", ""])
    if cards:
        lines.append("I’m going to call these tools:")
        lines.extend(cards)
        lines.append("")
    if plan:
        lines.append("The final tool route is: " + ", ".join(f"`{tool}`" for tool in plan) + ".")
        lines.append("")
    lines.append("**Answer:**")
    if findings:
        for finding in findings[:8]:
            lines.append(f"- {finding}")
    else:
        lines.append(report or "I could not produce a confident finding from this input.")
    lines.extend([
        "",
        f"I would treat this as research/prototype output, not a clinical diagnosis. {DISCLAIMER}",
        "",
        "<details><summary>Raw compact trace</summary>",
        "",
        "```json",
        json.dumps(_jsonable(_compact_tool_result(payload, max_items=8)), indent=2)[:6000],
        "```",
        "</details>",
    ])
    return "\n".join(lines)


def _chat_progress(text: str, tool_cards: list[str] | None = None) -> str:
    lines = [text]
    if tool_cards:
        lines.extend(["", *tool_cards])
    return "\n".join(lines)


def biosignal_chat_response(
    message: str,
    history: list[dict[str, Any]] | None,
    upload: Any,
    sampling_rate: float,
    modality_hint: str,
    trace_method: str,
):
    question = (message or "").strip() or DEFAULT_CSV_QUESTION
    sampling_rate = float(sampling_rate or 250.0)
    modality_hint = modality_hint or "auto"
    trace_method = trace_method or "path"
    upload_path = _file_path_from_gradio(upload)

    yield "I’ll take this step by step. First I need to understand the input and decide which biosignal route makes sense."
    time.sleep(0.35)

    if upload_path and _is_image_path(upload_path):
        yield _chat_progress(
            "This is an image input, so I’ll inspect text/axes first and then recover the waveform before using signal tools.",
            [_tool_call_card("Signal_read_image_text_ocr", {"image": "uploaded"}, {}, "🧾"), _tool_call_card("Signal_classify_modality_from_image_cnn", {"image": "uploaded"}, {}, "🧭")],
        )
        time.sleep(0.35)
        yield _chat_progress(
            "Next I’ll estimate the plot scale and digitize the visible trace. If the image classifier disagrees with OCR/title text, I’ll explain which route I trust more.",
            [_tool_call_card("Signal_estimate_image_scale", {"use_ocr": True}, {}, "📏"), _tool_call_card("Signal_digitize_waveform_image", {"trace": trace_method}, {}, "🔧")],
        )
        trajectory, report, payload, _plot, _csv = run_image_demo(
            upload_path,
            question,
            sampling_rate,
            modality_hint,
            None,
            None,
            trace_method,
        )
        time.sleep(0.2)
        yield _human_answer_from_pipeline("waveform image", question, report, payload)
        return

    if upload_path and _is_csv_path(upload_path):
        yield _chat_progress(
            "This is a signal table, so I’ll read the numeric channel, classify the modality, and then choose the relevant tools.",
            [_tool_call_card("Signal_classify_modality", {"input": "csv", "sampling_rate": sampling_rate}, {}, "🧭")],
        )
        trajectory, report, payload, _plot = run_csv_demo(upload_path, question, sampling_rate, modality_hint, "")
        time.sleep(0.2)
        yield _human_answer_from_pipeline("signal CSV", question, report, payload)
        return

    planner = PlanningBioSignalAgent()
    guessed_modality, matched = _modality_from_text_hint(question, "")
    route = modality_hint if modality_hint != "auto" else (guessed_modality or "unknown")
    plan = planner.plan(question, None if route == "unknown" else route)
    cards = [_tool_call_card(tool, {"planned_only": True}, {}, "🛠️") for tool in plan[:6]]
    yield _chat_progress(
        "I can plan the tool route from your question, but I need an image or CSV before I can execute the tools and give grounded measurements.",
        cards,
    )

def summarize_tool_universe():
    try:
        schemas = load_tool_schemas()
        by_modality: dict[str, int] = {}
        by_evidence: dict[str, int] = {}
        for schema in schemas:
            modality = str(schema.get("modality") or "unknown")
            evidence = str(schema.get("evidence_level") or schema.get("evidence") or "unspecified")
            by_modality[modality] = by_modality.get(modality, 0) + 1
            by_evidence[evidence] = by_evidence.get(evidence, 0) + 1
        lines = ["## BioSignalToolUniverse", "", f"Total tool schemas: **{len(schemas)}**", "", "### Tools by modality"]
        for key, value in sorted(by_modality.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{key}`: {value}")
        lines.append("")
        lines.append("### Evidence levels")
        for key, value in sorted(by_evidence.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{key}`: {value}")
        lines.append("")
        lines.append(f"Safety note: {DISCLAIMER}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Could not load tool schemas: {exc}"


def build_demo() -> gr.Blocks:
    placeholder = """
# BioSignalAgent

Upload a biosignal image or CSV, then ask a question.

Try: Classify this waveform, digitize it, estimate heart rate/HRV, and explain which tools you used.
"""
    with gr.Blocks(title="BioSignalAgent", fill_height=True) as demo:
        gr.Markdown(
            "# BioSignalAgent\n"
            "A TxAgent-style biosignal tool-use assistant for routing, digitization, tool calls, and grounded research-use reports."
        )
        with gr.Accordion("Signal attachment", open=True):
            with gr.Row():
                bot_upload = gr.File(label="Signal image or CSV", file_types=[".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".csv", ".tsv", ".txt"], type="filepath", scale=2)
                bot_sampling_rate = gr.Number(label="Sampling rate if known (Hz)", value=250, scale=1)
                bot_modality = gr.Dropdown(label="Modality hint", choices=MODALITIES, value="auto", scale=1)
                bot_trace_method = gr.Dropdown(label="Trace extraction", choices=["median", "path", "lazy", "fragmented", "momentum", "full"], value="path", scale=1)
        chatbot = gr.Chatbot(
            label="BioSignalAgent",
            height=800,
            placeholder=placeholder,
            buttons=["copy", "copy_all"],
            layout="bubble",
            show_label=False,
            sanitize_html=False,
        )
        gr.ChatInterface(
            fn=biosignal_chat_response,
            chatbot=chatbot,
            fill_height=True,
            fill_width=True,
            stop_btn=True,
            textbox=gr.Textbox(placeholder="Ask BioSignalAgent to analyze the uploaded biosignal...", lines=2, container=False),
            additional_inputs=[bot_upload, bot_sampling_rate, bot_modality, bot_trace_method],
            examples=[
                ["Classify this waveform image, digitize the trace, then estimate heart rate and explain the tools you used."],
                ["Analyze this ECG signal for R peaks, HRV, rhythm quality, and limitations."],
                ["This image may be low resolution. Recover the signal if possible and tell me whether the result is reliable."],
            ],
            cache_examples=False,
        )

        with gr.Accordion("Advanced pipeline views", open=False):
            with gr.Tab("CSV signal"):
                with gr.Row():
                    csv_file = gr.File(label="Signal CSV", file_types=[".csv"], type="filepath")
                    with gr.Column():
                        csv_question = gr.Textbox(label="Question", value=DEFAULT_CSV_QUESTION, lines=3)
                        csv_sampling_rate = gr.Number(label="Sampling rate (Hz)", value=250)
                        csv_modality = gr.Dropdown(label="Modality hint", choices=MODALITIES, value="auto")
                        csv_column = gr.Textbox(label="Column name (optional)", value="")
                        csv_button = gr.Button("Run CSV analysis", variant="primary")
                csv_trajectory = gr.Markdown(label="Agent trajectory")
                csv_report = gr.Markdown(label="Report")
                csv_plot = gr.Plot(label="Signal preview")
                csv_json = gr.JSON(label="Tool trace JSON")
                csv_button.click(run_csv_demo, [csv_file, csv_question, csv_sampling_rate, csv_modality, csv_column], [csv_trajectory, csv_report, csv_json, csv_plot])

            with gr.Tab("Waveform image"):
                with gr.Row():
                    image_file = gr.Image(label="Waveform image", type="filepath")
                    with gr.Column():
                        image_question = gr.Textbox(label="Question", value=DEFAULT_IMAGE_QUESTION, lines=3)
                        image_sampling_rate = gr.Number(label="Sampling rate if known (Hz)", value=250)
                        image_modality = gr.Dropdown(label="Modality hint", choices=MODALITIES, value="auto")
                        value_min = gr.Number(label="Y-axis min if known", value=None)
                        value_max = gr.Number(label="Y-axis max if known", value=None)
                        trace_method = gr.Dropdown(label="Trace extraction", choices=["median", "path", "lazy", "fragmented", "momentum", "full"], value="path")
                        image_button = gr.Button("Run image pipeline", variant="primary")
                image_trajectory = gr.Markdown(label="Agent trajectory")
                image_report = gr.Markdown(label="Report")
                image_plot = gr.Plot(label="Digitized signal preview")
                image_json = gr.JSON(label="Pipeline JSON")
                digitized_file = gr.File(label="Digitized CSV")
                image_button.click(run_image_demo, [image_file, image_question, image_sampling_rate, image_modality, value_min, value_max, trace_method], [image_trajectory, image_report, image_json, image_plot, digitized_file])

            with gr.Tab("ToolUniverse"):
                gr.Markdown(summarize_tool_universe())
    return demo

demo = build_demo()

if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE", "0").lower() in {"1", "true", "yes"}
    demo.launch(server_name="0.0.0.0", server_port=7860, share=share)
