from __future__ import annotations

import base64
import json
import os
import re
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
from biosignal_agent.tools.digitize_unet_tools import Signal_digitize_waveform_image_unet, Signal_digitize_waveform_image_unet_all, UNET_MODEL_PATH, build_waveform_segmentation_model, select_waveform_mask_area, select_waveform_mask_areas
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


def _numbers_from_text(text: str) -> list[float]:
    values: list[float] = []
    for token in re.findall(r"[-+]?\d+(?:\.\d+)?", str(text or "")):
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _ordered_axis_values_from_tokens(tokens: list[dict[str, Any]]) -> list[float]:
    raw = [token for token in tokens if str(token.get("variant", "")).startswith("raw_")]
    source = raw or tokens
    source = sorted(source, key=lambda token: float(token.get("x", 0.0)))
    collapsed: list[dict[str, Any]] = []
    for token in source:
        try:
            value = float(token.get("value"))
            x = float(token.get("x"))
        except Exception:
            continue
        if collapsed and abs(x - float(collapsed[-1].get("x", 0.0))) <= 8:
            # Prefer multi-character tokens because OCR often splits 10 into 1.
            prev_text = str(collapsed[-1].get("text", ""))
            text = str(token.get("text", ""))
            if len(text) > len(prev_text):
                collapsed[-1] = token
            continue
        collapsed.append(token)
    values = [float(token.get("value")) for token in collapsed]
    if len(values) >= 4:
        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1) if values[i + 1] > values[i]]
        positive_diffs = [d for d in diffs if d > 0]
        if positive_diffs:
            step = float(np.median(positive_diffs))
            if step > 0 and values[-1] <= values[-2]:
                values[-1] = values[-2] + step
    return sorted(set(float(v) for v in values if np.isfinite(v)))


def _clean_y_tick_values(values: list[float]) -> list[float]:
    vals = sorted(set(float(v) for v in values if np.isfinite(v)))
    if len(vals) <= 3:
        return vals
    if any(abs(v) < 1e-6 for v in vals):
        positives = sorted(v for v in vals if v > 0)
        for pos in positives:
            if any(abs(v + pos) < max(1e-6, abs(pos) * 0.05) for v in vals if v < 0):
                return [-float(pos), 0.0, float(pos)]
    return vals


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except Exception:
            continue
    return None


def _ollama_image_b64(path: str, max_side: int = 1400) -> str:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    width, height = image.size
    scale = min(1.0, float(max_side) / float(max(width, height)))
    if scale < 1.0:
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _distill_axes_with_ollama(image_path: str, panels: list[dict[str, Any]], ocr_text: str = "") -> dict[str, Any]:
    api_key = os.getenv("OLLAMA_API_KEY", "").strip()
    model = os.getenv("OLLAMA_AXIS_MODEL", "qwen3.5:4b")
    panel_hint = [{
        "panel_index": p.get("panel_index"),
        "plot_area": p.get("plot_area"),
        "x_axis_crop": p.get("x_axis_crop"),
        "y_axis_crop": p.get("y_axis_crop"),
        "x_ocr_text": p.get("x_ocr_text"),
        "y_ocr_text": p.get("y_ocr_text"),
        "x_tick_values_ocr": p.get("x_tick_values"),
        "y_tick_values_ocr": p.get("y_tick_values"),
    } for p in panels[:4]]
    prompt = (
        "You are a vision axis-label reader for biomedical waveform plot screenshots. "
        "Look directly at the attached image first, then use noisy OCR output and plot/panel geometry only as hints. "
        "Infer the true x-axis and y-axis ticks for each panel/subplot. "
        "Do not analyze the waveform. Ignore labels like (a), (b), grid fragments, and OCR duplicates. "
        "Return ONLY valid JSON with schema: "
        "{\"panels\":[{\"panel_index\":1,\"x_tick_values\":[...],\"y_tick_values\":[...],"
        "\"x_min\":number|null,\"x_max\":number|null,\"y_min\":number|null,\"y_max\":number|null,"
        "\"x_units\":string|null,\"y_units\":string|null,\"confidence\":0-1,\"reason\":string}],\"global_confidence\":0-1}. "
        "Hints: x-axis tick labels often appear along the bottom of each panel, e.g. 0, 500, 1000. "
        "y-axis tick labels often appear at the left side, e.g. -2, 0, 2. If OCR says 1 near a right edge after 0,500, infer 1000 only when visually/positionally consistent. "
        f"\nNoisy OCR/global text: {ocr_text[:900]}\nPanel OCR and geometry hints: {json.dumps(_jsonable(panel_hint))[:2200]}"
    )
    bodies = [
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt, "images": [_ollama_image_b64(image_path)]}],
            "stream": False,
            "think": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": 900},
        }
    ]
    endpoints = [
        os.getenv("OLLAMA_HOST", "").rstrip("/") + "/api/chat" if os.getenv("OLLAMA_HOST") else "",
        "http://127.0.0.1:11434/api/chat",
        # Cloud endpoint intentionally omitted by default; use local Ollama or set OLLAMA_HOST.
    ]
    endpoints = [endpoint for endpoint in endpoints if endpoint]
    last_error = None
    for endpoint in endpoints:
        for body in bodies:
            try:
                import requests

                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=body,
                    timeout=float(os.getenv("OLLAMA_AXIS_TIMEOUT", "20")),
                )
                if response.status_code >= 400:
                    last_error = f"{endpoint}_http_{response.status_code}:{response.text[:180]}"
                    continue
                data = response.json()
                content = data.get("message", {}).get("content") or data.get("response") or ""
                parsed = _extract_json_object(content)
                if parsed:
                    parsed["available"] = True
                    parsed["status"] = "ok"
                    parsed["model"] = model
                    parsed["provider"] = "ollama"
                    parsed["endpoint"] = endpoint.replace(api_key, "***")
                    return parsed
                last_error = f"{endpoint}_no_json_in_response:{content[:220]!r}"
            except Exception as exc:
                last_error = f"{type(exc).__name__}:{str(exc)[:180]}"
                continue
    return {"available": False, "status": "failed", "error": last_error, "model": model, "provider": "ollama"}


def _merge_llm_axis_panels(panels: list[dict[str, Any]], llm_axis: dict[str, Any]) -> list[dict[str, Any]]:
    if not llm_axis.get("available"):
        return panels
    llm_panels = llm_axis.get("panels") or []
    by_index = {}
    for item in llm_panels:
        try:
            by_index[int(item.get("panel_index"))] = item
        except Exception:
            continue
    merged = []
    for idx, panel in enumerate(panels, 1):
        item = by_index.get(int(panel.get("panel_index") or idx)) or (llm_panels[idx - 1] if idx - 1 < len(llm_panels) and isinstance(llm_panels[idx - 1], dict) else {})
        panel = dict(panel)
        llm_conf = float(item.get("confidence") or llm_axis.get("global_confidence") or 0.0) if item else 0.0
        if item and llm_conf >= 0.35:
            x_ticks = item.get("x_tick_values") or []
            y_ticks = item.get("y_tick_values") or []
            if len(x_ticks) >= 2:
                panel["x_tick_values"] = [float(v) for v in x_ticks]
            if len(y_ticks) >= 2:
                panel["y_tick_values"] = [float(v) for v in y_ticks]
            x_min = item.get("x_min") if item.get("x_min") is not None else (min(panel.get("x_tick_values") or []) if panel.get("x_tick_values") else None)
            x_max = item.get("x_max") if item.get("x_max") is not None else (max(panel.get("x_tick_values") or []) if panel.get("x_tick_values") else None)
            y_min = item.get("y_min") if item.get("y_min") is not None else (min(panel.get("y_tick_values") or []) if panel.get("y_tick_values") else None)
            y_max = item.get("y_max") if item.get("y_max") is not None else (max(panel.get("y_tick_values") or []) if panel.get("y_tick_values") else None)
            if x_min is not None and x_max is not None and float(x_max) > float(x_min):
                panel["duration_s"] = float(x_max) - float(x_min)
                width = max(1, int(panel.get("plot_area", {}).get("x_max", 1)) - int(panel.get("plot_area", {}).get("x_min", 0)))
                panel["sampling_rate"] = float(width) / float(panel["duration_s"])
            if y_min is not None and y_max is not None and float(y_max) > float(y_min):
                panel["value_min"] = float(y_min)
                panel["value_max"] = float(y_max)
            panel["x_units"] = item.get("x_units")
            panel["y_units"] = item.get("y_units")
            panel["llm_axis_reason"] = item.get("reason")
            panel["llm_axis_confidence"] = llm_conf
            panel["axis_status"] = "calibrated_xy_llm" if panel.get("duration_s") and panel.get("value_min") is not None and panel.get("value_max") is not None else "partial_llm"
        merged.append(panel)
    return merged


def _ocr_axis_crop(image, psm: int = 6) -> dict[str, Any]:
    try:
        import pytesseract
        from PIL import Image, ImageOps

        base = image.convert("L")
        base = ImageOps.autocontrast(base.resize((max(1, base.width * 4), max(1, base.height * 4))))
        arr = np.asarray(base, dtype=np.uint8)
        threshold = int(np.percentile(arr, 72))
        binary = Image.fromarray(np.where(arr < threshold, 0, 255).astype(np.uint8))
        variants = [(base, psm, f"raw_psm{int(psm)}"), (base, 11, "raw_psm11"), (binary, psm, f"binary_psm{int(psm)}"), (binary, 11, "binary_psm11")]
        texts: list[str] = []
        tokens: list[dict[str, Any]] = []
        seen_token_keys: set[tuple[float, int, int]] = set()
        for crop, mode, variant_name in variants:
            config = f"--psm {int(mode)} -c tessedit_char_whitelist=0123456789.-+"
            try:
                text = pytesseract.image_to_string(crop, config=config)
                if text.strip():
                    texts.append(" ".join(text.split()))
            except Exception:
                text = ""
            try:
                data = pytesseract.image_to_data(crop, config=config, output_type=pytesseract.Output.DICT)
            except Exception:
                continue
            n = len(data.get("text", []))
            for idx in range(n):
                raw = str(data["text"][idx]).strip()
                nums = _numbers_from_text(raw)
                if not nums:
                    continue
                try:
                    conf = float(data.get("conf", [0] * n)[idx])
                except Exception:
                    conf = 0.0
                x = (float(data["left"][idx]) + float(data["width"][idx]) / 2.0) / 4.0
                y = (float(data["top"][idx]) + float(data["height"][idx]) / 2.0) / 4.0
                for value in nums:
                    key = (round(float(value), 3), int(round(x)), int(round(y)))
                    if key in seen_token_keys:
                        continue
                    seen_token_keys.add(key)
                    tokens.append({"value": float(value), "text": raw, "x": x, "y": y, "confidence": conf, "variant": variant_name})
        number_values = sorted(set(float(token["value"]) for token in tokens))
        if not number_values:
            number_values = sorted(set(_numbers_from_text(" ".join(texts))))
        return {"available": True, "text": " | ".join(texts)[:400], "numbers": number_values, "tokens": tokens}
    except Exception as exc:
        return {"available": False, "text": "", "numbers": [], "tokens": [], "error": str(exc)}


def _extract_plot_axes_ocr(image_path: str, max_panels: int = 8) -> dict[str, Any]:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    panels: list[dict[str, Any]] = []
    areas: list[tuple[np.ndarray, dict[str, Any]]] = []
    try:
        import torch

        checkpoint = torch.load(UNET_MODEL_PATH, map_location="cpu", weights_only=False)
        input_height, input_width = checkpoint.get("input_size", [160, 384])
        rgb = np.asarray(image)
        resized = image.resize((int(input_width), int(input_height)), Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
        num_classes = int(checkpoint.get("num_classes", 1))
        model = build_waveform_segmentation_model(checkpoint.get("model_type") or checkpoint.get("backbone"), out_channels=num_classes)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        with torch.no_grad():
            logits = model(tensor)
            if num_classes > 1:
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                mask_small = np.argmax(probs, axis=0) == 1
            else:
                mask_small = torch.sigmoid(logits)[0, 0].cpu().numpy() >= 0.65
        mask = Image.fromarray((mask_small.astype(np.uint8) * 255), mode="L").resize((width, height), Image.NEAREST)
        raw_mask = np.asarray(mask, dtype=np.uint8) > 0
        areas, _summary = select_waveform_mask_areas(raw_mask, pad=max(3, int(height * 0.01)))
    except Exception:
        areas = []

    if not areas:
        scale = Signal_estimate_image_scale(image_path, use_ocr=False)
        bbox = scale.get("plot_area") or {"left": 0, "right": width - 1, "top": 0, "bottom": height - 1}
        areas = [(np.zeros((height, width), dtype=bool), {"x_min": int(bbox.get("left", 0)), "x_max": int(bbox.get("right", width - 1)), "y_min": int(bbox.get("top", 0)), "y_max": int(bbox.get("bottom", height - 1)), "panel_index": 1, "selection_mode": "scale_plot_area_fallback"})]

    for idx, (_mask, bbox) in enumerate(areas[:max(1, int(max_panels))], 1):
        x_min = max(0, int(bbox.get("x_min", 0)))
        x_max = min(width, int(bbox.get("x_max", width)))
        y_min = max(0, int(bbox.get("y_min", 0)))
        y_max = min(height, int(bbox.get("y_max", height)))
        pad_x = max(20, int(width * 0.10))
        pad_y = max(18, int(height * 0.08))
        x_crop_box = (max(0, x_min - int(pad_x * 0.2)), max(0, y_max - int(pad_y * 0.35)), min(width, x_max + int(pad_x * 0.05)), min(height, y_max + int(pad_y * 1.25)))
        y_crop_box = (max(0, x_min - int(pad_x * 1.4)), max(0, y_min - int(pad_y * 0.25)), min(width, x_min + int(pad_x * 0.65)), min(height, y_max + int(pad_y * 0.25)))
        x_ocr = _ocr_axis_crop(image.crop(x_crop_box), psm=6)
        y_ocr = _ocr_axis_crop(image.crop(y_crop_box), psm=6)
        x_values = _ordered_axis_values_from_tokens(x_ocr.get("tokens", [])) or sorted(set(float(v) for v in x_ocr.get("numbers", []) if np.isfinite(v)))
        y_raw_tokens = [token for token in y_ocr.get("tokens", []) if str(token.get("variant", "")).startswith("raw_")]
        y_source_values = [float(token.get("value")) for token in y_raw_tokens if np.isfinite(float(token.get("value")))]
        y_values = _clean_y_tick_values(sorted(set(y_source_values if len(set(y_source_values)) >= 2 else [float(v) for v in y_ocr.get("numbers", []) if np.isfinite(v)])))
        duration_s = None
        if len(x_values) >= 2:
            span = float(max(x_values) - min(x_values))
            if span > 0:
                duration_s = span
        y_min_value = None
        y_max_value = None
        if len(y_values) >= 2:
            y_min_value = float(min(y_values))
            y_max_value = float(max(y_values))
        plot_width = max(1, x_max - x_min)
        sampling_rate = float(plot_width / duration_s) if duration_s and duration_s > 0 else None
        panels.append({
            "panel_index": int(bbox.get("panel_index") or idx),
            "plot_area": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
            "x_axis_crop": {"left": x_crop_box[0], "top": x_crop_box[1], "right": x_crop_box[2], "bottom": x_crop_box[3]},
            "y_axis_crop": {"left": y_crop_box[0], "top": y_crop_box[1], "right": y_crop_box[2], "bottom": y_crop_box[3]},
            "x_ocr_text": x_ocr.get("text", ""),
            "y_ocr_text": y_ocr.get("text", ""),
            "x_tick_values": x_values,
            "y_tick_values": y_values,
            "duration_s": duration_s,
            "sampling_rate": sampling_rate,
            "value_min": y_min_value,
            "value_max": y_max_value,
            "axis_status": "calibrated_xy" if sampling_rate and y_min_value is not None and y_max_value is not None else "partial_or_unreadable",
        })

    needs_llm = any(
        len(panel.get("x_tick_values") or []) < 2 or len(panel.get("y_tick_values") or []) < 2
        for panel in panels
    )
    llm_axis = {"available": False, "status": "not_needed"}
    if needs_llm or os.getenv("BIOSIGNAL_ALWAYS_LLM_AXIS", "0").lower() in {"1", "true", "yes"}:
        full_image_ocr = _ocr_axis_crop(image, psm=11)
        llm_ocr_text = (full_image_ocr.get("text", "") + " " + " ".join(
            str(panel.get(key, ""))
            for panel in panels
            for key in ("x_ocr_text", "y_ocr_text")
        )).strip()
        llm_axis = _distill_axes_with_ollama(image_path, panels, ocr_text=llm_ocr_text)
        panels = _merge_llm_axis_panels(panels, llm_axis)

    primary = panels[0] if panels else {}
    confidence = 0.75 if str(primary.get("axis_status", "")).startswith("calibrated_xy") else 0.55 if primary.get("duration_s") else 0.35 if panels else 0.0
    if llm_axis.get("available") and llm_axis.get("global_confidence") is not None:
        confidence = max(confidence, min(0.85, float(llm_axis.get("global_confidence") or 0.0)))
    return {
        "tool": "Signal_extract_plot_axes_ocr",
        "image_path": str(image_path),
        "num_panels": len(panels),
        "panels": panels,
        "duration_s": primary.get("duration_s"),
        "sampling_rate": primary.get("sampling_rate"),
        "value_min": primary.get("value_min"),
        "value_max": primary.get("value_max"),
        "llm_axis_status": llm_axis.get("status"),
        "llm_axis_available": bool(llm_axis.get("available")),
        "llm_axis_model": llm_axis.get("model"),
        "llm_axis_provider": llm_axis.get("provider"),
        "llm_axis": llm_axis,
        "confidence": confidence,
        "disclaimer": "OCR/LLM axis extraction is heuristic; verify tick labels and units before using calibrated measurements.",
    }


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


def _signal_from_selected_color_mask(
    selected: np.ndarray,
    image_path: str,
    sampling_rate: float | None,
    out_csv: str,
    value_min: float | None,
    value_max: float | None,
    component_info: dict[str, Any],
    raw_fraction: float,
    panel_index: int | None = None,
) -> dict[str, Any]:
    ys, xs = np.nonzero(selected)
    if len(xs) == 0:
        return {"tool": "Signal_digitize_plot_image_color_trace", "error": "empty selected trace mask", "confidence": 0.0}
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
    result = {
        "tool": "Signal_digitize_plot_image_color_trace",
        "image_path": str(image_path),
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
    if panel_index is not None:
        result["panel_index"] = int(panel_index)
    return result


def _select_trace_components(mask: np.ndarray, max_panels: int = 8) -> tuple[list[tuple[np.ndarray, dict[str, Any]]], dict[str, Any]]:
    from scipy import ndimage

    h, w = mask.shape
    row_counts = mask.sum(axis=1)
    threshold = max(2, int(w * 0.004))
    active_rows = row_counts >= threshold
    active_rows = ndimage.binary_dilation(active_rows, iterations=max(2, int(h * 0.012)))
    row_labels, row_num = ndimage.label(active_rows)
    bands = []
    for label_id in range(1, row_num + 1):
        rows = np.flatnonzero(row_labels == label_id)
        if len(rows) < max(4, int(h * 0.02)):
            continue
        y1, y2 = int(rows.min()), int(rows.max()) + 1
        band = mask[y1:y2, :]
        ys, xs = np.nonzero(band)
        if len(xs) < 20:
            continue
        x_span = int(xs.max() - xs.min() + 1)
        if x_span < max(20, int(w * 0.08)):
            continue
        bands.append({
            "y_min": y1,
            "y_max": y2,
            "x_min": int(xs.min()),
            "x_max": int(xs.max()),
            "pixels": int(len(xs)),
            "x_span": x_span,
            "y_span": int(y2 - y1),
            "y_center": float((y1 + y2) / 2.0),
        })

    selected: list[tuple[np.ndarray, dict[str, Any]]] = []
    if len(bands) >= 2:
        for idx, band in enumerate(sorted(bands, key=lambda b: b["y_center"])[:max(1, int(max_panels))], 1):
            component_mask = np.zeros_like(mask, dtype=bool)
            component_mask[int(band["y_min"]):int(band["y_max"]), :] = mask[int(band["y_min"]):int(band["y_max"]), :]
            info = {"num_components": len(bands), "selected": band, "panel_policy": "all_row_bands", "panel_index": idx}
            selected.append((component_mask, info))
        return selected, {"num_components": len(bands), "candidate_components": bands, "selection_mode": "row_bands"}

    labels, num = ndimage.label(mask)
    if num <= 0:
        return [], {"num_components": 0, "candidate_components": bands, "selection_mode": "none"}
    components = []
    for label_id in range(1, num + 1):
        ys, xs = np.nonzero(labels == label_id)
        if len(xs) < 20:
            continue
        x_span = int(xs.max() - xs.min() + 1)
        y_span = int(ys.max() - ys.min() + 1)
        if x_span < max(20, int(w * 0.08)):
            continue
        components.append({
            "label": int(label_id),
            "pixels": int(len(xs)),
            "x_span": x_span,
            "y_span": y_span,
            "x_min": int(xs.min()),
            "x_max": int(xs.max()),
            "y_min": int(ys.min()),
            "y_max": int(ys.max()),
            "y_center": float(np.mean(ys)),
        })
    components = sorted(components, key=lambda c: (c["y_center"], c["x_min"]))
    for idx, comp in enumerate(components[:max(1, int(max_panels))], 1):
        component_mask = labels == int(comp["label"])
        info = {"num_components": int(num), "selected": comp, "panel_policy": "all_components", "panel_index": idx}
        selected.append((component_mask, info))
    return selected, {"num_components": int(num), "candidate_components": components[:max_panels], "candidate_bands": bands, "selection_mode": "components"}


def _digitize_color_trace_image_all(
    image_path: str,
    sampling_rate: float | None,
    out_csv: str,
    value_min: float | None = None,
    value_max: float | None = None,
    max_panels: int = 8,
) -> dict[str, Any]:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image)
    mask = _blue_trace_mask(rgb)
    raw_fraction = float(mask.mean()) if mask.size else 0.0
    components, component_summary = _select_trace_components(mask, max_panels=max_panels)
    if not components:
        return {"tool": "Signal_digitize_plot_image_color_trace_all", "error": "no blue/color trace component detected", "confidence": 0.0, "raw_mask_fraction": raw_fraction, "component_info": component_summary}
    base_out = Path(out_csv)
    signals = []
    for idx, (selected, component_info) in enumerate(components, 1):
        panel_out = base_out if len(components) == 1 else base_out.with_name(f"{base_out.stem}_panel_{idx:02d}{base_out.suffix}")
        panel = _signal_from_selected_color_mask(selected, image_path, sampling_rate, str(panel_out), value_min, value_max, component_info, raw_fraction, panel_index=idx)
        signals.append(panel)
    ok = [item for item in signals if not item.get("error")]
    if not ok:
        result = signals[0]
        result["tool"] = "Signal_digitize_plot_image_color_trace_all"
        result["signals"] = signals
        result["num_panels"] = len(signals)
        return result
    primary = ok[0]
    return {
        **primary,
        "tool": "Signal_digitize_plot_image_color_trace_all",
        "signals": signals,
        "num_panels": len(signals),
        "panel_csvs": [item.get("out_csv") for item in signals if item.get("out_csv")],
        "multi_panel": len(signals) > 1,
        "component_summary": component_summary,
        "confidence": float(np.mean([float(item.get("confidence") or 0.0) for item in ok])),
        "method": "color_trace_all_panel_digitizer",
    }


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


def _axis_panel_by_index(axis_ocr: dict[str, Any]) -> dict[int, dict[str, Any]]:
    panels = axis_ocr.get("panels") or []
    out: dict[int, dict[str, Any]] = {}
    for idx, panel in enumerate(panels, 1):
        try:
            out[int(panel.get("panel_index") or idx)] = panel
        except Exception:
            out[idx] = panel
    return out


def _signal_selected_bbox(signal: dict[str, Any]) -> dict[str, Any]:
    if signal.get("selected_bbox"):
        return signal.get("selected_bbox") or {}
    area = signal.get("selected_mask_area") or {}
    return area.get("selected") or {}


def _rewrite_csv_with_axis_calibration(csv_path: str, signal: dict[str, Any], panel_axis: dict[str, Any]) -> bool:
    if not csv_path or not Path(csv_path).exists():
        return False
    plot_area = panel_axis.get("plot_area") or {}
    bbox = _signal_selected_bbox(signal)
    required = ["x_min", "x_max", "y_min", "y_max"]
    if not all(k in plot_area for k in required) or not all(k in bbox for k in required):
        return False
    x_ticks = [float(v) for v in (panel_axis.get("x_tick_values") or []) if v is not None]
    y_ticks = [float(v) for v in (panel_axis.get("y_tick_values") or []) if v is not None]
    if len(x_ticks) < 2 or len(y_ticks) < 2:
        return False
    x0, x1 = min(x_ticks), max(x_ticks)
    y0, y1 = min(y_ticks), max(y_ticks)
    plot_x0, plot_x1 = float(plot_area["x_min"]), float(plot_area["x_max"])
    plot_y0, plot_y1 = float(plot_area["y_min"]), float(plot_area["y_max"])
    bbox_x0, bbox_x1 = float(bbox["x_min"]), float(bbox["x_max"])
    bbox_y0, bbox_y1 = float(bbox["y_min"]), float(bbox["y_max"])
    if plot_x1 <= plot_x0 or plot_y1 <= plot_y0 or bbox_x1 <= bbox_x0 or bbox_y1 <= bbox_y0 or y1 <= y0:
        return False
    frame = pd.read_csv(csv_path)
    if "signal" not in frame.columns or len(frame) < 2:
        return False
    values = frame["signal"].to_numpy(dtype=float)
    # Recover the digitized trace's relative y position inside its selected mask bbox,
    # then remap that absolute pixel position through the full plot-area y-axis.
    source_y0 = float(panel_axis.get("value_min") if panel_axis.get("value_min") is not None else y0)
    source_y1 = float(panel_axis.get("value_max") if panel_axis.get("value_max") is not None else y1)
    if source_y1 <= source_y0:
        return False
    normalized = 2.0 * (values - source_y0) / (source_y1 - source_y0) - 1.0
    y_rel = (1.0 - normalized) * max(1.0, bbox_y1 - bbox_y0) / 2.0
    y_abs = bbox_y0 + y_rel
    calibrated_values = y1 - ((y_abs - plot_y0) / (plot_y1 - plot_y0)) * (y1 - y0)
    n = len(values)
    x_abs = bbox_x0 + np.linspace(0.0, max(0.0, bbox_x1 - bbox_x0), n)
    calibrated_time = x0 + ((x_abs - plot_x0) / (plot_x1 - plot_x0)) * (x1 - x0)
    frame = pd.DataFrame({"time_s": calibrated_time, "signal": calibrated_values})
    frame.to_csv(csv_path, index=False)
    signal["sampling_rate"] = float((n - 1) / max(1e-12, calibrated_time[-1] - calibrated_time[0])) if n > 1 else None
    signal["value_min"] = float(np.nanmin(calibrated_values)) if n else None
    signal["value_max"] = float(np.nanmax(calibrated_values)) if n else None
    signal["time_min"] = float(np.nanmin(calibrated_time)) if n else None
    signal["time_max"] = float(np.nanmax(calibrated_time)) if n else None
    signal["scale"] = "axis_calibrated_from_vlm_plot_area"
    signal["axis_calibration"] = {
        "panel_index": panel_axis.get("panel_index"),
        "x_tick_values": x_ticks,
        "y_tick_values": y_ticks,
        "plot_area": plot_area,
        "trace_bbox": bbox,
    }
    return True


def _apply_axis_calibration_to_digitization(digitized: dict[str, Any], axis_ocr: dict[str, Any]) -> dict[str, Any]:
    if not digitized or digitized.get("error"):
        return digitized
    by_index = _axis_panel_by_index(axis_ocr)
    signals = digitized.get("signals") or [digitized]
    applied = 0
    for idx, signal in enumerate(signals, 1):
        panel_idx = int(signal.get("panel_index") or idx)
        panel_axis = by_index.get(panel_idx) or by_index.get(idx) or {}
        if _rewrite_csv_with_axis_calibration(str(signal.get("out_csv") or ""), signal, panel_axis):
            applied += 1
    if signals and signals[0] is not digitized:
        primary = signals[0]
        for key in ("sampling_rate", "value_min", "value_max", "time_min", "time_max", "scale", "axis_calibration"):
            if key in primary:
                digitized[key] = primary[key]
    digitized["axis_calibration_applied"] = int(applied)
    digitized["axis_calibration_source"] = axis_ocr.get("llm_axis_provider") or "ocr"
    return digitized


def _digitization_result_is_usable(result: dict[str, Any]) -> bool:
    if not result or result.get("error"):
        return False
    signals = result.get("signals") or [result]
    ok = [item for item in signals if not item.get("error") and int(item.get("num_points") or 0) >= 30]
    if not ok:
        return False
    coverage = [float(item.get("pixel_coverage") or 0.0) for item in ok]
    return max(coverage or [0.0]) >= 0.08


def _digitize_with_fallback(image_path: str, sampling_rate: float | None, out_csv: str, value_min: float | None, value_max: float | None, trace_method: str):
    unet = Signal_digitize_waveform_image_unet_all(
        image_path=image_path,
        sampling_rate=sampling_rate,
        out_csv=out_csv,
        value_min=value_min,
        value_max=value_max,
        probability_threshold=0.65,
        trace_method=trace_method,
        smooth_window=3,
        max_panels=8,
    )
    if _digitization_result_is_usable(unet):
        unet["fallback_priority"] = "segmentation_unet_first"
        return unet

    color = _digitize_color_trace_image_all(
        image_path,
        sampling_rate,
        out_csv,
        value_min=value_min,
        value_max=value_max,
        max_panels=8,
    )
    if _digitization_result_is_usable(color):
        color["fallback_priority"] = "color_trace_after_unet"
        color["fallback_from_unet_error"] = unet.get("error")
        color["fallback_from_unet_pixel_coverage"] = unet.get("pixel_coverage")
        return color

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
        fallback["fallback_from_color_error"] = color.get("error")
        fallback["fallback_from_ml_error"] = result.get("error")
        return fallback
    result["fallback_from_unet_error"] = unet.get("error")
    result["fallback_from_unet_pixel_coverage"] = unet.get("pixel_coverage")
    result["fallback_from_color_error"] = color.get("error")
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
        axis_ocr = _extract_plot_axes_ocr(image_file)
        axis_value_min = axis_ocr.get("value_min")
        axis_value_max = axis_ocr.get("value_max")
        calibrated_value_min = value_min if value_min is not None else axis_value_min
        calibrated_value_max = value_max if value_max is not None else axis_value_max
        user_sr = float(sampling_rate) if sampling_rate and sampling_rate > 0 else None
        axis_sr = axis_ocr.get("sampling_rate")
        # In the image demo, axis ticks are more trustworthy than the default UI
        # sampling-rate value because plot screenshots often use arbitrary x-scales.
        sr = axis_sr or user_sr or scale.get("sampling_rate")
        axis_panel_summaries = []
        for panel in (axis_ocr.get("panels") or [])[:4]:
            axis_panel_summaries.append(
                f"panel {panel.get('panel_index')}: x_ticks={panel.get('x_tick_values')}, y_ticks={panel.get('y_tick_values')}, "
                f"duration={panel.get('duration_s')}, value_range=({panel.get('value_min')}, {panel.get('value_max')})"
            )
        steps.append({
            "title": "Scale/OCR extraction",
            "items": [
                f"Sampling rate used: `{sr or 'default_for_tools_100Hz'}`",
                f"Sampling-rate source: `{'user_input' if user_sr else 'axis_or_scale_ocr'}`",
                f"Axis OCR confidence: `{axis_ocr.get('confidence')}`",
                f"Scale confidence: `{scale.get('confidence')}`",
                f"Legacy x-axis OCR status: `{scale.get('ocr_status')}`",
                *axis_panel_summaries,
            ],
        })
        digitized = _digitize_with_fallback(image_file, sr, str(out_csv), calibrated_value_min, calibrated_value_max, trace_method)
        digitized = _apply_axis_calibration_to_digitization(digitized, axis_ocr)
        sr = digitized.get("sampling_rate") or sr
        digitizer_route = str(digitized.get("fallback_priority") or digitized.get("method") or digitized.get("tool") or "segmentation_or_fallback_digitizer")
        if digitized.get("fallback_from_ml_error"):
            digitizer_route += " after ML model unavailable"
        steps.append({
            "title": "Panel and trace digitization",
            "items": [
                f"Digitizer: `{digitizer_route}`",
                f"Panels processed: `{digitized.get('num_panels', 1)}`",
                f"Points in primary panel: `{digitized.get('num_points')}`",
                f"Pixel coverage: `{digitized.get('pixel_coverage')}`",
                f"Selected box: `{digitized.get('selected_bbox') or (digitized.get('selected_mask_area') or {}).get('selected')}`",
                f"Failure: `{digitized.get('error')}`" if digitized.get("error") else "Digitization completed.",
            ],
        })
        if digitized.get("error"):
            payload = {"image_path": image_file, "ocr": ocr, "image_classifier": image_classifier, "scale": scale, "axis_ocr": axis_ocr, "digitization": digitized, "disclaimer": DISCLAIMER}
            return _trajectory_md(steps), "Digitization failed. Inspect JSON details.", _jsonable(payload), None, None
        primary_csv = digitized.get("out_csv") or str(out_csv)
        values, used_column = _read_signal(str(primary_csv), "signal")
        planner = PlanningBioSignalAgent()
        plan = planner.plan(question, fallback)
        steps.append({"title": "Tool planning", "items": [f"Selected `{len(plan)}` tools for `{fallback}`: " + ", ".join(f"`{tool}`" for tool in plan)]})
        report = planner.run(question, str(primary_csv), float(sr or 100.0), used_column, fallback)
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
            "axis_ocr": axis_ocr,
            "digitization": digitized,
            "signal_report": report,
            "disclaimer": DISCLAIMER,
        }
        return _trajectory_md(steps), _short_report(report), _jsonable(payload), _plot_csv_signal(str(primary_csv), sr, "Digitized waveform", y_limits=_axis_y_limits_from_signal(digitized)), str(primary_csv)
    except Exception as exc:
        steps.append({"title": "Failure", "status": "error", "detail": f"{type(exc).__name__}: {exc}"})
        payload = {"image_path": image_file, "error": str(exc), "error_type": type(exc).__name__, "stage": "image_demo", "disclaimer": DISCLAIMER}
        return _trajectory_md(steps), f"Error: {type(exc).__name__}: {exc}", payload, None, None




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


def _plot_csv_signal(csv_path: str, sampling_rate: float | None, title: str, y_limits: tuple[float, float] | None = None):
    frame = pd.read_csv(csv_path)
    values = frame["signal"].to_numpy(dtype=float) if "signal" in frame.columns else frame.select_dtypes(include=[np.number]).iloc[:, -1].to_numpy(dtype=float)
    if "time_s" in frame.columns:
        x = frame["time_s"].to_numpy(dtype=float)
        xlabel = "Time (s)"
    elif sampling_rate and sampling_rate > 0:
        x = np.arange(len(values), dtype=float) / float(sampling_rate)
        xlabel = "Time (s)"
    else:
        x = np.arange(len(values), dtype=float)
        xlabel = "Sample"
    fig, ax = plt.subplots(figsize=(6, 2.3))
    ax.plot(x, values, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Amplitude")
    if len(x):
        ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
    if y_limits is not None and y_limits[1] > y_limits[0]:
        pad = 0.03 * (float(y_limits[1]) - float(y_limits[0]))
        ax.set_ylim(float(y_limits[0]) - pad, float(y_limits[1]) + pad)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _axis_y_limits_from_signal(signal: dict[str, Any]) -> tuple[float, float] | None:
    calib = signal.get("axis_calibration") or {}
    ticks = [float(v) for v in (calib.get("y_tick_values") or []) if v is not None]
    if len(ticks) >= 2:
        return min(ticks), max(ticks)
    return None


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
        num_classes = int(checkpoint.get("num_classes", 1))
        model = build_waveform_segmentation_model(checkpoint.get("model_type") or checkpoint.get("backbone"), out_channels=num_classes)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        with torch.no_grad():
            logits = model(tensor)
            if num_classes > 1:
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred_class = np.argmax(probs, axis=0)
                target_small = pred_class == 1
                distractor_small = pred_class == 2
            else:
                prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
                target_small = prob >= float(probability_threshold)
                distractor_small = np.zeros_like(target_small, dtype=bool)
        target_mask_img = Image.fromarray((target_small.astype(np.uint8) * 255), mode="L").resize((width, height), Image.NEAREST)
        distractor_mask_img = Image.fromarray((distractor_small.astype(np.uint8) * 255), mode="L").resize((width, height), Image.NEAREST)
        raw_mask = np.asarray(target_mask_img, dtype=np.uint8) > 0
        distractor_mask = np.asarray(distractor_mask_img, dtype=np.uint8) > 0
        selected_areas, area_summary = select_waveform_mask_areas(raw_mask, pad=max(3, int(height * 0.01)))
        selected_mask = np.zeros_like(raw_mask, dtype=bool)
        for area_mask, _bbox in selected_areas:
            selected_mask |= area_mask
        bbox = selected_areas[0][1] if selected_areas else {}
        base = Image.fromarray(rgb).convert("RGBA")
        distractor_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        distractor_alpha = distractor_mask.astype(np.uint8) * 85
        distractor_layer.putalpha(Image.fromarray(distractor_alpha, mode="L"))
        # Color must be applied after alpha for PIL RGBA layers.
        distractor_arr = np.asarray(distractor_layer).copy()
        distractor_arr[..., 0] = 0
        distractor_arr[..., 1] = 145
        distractor_arr[..., 2] = 255
        distractor_layer = Image.fromarray(distractor_arr, mode="RGBA")
        target_all_layer = Image.new("RGBA", (width, height), (255, 35, 35, 0))
        target_all_layer.putalpha(Image.fromarray(raw_mask.astype(np.uint8) * 70, mode="L"))
        area_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        if selected_areas:
            from PIL import ImageDraw

            draw = ImageDraw.Draw(area_layer)
            for _area_mask, area_bbox in selected_areas:
                xy = [int(area_bbox["x_min"]), int(area_bbox["y_min"]), int(area_bbox["x_max"]), int(area_bbox["y_max"])]
                draw.rectangle(xy, fill=(255, 193, 7, 38), outline=(255, 152, 0, 220), width=3)
                draw.text((xy[0] + 4, xy[1] + 4), f"panel {area_bbox.get('panel_index', '?')}", fill=(180, 90, 0, 255))
        selected_layer = Image.new("RGBA", (width, height), (255, 35, 35, 0))
        selected_layer.putalpha(Image.fromarray(selected_mask.astype(np.uint8) * 185, mode="L"))
        overlay = Image.alpha_composite(base, distractor_layer)
        overlay = Image.alpha_composite(overlay, target_all_layer)
        overlay = Image.alpha_composite(overlay, area_layer)
        overlay = Image.alpha_composite(overlay, selected_layer).convert("RGB")
        target_fraction = float(raw_mask.mean()) if width and height else 0.0
        selected_fraction = float(selected_mask.mean()) if width and height else 0.0
        distractor_fraction = float(distractor_mask.mean()) if width and height else 0.0
        area_fractions = [float(area_bbox.get("area_fraction") or 0.0) for _area_mask, area_bbox in selected_areas]
        caption = (
            "Blue = non-target/distractor class such as axes, text, or grid; "
            "light red = all target-trace pixels; dark red = target pixels inside every selected digitization area; "
            "amber boxes = all plot areas that will be processed. "
            f"model={Path(UNET_MODEL_PATH).name}, classes={num_classes}, threshold={probability_threshold}, panels={len(selected_areas)}, "
            f"target_fraction={target_fraction:.4f}, selected_target_fraction={selected_fraction:.4f}, distractor_fraction={distractor_fraction:.4f}"
            + (f", area_fractions={[round(v, 4) for v in area_fractions]}." if area_fractions else ".")
        )
        return _visual_card("Segmentation classes and all selected areas", _pil_to_data_uri(overlay, fmt="JPEG"), caption)
    except Exception as exc:
        return f"<div style='border:1px solid #eee;border-radius:8px;padding:10px;margin:10px 0'><b>Segmentation overlay</b><br><span style='color:#999'>Unavailable: {exc}</span></div>"


def _signal_preview_card(csv_path: str | None, sampling_rate: float | None, title: str = "Digitized waveform", y_limits: tuple[float, float] | None = None) -> str:
    if not csv_path:
        return ""
    try:
        fig = _plot_csv_signal(csv_path, sampling_rate, title, y_limits=y_limits)
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
    signals = digitization.get("signals") or []
    if signals:
        for idx, signal in enumerate(signals[:8], 1):
            panel_csv = signal.get("out_csv")
            panel_sr = signal.get("sampling_rate") or sampling_rate
            title = f"Digitized waveform preview - panel {signal.get('panel_index') or idx}"
            caption_bits = []
            if signal.get("selected_bbox"):
                caption_bits.append(f"bbox={signal.get('selected_bbox')}")
            if signal.get("selected_mask_area"):
                selected = (signal.get("selected_mask_area") or {}).get("selected")
                if selected:
                    caption_bits.append(f"area={selected}")
            caption = "; ".join(caption_bits) or f"Recovered signal from `{Path(panel_csv or '').name}`."
            card = _signal_preview_card(panel_csv, panel_sr, title, y_limits=_axis_y_limits_from_signal(signal))
            if caption and "</div>" in card:
                card = card.replace("</div>", f"<div style='color:#777;font-size:0.85em;margin-top:4px'>{caption}</div></div>", 1)
            visuals.append(card)
    elif out_csv:
        visuals.append(_signal_preview_card(out_csv, sampling_rate, "Digitized waveform preview", y_limits=_axis_y_limits_from_signal(digitization)))
    return visuals


def _tool_call_card(name: str, args: dict[str, Any] | None = None, result: dict[str, Any] | None = None, icon: str = "T") -> str:
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
    result_html = f"<div class='bs-tool-result'>{result_text}</div>" if result_text else ""
    icon_text = re.sub(r"[^A-Za-z0-9]", "", str(icon or "T"))[:2].upper() or "T"
    return (
        "<div class='bs-tool-card'>"
        "<div class='bs-tool-row'>"
        f"<span class='bs-tool-icon'>{icon_text}</span>"
        f"<span class='bs-tool-name'>{name}</span>"
        f"<span class='bs-tool-args'>{arg_text}</span>"
        "</div>"
        f"{result_html}"
        "</div>"
    )


def _tool_cards_from_payload(kind: str, payload: dict[str, Any]) -> list[str]:
    cards: list[str] = []
    if kind == "waveform image":
        ocr = payload.get("ocr") or {}
        cards.append(_tool_call_card("Signal_read_image_text_ocr", {"image": "uploaded"}, {"available": ocr.get("available"), "text": (ocr.get("text") or "")[:120]}, "🧾"))
        cards.append(_tool_call_card("Signal_classify_modality_from_image_cnn", {"image": "uploaded"}, payload.get("image_classifier") or {}, "🧭"))
        cards.append(_tool_call_card("Signal_estimate_image_scale", {"use_ocr": True}, payload.get("scale") or {}, "📏"))
        cards.append(_tool_call_card("Signal_extract_plot_axes_ocr", {"x_axis": True, "y_axis": True}, payload.get("axis_ocr") or {}, "📐"))
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
        "I’ll inspect the image first, then show the tool calls I used before giving the result.",
        "",
    ]
    if kind == "waveform image":
        ocr = payload.get("ocr") or {}
        image_classifier = payload.get("image_classifier") or {}
        classifier_modality = image_classifier.get("predicted_modality")
        final_modality = modality if modality and modality != "unknown" else classifier_modality
        if final_modality and final_modality != "unknown":
            lines.extend([f"This looks like a `{final_modality}` waveform image. I’ll still check OCR/axis text because screenshots can fool the image classifier.", ""])
        else:
            lines.extend(["I can see this is a waveform plot image, but the modality classifier did not produce a confident label yet. I’ll continue with image OCR, axis reading, and waveform digitization rather than pretending the route is known.", ""])
        if payload.get("error"):
            lines.extend([f"The run stopped during `{payload.get('stage', 'image_demo')}` with `{payload.get('error_type', 'error')}`: {payload.get('error')}", "I’ll still show the uploaded image when available so we can debug the failure visually.", ""])
        if final_modality and image_classifier.get("predicted_modality") and image_classifier.get("predicted_modality") != final_modality:
            lines.extend([
                f"One important detail: the image CNN leaned toward `{image_classifier.get('predicted_modality')}`, but the OCR/title context pointed to `{final_modality}`, so I used the text-grounded route for this case.",
                "",
            ])
        if ocr.get("text"):
            lines.extend([f"OCR picked up: `{str(ocr.get('text'))[:180]}`", ""])
        axis_ocr = payload.get("axis_ocr") or {}
        if axis_ocr.get("panels"):
            first_axis = axis_ocr.get("panels", [{}])[0]
            lines.extend([
                "I also tried to read the plot axes before digitizing, because otherwise the recovered signal stays in pixel/normalized units.",
                f"For panel 1, I read x ticks `{first_axis.get('x_tick_values')}` and y ticks `{first_axis.get('y_tick_values')}`; inferred duration `{first_axis.get('duration_s')}` and y range `({first_axis.get('value_min')}, {first_axis.get('value_max')})`.",
                "",
            ])
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

def biosignal_chat_submit(
    message: str,
    history: list[dict[str, Any]] | None,
    upload: Any,
    sampling_rate: float,
    modality_hint: str,
    trace_method: str,
):
    question = (message or "").strip() or DEFAULT_CSV_QUESTION
    history = list(history or [])
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": ""})
    yield history, ""
    for chunk in biosignal_chat_response(question, history[:-1], upload, sampling_rate, modality_hint, trace_method):
        history[-1] = {"role": "assistant", "content": chunk}
        yield history, ""


def _clear_chat():
    return []



CUSTOM_CSS = """
:root {
  --bs-bg: #ffffff;
  --bs-sidebar: #f7f7f8;
  --bs-border: #ececec;
  --bs-text: #171717;
  --bs-muted: #777777;
  --bs-hover: #ededee;
  --bs-active: #e8e8e9;
}
.gradio-container {
  background: #ffffff !important;
  color: var(--bs-text) !important;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}
.contain { max-width: none !important; padding: 0 !important; }
.bs-shell {
  width: 100%;
  height: 100vh;
  margin: 0;
  padding: 0;
}
.bs-app-row { gap: 0 !important; height: 100vh; }
.bs-sidebar {
  width: 280px;
  max-width: 280px;
  min-width: 260px;
  height: 100vh;
  overflow-y: auto;
  background: var(--bs-sidebar);
  border-right: 1px solid var(--bs-border);
  border-radius: 0;
  padding: 18px 14px;
  box-shadow: none;
}
.bs-brand {
  font-size: 20px;
  font-weight: 650;
  padding: 0 8px 22px;
}
.bs-nav-item {
  display: flex;
  align-items: center;
  gap: 10px;
  height: 38px;
  padding: 0 10px;
  border-radius: 9px;
  color: #222;
  font-size: 14px;
  margin-bottom: 4px;
}
.bs-nav-item.active { background: var(--bs-active); }
.bs-nav-item:hover { background: var(--bs-hover); }
.bs-nav-dot {
  width: 18px;
  height: 18px;
  border-radius: 6px;
  border: 1px solid #222;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
}
.bs-section-label {
  color: #555;
  font-size: 12px;
  font-weight: 650;
  margin: 22px 8px 8px;
}
.bs-history-item {
  padding: 9px 10px;
  border-radius: 9px;
  color: #222;
  font-size: 14px;
  line-height: 1.25;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.bs-history-item.active { background: var(--bs-active); }
.bs-history-item:hover { background: var(--bs-hover); }
.bs-settings {
  margin-top: 18px;
  padding-top: 14px;
  border-top: 1px solid var(--bs-border);
}
.bs-settings .label-wrap span {
  font-size: 11px !important;
  color: var(--bs-muted) !important;
  font-weight: 500 !important;
}
.bs-settings input, .bs-settings textarea, .bs-settings select {
  border-radius: 10px !important;
  border-color: #dedede !important;
  background: #ffffff !important;
}
.bs-main {
  height: 100vh;
  min-width: 0;
  background: #ffffff;
  padding: 0;
}
.bs-main-inner {
  height: 100vh;
  display: flex;
  flex-direction: column;
  max-width: 980px;
  margin: 0 auto;
  padding: 34px 32px 22px;
}
.bs-topbar {
  border: 0;
  background: transparent;
  padding: 0;
  margin: 0 0 18px;
}
.bs-title {
  font-size: 18px;
  font-weight: 600;
  margin: 0;
}
.bs-subtitle {
  color: var(--bs-muted);
  font-size: 13px;
  line-height: 1.45;
  margin-top: 4px;
  max-width: 640px;
}
.bs-chat-panel {
  border: 0;
  background: transparent;
  border-radius: 0;
  box-shadow: none;
  padding: 0;
  flex: 1;
  min-height: 0;
}
.bs-chat-panel .chatbot {
  border: 0 !important;
  background: #ffffff !important;
  height: calc(100vh - 230px) !important;
}
.bs-chat-panel .message, .bs-chat-panel .prose, .bs-chat-panel p, .bs-chat-panel li {
  font-size: 15px !important;
  line-height: 1.65 !important;
}
.bs-composer {
  border: 1px solid #d9d9d9;
  background: #ffffff;
  border-radius: 28px;
  padding: 8px 10px 8px 16px;
  margin: 16px auto 0;
  width: min(860px, 100%);
  box-shadow: 0 12px 34px rgba(0,0,0,0.08);
}
.bs-composer textarea {
  border: 0 !important;
  box-shadow: none !important;
  background: transparent !important;
  font-size: 15px !important;
}
.bs-composer button.primary {
  min-width: 44px !important;
  height: 44px !important;
  border-radius: 50% !important;
  background: #111 !important;
  border-color: #111 !important;
  color: #fff !important;
  padding: 0 !important;
}
.bs-composer button:not(.primary) {
  height: 44px !important;
  border-radius: 22px !important;
  background: #f4f4f4 !important;
  border-color: transparent !important;
  color: #555 !important;
}
.bs-tool-card {
  border: 1px solid #e8e8e8;
  border-radius: 12px;
  padding: 10px 12px;
  margin: 8px 0;
  background: #fafafa;
}
.bs-tool-row { display: flex; align-items: center; gap: 8px; min-width: 0; }
.bs-tool-icon {
  width: 22px; height: 22px; border-radius: 50%;
  background: #eeeeef; color: #444;
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 11px; font-weight: 650; flex: 0 0 auto;
}
.bs-tool-name { font-weight: 600; color: var(--bs-text); }
.bs-tool-args { color: #8a8a8a; font-size: 13px; overflow-wrap: anywhere; }
.bs-tool-result { color: var(--bs-muted); font-size: 13px; padding-left: 30px; margin-top: 4px; overflow-wrap: anywhere; }
.bs-advanced {
  max-width: 980px;
  margin: 12px auto 0;
  border-top: 1px solid var(--bs-border);
  padding-top: 8px;
}
.examples { display: none !important; }
@media (max-width: 900px) {
  .bs-app-row { flex-direction: column !important; height: auto; }
  .bs-sidebar { width: 100%; max-width: none; height: auto; border-right: 0; border-bottom: 1px solid var(--bs-border); }
  .bs-main-inner { height: auto; min-height: 80vh; padding: 18px 14px; }
  .bs-chat-panel .chatbot { height: 560px !important; }
}
"""

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
"""
    with gr.Blocks(title="BioSignalAgent", fill_height=True) as demo:
        with gr.Row(elem_classes=["bs-app-row"]):
            with gr.Column(scale=0, min_width=280, elem_classes=["bs-sidebar"]):
                gr.HTML(
                    "<div class='bs-brand'>BioSignalAgent</div>"
                    "<div class='bs-nav-item active'><span class='bs-nav-dot'>+</span><span>New analysis</span></div>"
                    "<div class='bs-nav-item'><span class='bs-nav-dot'>⌕</span><span>Search runs</span></div>"
                    "<div class='bs-nav-item'><span class='bs-nav-dot'>□</span><span>ToolUniverse</span></div>"
                    "<div class='bs-section-label'>Recent</div>"
                    "<div class='bs-history-item active'>ECG image digitization</div>"
                    "<div class='bs-history-item'>Segmenting line plot</div>"
                    "<div class='bs-history-item'>PPG peak analysis</div>"
                    "<div class='bs-history-item'>PCG murmur screening</div>"
                    "<div class='bs-history-item'>Low-resolution recovery</div>"
                )
                with gr.Column(elem_classes=["bs-settings"]):
                    gr.Markdown("### Input")
                    bot_upload = gr.File(label="Signal image or CSV", file_types=[".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".csv", ".tsv", ".txt"], type="filepath")
                    bot_sampling_rate = gr.Number(label="Sampling rate if known (Hz)", value=250)
                    bot_modality = gr.Dropdown(label="Modality hint", choices=MODALITIES, value="auto")
                    bot_trace_method = gr.Dropdown(label="Trace extraction", choices=["median", "path", "lazy", "fragmented", "momentum", "full"], value="path")
            with gr.Column(scale=1, elem_classes=["bs-main"]):
                with gr.Column(elem_classes=["bs-main-inner"]):
                    gr.HTML(
                        "<div class='bs-topbar'>"
                        "<div class='bs-title'>BioSignalAgent</div>"
                        "<div class='bs-subtitle'>Upload a signal image or CSV, then ask for routing, digitization, tool calls, and a grounded report.</div>"
                        "</div>"
                    )
                    with gr.Column(elem_classes=["bs-chat-panel"]):
                        chatbot = gr.Chatbot(
                            label="BioSignalAgent",
                            height=680,
                            placeholder=placeholder,
                            buttons=["copy", "copy_all"],
                            layout="bubble",
                            show_label=False,
                            sanitize_html=False,
                        )
                    with gr.Row(elem_classes=["bs-composer"]):
                        chat_question = gr.Textbox(
                            label="Question",
                            placeholder="Ask anything about the uploaded biosignal",
                            lines=2,
                            scale=10,
                            autofocus=True,
                            container=False,
                        )
                        chat_clear = gr.Button("Clear", scale=1)
                        chat_send = gr.Button("↑", variant="primary", scale=1)
                    chat_inputs = [chat_question, chatbot, bot_upload, bot_sampling_rate, bot_modality, bot_trace_method]
                    chat_question.submit(biosignal_chat_submit, chat_inputs, [chatbot, chat_question])
                    chat_send.click(biosignal_chat_submit, chat_inputs, [chatbot, chat_question])
                    chat_clear.click(_clear_chat, outputs=chatbot)

                    with gr.Accordion("Advanced pipeline views", open=False, elem_classes=["bs-advanced"]):
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
    demo.launch(server_name="0.0.0.0", server_port=7860, share=share, css=CUSTOM_CSS)
