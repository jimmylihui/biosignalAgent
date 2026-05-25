from __future__ import annotations

from pathlib import Path
from typing import Any
import re
import shutil
import subprocess
import tempfile

import numpy as np
import joblib
import pandas as pd
from PIL import Image, ImageOps
from scipy import signal as scipy_signal
try:
    from skimage.morphology import skeletonize
except Exception:  # optional; path tracing falls back to raw masks
    skeletonize = None


def _interpolate_missing(values: np.ndarray) -> np.ndarray:
    x = np.arange(len(values))
    mask = np.isfinite(values)
    if mask.sum() == 0:
        raise ValueError("no waveform pixels detected")
    if mask.sum() == 1:
        values[~mask] = values[mask][0]
        return values
    values[~mask] = np.interp(x[~mask], x[mask], values[mask])
    return values


ML_MODEL_PATH = Path("/data1/jiahui/biosignal-agent/outputs/waveform_digitization_pixel_model.joblib")
SCALE_PRIOR_MODEL_PATH = Path("/data1/jiahui/biosignal-agent/outputs/image_scale_prior_aug/image_scale_prior_feature_model_aug.joblib")
SCALE_PRIOR_FALLBACK_MODEL_PATH = Path("/data1/jiahui/biosignal-agent/outputs/image_scale_prior/image_scale_prior_feature_model.joblib")
SCALE_PRIOR_PER_MODALITY_MODEL_PATH = Path("/data1/jiahui/biosignal-agent/outputs/image_scale_prior_aug/image_scale_prior_per_modality_models.joblib")



def _default_out_path(image_path: str) -> Path:
    stem = Path(image_path).stem
    return Path("/data1/jiahui/biosignal-agent/outputs/digitized") / f"{stem}_digitized.csv"


def _crop_rgb_image(image_path: str, crop_left: int, crop_right: int, crop_top: int, crop_bottom: int) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int]]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    left = max(0, int(crop_left))
    right = width - max(0, int(crop_right))
    top = max(0, int(crop_top))
    bottom = height - max(0, int(crop_bottom))
    if right <= left or bottom <= top:
        raise ValueError("invalid crop bounds")
    return np.asarray(image, dtype=np.uint8)[top:bottom, left:right], (left, width - right, top, height - bottom), (width, height)


def _crop_gray_image(image_path: str, crop_left: int, crop_right: int, crop_top: int, crop_bottom: int) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int]]:
    rgb, crop, size = _crop_rgb_image(image_path, crop_left, crop_right, crop_top, crop_bottom)
    gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.uint8)
    return gray, crop, size


def pixel_feature_matrix(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=float) / 255.0
    if arr.ndim == 2:
        gray_f = arr
        r = g = b = arr
    else:
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        gray_f = 0.299 * r + 0.587 * g + 0.114 * b
    h, w = gray_f.shape
    yy, xx = np.indices((h, w), dtype=float)
    row = yy / max(1, h - 1)
    col = xx / max(1, w - 1)
    grad_y = np.zeros_like(gray_f)
    grad_x = np.zeros_like(gray_f)
    grad_y[1:, :] = np.abs(gray_f[1:, :] - gray_f[:-1, :])
    grad_x[:, 1:] = np.abs(gray_f[:, 1:] - gray_f[:, :-1])
    max_rgb = np.maximum.reduce([r, g, b])
    min_rgb = np.minimum.reduce([r, g, b])
    saturation = max_rgb - min_rgb
    blue_excess = b - np.maximum(r, g)
    red_excess = r - np.maximum(g, b)
    return np.column_stack([
        gray_f.ravel(),
        (1.0 - gray_f).ravel(),
        r.ravel(),
        g.ravel(),
        b.ravel(),
        saturation.ravel(),
        blue_excess.ravel(),
        red_excess.ravel(),
        row.ravel(),
        col.ravel(),
        grad_y.ravel(),
        grad_x.ravel(),
    ])


def _median_trace_from_mask(mask: np.ndarray) -> np.ndarray:
    y_values = np.full(mask.shape[1], np.nan, dtype=float)
    for x in range(mask.shape[1]):
        ys = np.flatnonzero(mask[:, x])
        if len(ys):
            y_values[x] = float(np.median(ys))
    return y_values




def _mean_trace_from_mask(mask: np.ndarray) -> np.ndarray:
    y_values = np.full(mask.shape[1], np.nan, dtype=float)
    for x in range(mask.shape[1]):
        ys = np.flatnonzero(mask[:, x])
        if len(ys):
            y_values[x] = float(np.mean(ys))
    return y_values


def _lazy_trace_from_mask(mask: np.ndarray, max_jump_fraction: float = 0.12) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return np.full(mask.shape[1] if mask.ndim == 2 else 0, np.nan, dtype=float)
    work = skeletonize(mask).astype(bool) if skeletonize is not None else mask
    if not np.any(work):
        work = mask
    height, width = work.shape
    y_values = np.full(width, np.nan, dtype=float)
    columns = []
    for x in range(width):
        ys = np.flatnonzero(work[:, x])
        if len(ys) == 0:
            ys = np.flatnonzero(mask[:, x])
        columns.append(ys.astype(float))
    finite = [idx for idx, ys in enumerate(columns) if len(ys)]
    if not finite:
        return y_values
    prev_y = float(np.median(columns[finite[0]]))
    y_values[finite[0]] = prev_y
    max_jump = max(2.0, float(height) * float(max_jump_fraction))
    for x in finite[1:]:
        ys = columns[x]
        distances = np.abs(ys - prev_y)
        best = float(ys[int(np.argmin(distances))])
        if np.min(distances) > max_jump and len(ys) > 2:
            best = float(np.median(ys))
        y_values[x] = best
        prev_y = best
    return y_values


def _fragmented_trace_from_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return np.full(mask.shape[1] if mask.ndim == 2 else 0, np.nan, dtype=float)
    try:
        from scipy import ndimage

        labels, num = ndimage.label(mask)
        if num <= 0:
            return _median_trace_from_mask(mask)
        best_label = None
        best_score = -1.0
        for label_id in range(1, num + 1):
            ys, xs = np.nonzero(labels == label_id)
            if len(xs) == 0:
                continue
            width_span = int(xs.max() - xs.min() + 1)
            height_span = int(ys.max() - ys.min() + 1)
            area = int(len(xs))
            # Prefer long, mostly-horizontal connected fragments over grid/noise blobs.
            score = width_span * np.sqrt(area) / max(1.0, np.sqrt(height_span))
            if score > best_score:
                best_score = float(score)
                best_label = label_id
        if best_label is None:
            return _median_trace_from_mask(mask)
        component = labels == best_label
        if component.mean() <= 0 or np.any(component.sum(axis=0)) < max(8, int(mask.shape[1] * 0.08)):
            return _median_trace_from_mask(mask)
        return _median_trace_from_mask(component)
    except Exception:
        return _path_trace_from_mask(mask)


def _path_trace_from_mask(mask: np.ndarray, jump_penalty: float = 0.08, distance_penalty: float = 0.02) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return np.full(mask.shape[1] if mask.ndim == 2 else 0, np.nan, dtype=float)
    work = skeletonize(mask).astype(bool) if skeletonize is not None else mask
    if not np.any(work):
        work = mask
    height, width = work.shape
    columns: list[np.ndarray] = []
    for x in range(width):
        ys = np.flatnonzero(work[:, x])
        if len(ys) == 0:
            ys = np.flatnonzero(mask[:, x])
        if len(ys) > 0:
            # Keep representative candidates so dense vertical strokes remain tractable.
            if len(ys) > 9:
                q = np.linspace(0, 100, 9)
                ys = np.unique(np.percentile(ys, q).round().astype(int))
        columns.append(ys.astype(float))
    finite_cols = [i for i, ys in enumerate(columns) if len(ys)]
    if not finite_cols:
        return np.full(width, np.nan, dtype=float)
    start = finite_cols[0]
    costs = np.zeros(len(columns[start]), dtype=float)
    paths: list[list[int]] = [[i] for i in range(len(columns[start]))]
    prev_x = start
    prev_ys = columns[start]
    for x in finite_cols[1:]:
        ys = columns[x]
        new_costs = np.full(len(ys), np.inf, dtype=float)
        new_paths: list[list[int]] = [[] for _ in range(len(ys))]
        gap = max(1, x - prev_x)
        for j, y in enumerate(ys):
            transitions = costs + jump_penalty * (np.abs(prev_ys - y) / max(1, height - 1)) ** 2 / gap
            transitions += distance_penalty * gap / max(1, width - 1)
            best = int(np.argmin(transitions))
            new_costs[j] = transitions[best]
            new_paths[j] = paths[best] + [j]
        costs = new_costs
        paths = new_paths
        prev_x = x
        prev_ys = ys
    best_path = paths[int(np.argmin(costs))]
    y_values = np.full(width, np.nan, dtype=float)
    for x, candidate_idx in zip(finite_cols, best_path):
        y_values[x] = float(columns[x][candidate_idx])
    return y_values


def _momentum_trace_from_mask(mask: np.ndarray, max_candidates: int = 15) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return np.full(mask.shape[1] if mask.ndim == 2 else 0, np.nan, dtype=float)
    work = skeletonize(mask).astype(bool) if skeletonize is not None else mask
    if not np.any(work):
        work = mask
    height, width = work.shape
    y_values = np.full(width, np.nan, dtype=float)
    prev_y: float | None = None
    prev_dy = 0.0
    for x in range(width):
        ys = np.flatnonzero(work[:, x])
        if len(ys) == 0:
            ys = np.flatnonzero(mask[:, x])
        if len(ys) == 0:
            continue
        ys = ys.astype(float)
        if len(ys) > max_candidates:
            q = np.linspace(0, 100, max_candidates)
            ys = np.unique(np.percentile(ys, q).round()).astype(float)
        if prev_y is None:
            chosen = float(np.median(ys))
        else:
            predicted = prev_y + prev_dy
            # Prefer the candidate closest to the local momentum prediction, but
            # allow large jumps when the observed column only has distant pixels.
            chosen = float(ys[int(np.argmin(np.abs(ys - predicted)))])
            prev_dy = 0.65 * prev_dy + 0.35 * (chosen - prev_y)
        y_values[x] = chosen
        prev_y = chosen
    return y_values


def _signal_from_mask(mask: np.ndarray, sampling_rate: float | None, out_csv: str | None, image_path: str, value_min: float | None, value_max: float | None, smooth_window: int, tool_name: str, method: str, model_source: str | None = None, confidence_scale: float = 1.0, trace_method: str = "median") -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    if trace_method == "path":
        y_values = _path_trace_from_mask(mask)
    elif trace_method == "momentum":
        y_values = _momentum_trace_from_mask(mask)
    elif trace_method == "full":
        y_values = _mean_trace_from_mask(mask)
    elif trace_method == "lazy":
        y_values = _lazy_trace_from_mask(mask)
    elif trace_method == "fragmented":
        y_values = _fragmented_trace_from_mask(mask)
    else:
        y_values = _median_trace_from_mask(mask)
    coverage = float(np.isfinite(y_values).mean()) if len(y_values) else 0.0
    try:
        y_values = _interpolate_missing(y_values)
    except ValueError as exc:
        return {"tool": tool_name, "error": str(exc), "pixel_coverage": coverage, "confidence": 0.0}
    if smooth_window and smooth_window > 1:
        window = int(smooth_window)
        if window % 2 == 0:
            window += 1
        if window < len(y_values):
            y_values = scipy_signal.medfilt(y_values, kernel_size=window)
    normalized = 1.0 - 2.0 * (y_values / max(1, mask.shape[0] - 1))
    if value_min is not None and value_max is not None and float(value_max) != float(value_min):
        values = float(value_min) + (normalized + 1.0) / 2.0 * (float(value_max) - float(value_min))
        scale = "calibrated"
    else:
        values = normalized
        scale = "normalized_-1_to_1"
    out_path = Path(out_csv) if out_csv else _default_out_path(image_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"signal": values})
    if sampling_rate is not None:
        frame.insert(0, "time_s", np.arange(len(values), dtype=float) / float(sampling_rate))
    frame.to_csv(out_path, index=False)
    result = {
        "tool": tool_name,
        "image_path": str(image_path),
        "out_csv": str(out_path),
        "num_points": int(len(values)),
        "sampling_rate": float(sampling_rate) if sampling_rate is not None else None,
        "pixel_coverage": coverage,
        "value_min": float(np.nanmin(values)) if len(values) else None,
        "value_max": float(np.nanmax(values)) if len(values) else None,
        "scale": scale,
        "confidence": min(0.95, max(0.2, coverage * confidence_scale)),
        "method": method,
        "trace_method": trace_method,
        "disclaimer": "Baseline waveform-image digitizer; verify calibration and extracted morphology before clinical use.",
    }
    if model_source is not None:
        result["model_source"] = model_source
    return result


def Signal_digitize_waveform_image(
    image_path: str,
    sampling_rate: float | None = None,
    out_csv: str | None = None,
    value_min: float | None = None,
    value_max: float | None = None,
    crop_left: int = 0,
    crop_right: int = 0,
    crop_top: int = 0,
    crop_bottom: int = 0,
    threshold: int = 80,
    smooth_window: int = 1,
    trace_method: str = "median",
) -> dict[str, Any]:
    """Digitize a dark single waveform trace from a mostly light plot image.

    This baseline intentionally targets clean rendered plots first. If value_min and
    value_max are omitted, the returned CSV is normalized to approximately [-1, 1].
    """
    try:
        arr, crop, _ = _crop_gray_image(image_path, crop_left, crop_right, crop_top, crop_bottom)
    except Exception as exc:
        return {"tool": "Signal_digitize_waveform_image", "error": str(exc), "confidence": 0.0}
    result = _signal_from_mask(
        arr <= int(threshold),
        sampling_rate,
        out_csv,
        image_path,
        value_min,
        value_max,
        smooth_window,
        "Signal_digitize_waveform_image",
        "dark_trace_path_digitizer" if trace_method == "path" else "dark_trace_column_median_digitizer",
        confidence_scale=1.0,
        trace_method=trace_method,
    )
    result["crop"] = {"left": crop[0], "right": crop[1], "top": crop[2], "bottom": crop[3]}
    result["threshold"] = int(threshold)
    return result



def Signal_digitize_waveform_image_ml(
    image_path: str,
    sampling_rate: float | None = None,
    out_csv: str | None = None,
    value_min: float | None = None,
    value_max: float | None = None,
    crop_left: int = 0,
    crop_right: int = 0,
    crop_top: int = 0,
    crop_bottom: int = 0,
    model_path: str | None = None,
    probability_threshold: float = 0.5,
    smooth_window: int = 1,
    trace_method: str = "median",
) -> dict[str, Any]:
    model_file = Path(model_path) if model_path else ML_MODEL_PATH
    if not model_file.exists():
        return {"tool": "Signal_digitize_waveform_image_ml", "error": f"model not found: {model_file}", "confidence": 0.0}
    try:
        arr, crop, _ = _crop_rgb_image(image_path, crop_left, crop_right, crop_top, crop_bottom)
        bundle = joblib.load(model_file)
        features = pixel_feature_matrix(arr)
        model = bundle["model"]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[:, list(model.classes_).index(1)]
            mask = probs.reshape(arr.shape[:2]) >= float(probability_threshold)
            mean_probability = float(np.mean(probs[mask.ravel()])) if np.any(mask) else 0.0
        else:
            mask = model.predict(features).reshape(arr.shape[:2]).astype(bool)
            mean_probability = float(np.mean(mask))
    except Exception as exc:
        return {"tool": "Signal_digitize_waveform_image_ml", "error": str(exc), "confidence": 0.0, "model_source": str(model_file)}
    result = _signal_from_mask(
        mask,
        sampling_rate,
        out_csv,
        image_path,
        value_min,
        value_max,
        smooth_window,
        "Signal_digitize_waveform_image_ml",
        "trained_pixel_segmentation_path_digitizer" if trace_method == "path" else "trained_pixel_segmentation_digitizer",
        model_source=str(model_file),
        confidence_scale=max(0.3, mean_probability),
        trace_method=trace_method,
    )
    result["crop"] = {"left": crop[0], "right": crop[1], "top": crop[2], "bottom": crop[3]}
    result["probability_threshold"] = float(probability_threshold)
    result["mask_pixel_fraction"] = float(np.mean(mask)) if mask.size else 0.0
    return result



def _cluster_positions(values: list[float], tolerance: float = 3.0) -> list[float]:
    if not values:
        return []
    vals = sorted(float(v) for v in values)
    clusters: list[list[float]] = [[vals[0]]]
    for val in vals[1:]:
        if abs(val - np.mean(clusters[-1])) <= tolerance:
            clusters[-1].append(val)
        else:
            clusters.append([val])
    return [float(np.mean(group)) for group in clusters]


def _median_spacing(positions: list[float]) -> float | None:
    if len(positions) < 2:
        return None
    diffs = np.diff(sorted(positions))
    diffs = diffs[diffs > 4]
    if len(diffs) == 0:
        return None
    return float(np.median(diffs))



def _ocr_text_with_tesseract(image: Image.Image) -> tuple[str, str]:
    exe = shutil.which("tesseract")
    if exe is None:
        return "", "unavailable_tesseract_not_found"
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "axis_crop.png"
        image.save(src)
        cmd = [exe, str(src), "stdout", "--psm", "6", "-c", "tessedit_char_whitelist=0123456789.-+sSmM:/Hz"]
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        if proc.returncode != 0:
            return proc.stdout.strip(), f"tesseract_error:{proc.stderr.strip()[:160]}"
        return proc.stdout.strip(), "ok"


def _numbers_from_ocr_text(text: str) -> list[float]:
    values: list[float] = []
    for token in re.findall(r"[-+]?\d+(?:\.\d+)?", text):
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values



def _ocr_text_with_rapidocr_bottom(image: Image.Image) -> tuple[str, str]:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception as exc:
        return "", f"unavailable_rapidocr:{type(exc).__name__}"
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "axis_crop.png"
        image.save(src)
        try:
            result, _ = RapidOCR()(str(src))
        except Exception as exc:
            return "", f"rapidocr_error:{str(exc)[:160]}"
    if not result:
        return "", "rapidocr_no_text"
    height = image.height
    picked: list[tuple[float, str]] = []
    for item in result:
        if len(item) < 2:
            continue
        box, text = item[0], str(item[1]).strip()
        if not _numbers_from_ocr_text(text):
            continue
        try:
            center_y = float(np.mean([point[1] for point in box]))
            center_x = float(np.mean([point[0] for point in box]))
        except Exception:
            continue
        if center_y >= height * 0.55:
            picked.append((center_x, text))
    if not picked:
        return "", "rapidocr_no_bottom_tick_numbers"
    picked.sort(key=lambda item: item[0])
    return " ".join(text for _, text in picked), "ok"


def _ocr_xaxis_text(image: Image.Image) -> tuple[str, str]:
    # Require several tick labels before using OCR for calibration; two numbers
    # such as "8 10" are often only the right edge of the axis and imply a false
    # duration if interpreted as the full axis span.
    min_tick_numbers = 4
    text, status = _ocr_text_with_rapidocr_bottom(image)
    if status == "ok" and len(_numbers_from_ocr_text(text)) >= min_tick_numbers:
        return text, "rapidocr_ok"
    tess_text, tess_status = _ocr_text_with_tesseract(image)
    if tess_status == "ok" and len(_numbers_from_ocr_text(tess_text)) >= min_tick_numbers:
        return tess_text, "tesseract_ok"
    if status != "ok":
        return tess_text or text, f"rapidocr_{status};tesseract_{tess_status}"
    return tess_text or text, f"rapidocr_insufficient_tick_numbers;tesseract_{tess_status}"

def _duration_from_tick_numbers(values: list[float]) -> float | None:
    if len(values) < 4:
        return None
    vals = sorted(set(float(v) for v in values))
    span = vals[-1] - vals[0]
    if span <= 0:
        return None
    return float(span)

def Signal_estimate_image_scale(
    image_path: str,
    crop_left: int = 0,
    crop_right: int = 0,
    crop_top: int = 0,
    crop_bottom: int = 0,
    duration_s: float | None = None,
    seconds_per_major_grid: float | None = None,
    value_per_major_grid: float | None = None,
    value_units: str | None = None,
    use_ocr: bool = False,
) -> dict[str, Any]:
    """Estimate plot-area and axis scale cues from a waveform image.

    This intentionally separates image geometry from physical calibration. Grid/axis
    lines can give pixels-per-major-grid, but without readable tick labels or a known
    paper standard the image alone cannot determine seconds/mV units reliably.
    """
    try:
        arr, crop, original_size = _crop_rgb_image(image_path, crop_left, crop_right, crop_top, crop_bottom)
    except Exception as exc:
        return {"tool": "Signal_estimate_image_scale", "error": str(exc), "confidence": 0.0}
    gray = np.asarray(Image.fromarray(arr).convert("L"), dtype=np.uint8)
    height, width = gray.shape
    line_x: list[float] = []
    line_y: list[float] = []
    used_cv = False
    try:
        import cv2

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 40, 120)
        min_len = max(24, int(min(width, height) * 0.20))
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=max(20, int(min(width, height) * 0.08)), minLineLength=min_len, maxLineGap=6)
        if lines is not None:
            for item in lines[:, 0, :]:
                x1, y1, x2, y2 = [int(v) for v in item]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                # Treat only plot-spanning strokes as axes/grid. Shorter near-horizontal
                # or near-vertical segments are often the waveform itself.
                if dx >= max(80, width * 0.65) and dy <= max(3, height * 0.01):
                    line_y.append((y1 + y2) / 2.0)
                elif dy >= max(80, height * 0.65) and dx <= max(3, width * 0.01):
                    line_x.append((x1 + x2) / 2.0)
        used_cv = True
    except Exception:
        used_cv = False

    # Projection fallback catches faint grids and simple axes when Hough is sparse.
    dark = gray < 220
    row_density = dark.mean(axis=1)
    col_density = dark.mean(axis=0)
    # Projection fallback uses a high density bar threshold so dense waveform rows do
    # not masquerade as axes. This works best for simple rectangular plot borders.
    for y in np.flatnonzero(row_density > max(0.45, float(np.percentile(row_density, 99)))):
        line_y.append(float(y))
    for x in np.flatnonzero(col_density > max(0.45, float(np.percentile(col_density, 99)))):
        line_x.append(float(x))

    xs = _cluster_positions(line_x, tolerance=max(2.5, width * 0.003))
    ys = _cluster_positions(line_y, tolerance=max(2.5, height * 0.005))
    x_spacing = _median_spacing(xs)
    y_spacing = _median_spacing(ys)

    content = gray < 245
    if np.any(content):
        yy, xx = np.nonzero(content)
        content_bbox = {"left": int(xx.min()), "right": int(xx.max()), "top": int(yy.min()), "bottom": int(yy.max())}
    else:
        content_bbox = {"left": 0, "right": int(width - 1), "top": 0, "bottom": int(height - 1)}
    # Use the non-white bounding box as the conservative plot area. Detected grid
    # lines are still returned for scale hints, but we avoid using them as crop
    # bounds because steep waveform fragments can look like vertical grid lines.
    plot_bbox = content_bbox.copy()

    plot_width = max(1, int(plot_bbox["right"] - plot_bbox["left"] + 1))
    plot_height = max(1, int(plot_bbox["bottom"] - plot_bbox["top"] + 1))
    ocr_text = ""
    ocr_status = "disabled"
    ocr_numbers: list[float] = []
    ocr_duration = None
    if use_ocr:
        # Tick labels may sit inside the plot margin, below the axis, or under an
        # xlabel. Use a generous lower-band crop; OCR post-processing extracts only
        # numeric tick candidates.
        bottom_start = max(0, min(int(plot_bbox["bottom"]) - int(height * 0.28), int(height * 0.52)))
        axis_bottom = height
        axis_left = max(0, int(plot_bbox["left"]) - int(width * 0.05))
        axis_right = min(width, int(plot_bbox["right"]) + int(width * 0.05))
        axis_crop = Image.fromarray(arr[bottom_start:axis_bottom, axis_left:axis_right]).convert("L")
        axis_crop = ImageOps.autocontrast(axis_crop)
        axis_crop = axis_crop.resize((axis_crop.width * 3, axis_crop.height * 3), Image.Resampling.LANCZOS)
        axis_arr = np.asarray(axis_crop, dtype=np.uint8)
        thresh = int(np.percentile(axis_arr, 65))
        axis_crop = Image.fromarray(np.where(axis_arr < thresh, 0, 255).astype(np.uint8))
        ocr_text, ocr_status = _ocr_xaxis_text(axis_crop)
        ocr_numbers = _numbers_from_ocr_text(ocr_text)
        ocr_duration = _duration_from_tick_numbers(ocr_numbers)
    inferred_duration = float(duration_s) if duration_s is not None else None
    if inferred_duration is None and seconds_per_major_grid is not None and x_spacing:
        inferred_duration = plot_width / float(x_spacing) * float(seconds_per_major_grid)
    if inferred_duration is None and ocr_duration is not None:
        inferred_duration = ocr_duration
    sampling_rate = None
    if inferred_duration and inferred_duration > 0:
        sampling_rate = float(plot_width) / float(inferred_duration)
    value_span = None
    if value_per_major_grid is not None and y_spacing:
        value_span = plot_height / float(y_spacing) * float(value_per_major_grid)
    missing: list[str] = []
    if sampling_rate is None:
        missing.append("x-axis duration, seconds_per_major_grid, or readable x-axis tick labels")
    if value_span is None:
        missing.append("y-axis value_per_major_grid or readable tick labels")
    confidence = 0.25
    if x_spacing:
        confidence += 0.25
    if y_spacing:
        confidence += 0.20
    if sampling_rate is not None:
        confidence += 0.15
    if use_ocr and ocr_status == "ok" and ocr_numbers:
        confidence += 0.05
    if value_span is not None:
        confidence += 0.15
    confidence = float(min(0.95, confidence))
    return {
        "tool": "Signal_estimate_image_scale",
        "image_path": str(image_path),
        "original_size": {"width": int(original_size[0]), "height": int(original_size[1])},
        "crop": {"left": crop[0], "right": crop[1], "top": crop[2], "bottom": crop[3]},
        "analyzed_size": {"width": int(width), "height": int(height)},
        "plot_area": plot_bbox,
        "content_bbox": content_bbox,
        "detected_vertical_lines": [float(round(v, 2)) for v in xs[:80]],
        "detected_horizontal_lines": [float(round(v, 2)) for v in ys[:80]],
        "pixels_per_major_grid_x": x_spacing,
        "pixels_per_major_grid_y": y_spacing,
        "duration_s": inferred_duration,
        "sampling_rate": sampling_rate,
        "ocr_status": ocr_status,
        "ocr_text": ocr_text,
        "ocr_numbers": ocr_numbers,
        "ocr_duration_s": ocr_duration,
        "value_span": value_span,
        "value_units": value_units,
        "physical_scale_status": "calibrated" if not missing else "partial_requires_metadata_or_ocr",
        "missing_for_physical_scale": missing,
        "confidence": confidence,
        "method": "hough_lines_plus_projection" if used_cv else "projection_only",
        "disclaimer": "Image geometry can be estimated from axes/grid lines, but physical x/y units require tick labels, known paper scale, metadata, or user confirmation.",
    }


def _scale_prior_trace_from_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("L").resize((256, 96), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    dark = arr < 0.75
    ys: list[float] = []
    for x in range(dark.shape[1]):
        idx = np.flatnonzero(dark[:, x])
        if len(idx):
            ys.append(float(np.median(idx) / max(1, dark.shape[0] - 1)))
    if len(ys) < 4:
        return np.zeros(4, dtype=float)
    y = np.asarray(ys, dtype=float)
    return np.interp(np.linspace(0, len(y) - 1, 256), np.arange(len(y)), y)


def _scale_prior_extra_features(path: str) -> dict[str, float]:
    y = _scale_prior_trace_from_image(path)
    centered = y - np.mean(y)
    diff = np.diff(centered)
    peaks, _ = scipy_signal.find_peaks(-centered, distance=3, prominence=max(np.std(centered) * 0.2, 1e-8))
    fft = np.abs(np.fft.rfft(centered)) ** 2
    freqs = np.fft.rfftfreq(len(centered), d=1.0)
    power = float(np.sum(fft)) + 1e-12
    centroid = float(np.sum(freqs * fft) / power)
    entropy = float(-np.sum((fft / power) * np.log2(fft / power + 1e-12)) / np.log2(len(fft))) if len(fft) > 1 else 0.0
    return {
        "trace_peak_count_norm": float(len(peaks) / len(y)),
        "trace_zero_crossing_rate_resampled": float(np.mean(np.diff(np.signbit(centered)) != 0)),
        "trace_abs_slope_mean_resampled": float(np.mean(np.abs(diff))) if len(diff) else 0.0,
        "trace_slope_std_resampled": float(np.std(diff)) if len(diff) else 0.0,
        "trace_fft_centroid_resampled": centroid,
        "trace_fft_entropy_resampled": entropy,
        "trace_fft_low": float(np.sum(fft[freqs < 0.03]) / power),
        "trace_fft_mid": float(np.sum(fft[(freqs >= 0.03) & (freqs < 0.12)]) / power),
        "trace_fft_high": float(np.sum(fft[freqs >= 0.12]) / power),
    }


def _scale_prior_features(image_path: str, modality: str | None, feature_names: list[str]) -> np.ndarray:
    from biosignal_agent.tools.image_modality_tools import extract_image_modality_features

    features = extract_image_modality_features(image_path)
    features.update(_scale_prior_extra_features(image_path))
    mod = str(modality or "").lower().strip()
    for name in feature_names:
        if name.startswith("modality_"):
            features[name] = 1.0 if name == f"modality_{mod}" else 0.0
    return np.asarray([[float(features.get(name, 0.0)) for name in feature_names]], dtype=float)


def _plot_width_for_sampling_rate(image_path: str) -> int | None:
    try:
        image = Image.open(image_path).convert("L")
        gray = np.asarray(image, dtype=np.uint8)
        content = gray < 245
        if not np.any(content):
            return int(image.width)
        _, xx = np.nonzero(content)
        return int(xx.max() - xx.min() + 1)
    except Exception:
        return None


def Signal_predict_image_scale_prior(
    image_path: str,
    modality: str | None = None,
    top_k: int = 3,
    model_path: str | None = None,
    abstain_threshold: float = 0.90,
    model_scope: str = "auto",
    per_modality_model_path: str | None = None,
) -> dict[str, Any]:
    """Rank discrete duration candidates when OCR/grid/metadata cannot calibrate x-axis scale.

    This is a prior classifier, not a physical measurement. Use it only as a fallback
    and keep human confirmation or top-k evaluation when confidence is low.
    """
    scope = str(model_scope or "auto").lower().strip()
    model_file = Path(model_path) if model_path else SCALE_PRIOR_MODEL_PATH
    selected_scope = "unified"
    try:
        bundle = None
        if scope in {"auto", "per_modality", "per-modality"} and modality:
            per_file = Path(per_modality_model_path) if per_modality_model_path else SCALE_PRIOR_PER_MODALITY_MODEL_PATH
            if per_file.exists():
                per_bundle = joblib.load(per_file)
                per_models = per_bundle.get("models", {})
                mod_key = str(modality).lower().strip()
                if mod_key in per_models:
                    bundle = per_models[mod_key]
                    model_file = per_file
                    selected_scope = f"per_modality:{mod_key}"
        if bundle is None:
            if not model_file.exists() and SCALE_PRIOR_FALLBACK_MODEL_PATH.exists():
                model_file = SCALE_PRIOR_FALLBACK_MODEL_PATH
            if not model_file.exists():
                return {"tool": "Signal_predict_image_scale_prior", "error": f"model not found: {model_file}", "confidence": 0.0}
            bundle = joblib.load(model_file)
        model = bundle["model"]
        durations = [float(v) for v in bundle["durations"]]
        feature_names = list(bundle["feature_names"])
        X = _scale_prior_features(image_path, modality, feature_names)
        if not hasattr(model, "predict_proba"):
            return {"tool": "Signal_predict_image_scale_prior", "error": "model does not provide predict_proba", "confidence": 0.0}
        proba = np.asarray(model.predict_proba(X)[0], dtype=float)
        classes = [int(c) for c in getattr(model, "classes_", range(len(durations)))]
        candidates = []
        plot_width = _plot_width_for_sampling_rate(image_path)
        for rank, idx in enumerate(np.argsort(proba)[::-1][: max(1, int(top_k))], start=1):
            duration_idx = classes[int(idx)] if int(idx) < len(classes) else int(idx)
            if duration_idx < 0 or duration_idx >= len(durations):
                continue
            duration = float(durations[duration_idx])
            item = {"rank": int(rank), "duration_s": duration, "probability": float(proba[int(idx)])}
            if plot_width and duration > 0:
                item["sampling_rate_hz_if_used"] = float(plot_width / duration)
            candidates.append(item)
        confidence = float(candidates[0]["probability"]) if candidates else 0.0
        return {
            "tool": "Signal_predict_image_scale_prior",
            "image_path": str(image_path),
            "modality": str(modality).lower() if modality else None,
            "duration_candidates": candidates,
            "best_duration_s": candidates[0]["duration_s"] if candidates else None,
            "confidence": confidence,
            "requires_confirmation": bool(confidence < float(abstain_threshold)),
            "scale_source": "model_prior_classifier",
            "model_source": str(model_file),
            "model_scope": selected_scope,
            "model_type": str(bundle.get("best_model", "unknown")),
            "trained_duration_classes_s": durations,
            "top_k_accuracy_hint": float(bundle.get("results", {}).get(bundle.get("best_model", ""), {}).get("top3_accuracy", 0.0)),
            "disclaimer": "Fallback prior only: without OCR/grid/metadata, absolute x-axis scale is ambiguous. Use top-k candidates or user confirmation before clinical interpretation.",
        }
    except Exception as exc:
        return {"tool": "Signal_predict_image_scale_prior", "error": str(exc), "confidence": 0.0, "model_source": str(model_file)}


def _duration_candidate_values(duration_candidates: list[Any] | None) -> list[dict[str, float]]:
    values: list[dict[str, float]] = []
    for idx, item in enumerate(duration_candidates or []):
        if isinstance(item, dict):
            duration = item.get("duration_s")
            probability = item.get("probability", item.get("score", None))
        else:
            duration = item
            probability = None
        try:
            duration_f = float(duration)
        except Exception:
            continue
        if duration_f <= 0:
            continue
        values.append({"duration_s": duration_f, "prior_probability": float(probability) if probability is not None else float("nan"), "prior_rank": float(idx + 1)})
    return values


def _detect_periodic_peaks_for_scale(values: np.ndarray, sampling_rate: float, modality: str) -> dict[str, Any]:
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 8 or sampling_rate <= 0:
        return {"num_peaks": 0, "rate_bpm": None, "interval_cv": None, "peak_confidence": 0.0, "polarity": None}
    y = y - float(np.nanmedian(y))
    std = float(np.nanstd(y))
    if std <= 1e-10:
        return {"num_peaks": 0, "rate_bpm": None, "interval_cv": None, "peak_confidence": 0.0, "polarity": None}
    mod = str(modality or "").lower()
    ranges = {
        "ecg": (35.0, 220.0, 0.5, 35.0),
        "ppg": (35.0, 220.0, 0.4, 8.0),
        "abp": (35.0, 220.0, 0.4, 12.0),
        "bcg": (35.0, 220.0, 0.6, 18.0),
        "scg": (35.0, 220.0, 0.8, 25.0),
        "resp": (6.0, 40.0, 0.05, 0.8),
    }
    min_bpm, max_bpm, low_hz, high_hz = ranges.get(mod, (35.0, 220.0, 0.4, 12.0))
    high = min(float(high_hz), 0.45 * float(sampling_rate))
    filtered = y
    try:
        if high > low_hz and len(y) > max(12, int(sampling_rate)):
            sos = scipy_signal.butter(3, [low_hz / (0.5 * sampling_rate), high / (0.5 * sampling_rate)], btype="bandpass", output="sos")
            filtered = scipy_signal.sosfiltfilt(sos, y)
    except Exception:
        filtered = y
    min_distance = max(1, int(round(sampling_rate * 60.0 / max_bpm * 0.8)))
    prominence = max(float(np.nanstd(filtered)) * 0.25, 1e-8)
    best: dict[str, Any] | None = None
    for polarity, signal_values in [("positive", filtered), ("negative", -filtered)]:
        peaks, props = scipy_signal.find_peaks(signal_values, distance=min_distance, prominence=prominence)
        rate_bpm = None
        interval_cv = None
        regularity = 0.0
        if len(peaks) >= 2:
            intervals_s = np.diff(peaks) / float(sampling_rate)
            intervals_s = intervals_s[np.isfinite(intervals_s) & (intervals_s > 0)]
            if len(intervals_s):
                rate_bpm = float(60.0 / np.median(intervals_s))
                mean_interval = float(np.mean(intervals_s))
                interval_cv = float(np.std(intervals_s) / mean_interval) if mean_interval > 0 else None
                regularity = float(max(0.0, 1.0 - min(interval_cv or 1.0, 1.0)))
        plausibility = 0.0
        if rate_bpm is not None:
            if min_bpm <= rate_bpm <= max_bpm:
                plausibility = 1.0
            else:
                # Softly penalize just-outside-range candidates, strongly penalize absurd rates.
                if rate_bpm < min_bpm:
                    plausibility = max(0.0, 1.0 - (min_bpm - rate_bpm) / max(min_bpm, 1.0))
                else:
                    plausibility = max(0.0, 1.0 - (rate_bpm - max_bpm) / max(max_bpm, 1.0))
        expected_min_peaks = max(2, int(np.floor(min_bpm / 60.0 * len(y) / sampling_rate * 0.6)))
        expected_max_peaks = max(expected_min_peaks + 1, int(np.ceil(max_bpm / 60.0 * len(y) / sampling_rate * 1.4)))
        count_score = 1.0 if expected_min_peaks <= len(peaks) <= expected_max_peaks else 0.4 if len(peaks) >= 2 else 0.0
        prom = np.asarray(props.get("prominences", []), dtype=float)
        prom_score = float(np.tanh(float(np.nanmedian(prom)) / (float(np.nanstd(filtered)) + 1e-8))) if len(prom) else 0.0
        score = 0.45 * plausibility + 0.25 * regularity + 0.20 * count_score + 0.10 * prom_score
        candidate = {
            "num_peaks": int(len(peaks)),
            "rate_bpm": rate_bpm,
            "interval_cv": interval_cv,
            "peak_confidence": float(max(0.0, min(1.0, score))),
            "polarity": polarity,
            "plausibility_score": float(plausibility),
            "regularity_score": float(regularity),
            "count_score": float(count_score),
            "prominence_score": float(prom_score),
            "peak_indices": peaks.tolist()[:200],
        }
        if best is None or candidate["peak_confidence"] > best["peak_confidence"]:
            best = candidate
    return best or {"num_peaks": 0, "rate_bpm": None, "interval_cv": None, "peak_confidence": 0.0, "polarity": None}


def Signal_select_image_scale_by_peak_plausibility(
    image_path: str,
    modality: str | None = None,
    duration_candidates: list[Any] | None = None,
    top_k: int = 3,
    crop_left: int = 0,
    crop_right: int = 0,
    crop_top: int = 0,
    crop_bottom: int = 0,
    threshold: int = 80,
    trace_method: str = "median",
    smooth_window: int = 1,
    model_scope: str = "auto",
) -> dict[str, Any]:
    """Choose the most plausible x-axis duration using peak-derived physiological rate checks.

    Use this after OCR/grid calibration fails. It is most useful for ECG/PPG/ABP/BCG/SCG/RESP;
    dense or non-periodic modalities should keep top-k candidates rather than a hard decision.
    """
    mod = str(modality or "").lower().strip()
    peak_supported = {"ecg", "ppg", "abp", "bcg", "scg", "resp"}
    if duration_candidates is None:
        prior = Signal_predict_image_scale_prior(image_path=image_path, modality=mod or None, top_k=max(3, int(top_k)), model_scope=model_scope)
        duration_candidates = prior.get("duration_candidates", [])
    else:
        prior = {"tool": "Signal_predict_image_scale_prior", "duration_candidates": duration_candidates}
    candidates = _duration_candidate_values(duration_candidates)
    if not candidates:
        return {"tool": "Signal_select_image_scale_by_peak_plausibility", "error": "no duration candidates", "confidence": 0.0}
    try:
        arr, crop, _ = _crop_gray_image(image_path, crop_left, crop_right, crop_top, crop_bottom)
    except Exception as exc:
        return {"tool": "Signal_select_image_scale_by_peak_plausibility", "error": str(exc), "confidence": 0.0}
    scored = []
    for item in candidates[: max(1, int(top_k))]:
        duration = float(item["duration_s"])
        sampling_rate = float(arr.shape[1]) / duration
        digitized = _signal_from_mask(
            arr <= int(threshold), sampling_rate, None, image_path, None, None, int(smooth_window),
            "Signal_select_image_scale_by_peak_plausibility", "temporary_dark_trace_digitizer", trace_method=trace_method,
        )
        if digitized.get("error"):
            peak = {"num_peaks": 0, "rate_bpm": None, "interval_cv": None, "peak_confidence": 0.0, "polarity": None}
            coverage = 0.0
        else:
            df = pd.read_csv(digitized["out_csv"])
            values = df["signal"].to_numpy(dtype=float)
            peak = _detect_periodic_peaks_for_scale(values, sampling_rate, mod)
            coverage = float(digitized.get("pixel_coverage", 0.0))
        prior_prob = item.get("prior_probability")
        prior_score = 0.0 if prior_prob is None or not np.isfinite(prior_prob) else float(prior_prob)
        if mod in peak_supported:
            # Peak detection is used as a physiological veto, not as the primary
            # scale estimator: neighboring durations often all produce plausible
            # heart/respiratory rates. Keep the classifier ordering unless the
            # candidate has weak/implausible peak evidence.
            peak_conf = float(peak.get("peak_confidence", 0.0))
            plausible = float(peak.get("plausibility_score", 0.0)) >= 0.75 and int(peak.get("num_peaks", 0)) >= 2
            if plausible:
                peak_gate = 0.78 + 0.22 * peak_conf
            elif int(peak.get("num_peaks", 0)) >= 2:
                peak_gate = 0.45 + 0.25 * peak_conf
            else:
                peak_gate = 0.18
            rank_prior = 1.0 / max(1.0, float(item.get("prior_rank", 1.0)))
            effective_prior = prior_score if prior_score > 0 else rank_prior
            combined = 0.82 * effective_prior * peak_gate + 0.10 * peak_conf + 0.08 * min(1.0, coverage)
        else:
            combined = 0.75 * prior_score + 0.25 * min(1.0, coverage)
        scored.append({
            "duration_s": duration,
            "sampling_rate_hz_if_used": sampling_rate,
            "combined_score": float(max(0.0, min(1.0, combined))),
            "prior_probability": None if prior_prob is None or not np.isfinite(prior_prob) else float(prior_prob),
            "pixel_coverage": coverage,
            **peak,
        })
    scored.sort(key=lambda row: row["combined_score"], reverse=True)
    best = scored[0]
    margin = float(best["combined_score"] - scored[1]["combined_score"]) if len(scored) > 1 else float(best["combined_score"])
    needs_confirmation = bool(mod not in peak_supported or best["combined_score"] < 0.72 or margin < 0.12)
    return {
        "tool": "Signal_select_image_scale_by_peak_plausibility",
        "image_path": str(image_path),
        "modality": mod or None,
        "selected_duration_s": float(best["duration_s"]),
        "selected_sampling_rate_hz": float(best["sampling_rate_hz_if_used"]),
        "selected_rate_bpm": best.get("rate_bpm"),
        "ranked_candidates": scored,
        "confidence": float(best["combined_score"]),
        "score_margin": margin,
        "requires_confirmation": needs_confirmation,
        "peak_supported": mod in peak_supported,
        "scale_source": "peak_plausibility_plus_duration_prior",
        "prior": prior,
        "disclaimer": "Fallback scale selector: peak plausibility can reject physiologically impossible scales, but cannot prove absolute scale when multiple candidates produce plausible rates.",
    }

def estimate_digitization_resolution_risk(
    image_width: int,
    duration_s: float | None = None,
    expected_signal_bandwidth_hz: float | None = None,
    modality: str | None = None,
) -> dict[str, Any]:
    """Estimate whether an image has enough horizontal resolution to digitize a waveform.

    This is a heuristic Nyquist-style screen for image digitization. If the trace is
    rendered with too few pixels per second for the signal bandwidth, even a perfect
    segmentation mask cannot recover the original waveform reliably.
    """
    modality_bandwidth = {
        "spo2": 0.5,
        "eda": 1.0,
        "resp": 2.0,
        "ppg": 8.0,
        "abp": 12.0,
        "bcg": 15.0,
        "acc": 20.0,
        "ecg": 40.0,
        "scg": 50.0,
        "eeg": 45.0,
        "pcg": 150.0,
        "emg": 250.0,
    }
    mod = str(modality or "").lower()
    bandwidth = float(expected_signal_bandwidth_hz or modality_bandwidth.get(mod, 40.0))
    pixels_per_second = None
    pixels_per_cycle = None
    if duration_s and duration_s > 0:
        pixels_per_second = float(image_width) / float(duration_s)
        pixels_per_cycle = pixels_per_second / max(bandwidth, 1e-9)
    risk = "unknown"
    recommendation = "provide duration_s and modality/bandwidth for resolution screening"
    if pixels_per_cycle is not None:
        if pixels_per_cycle >= 8:
            risk = "low"
            recommendation = "waveform reconstruction is plausible if calibration and trace extraction are correct"
        elif pixels_per_cycle >= 3:
            risk = "medium"
            recommendation = "expect morphology loss; prefer higher-resolution image or task-level features"
        else:
            risk = "high"
            recommendation = "waveform reconstruction is under-resolved; use higher-resolution rendering/source data or image/spectrogram task classifier"
    return {
        "tool": "estimate_digitization_resolution_risk",
        "modality": mod or None,
        "image_width": int(image_width),
        "duration_s": float(duration_s) if duration_s is not None else None,
        "expected_signal_bandwidth_hz": bandwidth,
        "pixels_per_second": pixels_per_second,
        "pixels_per_cycle_at_bandwidth": pixels_per_cycle,
        "risk": risk,
        "recommendation": recommendation,
    }


def recommend_image_signal_strategy(
    image_width: int,
    duration_s: float | None = None,
    modality: str | None = None,
    expected_signal_bandwidth_hz: float | None = None,
) -> dict[str, Any]:
    """Recommend how an agent should handle a signal image before tool execution."""
    risk = estimate_digitization_resolution_risk(image_width, duration_s, expected_signal_bandwidth_hz, modality)
    mod = str(modality or "").lower()
    if duration_s is None:
        strategy = "require_scale_before_digitization"
        tools = ["Signal_estimate_image_scale", "Signal_predict_image_scale_prior"]
        reason = "Physical x-axis scale is missing; use OCR/grid/metadata first, then defer the continue/confirm decision to the trained scale model."
    elif mod == "emg":
        strategy = "image_or_spectrogram_task_model"
        tools = ["EMG_summarize_activation", "EMG_detect_bursts", "EMG_estimate_fatigue"]
        reason = "EMG is often too dense for faithful waveform reconstruction from plot images."
    elif mod == "pcg" and risk["risk"] in {"high", "medium"}:
        strategy = "prefer_audio_or_spectrogram_task_model"
        tools = ["PCG_extract_murmur_features", "PCG_screen_murmur_proxy", "PCG_detect_heart_sounds"]
        reason = "PCG waveform plots need high horizontal resolution; murmur tasks can often use audio/spectrogram features directly."
    elif risk["risk"] == "high":
        strategy = "request_higher_resolution_or_source_signal"
        tools = ["estimate_digitization_resolution_risk", "Signal_digitize_waveform_image_ml"]
        reason = "The image is under-resolved for expected signal bandwidth."
    else:
        strategy = "digitize_waveform_then_run_signal_tools"
        tools = ["Signal_digitize_waveform_image_ml", "Signal_classify_modality"]
        reason = "Image resolution is plausible for waveform reconstruction."
    return {
        "tool": "recommend_image_signal_strategy",
        "strategy": strategy,
        "recommended_tools": tools,
        "reason": reason,
        "resolution_risk": risk,
    }
