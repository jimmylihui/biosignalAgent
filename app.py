from __future__ import annotations

import json
import os
import tempfile
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
)
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
    if not csv_file:
        return "Upload a CSV file first.", {}, None
    if not sampling_rate or sampling_rate <= 0:
        return "Sampling rate must be a positive number.", {}, None
    question = (question or DEFAULT_CSV_QUESTION).strip()
    fallback = None if modality_hint == "auto" else modality_hint
    try:
        values, used_column = _read_signal(csv_file, column or None)
        classifier = Signal_classify_modality(csv_file, float(sampling_rate), column=used_column)
        if fallback is None:
            fallback = classifier.get("predicted_modality")
        result = PlanningBioSignalAgent().run(question, csv_file, float(sampling_rate), used_column, fallback)
        result["input_column"] = used_column
        result["modality_classifier"] = classifier
        result["disclaimer"] = DISCLAIMER
        return _short_report(result), _jsonable(result), _plot_signal(values, sampling_rate, "Uploaded signal")
    except Exception as exc:
        return f"Error: {type(exc).__name__}: {exc}", {"error": str(exc), "stage": "csv_demo"}, None


def _digitize_with_fallback(image_path: str, sampling_rate: float | None, out_csv: str, value_min: float | None, value_max: float | None, trace_method: str):
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
        fallback["fallback_from_ml_error"] = result.get("error")
        return fallback
    return result


def run_image_demo(image_file: str | None, question: str, sampling_rate: float | None, modality_hint: str, value_min: float | None, value_max: float | None, trace_method: str):
    if not image_file:
        return "Upload an image first.", {}, None, None
    question = (question or DEFAULT_IMAGE_QUESTION).strip()
    fallback = None if modality_hint == "auto" else modality_hint
    try:
        work_dir = Path(tempfile.mkdtemp(prefix="biosignalagent_demo_"))
        out_csv = work_dir / "digitized_signal.csv"
        image_classifier = Signal_classify_modality_from_image(image_file)
        if fallback is None:
            fallback = image_classifier.get("predicted_modality")
        scale = Signal_estimate_image_scale(image_file, duration_s=None, use_ocr=True)
        sr = float(sampling_rate) if sampling_rate and sampling_rate > 0 else scale.get("sampling_rate")
        digitized = _digitize_with_fallback(image_file, sr, str(out_csv), value_min, value_max, trace_method)
        if digitized.get("error"):
            payload = {"image_classifier": image_classifier, "scale": scale, "digitization": digitized, "disclaimer": DISCLAIMER}
            return "Digitization failed. Inspect JSON details.", _jsonable(payload), None, None
        values, used_column = _read_signal(str(out_csv), "signal")
        report = PlanningBioSignalAgent().run(question, str(out_csv), float(sr or 100.0), used_column, fallback)
        payload = {
            "image_classifier": image_classifier,
            "scale": scale,
            "digitization": digitized,
            "signal_report": report,
            "disclaimer": DISCLAIMER,
        }
        return _short_report(report), _jsonable(payload), _plot_signal(values, sr, "Digitized waveform"), str(out_csv)
    except Exception as exc:
        return f"Error: {type(exc).__name__}: {exc}", {"error": str(exc), "stage": "image_demo"}, None, None


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
    with gr.Blocks(title="BioSignalAgent Demo") as demo:
        gr.Markdown(
            "# BioSignalAgent Demo\n"
            "Upload a biosignal CSV or waveform image, run offline tool planning/execution, and inspect grounded tool outputs. "
            "This public demo is for research prototyping only and does not provide medical diagnosis."
        )
        with gr.Tab("CSV signal"):
            with gr.Row():
                csv_file = gr.File(label="Signal CSV", file_types=[".csv"], type="filepath")
                with gr.Column():
                    csv_question = gr.Textbox(label="Question", value=DEFAULT_CSV_QUESTION, lines=3)
                    csv_sampling_rate = gr.Number(label="Sampling rate (Hz)", value=250)
                    csv_modality = gr.Dropdown(label="Modality hint", choices=MODALITIES, value="auto")
                    csv_column = gr.Textbox(label="Column name (optional)", value="")
                    csv_button = gr.Button("Run CSV analysis", variant="primary")
            csv_report = gr.Markdown(label="Report")
            csv_plot = gr.Plot(label="Signal preview")
            csv_json = gr.JSON(label="Tool trace JSON")
            csv_button.click(run_csv_demo, [csv_file, csv_question, csv_sampling_rate, csv_modality, csv_column], [csv_report, csv_json, csv_plot])

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
            image_report = gr.Markdown(label="Report")
            image_plot = gr.Plot(label="Digitized signal preview")
            image_json = gr.JSON(label="Pipeline JSON")
            digitized_file = gr.File(label="Digitized CSV")
            image_button.click(run_image_demo, [image_file, image_question, image_sampling_rate, image_modality, value_min, value_max, trace_method], [image_report, image_json, image_plot, digitized_file])

        with gr.Tab("ToolUniverse"):
            gr.Markdown(summarize_tool_universe())
    return demo


demo = build_demo()

if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE", "0").lower() in {"1", "true", "yes"}
    demo.launch(server_name="0.0.0.0", server_port=7860, share=share)
