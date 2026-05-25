# Online Demo

This repository includes a Gradio demo in `app.py`. It is designed for Hugging Face Spaces or a small VM.

## Local run

```bash
python -m pip install -r demo_requirements.txt
python app.py
```

Then open `http://localhost:7860`. To request a temporary public Gradio link, run:

```bash
GRADIO_SHARE=1 python app.py
```

## Hugging Face Spaces

Create a new Space with SDK `Gradio`, then upload or sync this repository. For Spaces, either rename `demo_requirements.txt` to `requirements.txt` in the Space, or set the Space build to install it. The demo does not require OpenRouter keys.

## Supported demo paths

- AI bot: upload a waveform image or CSV, ask a natural-language question, and receive a chat answer with modality routing, tool planning, tool execution, compact trace JSON, and grounded report.
- Advanced CSV signal analysis: upload a one-dimensional signal CSV, provide sampling rate and optional modality hint, then run offline ToolRAG-style rule planning and local tools.
- Advanced waveform image pipeline: upload a plot image, optionally provide sampling rate and y-axis bounds, then run image modality classification, image-to-signal digitization, planning, tools, and grounded report.
- ToolUniverse summary: display current schema counts by modality/evidence level.

Outputs are research-use only and not clinical diagnoses.

## TxAgent-style chat UI

The Gradio app opens on an AI bot tab rather than a form-only pipeline. The user uploads a biosignal file and asks naturally; the bot response includes the agent trajectory, selected tools, compact tool trace, final report, and limitation notes in one conversational answer. The older CSV/image forms remain as advanced debugging tabs.

For image inputs, the trajectory shows OCR/title hints, image-classifier output, final modality route, scale/OCR status, panel-aware color-trace digitization, selected tools, and tool execution summaries. This is intentionally closer to the TxAgent demo pattern: users can inspect why a tool route was selected and where the pipeline may have failed.

For Matplotlib-style waveform screenshots with blue traces and multiple panels, the demo first tries `Signal_digitize_plot_image_color_trace`, which extracts the colored waveform component and prefers the lower filtered panel when multiple similarly wide panels are present. If no colored trace is detected, it falls back to the existing dark/ML trace digitizers.

## Waveform segmentation digitizer

For low-resolution or non-blue waveform plots, the demo now has a trained segmentation fallback: `Signal_digitize_waveform_image_unet`. The model predicts a curve mask from the plot image, then converts the mask to a one-dimensional waveform with the same path/median/lazy extraction options used by the classical digitizers.

Multi-panel plot images are processed panel-by-panel. The target-aware segmentation path is tried first because screenshots often contain colored grid/text artifacts that confuse simple color thresholding. Color-trace, ML, and dark-trace digitizers are fallback routes. Successful multi-panel runs return a `signals[]` list, `num_panels`, and per-panel CSV files such as `digitized_signal_panel_01.csv`, while the primary CSV remains the first recovered panel for backward-compatible downstream tool execution. The chat UI renders every recovered panel preview instead of silently showing only the selected/bottom trace.

A lightweight comparison on the rendered digitization benchmark favored an augmented U-Net over the current tiny DeepLabV3-style and SegFormer-lite prototypes. The current default model is a weighted 3-class target-aware U-Net trained on 648 synthetic rendered plots covering clean/grid/color/dark/low-resolution styles plus multi-trace and multi-panel screenshots. It predicts `background`, `target_trace`, and `distractor_trace_axes_text_grid`. On a source-record held-out validation split, it reached mean Dice 0.8337 and mean IoU 0.7201 for the target trace, with multi-panel + multi-trace Dice improving to 0.7777. This is still a research prototype; real screenshots with unusual axes, multiple traces, or severe compression should be verified visually.

The chat demo also embeds visual checkpoints in image runs: the uploaded plot, a multi-class segmentation overlay: blue non-target/distractor pixels, red target-trace pixels, and amber selected mask-area/panel, and the recovered digitized waveform preview. These are included directly in the agent answer so users can inspect whether the correct mask area, panel, and trace were selected before trusting downstream measurements.

Axis OCR is now a separate visible step in the image pipeline. `Signal_extract_plot_axes_ocr` crops each selected panel's x-axis and y-axis bands, reads tick labels with Tesseract, applies conservative post-processing for common OCR errors such as `10` -> `1` and `-1` -> `4`, and passes inferred x-axis duration/sampling rate plus y-axis value bounds into digitization when readable. If tick labels are unreadable, the report explicitly marks the axis calibration as partial instead of silently pretending the physical scale is known.

## Target-aware segmentation experiment

A weighted 3-class target-aware U-Net is now the default image digitization segmentation model. Compared with the previous binary multi-style model on the same target-aware validation split, it improved mean Dice from 0.8260 to 0.8337 and mean IoU from 0.7115 to 0.7201. The largest gain is on multi-panel + multi-trace plots, where Dice improved from 0.6841 to 0.7777.
