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

A lightweight comparison on the rendered digitization benchmark favored an augmented U-Net over the current tiny DeepLabV3-style and SegFormer-lite prototypes. The current default model is trained on 648 synthetic rendered plots covering clean/grid/color/dark/low-resolution styles plus multi-trace and multi-panel screenshots. On a source-record held-out validation split of this multi-style benchmark, it reached mean Dice 0.8153 and mean IoU 0.6970 at threshold 0.65. This is still a research prototype; real screenshots with unusual axes, multiple traces, or severe compression should be verified visually.

The chat demo also embeds visual checkpoints in image runs: the uploaded plot, a red U-Net segmentation overlay, and the recovered digitized waveform preview. These are included directly in the agent answer so users can inspect whether the correct panel and trace were selected before trusting downstream measurements.
