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

- CSV signal analysis: upload a one-dimensional signal CSV, provide sampling rate and optional modality hint, then run offline ToolRAG-style rule planning and local tools.
- Waveform image pipeline: upload a plot image, optionally provide sampling rate and y-axis bounds, then run image modality classification, image-to-signal digitization, planning, tools, and grounded report.
- ToolUniverse summary: display current schema counts by modality/evidence level.

Outputs are research-use only and not clinical diagnoses.
