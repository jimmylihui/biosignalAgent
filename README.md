# BioSignalAgent

Tool-first prototype for ECG, PPG, and BCG signal reasoning.

This first version keeps the LLM out of the critical path: signal analysis is done by explicit Python tools, and the agent selects a basic workflow from the requested modality. Later, the same tool schemas can be used for tool calling, ToolRAG, and instruction tuning traces.

## Example

```bash
python examples/run_basic_report.py --csv path/to/signal.csv --modality ecg --sampling-rate 250
```

The CSV should contain one numeric column, or a column named `signal`.
