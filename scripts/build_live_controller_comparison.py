#!/usr/bin/env python3
"""Build a paper-ready comparison table for live BioSignalAgent controllers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def controller_row(label: str, path: str | Path, note: str = "") -> list[str]:
    data = load_json(path)
    return [
        label,
        fmt(data.get("num_cases")),
        fmt(data.get("planner_strict_parse_rate")),
        fmt(data.get("planner_parse_rate")),
        fmt(data.get("planning_accuracy")),
        fmt(data.get("tool_f1")),
        fmt(data.get("execution_success")),
        fmt(data.get("report_factuality_score")),
        fmt(data.get("overall_hmean")),
        note,
    ]


def openrouter_row(label: str, path: str | Path, note: str = "") -> list[str]:
    data = load_json(path)
    return [
        label,
        fmt(data.get("num_cases")),
        "",
        fmt(data.get("parse_rate")),
        fmt(data.get("planning_accuracy")),
        fmt(data.get("tool_f1")),
        "",
        "",
        "",
        note,
    ]


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-md", default="/data1/jiahui/biosignal-agent/outputs/paper_tables/table13_live_controller_comparison.md")
    ap.add_argument("--replay", default="/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_replay.json")
    ap.add_argument("--live-v3", default="/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_live.json")
    ap.add_argument("--live-v4", default="/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_live_v4.json")
    ap.add_argument("--openrouter", default="/data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_openrouter_owl_alpha.json")
    args = ap.parse_args()

    rows = [
        openrouter_row("OpenRouter owl-alpha planner", args.openrouter, "external free LLM planner only"),
        controller_row("Replay SFT controller", args.replay, "artifact upper bound; planner/report replay"),
        controller_row("Live SFT controller v3", args.live_v3, "first real live LoRA controller"),
        controller_row("Live SFT controller v4", args.live_v4, "hard-case live-controller SFT"),
    ]
    headers = [
        "Method",
        "Cases",
        "Strict parse",
        "Recovered parse",
        "Planning",
        "Tool F1",
        "Exec success",
        "Report score",
        "Overall H-mean",
        "Note",
    ]
    text = "# Table 13. Live Controller Comparison\n\n"
    text += "This table separates external planner-only and controller-shaped evaluations from replay upper bounds. Live rows generate planner outputs during evaluation instead of reading saved SFT outputs.\n\n"
    text += markdown_table(headers, rows)
    text += "\nInterpretation: v4 closes most of the replay-to-live gap for tool selection while preserving live execution success. Remaining failures are concentrated in multimodal/session exact tool-set matching and report wording coverage.\n"

    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(out)


if __name__ == "__main__":
    main()
