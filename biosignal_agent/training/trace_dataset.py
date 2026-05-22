from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def iter_trace_files(trace_dir: str | Path) -> Iterable[Path]:
    trace_dir = Path(trace_dir)
    for path in sorted(trace_dir.glob("*.json")):
        if path.is_file():
            yield path


def load_trace(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def trace_to_planning_sample(trace: dict[str, Any]) -> dict[str, Any] | None:
    question = trace.get("question")
    tool_plan = trace.get("tool_plan")
    if not question or not tool_plan:
        return None
    user_payload = {
        "question": question,
        "signal": trace.get("signal"),
        "retrieved_tools": trace.get("retrieved_tools"),
    }
    assistant_payload = {
        "modality": trace.get("modality"),
        "tool_calls": tool_plan,
    }
    return {
        "task": "biosignal_tool_planning",
        "messages": [
            {"role": "system", "content": "Plan ECG, PPG, and BCG analysis by selecting valid local tools. Return strict JSON only."},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
            {"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=True)},
        ],
        "metadata": {
            "planner": trace.get("planner"),
            "model": trace.get("model"),
            "trace_path": trace.get("trace_path"),
        },
    }


def trace_to_report_sample(trace: dict[str, Any]) -> dict[str, Any] | None:
    question = trace.get("question")
    tool_results = trace.get("tool_results")
    final_report = trace.get("final_report")
    if not question or not tool_results or not final_report:
        return None
    user_payload = {
        "question": question,
        "tool_results": tool_results,
        "disclaimer_required": True,
    }
    return {
        "task": "biosignal_report_generation",
        "messages": [
            {"role": "system", "content": "Write a concise research-use biosignal report from tool outputs. Do not make a clinical diagnosis."},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
            {"role": "assistant", "content": final_report},
        ],
        "metadata": {
            "planner": trace.get("planner"),
            "model": trace.get("model"),
            "trace_path": trace.get("trace_path"),
        },
    }


def export_trace_dataset(trace_dir: str | Path, output_jsonl: str | Path, include_reports: bool = True) -> dict[str, int]:
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    counts = {"trace_files": 0, "planning_samples": 0, "report_samples": 0, "total_samples": 0}
    with output_jsonl.open("w") as handle:
        for path in iter_trace_files(trace_dir):
            counts["trace_files"] += 1
            trace = load_trace(path)
            trace.setdefault("trace_path", str(path))
            samples = [trace_to_planning_sample(trace)]
            if include_reports:
                samples.append(trace_to_report_sample(trace))
            for sample in samples:
                if sample is None:
                    continue
                task = sample["task"]
                if task == "biosignal_tool_planning":
                    counts["planning_samples"] += 1
                elif task == "biosignal_report_generation":
                    counts["report_samples"] += 1
                counts["total_samples"] += 1
                handle.write(json.dumps(sample, ensure_ascii=True) + "\n")
    return counts
