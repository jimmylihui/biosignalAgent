from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from biosignal_agent.agent.schema_loader import load_tool_schemas
from biosignal_agent.agent.tool_registry import WORKFLOWS


def iter_trace_files(trace_dir: str | Path) -> Iterable[Path]:
    trace_dir = Path(trace_dir)
    for path in sorted(trace_dir.glob("*.json")):
        if path.is_file():
            yield path


def load_trace(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _metadata(trace: dict[str, Any], task: str) -> dict[str, Any]:
    return {
        "task": task,
        "planner": trace.get("planner"),
        "model": trace.get("model"),
        "trace_path": trace.get("trace_path"),
        "trace_id": trace.get("trace_id"),
    }


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
            {"role": "system", "content": "Plan physiological signal analysis by selecting valid local tools. Return strict JSON only."},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
            {"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=True)},
        ],
        "metadata": _metadata(trace, "biosignal_tool_planning"),
    }


def trace_to_tool_use_sample(trace: dict[str, Any]) -> dict[str, Any] | None:
    question = trace.get("question")
    tool_plan = trace.get("tool_plan")
    tool_results = trace.get("tool_results")
    if not question or not tool_plan or not tool_results:
        return None
    user_payload = {
        "question": question,
        "signal": trace.get("signal"),
        "tool_plan": tool_plan,
    }
    assistant_payload = {"tool_results": tool_results}
    return {
        "task": "biosignal_tool_execution_trace",
        "messages": [
            {"role": "system", "content": "Execute planned biosignal tools and return structured tool results."},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
            {"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=True)},
        ],
        "metadata": _metadata(trace, "biosignal_tool_execution_trace"),
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
        "metadata": _metadata(trace, "biosignal_report_generation"),
    }


def session_to_planning_sample(trace: dict[str, Any]) -> dict[str, Any] | None:
    session = trace.get("session")
    runs = trace.get("runs") or []
    if not session or not runs:
        return None
    assistant_payload = {
        "signal_plans": [
            {
                "signal_label": run.get("signal_label"),
                "modality": run.get("modality"),
                "retrieved_tools": run.get("retrieved_tools"),
                "tool_calls": run.get("tool_plan"),
            }
            for run in runs
        ]
    }
    return {
        "task": "biosignal_session_tool_planning",
        "messages": [
            {"role": "system", "content": "Plan a multi-signal physiological analysis session. Route each signal to valid local tools and return strict JSON."},
            {"role": "user", "content": json.dumps(session, ensure_ascii=True)},
            {"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=True)},
        ],
        "metadata": _metadata(trace, "biosignal_session_tool_planning"),
    }


def session_to_tool_use_sample(trace: dict[str, Any]) -> dict[str, Any] | None:
    session = trace.get("session")
    runs = trace.get("runs") or []
    if not session or not runs:
        return None
    user_payload = {
        "session": session,
        "signal_plans": [
            {"signal_label": run.get("signal_label"), "modality": run.get("modality"), "tool_plan": run.get("tool_plan")}
            for run in runs
        ],
    }
    assistant_payload = {
        "signal_tool_results": [
            {"signal_label": run.get("signal_label"), "modality": run.get("modality"), "tool_results": run.get("tool_results")}
            for run in runs
        ]
    }
    return {
        "task": "biosignal_session_tool_execution_trace",
        "messages": [
            {"role": "system", "content": "Execute a multi-signal biosignal tool plan and return structured per-signal tool results."},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
            {"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=True)},
        ],
        "metadata": _metadata(trace, "biosignal_session_tool_execution_trace"),
    }


def session_to_report_sample(trace: dict[str, Any]) -> dict[str, Any] | None:
    session = trace.get("session")
    runs = trace.get("runs") or []
    if not session or not runs:
        return None
    user_payload = {
        "session": session,
        "signal_tool_results": [
            {"signal_label": run.get("signal_label"), "modality": run.get("modality"), "tool_results": run.get("tool_results")}
            for run in runs
        ],
        "disclaimer_required": True,
    }
    assistant_payload = {
        "session_summary": "\n\n".join(run.get("final_report", "") for run in runs if run.get("final_report"))
    }
    return {
        "task": "biosignal_session_report_generation",
        "messages": [
            {"role": "system", "content": "Write a concise multi-signal research-use biosignal report from per-signal tool outputs. Do not make a clinical diagnosis."},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
            {"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=True)},
        ],
        "metadata": _metadata(trace, "biosignal_session_report_generation"),
    }


def negative_planning_samples(limit: int | None = None) -> list[dict[str, Any]]:
    schemas = load_tool_schemas()
    known_modalities = sorted(WORKFLOWS)
    samples = [
        {
            "task": "biosignal_negative_tool_planning",
            "messages": [
                {"role": "system", "content": "Reject unsupported or unsafe biosignal tool-planning requests. Return strict JSON."},
                {"role": "user", "content": json.dumps({"question": "Diagnose myocardial infarction from this waveform and prescribe treatment.", "signal": {"modality": "ecg"}}, ensure_ascii=True)},
                {"role": "assistant", "content": json.dumps({"status": "refuse_clinical_diagnosis", "reason": "The framework can run signal-processing tools but cannot diagnose or prescribe treatment.", "allowed_actions": WORKFLOWS["ecg"]}, ensure_ascii=True)},
            ],
            "metadata": {"task": "biosignal_negative_tool_planning", "negative_type": "clinical_diagnosis"},
        },
        {
            "task": "biosignal_negative_tool_planning",
            "messages": [
                {"role": "system", "content": "Reject unsupported or unsafe biosignal tool-planning requests. Return strict JSON."},
                {"role": "user", "content": json.dumps({"question": "Analyze this ultrasound image with the ECG tools.", "signal": {"modality": "ultrasound"}, "available_modalities": known_modalities}, ensure_ascii=True)},
                {"role": "assistant", "content": json.dumps({"status": "unsupported_modality", "supported_modalities": known_modalities}, ensure_ascii=True)},
            ],
            "metadata": {"task": "biosignal_negative_tool_planning", "negative_type": "unsupported_modality"},
        },
        {
            "task": "biosignal_negative_tool_planning",
            "messages": [
                {"role": "system", "content": "Reject invalid tool selections. Return strict JSON."},
                {"role": "user", "content": json.dumps({"question": "Compute EEG bandpower", "signal": {"modality": "eeg"}, "invalid_tool": "ECG_compute_hrv"}, ensure_ascii=True)},
                {"role": "assistant", "content": json.dumps({"status": "invalid_tool_for_modality", "modality": "eeg", "valid_tools": WORKFLOWS["eeg"]}, ensure_ascii=True)},
            ],
            "metadata": {"task": "biosignal_negative_tool_planning", "negative_type": "wrong_tool_modality"},
        },
    ]
    if limit is not None:
        return samples[:limit]
    return samples


def samples_from_trace(trace: dict[str, Any], include_reports: bool = True, include_tool_use: bool = True) -> list[dict[str, Any]]:
    if "session" in trace:
        samples = [session_to_planning_sample(trace)]
        if include_tool_use:
            samples.append(session_to_tool_use_sample(trace))
        if include_reports:
            samples.append(session_to_report_sample(trace))
        return [sample for sample in samples if sample is not None]
    samples = [trace_to_planning_sample(trace)]
    if include_tool_use:
        samples.append(trace_to_tool_use_sample(trace))
    if include_reports:
        samples.append(trace_to_report_sample(trace))
    return [sample for sample in samples if sample is not None]


def export_trace_dataset(
    trace_dir: str | Path,
    output_jsonl: str | Path,
    include_reports: bool = True,
    include_tool_use: bool = True,
    include_negative: bool = False,
) -> dict[str, int]:
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    counts = {"trace_files": 0, "total_samples": 0}
    with output_jsonl.open("w") as handle:
        for path in iter_trace_files(trace_dir):
            counts["trace_files"] += 1
            trace = load_trace(path)
            trace.setdefault("trace_path", str(path))
            for sample in samples_from_trace(trace, include_reports=include_reports, include_tool_use=include_tool_use):
                task = sample["task"]
                counts[task] = counts.get(task, 0) + 1
                counts["total_samples"] += 1
                handle.write(json.dumps(sample, ensure_ascii=True) + "\n")
        if include_negative:
            for sample in negative_planning_samples():
                task = sample["task"]
                counts[task] = counts.get(task, 0) + 1
                counts["total_samples"] += 1
                handle.write(json.dumps(sample, ensure_ascii=True) + "\n")
    return counts
