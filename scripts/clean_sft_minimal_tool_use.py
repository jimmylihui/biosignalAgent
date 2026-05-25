#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

QUALITY_INTENT_RE = re.compile(
    r"\b(quality|artifact|artefact|noise|noisy|confidence|confident|reliable|reliability|limitation|limitations|valid|validity|usable|trust|low[- ]?res|low resolution|resolution|risk|robust|uncertain)\b",
    re.I,
)
HR_INTENT_RE = re.compile(r"\b(hr|heart rate|pulse rate|cardiac rate|beat rate|beats per minute|bpm)\b", re.I)
RATE_INTENT_RE = re.compile(r"\b(respiratory rate|breathing rate|respiration rate|rate)\b", re.I)
HRV_INTENT_RE = re.compile(r"\b(hrv|heart rate variability|prv|pulse rate variability)\b", re.I)
QUALITY_TOOL_RE = re.compile(
    r"(^|_)(assess_quality|detect_artifacts|artifact|quality)$|_assess_quality$|_detect_artifacts$|estimate_digitization_resolution_risk|recommend_image_signal_strategy",
    re.I,
)

# Tools that mostly serve as broad preflight/context rather than the requested measurement.
OPTIONAL_PREFLIGHT_RE = re.compile(
    r"^(ECG|PPG|PCG|SCG|BCG|EMG|EDA|EEG|RESP|SpO2|ABP|ACC)_assess_quality$|"
    r"^Signal_detect_artifacts$|^estimate_digitization_resolution_risk$|^recommend_image_signal_strategy$",
    re.I,
)

# If a precise measurement task is requested, keep these compact measurement tools.
MEASUREMENT_KEEPERS = {
    "ecg": {
        "hr": {"ECG_detect_r_peaks"},
        "hrv": {"ECG_detect_r_peaks", "ECG_compute_hrv"},
    },
    "ppg": {
        "hr": {"PPG_detect_peaks"},
        "hrv": {"PPG_detect_peaks", "PPG_compute_prv"},
    },
    "bcg": {"hr": {"BCG_detect_j_peaks"}},
    "scg": {"hr": {"SCG_detect_j_peaks"}},
    "pcg": {"hr": {"PCG_detect_heart_sounds", "PCG_estimate_heart_rate"}},
    "abp": {"hr": {"ABP_detect_pulses"}},
    "resp": {"rate": {"RESP_estimate_rate"}},
}

# Keep planning/digitization bootstrap tools for image cases; these are not redundant even if not the final biomedical task.
IMAGE_BOOTSTRAP_PREFIXES = (
    "Signal_classify_modality",
    "Signal_classify_modality_from_image",
    "Signal_read_image_text_ocr",
    "Signal_extract_plot_axes_ocr",
    "Signal_estimate_image_scale",
    "Signal_digitize_waveform_image",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def extract_user_payload(row: dict[str, Any]) -> dict[str, Any]:
    for msg in row.get("messages") or []:
        if msg.get("role") == "user":
            content = msg.get("content") or "{}"
            try:
                parsed = json.loads(content)
                return parsed if isinstance(parsed, dict) else {"question": content}
            except Exception:
                return {"question": content}
    return {}


def question_text(row: dict[str, Any]) -> str:
    payload = extract_user_payload(row)
    bits = [str(payload.get("question") or "")]
    for key in ("planner_instruction", "input_type", "modality_hint", "modality"):
        if payload.get(key):
            bits.append(str(payload.get(key)))
    return " ".join(bits)


def explicit_quality_requested(text: str) -> bool:
    return bool(QUALITY_INTENT_RE.search(text or ""))


def modality_from_tool(tool: str) -> str:
    prefix = str(tool or "").split("_", 1)[0].lower()
    return {"spo2": "spo2"}.get(prefix, prefix)


def compact_measurement_intent(text: str) -> set[str]:
    intents = set()
    if HR_INTENT_RE.search(text or ""):
        intents.add("hr")
    if HRV_INTENT_RE.search(text or ""):
        intents.add("hrv")
    if RATE_INTENT_RE.search(text or "") and not HR_INTENT_RE.search(text or ""):
        intents.add("rate")
    return intents


def should_drop_tool(tool: str, question: str) -> bool:
    if not tool:
        return False
    if explicit_quality_requested(question):
        return False
    if any(str(tool).startswith(prefix) for prefix in IMAGE_BOOTSTRAP_PREFIXES):
        return False
    if OPTIONAL_PREFLIGHT_RE.search(str(tool)) or QUALITY_TOOL_RE.search(str(tool)):
        return True
    return False


def minimalize_tool_calls(calls: list[dict[str, Any]], question: str) -> tuple[list[dict[str, Any]], list[str]]:
    removed: list[str] = []
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    intents = compact_measurement_intent(question)
    for call in calls:
        if not isinstance(call, dict):
            kept.append(call)
            continue
        name = call.get("name") or call.get("tool")
        if should_drop_tool(str(name), question):
            removed.append(str(name))
            continue
        # For HR-only questions, avoid broad companion tools from the same modality unless they are direct keepers.
        mod = modality_from_tool(str(name))
        if intents and not explicit_quality_requested(question):
            allowed = set()
            for intent in intents:
                allowed |= MEASUREMENT_KEEPERS.get(mod, {}).get(intent, set())
            if allowed and str(name) not in allowed and not any(str(name).startswith(prefix) for prefix in IMAGE_BOOTSTRAP_PREFIXES):
                # Do not drop non-quality specialized tools for non-HR intents; this guard only handles known measurement bundles.
                if any(str(name).startswith(f"{mod.upper()}_") or str(name).startswith(f"{mod.capitalize()}_") for mod in [mod]):
                    removed.append(str(name))
                    continue
        if str(name) in seen:
            removed.append(str(name))
            continue
        seen.add(str(name))
        kept.append(call)
    return kept, removed


def clean_assistant_json(value: Any, question: str) -> tuple[Any, list[str]]:
    removed_all: list[str] = []
    if isinstance(value, dict):
        value = deepcopy(value)
        if isinstance(value.get("tool_calls"), list):
            value["tool_calls"], removed = minimalize_tool_calls(value["tool_calls"], question)
            removed_all.extend(removed)
        if isinstance(value.get("tool_plan"), list):
            value["tool_plan"], removed = minimalize_tool_calls(value["tool_plan"], question)
            removed_all.extend(removed)
        if isinstance(value.get("tool_results"), list):
            value["tool_results"], removed = minimalize_tool_results(value["tool_results"], question)
            removed_all.extend(removed)
        if isinstance(value.get("signal_plans"), list):
            for plan in value["signal_plans"]:
                if isinstance(plan, dict) and isinstance(plan.get("tool_calls"), list):
                    plan["tool_calls"], removed = minimalize_tool_calls(plan["tool_calls"], question)
                    removed_all.extend(removed)
        return value, removed_all
    return value, removed_all


def minimalize_tool_results(results: list[dict[str, Any]], question: str) -> tuple[list[dict[str, Any]], list[str]]:
    kept = []
    removed = []
    seen = set()
    for result in results:
        if not isinstance(result, dict):
            kept.append(result)
            continue
        name = result.get("tool") or result.get("name")
        if should_drop_tool(str(name), question):
            removed.append(str(name))
            continue
        if str(name) in seen:
            removed.append(str(name))
            continue
        seen.add(str(name))
        kept.append(result)
    return kept, removed


def clean_row(row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    row = deepcopy(row)
    q = question_text(row)
    removed_all: list[str] = []
    for msg in row.get("messages") or []:
        if msg.get("role") not in {"assistant", "user"}:
            continue
        content = msg.get("content")
        try:
            parsed = json.loads(content)
        except Exception:
            continue
        cleaned, removed = clean_assistant_json(parsed, q)
        if removed:
            msg["content"] = json.dumps(cleaned, ensure_ascii=False, sort_keys=True)
            removed_all.extend(removed)
    if removed_all:
        meta = row.setdefault("metadata", {})
        meta["minimal_tool_cleaning"] = {
            "removed_tools": removed_all,
            "policy": "drop redundant quality/artifact/risk preflight unless explicitly requested; dedupe repeated calls",
        }
    return row, removed_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Create minimal-tool-use SFT JSONL files by removing redundant preflight tools.")
    parser.add_argument("inputs", nargs="+", help="Input JSONL files")
    parser.add_argument("--out-dir", default="/data1/jiahui/biosignal-agent/outputs/minimal_tool_sft")
    parser.add_argument("--summary", default="/data1/jiahui/biosignal-agent/outputs/minimal_tool_sft/summary.json")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    summary = {"files": [], "total_rows": 0, "changed_rows": 0, "removed_tool_counts": {}}
    for inp in args.inputs:
        in_path = Path(inp)
        rows = read_jsonl(in_path)
        cleaned_rows = []
        changed = 0
        for row in rows:
            cleaned, removed = clean_row(row)
            cleaned_rows.append(cleaned)
            if removed:
                changed += 1
                for tool in removed:
                    summary["removed_tool_counts"][tool] = summary["removed_tool_counts"].get(tool, 0) + 1
        out_path = out_dir / f"{in_path.stem}_minimal_tools.jsonl"
        write_jsonl(out_path, cleaned_rows)
        summary["files"].append({"input": str(in_path), "output": str(out_path), "rows": len(rows), "changed_rows": changed})
        summary["total_rows"] += len(rows)
        summary["changed_rows"] += changed
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
