from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from biosignal_agent.agent.llm_agent import OpenRouterBioSignalAgent
from biosignal_agent.agent.openrouter_client import DEFAULT_MODEL
from biosignal_agent.agent.planning_agent import PlanningBioSignalAgent
from biosignal_agent.agent.tool_retriever import ToolRetriever
from biosignal_agent.agent.tool_registry import TOOLS
from biosignal_agent.evaluation.planning_cases import DEFAULT_PLANNING_CASES, PlanningCase


def evaluate_cases(
    cases: list[PlanningCase] | None = None,
    planner_name: str = "rule",
    model: str = DEFAULT_MODEL,
    retrieved_tool_count: int = 5,
    execute: bool = False,
    llm_timeout: int = 30,
    llm_retry_max: int = 1,
    llm_retry_delay: float = 2.0,
    llm_fallback_to_rules: bool = True,
    signal_paths: dict[str, str] | None = None,
    sampling_rates: dict[str, float] | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    cases = cases or DEFAULT_PLANNING_CASES
    signal_paths = signal_paths or {}
    sampling_rates = sampling_rates or {}
    retriever = ToolRetriever()
    rule_agent = PlanningBioSignalAgent()
    llm_agent = OpenRouterBioSignalAgent(
        model=model,
        retrieved_tool_count=retrieved_tool_count,
        llm_timeout=llm_timeout,
        llm_retry_max=llm_retry_max,
        llm_retry_delay=llm_retry_delay,
        fallback_to_rules=llm_fallback_to_rules,
    )
    rows = []
    for index, case in enumerate(cases, start=1):
        if progress:
            print(f"[{index}/{len(cases)}] {planner_name} {case.case_id}", flush=True)
        retrieved = [
            schema["name"]
            for schema in retriever.retrieve(case.question, top_k=retrieved_tool_count, modality=case.modality)
        ]
        planner_error = None
        if planner_name == "openrouter":
            signal_path = signal_paths.get(case.modality, "placeholder.csv")
            sampling_rate = sampling_rates.get(case.modality, 100.0)
            try:
                plan = llm_agent.plan(case.question, signal_path, sampling_rate, fallback_modality=case.modality)
                planned_tools = [call["name"] for call in plan["tool_calls"]]
                actual_planner = plan.get("planner", "openrouter")
                planner_error = plan.get("fallback_reason")
            except Exception as exc:
                planned_tools = []
                actual_planner = "openrouter_error"
                planner_error = str(exc)
        else:
            planned_tools = rule_agent.plan(case.question, case.modality)
            actual_planner = "rule"
        expected = set(case.expected_tools)
        planned = set(planned_tools)
        retrieved_set = set(retrieved)
        missing_from_plan = sorted(expected - planned)
        unexpected_tools = sorted(planned - expected)
        missing_from_retrieval = sorted(expected - retrieved_set)
        execution_ok = None
        execution_errors: list[str] = []
        if execute:
            signal_path = signal_paths.get(case.modality)
            sampling_rate = sampling_rates.get(case.modality)
            if not signal_path or not sampling_rate:
                execution_ok = False
                execution_errors.append("missing signal path or sampling rate")
            else:
                execution_ok = True
                for tool_name in planned_tools:
                    try:
                        result = TOOLS[tool_name](signal_path=signal_path, sampling_rate=sampling_rate, column=None)
                        if isinstance(result, dict) and result.get("error"):
                            execution_ok = False
                            execution_errors.append(f"{tool_name}: {result['error']}")
                    except Exception as exc:  # pragma: no cover - command-line safety net
                        execution_ok = False
                        execution_errors.append(f"{tool_name}: {exc}")
        rows.append({
            "case_id": case.case_id,
            "question": case.question,
            "modality": case.modality,
            "expected_tools": list(case.expected_tools),
            "retrieved_tools": retrieved,
            "planned_tools": planned_tools,
            "planner": actual_planner,
            "retrieval_pass": not missing_from_retrieval,
            "planning_pass": not missing_from_plan and not unexpected_tools,
            "execution_ok": execution_ok,
            "missing_from_retrieval": missing_from_retrieval,
            "missing_from_plan": missing_from_plan,
            "unexpected_tools": unexpected_tools,
            "execution_errors": execution_errors,
            "planner_error": planner_error,
        })
    retrieval_passes = sum(1 for row in rows if row["retrieval_pass"])
    planning_passes = sum(1 for row in rows if row["planning_pass"])
    planner_backend_counts: dict[str, int] = {}
    for row in rows:
        planner_backend_counts[row["planner"]] = planner_backend_counts.get(row["planner"], 0) + 1
    executable_rows = [row for row in rows if row["execution_ok"] is not None]
    execution_passes = sum(1 for row in executable_rows if row["execution_ok"])
    return {
        "planner": planner_name,
        "model": model if planner_name == "openrouter" else None,
        "retrieved_tool_count": retrieved_tool_count,
        "num_cases": len(rows),
        "retrieval_accuracy": retrieval_passes / len(rows) if rows else 0.0,
        "planning_accuracy": planning_passes / len(rows) if rows else 0.0,
        "execution_accuracy": execution_passes / len(executable_rows) if executable_rows else None,
        "planner_backend_counts": planner_backend_counts,
        "cases": rows,
    }


def write_eval_outputs(report: dict[str, Any], output_json: str | Path, output_csv: str | Path | None = None) -> None:
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2))
    if output_csv is None:
        return
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "modality",
        "planner",
        "retrieval_pass",
        "planning_pass",
        "execution_ok",
        "expected_tools",
        "retrieved_tools",
        "planned_tools",
        "missing_from_retrieval",
        "missing_from_plan",
        "unexpected_tools",
        "execution_errors",
        "planner_error",
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["cases"]:
            writer.writerow({
                key: json.dumps(row[key]) if isinstance(row.get(key), list) else row.get(key)
                for key in fieldnames
            })
