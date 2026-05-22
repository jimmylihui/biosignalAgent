from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_instruction_dataset import validate


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "missing": True}
    return json.loads(path.read_text())


def manifest_summary(path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("missing"):
        return payload
    records = payload.get("records", [])
    modalities = Counter(record.get("modality") for record in records)
    datasets = Counter(record.get("dataset") for record in records)
    return {
        "path": str(path),
        "num_records": len(records),
        "modalities": dict(sorted(modalities.items())),
        "datasets": dict(sorted(datasets.items())),
    }


def framework_summary(name: str, path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("missing"):
        return {"name": name, **payload}
    modalities = Counter(row.get("modality") for row in payload.get("cases", []))
    return {
        "name": name,
        "path": str(path),
        "planner": payload.get("planner"),
        "model": payload.get("model"),
        "num_records": payload.get("num_records"),
        "num_case_runs": payload.get("num_case_runs"),
        "retrieval_accuracy": payload.get("retrieval_accuracy"),
        "planning_accuracy": payload.get("planning_accuracy"),
        "execution_accuracy": payload.get("execution_accuracy"),
        "planner_backend_counts": payload.get("planner_backend_counts"),
        "modalities": dict(sorted(modalities.items())),
    }


def session_summary(name: str, path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("missing"):
        return {"name": name, **payload}
    rows = [row for case in payload.get("cases", []) for row in case.get("rows", [])]
    modalities = Counter(row.get("modality") for row in rows)
    return {
        "name": name,
        "path": str(path),
        "planner": payload.get("planner"),
        "model": payload.get("model"),
        "num_sessions": payload.get("num_sessions"),
        "num_signal_runs": payload.get("num_signal_runs"),
        "retrieval_accuracy": payload.get("retrieval_accuracy"),
        "planning_accuracy": payload.get("planning_accuracy"),
        "execution_accuracy": payload.get("execution_accuracy"),
        "modalities": dict(sorted(modalities.items())),
    }


def audit_summary(path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("missing"):
        return payload
    return {
        "path": str(path),
        "num_records": payload.get("num_records"),
        "num_tool_runs": payload.get("num_tool_runs"),
        "by_modality": payload.get("by_modality", {}),
    }


def instruction_summary(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "missing": True}
    report = validate(path)
    report["path"] = str(path)
    return report


def fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# BioSignalAgent Benchmark Report",
        "",
        "## Dataset Manifests",
        "",
        "| Manifest | Records | Modalities |",
        "| --- | ---: | --- |",
    ]
    for item in report["manifests"]:
        modalities = ", ".join(f"{key}:{value}" for key, value in item.get("modalities", {}).items())
        lines.append(f"| {item['name']} | {fmt_metric(item.get('num_records'))} | {modalities} |")

    lines.extend([
        "",
        "## Framework Evals",
        "",
        "| Eval | Planner | Records | Case Runs | Retrieval | Planning | Execution |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for item in report["framework_evals"]:
        lines.append(
            f"| {item['name']} | {item.get('planner')} | {fmt_metric(item.get('num_records'))} | "
            f"{fmt_metric(item.get('num_case_runs'))} | {fmt_metric(item.get('retrieval_accuracy'))} | "
            f"{fmt_metric(item.get('planning_accuracy'))} | {fmt_metric(item.get('execution_accuracy'))} |"
        )

    lines.extend([
        "",
        "## Session Evals",
        "",
        "| Eval | Planner | Sessions | Signal Runs | Retrieval | Planning | Execution |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for item in report["session_evals"]:
        lines.append(
            f"| {item['name']} | {item.get('planner')} | {fmt_metric(item.get('num_sessions'))} | "
            f"{fmt_metric(item.get('num_signal_runs'))} | {fmt_metric(item.get('retrieval_accuracy'))} | "
            f"{fmt_metric(item.get('planning_accuracy'))} | {fmt_metric(item.get('execution_accuracy'))} |"
        )

    audit = report["tool_audit"]
    lines.extend([
        "",
        "## Tool Audit",
        "",
        f"Records: {fmt_metric(audit.get('num_records'))}; tool runs: {fmt_metric(audit.get('num_tool_runs'))}.",
        "",
        "| Modality | Runs | OK Rate | Errors | Low Confidence |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for modality, stats in sorted(audit.get("by_modality", {}).items()):
        lines.append(
            f"| {modality} | {stats.get('tool_runs')} | {fmt_metric(stats.get('ok_rate'))} | "
            f"{stats.get('errors')} | {stats.get('low_confidence')} |"
        )

    lines.extend([
        "",
        "## Instruction Data",
        "",
        "| Dataset | Samples | Validation Errors | Task Counts |",
        "| --- | ---: | ---: | --- |",
    ])
    for item in report["instruction_data"]:
        tasks = ", ".join(f"{key}:{value}" for key, value in item.get("task_counts", {}).items())
        lines.append(f"| {item['name']} | {fmt_metric(item.get('num_samples'))} | {fmt_metric(item.get('num_errors'))} | {tasks} |")
    lines.append("")
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "manifests": [
            {"name": "real_world", **manifest_summary(args.real_manifest)},
            {"name": "dedicated_common", **manifest_summary(args.dedicated_manifest)},
            {"name": "dedicated_bcg", **manifest_summary(args.bcg_manifest)},
        ],
        "framework_evals": [
            framework_summary("real_world_optimized", args.real_eval),
            framework_summary("dedicated_common_optimized", args.dedicated_eval),
            framework_summary("dedicated_bcg_optimized", args.bcg_eval),
        ],
        "session_evals": [session_summary("cross_modality_with_bcg_optimized", args.session_eval)],
        "tool_audit": audit_summary(args.tool_audit),
        "instruction_data": [
            {"name": "full_sft", **instruction_summary(args.full_sft)},
            {"name": "planning_sft", **instruction_summary(args.planning_sft)},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a BioSignalAgent benchmark index from existing eval artifacts.")
    parser.add_argument("--real-manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/real_world_manifest.json")
    parser.add_argument("--dedicated-manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/dedicated_common_manifest.json")
    parser.add_argument("--bcg-manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/dedicated_bcg_manifest.json")
    parser.add_argument("--real-eval", default="/data1/jiahui/biosignal-agent/outputs/real_dataset_framework_eval_rule_common_modalities_optimized.json")
    parser.add_argument("--dedicated-eval", default="/data1/jiahui/biosignal-agent/outputs/dedicated_common_framework_eval_rule_optimized.json")
    parser.add_argument("--bcg-eval", default="/data1/jiahui/biosignal-agent/outputs/dedicated_bcg_framework_eval_rule_optimized.json")
    parser.add_argument("--session-eval", default="/data1/jiahui/biosignal-agent/outputs/session_eval_rule_with_bcg_optimized.json")
    parser.add_argument("--tool-audit", default="/data1/jiahui/biosignal-agent/outputs/tool_output_audit_optimized.json")
    parser.add_argument("--full-sft", default="/data1/jiahui/biosignal-agent/outputs/biosignal_txagent_sft.jsonl")
    parser.add_argument("--planning-sft", default="/data1/jiahui/biosignal-agent/outputs/biosignal_txagent_planning_sft.jsonl")
    parser.add_argument("--out-json", default="/data1/jiahui/biosignal-agent/outputs/benchmark_report.json")
    parser.add_argument("--out-md", default="/data1/jiahui/biosignal-agent/outputs/benchmark_report.md")
    args = parser.parse_args()

    report = build_report(args)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(markdown_report(report))
    print(json.dumps({"out_json": str(out_json), "out_md": str(out_md)}, indent=2))


if __name__ == "__main__":
    main()
