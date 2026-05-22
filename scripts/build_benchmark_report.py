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
        "session_tool_accuracy": payload.get("session_tool_accuracy"),
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


def labeled_arrhythmia_summary(path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("missing"):
        return {"name": "mitdb_arrhythmia_windows", **payload}
    metrics = payload.get("metrics", {})
    return {
        "name": "mitdb_arrhythmia_windows",
        "path": str(path),
        "num_windows": payload.get("num_windows"),
        "truth_counts": payload.get("truth_counts", {}),
        "prediction_counts": payload.get("prediction_counts", {}),
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall_sensitivity": metrics.get("recall_sensitivity"),
        "specificity": metrics.get("specificity"),
        "f1": metrics.get("f1"),
    }


def labeled_apnea_ecg_summary(path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("missing"):
        return {"name": "apnea_ecg_minutes", **payload}
    metrics = payload.get("metrics", {})
    return {
        "name": "apnea_ecg_minutes",
        "path": str(path),
        "num_windows": payload.get("num_windows"),
        "truth_counts": payload.get("truth_counts", {}),
        "prediction_counts": payload.get("prediction_counts", {}),
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall_sensitivity": metrics.get("recall_sensitivity"),
        "specificity": metrics.get("specificity"),
        "f1": metrics.get("f1"),
    }


def labeled_ucddb_summary(path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("missing"):
        return {"name": "ucddb_resp_spo2_windows", **payload}
    metrics = payload.get("metrics", {})
    return {
        "name": "ucddb_resp_spo2_windows",
        "path": str(path),
        "num_windows": payload.get("num_windows"),
        "truth_counts": payload.get("truth_counts", {}),
        "prediction_counts": payload.get("prediction_counts", {}),
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall_sensitivity": metrics.get("recall_sensitivity"),
        "specificity": metrics.get("specificity"),
        "f1": metrics.get("f1"),
    }


def labeled_ecg_rhythm_summary(path: str | Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if payload.get("missing"):
        return [{"name": "ecg_rhythm_af_windows", **payload}, {"name": "ecg_beat_abnormal", **payload}]
    af = payload.get("af_metrics", {})
    beat = payload.get("beat_abnormal_metrics", {})
    return [
        {
            "name": "ecg_rhythm_af_windows",
            "path": str(path),
            "num_windows": payload.get("num_windows"),
            "truth_counts": payload.get("rhythm_truth_counts", {}),
            "prediction_counts": payload.get("rhythm_prediction_counts", {}),
            "accuracy": af.get("accuracy"),
            "precision": af.get("precision"),
            "recall_sensitivity": af.get("recall_sensitivity"),
            "specificity": af.get("specificity"),
            "f1": af.get("f1"),
        },
        {
            "name": "ecg_beat_abnormal",
            "path": str(path),
            "num_windows": payload.get("num_beats"),
            "truth_counts": payload.get("beat_truth_counts", {}),
            "prediction_counts": payload.get("beat_prediction_counts", {}),
            "accuracy": beat.get("accuracy"),
            "precision": beat.get("precision"),
            "recall_sensitivity": beat.get("recall_sensitivity"),
            "specificity": beat.get("specificity"),
            "f1": beat.get("f1"),
            "beat_detection_recall": payload.get("beat_detection_recall"),
        },
    ]


def labeled_psg_sleep_summary(path: str | Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if payload.get("missing"):
        return [{"name": "psg_sleep_stage", **payload}, {"name": "psg_respiratory_events", **payload}]
    sleep = payload.get("sleep_stage_metrics", {})
    resp = payload.get("respiratory_event_metrics", {})
    return [
        {
            "name": "psg_sleep_stage",
            "path": str(path),
            "num_windows": payload.get("num_windows"),
            "truth_counts": payload.get("sleep_truth_counts", {}),
            "prediction_counts": payload.get("sleep_prediction_counts", {}),
            "accuracy": sleep.get("accuracy"),
            "precision": None,
            "recall_sensitivity": None,
            "specificity": None,
            "f1": sleep.get("macro_f1"),
        },
        {
            "name": "psg_respiratory_events",
            "path": str(path),
            "num_windows": payload.get("num_windows"),
            "truth_counts": payload.get("resp_truth_counts", {}),
            "prediction_counts": payload.get("resp_prediction_counts", {}),
            "accuracy": resp.get("accuracy"),
            "precision": resp.get("precision"),
            "recall_sensitivity": resp.get("recall_sensitivity"),
            "specificity": resp.get("specificity"),
            "f1": resp.get("f1"),
        },
    ]


def labeled_pcg_murmur_summary(path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("missing"):
        return {"name": "pcg_murmur_normal_abnormal", **payload}
    metrics = payload.get("metrics", {})
    return {
        "name": "pcg_murmur_normal_abnormal",
        "path": str(path),
        "num_windows": payload.get("num_records"),
        "truth_counts": payload.get("truth_counts", {}),
        "prediction_counts": payload.get("prediction_counts", {}),
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall_sensitivity": metrics.get("recall_sensitivity"),
        "specificity": metrics.get("specificity"),
        "f1": metrics.get("f1"),
    }


def generic_labeled_summary(name: str, path: str | Path, metrics_key: str = "metrics") -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("missing"):
        return {"name": name, **payload}
    metrics = payload.get(metrics_key, payload.get("metrics", {}))
    return {
        "name": name,
        "path": str(path),
        "num_windows": payload.get("num_windows") or payload.get("num_records"),
        "truth_counts": payload.get("truth_counts", payload.get("label_counts", {})),
        "prediction_counts": payload.get("prediction_counts", {}),
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall_sensitivity": metrics.get("recall_sensitivity"),
        "specificity": metrics.get("specificity"),
        "f1": metrics.get("f1", metrics.get("macro_f1")),
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
        "## Labeled Benchmarks",
        "",
        "| Benchmark | Windows | Accuracy | Precision | Recall | Specificity | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for item in report.get("labeled_benchmarks", []):
        lines.append(
            f"| {item['name']} | {fmt_metric(item.get('num_windows'))} | {fmt_metric(item.get('accuracy'))} | "
            f"{fmt_metric(item.get('precision'))} | {fmt_metric(item.get('recall_sensitivity'))} | "
            f"{fmt_metric(item.get('specificity'))} | {fmt_metric(item.get('f1'))} |"
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
            framework_summary("real_world_major_tasks", args.real_eval),
            framework_summary("dedicated_common_major_tasks", args.dedicated_eval),
            framework_summary("dedicated_bcg_major_tasks", args.bcg_eval),
        ],
        "session_evals": [session_summary("cross_modality_major_tasks", args.session_eval)],
        "tool_audit": audit_summary(args.tool_audit),
        "labeled_benchmarks": [
            labeled_arrhythmia_summary(args.arrhythmia_eval),
            labeled_apnea_ecg_summary(args.apnea_ecg_eval),
            labeled_ucddb_summary(args.ucddb_eval),
            *labeled_ecg_rhythm_summary(args.ecg_rhythm_eval),
            *labeled_psg_sleep_summary(args.psg_sleep_eval),
            labeled_pcg_murmur_summary(args.pcg_murmur_eval),
            labeled_pcg_murmur_summary(args.pcg_murmur_v2_eval) | {"name": "pcg_murmur_feature_logreg"},
            generic_labeled_summary("ppg_af_irregularity", args.ppg_af_eval),
            generic_labeled_summary("wesad_stress_eda", args.wesad_stress_eval),
            generic_labeled_summary("acc_activity_uci_har", args.acc_activity_eval),
            generic_labeled_summary("acc_fall_unimib", args.acc_fall_eval),
            generic_labeled_summary("chbmit_seizure_eeg", args.chbmit_seizure_eval),
        ],
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
    parser.add_argument("--real-eval", default="/data1/jiahui/biosignal-agent/outputs/real_dataset_framework_eval_rule_major_tasks.json")
    parser.add_argument("--dedicated-eval", default="/data1/jiahui/biosignal-agent/outputs/dedicated_common_framework_eval_rule_major_tasks.json")
    parser.add_argument("--bcg-eval", default="/data1/jiahui/biosignal-agent/outputs/dedicated_bcg_framework_eval_rule_major_tasks.json")
    parser.add_argument("--session-eval", default="/data1/jiahui/biosignal-agent/outputs/session_eval_rule_major_tasks.json")
    parser.add_argument("--tool-audit", default="/data1/jiahui/biosignal-agent/outputs/tool_output_audit_major_tasks.json")
    parser.add_argument("--arrhythmia-eval", default="/data1/jiahui/biosignal-agent/outputs/labeled_arrhythmia_eval.json")
    parser.add_argument("--apnea-ecg-eval", default="/data1/jiahui/biosignal-agent/outputs/apnea_ecg_eval.json")
    parser.add_argument("--ucddb-eval", default="/data1/jiahui/biosignal-agent/outputs/ucddb_resp_spo2_eval_more_tasks.json")
    parser.add_argument("--ecg-rhythm-eval", default="/data1/jiahui/biosignal-agent/outputs/ecg_rhythm_beat_eval.json")
    parser.add_argument("--psg-sleep-eval", default="/data1/jiahui/biosignal-agent/outputs/psg_sleep_eval.json")
    parser.add_argument("--pcg-murmur-eval", default="/data1/jiahui/biosignal-agent/outputs/pcg_murmur_eval.json")
    parser.add_argument("--pcg-murmur-v2-eval", default="/data1/jiahui/biosignal-agent/outputs/pcg_murmur_v2_eval.json")
    parser.add_argument("--ppg-af-eval", default="/data1/jiahui/biosignal-agent/outputs/ppg_af_eval.json")
    parser.add_argument("--wesad-stress-eval", default="/data1/jiahui/biosignal-agent/outputs/wesad_stress_eval.json")
    parser.add_argument("--acc-activity-eval", default="/data1/jiahui/biosignal-agent/outputs/acc_activity_eval.json")
    parser.add_argument("--acc-fall-eval", default="/data1/jiahui/biosignal-agent/outputs/acc_fall_eval.json")
    parser.add_argument("--chbmit-seizure-eval", default="/data1/jiahui/biosignal-agent/outputs/chbmit_seizure_eval.json")
    parser.add_argument("--full-sft", default="/data1/jiahui/biosignal-agent/outputs/biosignal_txagent_sft.jsonl")
    parser.add_argument("--planning-sft", default="/data1/jiahui/biosignal-agent/outputs/biosignal_txagent_planning_sft.jsonl")
    parser.add_argument("--out-json", default="/data1/jiahui/biosignal-agent/outputs/benchmark_report_major_tasks.json")
    parser.add_argument("--out-md", default="/data1/jiahui/biosignal-agent/outputs/benchmark_report_major_tasks.md")
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
