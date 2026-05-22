from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.agent.openrouter_client import DEFAULT_MODEL
from biosignal_agent.evaluation.framework_eval import evaluate_cases, write_eval_outputs
from biosignal_agent.evaluation.planning_cases import DEFAULT_PLANNING_CASES


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BioSignalAgent retrieval, planning, and optional execution.")
    parser.add_argument("--planner", choices=["rule", "openrouter"], default="rule")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--retrieved-tool-count", type=int, default=5)
    parser.add_argument("--llm-timeout", type=int, default=30)
    parser.add_argument("--llm-retry-max", type=int, default=1)
    parser.add_argument("--llm-retry-delay", type=float, default=2.0)
    parser.add_argument("--no-rule-fallback", action="store_true", help="Record OpenRouter failures instead of falling back to the rule planner.")
    parser.add_argument("--progress", action="store_true", help="Print case progress while evaluating.")
    parser.add_argument("--case-id", action="append", default=None, help="Evaluate only selected planning case IDs. Can be repeated.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N selected cases.")
    parser.add_argument("--execute", action="store_true", help="Run planned tools on provided modality signals.")
    parser.add_argument("--ecg-csv", default=None)
    parser.add_argument("--ppg-csv", default=None)
    parser.add_argument("--bcg-csv", default=None)
    parser.add_argument("--ecg-sampling-rate", type=float, default=360.0)
    parser.add_argument("--ppg-sampling-rate", type=float, default=100.0)
    parser.add_argument("--bcg-sampling-rate", type=float, default=100.0)
    parser.add_argument("--out-json", default="/data1/jiahui/biosignal-agent/outputs/framework_eval.json")
    parser.add_argument("--out-csv", default="/data1/jiahui/biosignal-agent/outputs/framework_eval.csv")
    args = parser.parse_args()

    signal_paths = {
        modality: path
        for modality, path in {"ecg": args.ecg_csv, "ppg": args.ppg_csv, "bcg": args.bcg_csv}.items()
        if path
    }
    sampling_rates = {
        "ecg": args.ecg_sampling_rate,
        "ppg": args.ppg_sampling_rate,
        "bcg": args.bcg_sampling_rate,
    }
    cases = DEFAULT_PLANNING_CASES
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case.case_id in wanted]
    if args.limit is not None:
        cases = cases[: args.limit]
    report = evaluate_cases(
        cases=cases,
        planner_name=args.planner,
        model=args.model,
        retrieved_tool_count=args.retrieved_tool_count,
        execute=args.execute,
        llm_timeout=args.llm_timeout,
        llm_retry_max=args.llm_retry_max,
        llm_retry_delay=args.llm_retry_delay,
        llm_fallback_to_rules=not args.no_rule_fallback,
        signal_paths=signal_paths,
        sampling_rates=sampling_rates,
        progress=args.progress,
    )
    write_eval_outputs(report, args.out_json, args.out_csv)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
