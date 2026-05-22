from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.agent.llm_agent import OpenRouterBioSignalAgent
from biosignal_agent.agent.openrouter_client import DEFAULT_MODEL
from biosignal_agent.agent.planning_agent import PlanningBioSignalAgent
from biosignal_agent.session.schema import BioSignalSession
from biosignal_agent.session.trace_logger import save_trace
from biosignal_agent.agent.tool_registry import TOOLS


def run_rule_planner(question: str, signal) -> dict:
    planner = PlanningBioSignalAgent()
    plan_names = planner.plan(question, signal.modality)
    tool_plan = [
        {"name": name, "arguments": {"signal_path": signal.path, "sampling_rate": signal.sampling_rate, "column": signal.column}}
        for name in plan_names
    ]
    tool_results = []
    for call in tool_plan:
        result = TOOLS[call["name"]](**call["arguments"])
        tool_results.append({"tool": call["name"], "arguments": call["arguments"], "result": result})
    report = OpenRouterBioSignalAgent().deterministic_report(question, tool_results)
    return {
        "question": question,
        "planner": "rule",
        "modality": signal.modality,
        "signal": signal.to_dict(),
        "tool_plan": tool_plan,
        "tool_results": tool_results,
        "final_report": report,
        "disclaimer": "Prototype output for research use only; not a clinical diagnosis.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', required=True, help='Path to a BioSignalSession JSON file.')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--llm-planner', action='store_true', help='Use OpenRouter planner for each signal. Default uses rule planner for speed.')
    parser.add_argument('--llm-report', action='store_true')
    args = parser.parse_args()

    session = BioSignalSession.from_json_file(args.session)
    llm_agent = OpenRouterBioSignalAgent(model=args.model, use_llm_report=args.llm_report)
    runs = []
    for signal in session.signals:
        if args.llm_planner:
            run = llm_agent.run(
                question=session.question,
                signal_path=signal.path,
                sampling_rate=signal.sampling_rate,
                column=signal.column,
                fallback_modality=signal.modality,
                save_trace_path=False,
            )
        else:
            run = run_rule_planner(session.question, signal)
        run['signal_label'] = signal.label
        runs.append(run)
    trace = {'session': session.to_dict(), 'runs': runs, 'model': args.model if args.llm_planner else None, 'planner': 'openrouter' if args.llm_planner else 'rule'}
    trace['trace_path'] = str(save_trace(trace))
    print(json.dumps(trace, indent=2))


if __name__ == '__main__':
    main()
