from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from biosignal_agent.agent.llm_agent import OpenRouterBioSignalAgent
from biosignal_agent.agent.openrouter_client import DEFAULT_MODEL, DEFAULT_TIMEOUT
from biosignal_agent.agent.planning_agent import PlanningBioSignalAgent
from biosignal_agent.agent.tool_retriever import ToolRetriever
from biosignal_agent.agent.tool_registry import TOOLS
from biosignal_agent.session.schema import BioSignalSession, SignalInput
from biosignal_agent.session.trace_logger import save_trace

PlannerName = Literal["rule", "openrouter"]


@dataclass
class BioSignalAgentConfig:
    planner: PlannerName = "rule"
    model: str = DEFAULT_MODEL
    retrieved_tool_count: int = 5
    use_llm_report: bool = False
    llm_timeout: int = DEFAULT_TIMEOUT
    llm_retry_max: int = 3
    llm_retry_delay: float = 8.0
    save_traces: bool = True


class BioSignalAgentFramework:
    """Unified TxAgent-style loop for retrieval, planning, execution, and tracing."""

    def __init__(self, config: BioSignalAgentConfig | None = None) -> None:
        self.config = config or BioSignalAgentConfig()
        self.retriever = ToolRetriever()
        self.rule_agent = PlanningBioSignalAgent()
        self.llm_agent = OpenRouterBioSignalAgent(
            model=self.config.model,
            use_llm_report=self.config.use_llm_report,
            retrieved_tool_count=self.config.retrieved_tool_count,
            llm_timeout=self.config.llm_timeout,
            llm_retry_max=self.config.llm_retry_max,
            llm_retry_delay=self.config.llm_retry_delay,
        )

    def run_signal(self, question: str, signal: SignalInput) -> dict:
        if self.config.planner == "openrouter":
            return self.llm_agent.run(
                question=question,
                signal_path=signal.path,
                sampling_rate=signal.sampling_rate,
                column=signal.column,
                fallback_modality=signal.modality,
                save_trace_path=self.config.save_traces,
            )
        return self._run_rule_signal(question, signal)

    def run_session(self, session: BioSignalSession) -> dict:
        runs = []
        for signal in session.signals:
            run = self.run_signal(session.question, signal)
            run["signal_label"] = signal.label
            runs.append(run)
        trace = {
            "session": session.to_dict(),
            "runs": runs,
            "planner": self.config.planner,
            "model": self.config.model if self.config.planner == "openrouter" else None,
        }
        if self.config.save_traces:
            trace["trace_path"] = str(save_trace(trace))
        return trace

    def _run_rule_signal(self, question: str, signal: SignalInput) -> dict:
        retrieved_tools = [
            schema["name"]
            for schema in self.retriever.retrieve(question, top_k=self.config.retrieved_tool_count, modality=signal.modality)
        ]
        plan_names = self.rule_agent.plan(question, signal.modality)
        tool_plan = [
            {"name": name, "arguments": {"signal_path": signal.path, "sampling_rate": signal.sampling_rate, "column": signal.column}}
            for name in plan_names
        ]
        tool_results = []
        for call in tool_plan:
            result = TOOLS[call["name"]](**call["arguments"])
            tool_results.append({"tool": call["name"], "arguments": call["arguments"], "result": result})
        final_report = self.llm_agent.deterministic_report(question, tool_results)
        trace = {
            "question": question,
            "model": None,
            "planner": "rule",
            "modality": signal.modality,
            "signal": signal.to_dict(),
            "retrieved_tools": retrieved_tools,
            "tool_plan": tool_plan,
            "tool_results": tool_results,
            "final_report": final_report,
            "disclaimer": "Prototype output for research use only; not a clinical diagnosis.",
        }
        if self.config.save_traces:
            trace["trace_path"] = str(save_trace(trace))
        return trace
