from __future__ import annotations

from dataclasses import dataclass

from .tool_registry import TOOLS, WORKFLOWS


@dataclass
class BasicBioSignalAgent:
    """Small deterministic agent for validating tools before LLM integration."""

    def run_report(self, signal_path: str, modality: str, sampling_rate: float, column: str | None = None) -> dict:
        modality_key = modality.lower()
        if modality_key not in WORKFLOWS:
            raise ValueError(f"Unsupported modality: {modality}. Choose one of {sorted(WORKFLOWS)}.")
        calls = []
        for tool_name in WORKFLOWS[modality_key]:
            result = TOOLS[tool_name](signal_path=signal_path, sampling_rate=sampling_rate, column=column)
            calls.append({"tool": tool_name, "result": result})
        return {"modality": modality_key, "signal_path": signal_path, "sampling_rate": sampling_rate, "tool_calls": calls, "findings": self._summarize(modality_key, calls), "disclaimer": "Prototype output for research use only; not a clinical diagnosis."}

    def _summarize(self, modality: str, calls: list[dict]) -> list[str]:
        findings = []
        for call in calls:
            result = call["result"]
            if "quality" in result:
                findings.append(f"{modality.upper()} signal quality is {result['quality']} with confidence {result['confidence']}.")
            if "heart_rate_bpm" in result and result["heart_rate_bpm"] is not None:
                findings.append(f"{call['tool']} estimated heart rate at {result['heart_rate_bpm']:.1f} bpm.")
            if "sdnn_ms" in result:
                rmssd = result["rmssd_ms"]
                findings.append(f"HRV SDNN is {result['sdnn_ms']:.1f} ms and RMSSD is {rmssd:.1f} ms.")
            if "error" in result:
                findings.append(f"{call['tool']} could not complete: {result['error']}.")
        return findings
