from __future__ import annotations

from dataclasses import dataclass

from .schema_loader import find_tool_schemas
from .tool_registry import TOOLS


MODALITY_KEYWORDS = {
    "ecg": {"ecg", "ekg", "electrocardiogram", "r-peak", "r peak", "qrs", "hrv", "rr"},
    "ppg": {"ppg", "photoplethysmography", "pulse", "spo2", "pleth"},
    "bcg": {"bcg", "ballistocardiogram", "ballistocardiography", "j-peak", "j peak"},
}


@dataclass
class PlanningBioSignalAgent:
    """Question-driven planner that selects tools before execution."""

    def infer_modality(self, question: str, fallback: str | None = None) -> str:
        text = question.lower()
        scores = {modality: sum(1 for key in keys if key in text) for modality, keys in MODALITY_KEYWORDS.items()}
        best_modality, best_score = max(scores.items(), key=lambda item: item[1])
        if best_score > 0:
            return best_modality
        if fallback is not None:
            return fallback.lower()
        raise ValueError("Could not infer modality from question. Pass fallback_modality as ecg, ppg, or bcg.")

    def plan(self, question: str, fallback_modality: str | None = None) -> list[str]:
        modality = self.infer_modality(question, fallback_modality)
        text = question.lower()
        selected = []
        quality_tool = f"{modality.upper()}_assess_quality"
        if quality_tool in TOOLS:
            selected.append(quality_tool)

        wants_hrv = "hrv" in text or "heart rate variability" in text or "rmssd" in text or "sdnn" in text
        wants_peaks = any(term in text for term in ["peak", "peaks", "heart rate", "bpm", "rate", "rr", "pulse"])
        wants_general = any(term in text for term in ["analyze", "report", "summary", "quality", "what"])

        if modality == "ecg":
            if wants_peaks or wants_hrv or wants_general:
                selected.append("ECG_detect_r_peaks")
            if wants_hrv or wants_general:
                selected.append("ECG_compute_hrv")
        elif modality == "ppg":
            if wants_peaks or wants_general:
                selected.append("PPG_detect_peaks")
        elif modality == "bcg":
            if wants_peaks or wants_general:
                selected.append("BCG_detect_j_peaks")

        if len(selected) == 1:
            retrieved = [schema["name"] for schema in find_tool_schemas(question, top_k=3)]
            for tool_name in retrieved:
                if tool_name in TOOLS and tool_name.startswith(modality.upper()) and tool_name not in selected:
                    selected.append(tool_name)
        return selected

    def run(self, question: str, signal_path: str, sampling_rate: float, column: str | None = None, fallback_modality: str | None = None) -> dict:
        modality = self.infer_modality(question, fallback_modality)
        plan = self.plan(question, modality)
        calls = []
        for tool_name in plan:
            result = TOOLS[tool_name](signal_path=signal_path, sampling_rate=sampling_rate, column=column)
            calls.append({"tool": tool_name, "result": result})
        return {
            "question": question,
            "modality": modality,
            "signal_path": signal_path,
            "sampling_rate": sampling_rate,
            "plan": plan,
            "tool_calls": calls,
            "findings": self._summarize(calls),
            "disclaimer": "Prototype output for research use only; not a clinical diagnosis.",
        }

    def _summarize(self, calls: list[dict]) -> list[str]:
        findings = []
        for call in calls:
            result = call["result"]
            if "quality" in result:
                findings.append(f"{call['tool']} reports {result['quality']} quality with confidence {result['confidence']}.")
            if "heart_rate_bpm" in result and result["heart_rate_bpm"] is not None:
                findings.append(f"{call['tool']} estimates heart rate at {result['heart_rate_bpm']:.1f} bpm using {result.get('method', 'unknown method')}.")
            if "sdnn_ms" in result:
                findings.append(f"HRV: mean RR {result['mean_rr_ms']:.1f} ms, SDNN {result['sdnn_ms']:.1f} ms, RMSSD {result['rmssd_ms']:.1f} ms.")
            if "error" in result:
                findings.append(f"{call['tool']} could not complete: {result['error']}.")
        return findings
