from __future__ import annotations

from dataclasses import dataclass

from .schema_loader import find_tool_schemas
from .tool_registry import TOOLS, WORKFLOWS


MODALITY_KEYWORDS = {
    "ecg": {"ecg", "ekg", "electrocardiogram", "r-peak", "r peak", "qrs", "hrv", "rr"},
    "ppg": {"ppg", "photoplethysmography", "pulse", "pleth"},
    "bcg": {"bcg", "ballistocardiogram", "ballistocardiography", "j-peak", "j peak"},
    "scg": {"scg", "seismocardiogram", "seismocardiography", "mechanical cardiac", "j-peak", "j peak"},
    "resp": {"resp", "respiration", "respiratory", "breath", "breathing"},
    "spo2": {"spo2", "oxygen", "saturation", "oximetry", "desaturation"},
    "abp": {"abp", "arterial blood pressure", "blood pressure", "systolic", "diastolic"},
    "pcg": {"pcg", "phonocardiogram", "heart sound", "heart sounds", "s1", "s2"},
    "acc": {"acc", "accelerometer", "acceleration", "activity", "motion"},
    "eda": {"eda", "gsr", "electrodermal", "skin conductance", "stress"},
    "eeg": {"eeg", "electroencephalogram", "brain", "alpha", "beta", "theta", "delta", "bandpower"},
    "emg": {"emg", "electromyography", "muscle", "activation", "rms"},
}

BASIC_ANALYSIS_TOOLS = {
    "ecg": ["ECG_detect_r_peaks"],
    "ppg": ["PPG_detect_peaks"],
    "bcg": ["BCG_detect_j_peaks"],
    "scg": ["SCG_detect_j_peaks"],
    "resp": ["RESP_estimate_rate"],
    "spo2": ["SpO2_summarize"],
    "abp": ["ABP_detect_pulses"],
    "pcg": ["PCG_detect_heart_sounds"],
    "acc": ["ACC_summarize_activity"],
    "eda": ["EDA_summarize"],
    "eeg": ["EEG_compute_bandpower"],
    "emg": ["EMG_summarize_activation"],
}

TASK_TOOL_RULES = {
    "ecg": [
        ({"arrhythmia", "rhythm", "irregular", "afib", "atrial fibrillation", "bradycardia", "tachycardia", "pause"}, ["ECG_detect_r_peaks", "ECG_compute_hrv", "ECG_screen_arrhythmia"]),
    ],
    "resp": [
        ({"apnea", "apnoea", "hypopnea", "hypopnoea", "sleep disordered", "cessation"}, ["RESP_estimate_rate", "RESP_detect_apnea"]),
    ],
    "spo2": [
        ({"desaturation", "desat", "odi", "oxygen drop", "below 90", "hypoxemia", "hypoxaemia", "apnea"}, ["SpO2_summarize", "SpO2_detect_desaturation"]),
    ],
    "eeg": [
        ({"sleep stage", "sleep staging", "n1", "n2", "n3", "rem", "wake", "slow wave"}, ["EEG_compute_bandpower", "EEG_estimate_sleep_stage_features"]),
    ],
    "acc": [
        ({"sleep", "wake", "actigraphy", "rest", "sleep wake"}, ["ACC_summarize_activity", "ACC_estimate_sleep_wake"]),
    ],
}


@dataclass
class PlanningBioSignalAgent:
    """Question-driven planner that selects tools before execution."""

    def infer_modality(self, question: str, fallback: str | None = None) -> str:
        if fallback is not None:
            fallback = fallback.lower()
            if fallback in WORKFLOWS:
                return fallback
        text = question.lower()
        scores = {modality: sum(1 for key in keys if key in text) for modality, keys in MODALITY_KEYWORDS.items()}
        best_modality, best_score = max(scores.items(), key=lambda item: item[1])
        if best_score > 0:
            return best_modality
        raise ValueError(f"Could not infer modality from question. Pass fallback_modality as one of {sorted(WORKFLOWS)}.")

    def plan(self, question: str, fallback_modality: str | None = None) -> list[str]:
        modality = self.infer_modality(question, fallback_modality)
        text = question.lower()
        workflow = WORKFLOWS.get(modality, [])
        selected = []
        if workflow and workflow[0] in TOOLS:
            selected.append(workflow[0])

        wants_hrv = any(term in text for term in ["hrv", "heart rate variability", "rmssd", "sdnn"])
        wants_analysis = any(
            term in text
            for term in [
                "analyze",
                "report",
                "summary",
                "summarize",
                "what",
                "estimate",
                "detect",
                "peak",
                "peaks",
                "heart rate",
                "bpm",
                "rate",
                "pulse",
                "breath",
                "respiratory",
                "saturation",
                "oxygen",
                "pressure",
                "activity",
                "bandpower",
                "activation",
            ]
        )
        if modality == "ecg":
            if wants_analysis or wants_hrv:
                selected.append("ECG_detect_r_peaks")
            if wants_hrv or any(term in text for term in ["variability", "hrv", "summary", "summarize", "analyze"]):
                selected.append("ECG_compute_hrv")
        elif wants_analysis:
            selected.extend(BASIC_ANALYSIS_TOOLS.get(modality, []))

        for terms, tools in TASK_TOOL_RULES.get(modality, []):
            if any(term in text for term in terms):
                selected.extend(tools)

        selected = [tool for idx, tool in enumerate(selected) if tool in TOOLS and tool not in selected[:idx]]
        if len(selected) == 1 and wants_analysis:
            retrieved = [schema["name"] for schema in find_tool_schemas(question, top_k=3)]
            for tool_name in retrieved:
                if tool_name in TOOLS and tool_name in workflow and tool_name not in selected:
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
            if "respiratory_rate_bpm" in result and result["respiratory_rate_bpm"] is not None:
                findings.append(f"{call['tool']} estimates respiratory rate at {result['respiratory_rate_bpm']:.1f} bpm.")
            if "sdnn_ms" in result:
                findings.append(f"HRV: mean RR {result['mean_rr_ms']:.1f} ms, SDNN {result['sdnn_ms']:.1f} ms, RMSSD {result['rmssd_ms']:.1f} ms.")
            if "error" in result:
                findings.append(f"{call['tool']} could not complete: {result['error']}.")
        return findings
