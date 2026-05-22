from __future__ import annotations

from dataclasses import dataclass

from .schema_loader import find_tool_schemas
from .tool_registry import TOOLS, WORKFLOWS


MODALITY_KEYWORDS = {
    "ecg": {"ecg", "ekg", "electrocardiogram", "r-peak", "r peak", "qrs", "hrv", "rr", "qt", "qtc", "st", "pr interval", "p wave", "t wave"},
    "ppg": {"ppg", "photoplethysmography", "pulse", "pleth", "respiration modulation", "respiratory modulation", "ppg respiration", "irregular pulse", "pulse irregularity", "af", "afib"},
    "bcg": {"bcg", "ballistocardiogram", "ballistocardiography", "j-peak", "j peak", "bcg respiration", "bcg breathing", "bed-based"},
    "scg": {"scg", "seismocardiogram", "seismocardiography", "mechanical cardiac", "j-peak", "j peak", "scg respiration", "scg breathing"},
    "resp": {"resp", "respiration", "respiratory", "breath", "breathing", "tachypnea", "bradypnea", "periodic breathing"},
    "spo2": {"spo2", "oxygen", "saturation", "oximetry", "desaturation", "hypoxemia", "hypoxaemia"},
    "abp": {"abp", "arterial blood pressure", "blood pressure", "systolic", "diastolic"},
    "pcg": {"pcg", "phonocardiogram", "heart sound", "heart sounds", "s1", "s2", "murmur", "valve", "segmentation", "systole", "diastole"},
    "acc": {"acc", "accelerometer", "acceleration", "activity", "motion", "actigraphy", "activity bout", "fall", "impact", "sedentary"},
    "eda": {"eda", "gsr", "electrodermal", "skin conductance", "stress"},
    "eeg": {"eeg", "electroencephalogram", "brain", "alpha", "beta", "theta", "delta", "bandpower", "seizure", "spike", "epileptiform", "sleep stage", "drowsiness", "vigilance", "eeg artifact", "blink"},
    "emg": {"emg", "electromyography", "muscle", "activation", "rms", "fatigue", "median frequency", "burst", "onset", "contraction"},
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
    "artifact": [
        ({"artifact", "noise", "noisy", "motion artifact", "clipping", "flatline", "signal dropout", "dropout"}, ["Signal_detect_artifacts"]),
    ],

    "ppg": [
        ({"perfusion", "low perfusion", "pulse variability", "pulse amplitude", "vascular"}, ["PPG_detect_peaks", "PPG_assess_perfusion_variability"]),
        ({"irregular pulse", "pulse irregularity", "af", "afib", "atrial fibrillation"}, ["PPG_detect_peaks", "PPG_screen_pulse_irregularity"]),
        ({"respiration", "respiratory modulation", "ppg respiration", "breathing"}, ["PPG_detect_peaks", "PPG_estimate_respiration_modulation"]),
    ],

    "bcg": [
        ({"respiration", "respiratory", "breathing", "breath"}, ["BCG_estimate_respiration"]),
    ],
    "scg": [
        ({"respiration", "respiratory", "breathing", "breath"}, ["SCG_estimate_respiration"]),
    ],
    "abp": [
        ({"hypotension", "hypertension", "pressure event", "shock", "high blood pressure", "low blood pressure"}, ["ABP_detect_pulses", "ABP_screen_pressure_events"]),
        ({"map", "mean arterial pressure", "pulse pressure", "hemodynamic", "haemodynamic", "perfusion pressure"}, ["ABP_detect_pulses", "ABP_compute_hemodynamics"]),
    ],
    "pcg": [
        ({"murmur", "valve", "abnormal heart sound"}, ["PCG_detect_heart_sounds", "PCG_screen_murmur_proxy", "PCG_extract_murmur_features"]),
        ({"s1", "s2", "segmentation", "systole", "diastole"}, ["PCG_detect_heart_sounds", "PCG_segment_s1_s2_proxy"]),
    ],
    "eda": [
        ({"arousal", "sympathetic", "stress event", "skin conductance response", "scr"}, ["EDA_summarize", "EDA_detect_arousal_events"]),
        ({"stress", "stress classification", "stress level", "mental stress"}, ["EDA_summarize", "EDA_detect_arousal_events", "EDA_screen_stress_proxy"]),
    ],
    "emg": [
        ({"fatigue", "median frequency", "muscle fatigue"}, ["EMG_summarize_activation", "EMG_estimate_fatigue"]),
        ({"burst", "bursts", "onset", "contraction", "muscle contraction"}, ["EMG_summarize_activation", "EMG_detect_bursts"]),
    ],
    "ecg": [
        ({"arrhythmia", "rhythm", "irregular", "afib", "atrial fibrillation", "bradycardia", "tachycardia", "pause"}, ["ECG_detect_r_peaks", "ECG_compute_hrv", "ECG_screen_arrhythmia"]),
        ({"apnea", "apnoea", "sleep disordered", "sleep breathing", "sleep apnea"}, ["ECG_detect_r_peaks", "ECG_compute_hrv", "ECG_screen_sleep_apnea"]),
        ({"morphology", "interval", "intervals", "qrs", "qt", "qtc", "st elevation", "st depression", "pr interval", "p wave", "t wave"}, ["ECG_detect_r_peaks", "ECG_measure_morphology_intervals"]),
    ],
    "resp": [
        ({"apnea", "apnoea", "sleep disordered", "cessation"}, ["RESP_estimate_rate", "RESP_detect_apnea"]),
        ({"hypopnea", "hypopnoea", "shallow breathing", "airflow reduction", "reduced respiration"}, ["RESP_estimate_rate", "RESP_detect_hypopnea"]),
        ({"tachypnea", "bradypnea", "periodic breathing", "irregular breathing", "respiratory pattern", "breathing pattern"}, ["RESP_estimate_rate", "RESP_screen_rate_pattern"]),
    ],
    "spo2": [
        ({"desaturation", "desat", "odi", "oxygen drop", "below 90", "apnea"}, ["SpO2_summarize", "SpO2_detect_desaturation"]),
        ({"hypoxemia", "hypoxaemia", "oxygen burden", "below 88", "low oxygen burden"}, ["SpO2_summarize", "SpO2_assess_hypoxemia_burden"]),
    ],
    "eeg": [
        ({"sleep stage", "sleep staging", "n1", "n2", "n3", "rem", "wake", "slow wave"}, ["EEG_compute_bandpower", "EEG_estimate_sleep_stage_features"]),
        ({"seizure", "epileptiform", "spike", "spikes", "abnormal eeg"}, ["EEG_compute_bandpower", "EEG_screen_seizure_like_activity"]),
        ({"drowsiness", "vigilance", "alertness", "sleepiness"}, ["EEG_compute_bandpower", "EEG_estimate_drowsiness"]),
        ({"artifact", "blink", "eye movement", "muscle artifact", "eeg artifact"}, ["Signal_detect_artifacts", "EEG_detect_artifact_proxy"]),
    ],
    "acc": [
        ({"sleep", "wake", "actigraphy", "rest", "sleep wake"}, ["ACC_summarize_activity", "ACC_estimate_sleep_wake"]),
        ({"activity bout", "activity bouts", "bouts", "sedentary", "active periods"}, ["ACC_summarize_activity", "ACC_detect_activity_bouts"]),
        ({"fall", "impact", "fall detection", "impact event"}, ["ACC_summarize_activity", "ACC_detect_fall_proxy"]),
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
        wants_artifact = any(term in text for term in ["artifact", "noise", "noisy", "motion artifact", "clipping", "flatline", "signal dropout", "dropout"])
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
                "perfusion",
                "irregular pulse",
                "pulse irregularity",
                "hypotension",
                "hypertension",
                "murmur",
                "arousal",
                "fatigue",
                "seizure",
                "hypopnea",
                "hypoxemia",
                "artifact",
                "noise",
                "morphology",
                "qtc",
                "tachypnea",
                "bradypnea",
                "map",
                "pulse pressure",
                "respiration modulation",
                "breathing",
                "s1",
                "s2",
                "systole",
                "diastole",
                "bout",
                "fall",
                "impact",
                "drowsiness",
                "vigilance",
                "blink",
                "burst",
                "onset",
                "contraction",
            ]
        )
        if modality == "ecg":
            if wants_analysis or wants_hrv:
                selected.append("ECG_detect_r_peaks")
            if wants_hrv or any(term in text for term in ["variability", "hrv", "summary", "summarize", "analyze"]):
                selected.append("ECG_compute_hrv")
        elif wants_analysis and not wants_artifact:
            respiration_only = modality in {"bcg", "scg"} and any(term in text for term in ["respiration", "respiratory", "breathing", "breath"])
            if not respiration_only:
                selected.extend(BASIC_ANALYSIS_TOOLS.get(modality, []))

        for terms, tools in TASK_TOOL_RULES.get(modality, []):
            if any(term in text for term in terms):
                selected.extend(tools)
        for terms, tools in TASK_TOOL_RULES.get("artifact", []):
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
