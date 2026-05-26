from __future__ import annotations

from dataclasses import dataclass

from .schema_loader import find_tool_schemas
from .tool_registry import TOOLS, WORKFLOWS


MODALITY_KEYWORDS = {
    "ecg": {"ecg", "ekg", "electrocardiogram", "r-peak", "r peak", "qrs", "hrv", "rr", "qt", "qtc", "st", "pr interval", "p wave", "t wave"},
    "ppg": {"ppg", "photoplethysmography", "pulse", "pleth", "prv", "pulse rate variability", "spo2", "oxygen", "respiration modulation", "respiratory modulation", "ppg respiration", "irregular pulse", "pulse irregularity", "af", "afib", "blood pressure", "vascular", "perfusion", "sleep", "stress", "exercise", "shock"},
    "bcg": {"bcg", "ballistocardiogram", "ballistocardiography", "j-peak", "j peak", "bcg respiration", "bcg breathing", "bed-based"},
    "scg": {"scg", "seismocardiogram", "seismocardiography", "mechanical cardiac", "j-peak", "j peak", "scg respiration", "scg breathing"},
    "resp": {"resp", "respiration", "respiratory", "breath", "breathing", "tachypnea", "bradypnea", "periodic breathing"},
    "spo2": {"spo2", "oxygen", "saturation", "oximetry", "desaturation", "hypoxemia", "hypoxaemia"},
    "abp": {"abp", "arterial blood pressure", "blood pressure", "systolic", "diastolic"},
    "pcg": {"pcg", "phonocardiogram", "heart sound", "heart sounds", "s1", "s2", "s3", "s4", "murmur", "valve", "congenital", "chd", "rhythm", "irregular", "segmentation", "systole", "diastole", "spectrogram", "heart sound classification"},
    "acc": {"acc", "accelerometer", "acceleration", "activity", "motion", "actigraphy", "activity bout", "fall", "impact", "sedentary"},
    "eda": {"eda", "gsr", "electrodermal", "skin conductance", "stress"},
    "eeg": {"eeg", "electroencephalogram", "brain", "alpha", "beta", "theta", "delta", "bandpower", "seizure", "spike", "epileptiform", "sleep stage", "drowsiness", "vigilance", "eeg artifact", "blink"},
    "emg": {"emg", "electromyography", "muscle", "activation", "rms", "fatigue", "median frequency", "burst", "onset", "contraction", "spectrogram", "myopathy", "neuropathy", "condition classification"},
}

BASIC_ANALYSIS_TOOLS = {
    "ecg": ["ECG_detect_r_peaks"],
    "ppg": ["PPG_detect_peaks"],
    "bcg": ["BCG_detect_j_peaks"],
    "scg": ["SCG_detect_j_peaks"],
    "resp": ["RESP_estimate_rate"],
    "spo2": ["SpO2_summarize"],
    "abp": ["ABP_detect_fiducial_points"],
    "pcg": ["PCG_detect_heart_sounds", "PCG_estimate_heart_rate"],
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
        ({"heart rate", "hr", "bpm", "pulse rate"}, ["PPG_detect_peaks"]),
        ({"prv", "pulse rate variability", "hrv", "pulse variability", "rmssd", "sdnn"}, ["PPG_detect_peaks", "PPG_compute_prv"]),
        ({"fiducial", "onset", "dicrotic", "notch", "systolic peak", "diastolic peak", "pulse morphology"}, ["PPG_detect_fiducial_points"]),
        ({"spo2", "oxygen saturation", "blood oxygen", "red infrared", "red/ir"}, ["PPG_estimate_spo2"]),
        ({"blood pressure", "bp", "cuffless", "pat", "ptt"}, ["PPG_detect_fiducial_points", "PPG_estimate_bp_proxy"]),
        ({"perfusion", "low perfusion", "pulse amplitude"}, ["PPG_assess_perfusion_variability"]),
        ({"shock", "low-perfusion", "low perfusion shock", "hypoperfusion"}, ["PPG_screen_low_perfusion_shock_risk"]),
        ({"irregular pulse", "pulse irregularity", "af", "afib", "atrial fibrillation"}, ["PPG_detect_peaks", "PPG_screen_pulse_irregularity", "PPG_detect_afib"]),
        ({"respiration", "respiratory modulation", "ppg respiration", "breathing", "respiratory rate"}, ["PPG_detect_peaks", "PPG_estimate_respiration_modulation"]),
        ({"sleep", "sleep monitoring", "sleep state", "recovery"}, ["PPG_detect_peaks", "PPG_compute_prv", "PPG_estimate_respiration_modulation", "PPG_estimate_sleep_features"]),
        ({"stress", "emotion", "mental workload", "strain"}, ["PPG_detect_peaks", "PPG_compute_prv", "PPG_assess_stress_prv"]),
        ({"exercise", "activity intensity", "workout", "sport", "fitness"}, ["PPG_detect_peaks", "PPG_estimate_heart_rate", "PPG_estimate_exercise_intensity"]),
        ({"vascular", "arterial stiffness", "vascular health", "vascular aging", "pulse wave"}, ["PPG_detect_fiducial_points", "PPG_assess_vascular_health"]),
    ],

    "bcg": [
        ({"respiration", "respiratory", "breathing", "breath"}, ["BCG_estimate_respiration"]),
    ],
    "scg": [
        ({"respiration", "respiratory", "breathing", "breath"}, ["SCG_estimate_respiration"]),
    ],
    "abp": [
        ({"hypotension", "hypertension", "pressure event", "shock", "high blood pressure", "low blood pressure"}, ["ABP_detect_fiducial_points", "ABP_screen_pressure_events"]),
        ({"map", "mean arterial pressure", "pulse pressure", "hemodynamic", "haemodynamic", "perfusion pressure"}, ["ABP_detect_fiducial_points", "ABP_compute_hemodynamics"]),
    ],
    "pcg": [
        ({"heart rate", "hr", "bpm"}, ["PCG_detect_heart_sounds", "PCG_estimate_heart_rate"]),
        ({"murmur", "abnormal heart sound"}, ["PCG_detect_heart_sounds", "Signal_extract_spectrogram_features", "Signal_render_spectrogram_image", "PCG_screen_murmur_proxy", "PCG_screen_murmur_patient_multisite", "PCG_extract_murmur_features"]),
        ({"valve", "valvular", "aortic", "mitral", "tricuspid"}, ["PCG_detect_heart_sounds", "PCG_segment_s1_s2_proxy", "PCG_extract_murmur_features", "PCG_screen_murmur_proxy", "PCG_screen_valve_disease_proxy"]),
        ({"congenital", "chd", "pediatric structural", "structural abnormality"}, ["PCG_detect_heart_sounds", "PCG_screen_murmur_proxy", "PCG_screen_congenital_abnormality_proxy"]),
        ({"s3", "s4", "extra heart sound", "gallop"}, ["PCG_detect_heart_sounds", "PCG_segment_s1_s2_proxy", "PCG_detect_s3_s4_proxy"]),
        ({"rhythm", "irregular", "arrhythmia", "cycle variability"}, ["PCG_detect_heart_sounds", "PCG_estimate_heart_rate", "PCG_assess_rhythm_irregularity"]),
        ({"heart function", "cardiac function", "monitoring", "longitudinal", "trend"}, ["PCG_segment_s1_s2_proxy", "PCG_extract_murmur_features", "PCG_monitor_heart_function_proxy"]),
        ({"spectrogram", "heart sound classification", "pcg classification"}, ["Signal_extract_spectrogram_features", "Signal_render_spectrogram_image", "PCG_screen_murmur_proxy", "PCG_screen_murmur_patient_multisite", "PCG_extract_murmur_features"]),
        ({"s1", "s2", "segmentation", "systole", "diastole"}, ["PCG_detect_heart_sounds", "PCG_segment_s1_s2_proxy"]),
    ],
    "eda": [
        ({"arousal", "sympathetic", "stress event", "skin conductance response", "scr"}, ["EDA_summarize", "EDA_detect_arousal_events"]),
        ({"stress", "stress classification", "stress level", "mental stress"}, ["EDA_summarize", "EDA_detect_arousal_events", "EDA_screen_stress_proxy"]),
    ],
    "emg": [
        ({"fatigue", "median frequency", "muscle fatigue"}, ["EMG_summarize_activation", "EMG_estimate_fatigue"]),
        ({"burst", "bursts", "onset", "contraction", "muscle contraction"}, ["EMG_summarize_activation", "EMG_detect_bursts"]),
        ({"spectrogram", "classification", "condition", "condition classification", "myopathy", "neuropathy", "healthy"}, ["Signal_extract_spectrogram_features", "Signal_render_spectrogram_image", "EMG_summarize_activation"]),
    ],
    "ecg": [
        ({"heart rate", "bpm", "r-peak", "r peak", "qrs detection"}, ["ECG_detect_r_peaks"]),
        ({"beat classification", "beat-level", "beat level", "pvc", "pac", "sveb", "veb", "ectopy"}, ["ECG_detect_r_peaks", "ECG_classify_beats", "ECG_screen_arrhythmia"]),
        ({"afib", "atrial fibrillation", "af detection", "af screening"}, ["ECG_detect_r_peaks", "ECG_compute_hrv", "ECG_detect_afib", "ECG_classify_rhythm_segment"]),
        ({"arrhythmia", "rhythm", "irregular", "bradycardia", "tachycardia", "pause"}, ["ECG_detect_r_peaks", "ECG_compute_hrv", "ECG_classify_rhythm_segment", "ECG_screen_arrhythmia"]),
        ({"apnea", "apnoea", "sleep disordered", "sleep breathing", "sleep apnea"}, ["ECG_detect_r_peaks", "ECG_compute_hrv", "ECG_screen_sleep_apnea"]),
        ({"qt", "qtc", "long qt", "qt prolongation"}, ["ECG_detect_r_peaks", "ECG_measure_morphology_intervals", "ECG_delineate_waves_dl", "ECG_analyze_qt_interval"]),
        ({"conduction", "bundle branch", "av block", "pr interval", "qrs duration"}, ["ECG_detect_r_peaks", "ECG_measure_morphology_intervals", "ECG_delineate_waves_dl", "ECG_screen_conduction_block"]),
        ({"ischemia", "ischaemia", "st elevation", "st depression", "st abnormality"}, ["ECG_detect_r_peaks", "ECG_measure_morphology_intervals", "ECG_delineate_waves_dl", "ECG_screen_ischemia_st"]),
        ({"stress", "fatigue", "recovery", "autonomic"}, ["ECG_detect_r_peaks", "ECG_compute_hrv", "ECG_assess_stress_fatigue_hrv"]),
        ({"morphology", "interval", "intervals", "p wave", "t wave"}, ["ECG_detect_r_peaks", "ECG_measure_morphology_intervals", "ECG_delineate_waves_dl", "ECG_analyze_qt_interval", "ECG_screen_conduction_block", "ECG_screen_ischemia_st"]),
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

        wants_hrv = any(term in text for term in ["hrv", "heart rate variability", "rmssd", "sdnn"])
        wants_quality = any(term in text for term in ["quality", "signal quality", "reliable", "reliability", "confidence", "confident", "limitation", "limitations", "trust", "usable", "validity"])
        wants_artifact = any(term in text for term in ["artifact", "noise", "noisy", "motion artifact", "clipping", "flatline", "signal dropout", "dropout"])
        if (wants_quality or wants_artifact) and workflow and workflow[0] in TOOLS:
            selected.append(workflow[0])
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
                "screen",
                "classify",
                "classification",
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
                "spectrogram",
                "condition",
                "myopathy",
                "neuropathy",
            ]
        )
        if modality == "ecg":
            if wants_analysis or wants_hrv:
                selected.append("ECG_detect_r_peaks")
            if wants_hrv or any(term in text for term in ["variability", "hrv", "summary", "summarize", "analyze"]):
                selected.append("ECG_compute_hrv")
        elif wants_analysis and not wants_artifact:
            respiration_only = modality in {"bcg", "scg"} and any(term in text for term in ["respiration", "respiratory", "breathing", "breath"])
            ppg_specific_morphology = modality == "ppg" and any(
                term in text
                for term in [
                    "fiducial",
                    "onset",
                    "dicrotic",
                    "notch",
                    "systolic peak",
                    "diastolic peak",
                    "pulse morphology",
                    "vascular",
                    "arterial stiffness",
                    "pulse wave",
                    "blood pressure",
                    "cuffless",
                    "perfusion",
                    "shock",
                ]
            )
            if not respiration_only and not ppg_specific_morphology:
                selected.extend(BASIC_ANALYSIS_TOOLS.get(modality, []))

        for terms, tools in TASK_TOOL_RULES.get(modality, []):
            if any(term in text for term in terms):
                selected.extend(tools)
        for terms, tools in TASK_TOOL_RULES.get("artifact", []):
            if any(term in text for term in terms):
                selected.extend(tools)

        selected = [tool for idx, tool in enumerate(selected) if tool in TOOLS and tool not in selected[:idx]]
        if not selected and wants_analysis:
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
            if "sdnn_ms" in result and not call["tool"].startswith("PPG_"):
                mean_rr = result.get("mean_rr_ms")
                rmssd = result.get("rmssd_ms")
                if mean_rr is not None and rmssd is not None:
                    findings.append(f"HRV: mean RR {mean_rr:.1f} ms, SDNN {result['sdnn_ms']:.1f} ms, RMSSD {rmssd:.1f} ms.")
                else:
                    findings.append(f"Interval variability: SDNN {result['sdnn_ms']:.1f} ms with available derived variability features.")
            if call["tool"] == "ECG_classify_beats" and "num_returned_beats" in result:
                findings.append(f"Beat classification returned {result['num_returned_beats']} of {result.get('num_beats', 0)} beats using {result.get('model_name', 'beat model')}.")
            if call["tool"] == "ECG_classify_rhythm_segment" and "predicted_rhythm" in result:
                findings.append(f"Rhythm segment classifier predicts {result['predicted_rhythm']} with confidence {result.get('confidence')}.")
            if call["tool"] == "ECG_detect_afib" and "afib_risk" in result:
                findings.append(f"AF screen: {result['afib_risk']} risk (probability {result.get('afib_probability')}).")
            if call["tool"] == "ECG_delineate_waves_dl" and "qrs_complex_count" in result:
                findings.append(f"ECG DL delineation found {result.get('p_wave_count', 0)} P-wave, {result.get('qrs_complex_count', 0)} QRS, and {result.get('t_wave_count', 0)} T-wave segments.")
            if call["tool"] == "ECG_analyze_qt_interval" and "qt_risk" in result:
                findings.append(f"QT analysis: {result['qt_risk']} risk with QTc {result.get('qtc_interval_ms')} ms.")
            if call["tool"] == "ECG_screen_conduction_block" and "conduction_risk" in result:
                findings.append(f"Conduction screen: {result['conduction_risk']} risk; flags {result.get('conduction_flags', [])}.")
            if call["tool"] == "ECG_screen_ischemia_st" and "ischemia_st_risk" in result:
                findings.append(f"ST/ischemia screen: {result['ischemia_st_risk']} risk; flags {result.get('ischemia_st_flags', [])}.")
            if call["tool"] == "ECG_assess_stress_fatigue_hrv" and "stress_fatigue_level" in result:
                findings.append(f"HRV stress/fatigue proxy: {result['stress_fatigue_level']} with confidence {result.get('confidence')}.")
            if call["tool"] == "PPG_compute_prv" and "sdnn_ms" in result:
                findings.append(f"PPG PRV: SDNN {result['sdnn_ms']:.1f} ms, RMSSD {result.get('rmssd_ms')} ms.")
            if call["tool"] == "PPG_detect_fiducial_points" and "num_morphology_pulses" in result:
                findings.append(f"PPG fiducials: {result.get('num_beats') or result.get('num_morphology_pulses')} pulses with onset, systolic peak, dicrotic notch, and diastolic peak points.")
            if call["tool"] == "PPG_estimate_spo2":
                if "spo2_percent_proxy" in result:
                    findings.append(f"PPG SpO2 proxy: {result['spo2_percent_proxy']:.1f}% from red/IR ratio-of-ratios.")
                elif "error" in result:
                    findings.append(f"PPG SpO2 unavailable: {result['error']}.")
            if call["tool"] == "PPG_estimate_bp_proxy" and "bp_proxy_risk" in result:
                findings.append(f"PPG BP proxy: {result['bp_proxy_risk']} with flags {result.get('bp_proxy_flags', [])}.")
            if call["tool"] == "PPG_detect_afib" and "afib_risk" in result:
                findings.append(f"PPG AF screen: {result['afib_risk']} risk, probability {result.get('af_probability')}.")
            if call["tool"] == "PPG_estimate_sleep_features" and "sleep_proxy" in result:
                findings.append(f"PPG sleep/rest proxy: {result['sleep_proxy']} with flags {result.get('sleep_feature_flags', [])}.")
            if call["tool"] == "PPG_assess_stress_prv" and "stress_prv_level" in result:
                findings.append(f"PPG stress/recovery proxy: {result['stress_prv_level']} with confidence {result.get('confidence')}.")
            if call["tool"] == "PPG_estimate_exercise_intensity" and "exercise_intensity_zone" in result:
                findings.append(f"PPG exercise intensity: {result['exercise_intensity_zone']} from {result.get('heart_rate_bpm')} bpm.")
            if call["tool"] == "PPG_assess_vascular_health" and "vascular_stiffness_proxy" in result:
                findings.append(f"PPG vascular proxy: {result['vascular_stiffness_proxy']} with flags {result.get('vascular_flags', [])}.")
            if call["tool"] == "PPG_screen_low_perfusion_shock_risk" and "shock_perfusion_risk" in result:
                findings.append(f"PPG low-perfusion/shock proxy: {result['shock_perfusion_risk']} with flags {result.get('shock_perfusion_flags', [])}.")
            if call["tool"] == "Signal_extract_spectrogram_features" and "num_windows" in result:
                findings.append(f"Extracted spectrogram features over {result['num_windows']} windows for time-frequency analysis.")
            if call["tool"] == "Signal_render_spectrogram_image" and "image_path" in result:
                findings.append(f"Rendered spectrogram image to {result['image_path']}.")
            if "error" in result:
                findings.append(f"{call['tool']} could not complete: {result['error']}.")
        return findings
