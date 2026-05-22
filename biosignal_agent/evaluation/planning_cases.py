from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlanningCase:
    case_id: str
    question: str
    modality: str
    expected_tools: tuple[str, ...]


DEFAULT_PLANNING_CASES = [
    PlanningCase(
        case_id="ecg_hrv",
        question="Estimate ECG heart rate and HRV from this signal",
        modality="ecg",
        expected_tools=("ECG_assess_quality", "ECG_detect_r_peaks", "ECG_compute_hrv"),
    ),
    PlanningCase(
        case_id="ecg_r_peaks",
        question="Detect R peaks and estimate heart rate in this ECG waveform",
        modality="ecg",
        expected_tools=("ECG_assess_quality", "ECG_detect_r_peaks"),
    ),
    PlanningCase(
        case_id="ppg_pulse",
        question="Find pulse peaks and estimate heart rate from this PPG waveform",
        modality="ppg",
        expected_tools=("PPG_assess_quality", "PPG_detect_peaks"),
    ),
    PlanningCase(
        case_id="bcg_j_peaks",
        question="Detect BCG J peaks and estimate heart rate",
        modality="bcg",
        expected_tools=("BCG_assess_quality", "BCG_detect_j_peaks"),
    ),
]
