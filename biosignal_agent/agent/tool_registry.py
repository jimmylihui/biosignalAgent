from __future__ import annotations

from biosignal_agent.tools.bcg_tools import BCG_assess_quality, BCG_detect_j_peaks
from biosignal_agent.tools.ecg_tools import ECG_assess_quality, ECG_compute_hrv, ECG_detect_r_peaks
from biosignal_agent.tools.ppg_tools import PPG_assess_quality, PPG_detect_peaks

TOOLS = {
    "ECG_assess_quality": ECG_assess_quality,
    "ECG_detect_r_peaks": ECG_detect_r_peaks,
    "ECG_compute_hrv": ECG_compute_hrv,
    "PPG_assess_quality": PPG_assess_quality,
    "PPG_detect_peaks": PPG_detect_peaks,
    "BCG_assess_quality": BCG_assess_quality,
    "BCG_detect_j_peaks": BCG_detect_j_peaks,
}

WORKFLOWS = {
    "ecg": ["ECG_assess_quality", "ECG_detect_r_peaks", "ECG_compute_hrv"],
    "ppg": ["PPG_assess_quality", "PPG_detect_peaks"],
    "bcg": ["BCG_assess_quality", "BCG_detect_j_peaks"],
}
