from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biosignal_agent.agent.planning_agent import PlanningBioSignalAgent
from biosignal_agent.session.schema import BioSignalSession, SignalInput

REAL_MANIFEST = Path('/data1/jiahui/biosignal-agent/datasets/processed/real_world_manifest.json')
DEDICATED_MANIFEST = Path('/data1/jiahui/biosignal-agent/datasets/processed/dedicated_common_manifest.json')
DEDICATED_BCG_MANIFEST = Path('/data1/jiahui/biosignal-agent/datasets/processed/dedicated_bcg_manifest.json')
ECG_RECORD = {
    'dataset': 'mitdb',
    'record': '100',
    'modality': 'ecg',
    'path': '/data1/jiahui/biosignal-agent/datasets/processed/mitdb_100_mlii_60s.csv',
    'sampling_rate': 360.0,
    'source_channel': 'MLII',
}


@dataclass(frozen=True)
class SessionSignalExpectation:
    label: str
    modality: str
    expected_tools: tuple[str, ...]


@dataclass(frozen=True)
class SessionCase:
    case_id: str
    question: str
    signals: tuple[SignalInput, ...]
    expectations: tuple[SessionSignalExpectation, ...]
    expected_session_tools: tuple[str, ...] = ()


def load_records(paths: list[Path] | None = None) -> list[dict[str, Any]]:
    paths = paths or [REAL_MANIFEST, DEDICATED_MANIFEST, DEDICATED_BCG_MANIFEST]
    records = [ECG_RECORD]
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        records.extend(payload.get('records', []))
    return records


def first_record_by_modality(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_modality: dict[str, dict[str, Any]] = {}
    for record in records:
        by_modality.setdefault(record['modality'], record)
    return by_modality


def find_record(records: list[dict[str, Any]], dataset: str, record_name: str, modality: str) -> dict[str, Any] | None:
    for record in records:
        if record.get('dataset') == dataset and record.get('record') == record_name and record.get('modality') == modality:
            return record
    return None


def find_synchronized_pair(records: list[dict[str, Any]], dataset: str, modalities: tuple[str, ...]) -> tuple[dict[str, Any], ...] | None:
    record_names = sorted({record.get('record') for record in records if record.get('dataset') == dataset})
    for record_name in record_names:
        matched = tuple(find_record(records, dataset, record_name, modality) for modality in modalities)
        if all(item is not None for item in matched):
            return matched  # type: ignore[return-value]
    return None


def make_signal(record: dict[str, Any], label: str) -> SignalInput:
    return SignalInput(
        modality=record['modality'],
        path=record['path'],
        sampling_rate=float(record['sampling_rate']),
        column=None,
        label=label,
    )


def expected(label: str, modality: str, question: str) -> SessionSignalExpectation:
    tools = tuple(PlanningBioSignalAgent().plan(question, modality))
    return SessionSignalExpectation(label=label, modality=modality, expected_tools=tools)


def build_default_session_cases(records: list[dict[str, Any]] | None = None) -> list[SessionCase]:
    records = records or load_records()
    by_modality = first_record_by_modality(records)
    specs = [
        (
            'cardiorespiratory_vitals',
            'Estimate cardiac rate, respiratory rate, oxygen saturation, and summarize confidence across these signals.',
            [('ecg', 'ecg'), ('ppg', 'ppg'), ('resp', 'resp'), ('spo2', 'spo2')],
        ),
        (
            'hemodynamic_monitoring',
            'Analyze ECG, arterial blood pressure, and SpO2 to summarize heart rate and oxygenation.',
            [('ecg', 'ecg'), ('abp', 'abp'), ('spo2', 'spo2')],
        ),
        (
            'wearable_stress_context',
            'Analyze pulse, activity, and skin conductance to summarize wearable physiological state.',
            [('ppg', 'ppg'), ('acc', 'acc'), ('eda', 'eda')],
        ),
        (
            'neuro_muscle_activity',
            'Compute EEG bandpower and EMG activation, using accelerometer activity as movement context.',
            [('eeg', 'eeg'), ('emg', 'emg'), ('acc', 'acc')],
        ),
        (
            'mechanical_cardiorespiratory',
            'Estimate mechanical cardiac rate from SCG and respiratory rate from the breathing signal.',
            [('scg', 'scg'), ('resp', 'resp')],
        ),
        (
            'bed_bcg_cardiorespiratory',
            'Estimate bed-based BCG mechanical cardiac rate, respiratory rate, and ECG heart rate variability evidence.',
            [('bcg', 'bcg'), ('resp', 'resp'), ('ecg', 'ecg')],
        ),
        (
            'heart_sound_hemodynamics',
            'Analyze PCG heart sounds together with ABP and ECG heart-rate evidence.',
            [('pcg', 'pcg'), ('abp', 'abp'), ('ecg', 'ecg')],
        ),
    ]
    cases: list[SessionCase] = []
    for case_id, question, modality_labels in specs:
        missing = [modality for modality, _ in modality_labels if modality not in by_modality]
        if missing:
            continue
        signals = tuple(make_signal(by_modality[modality], label) for modality, label in modality_labels)
        expectations = tuple(expected(label, modality, question) for modality, label in modality_labels)
        cases.append(SessionCase(case_id=case_id, question=question, signals=signals, expectations=expectations))
    synchronized_ecg_ppg = find_synchronized_pair(records, 'bidmc', ('ecg', 'ppg'))
    if synchronized_ecg_ppg is not None:
        question = 'Compute ECG PPG pulse arrival time and pulse transit timing proxy from synchronized signals.'
        modality_labels = [('ecg', 'ecg'), ('ppg', 'ppg')]
        signals = tuple(make_signal(record, label) for record, (_, label) in zip(synchronized_ecg_ppg, modality_labels))
        expectations = tuple(expected(label, modality, question) for modality, label in modality_labels)
        cases.append(SessionCase(
            case_id='ecg_ppg_pulse_arrival_timing',
            question=question,
            signals=signals,
            expectations=expectations,
            expected_session_tools=('Session_compute_ecg_ppg_pulse_arrival',),
        ))
    if {'ecg', 'resp', 'spo2'}.issubset(by_modality):
        question = 'Screen sleep apnea and hypopnea using ECG HRV, respiration, and SpO2 desaturation evidence.'
        modality_labels = [('ecg', 'ecg'), ('resp', 'resp'), ('spo2', 'spo2')]
        signals = tuple(make_signal(by_modality[modality], label) for modality, label in modality_labels)
        expectations = tuple(expected(label, modality, question) for modality, label in modality_labels)
        cases.append(SessionCase(
            case_id='multimodal_sleep_apnea_screening',
            question=question,
            signals=signals,
            expectations=expectations,
            expected_session_tools=('Session_screen_sleep_apnea_multimodal',),
        ))
    return cases


def case_to_session(case: SessionCase) -> BioSignalSession:
    return BioSignalSession(question=case.question, signals=list(case.signals))
