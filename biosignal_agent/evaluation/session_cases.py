from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biosignal_agent.agent.tool_registry import WORKFLOWS
from biosignal_agent.session.schema import BioSignalSession, SignalInput

REAL_MANIFEST = Path('/data1/jiahui/biosignal-agent/datasets/processed/real_world_manifest.json')
DEDICATED_MANIFEST = Path('/data1/jiahui/biosignal-agent/datasets/processed/dedicated_common_manifest.json')
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


def load_records(paths: list[Path] | None = None) -> list[dict[str, Any]]:
    paths = paths or [REAL_MANIFEST, DEDICATED_MANIFEST]
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


def make_signal(record: dict[str, Any], label: str) -> SignalInput:
    return SignalInput(
        modality=record['modality'],
        path=record['path'],
        sampling_rate=float(record['sampling_rate']),
        column=None,
        label=label,
    )


def expected(label: str, modality: str) -> SessionSignalExpectation:
    return SessionSignalExpectation(label=label, modality=modality, expected_tools=tuple(WORKFLOWS[modality]))


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
        expectations = tuple(expected(label, modality) for modality, label in modality_labels)
        cases.append(SessionCase(case_id=case_id, question=question, signals=signals, expectations=expectations))
    return cases


def case_to_session(case: SessionCase) -> BioSignalSession:
    return BioSignalSession(question=case.question, signals=list(case.signals))
