#!/usr/bin/env python
"""Build a larger BioSignalBench expansion from existing artifacts.

This does not replace frozen BioSignalBench v1. It creates a larger stress/scale
manifest that can be split and evaluated with the same evaluator/controller.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.agent.tool_registry import TOOLS
from biosignal_agent.evaluation.biosignalbench import validate_bench_cases, write_json, write_jsonl, markdown_table, read_jsonl
from scripts.build_biosignalbench_v1 import (
    DATA_ROOT,
    case,
    digitization_cases,
    negative_cases,
    planning_cases,
    ptbxl_12lead_cases,
    sanitize,
    session_cases,
    trace_cases,
)

OUTPUT_ROOT = Path('/data1/jiahui/biosignal-agent/outputs')


def read_json(path: str | Path) -> Any | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def tool_metric_cases(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not payload:
        return []
    rows = []
    for idx, item in enumerate(payload.get('rows', [])):
        tool = item.get('tool')
        if not tool or tool not in TOOLS:
            continue
        modality = str(item.get('modality', 'unknown')).lower()
        task = str(item.get('task') or tool)
        rows.append(case(
            f'tool_metric_{sanitize(modality)}_{sanitize(tool)}_{idx:03d}',
            'tool_execution',
            f"Select the BioSignalAgent tool for {modality.upper()} task: {task}. Existing benchmark evidence uses {item.get('dataset', 'a benchmark')} with metric {item.get('metric', 'metric')}.",
            'csv',
            modality,
            [tool],
            str(path),
            signal={'path': None, 'sampling_rate': None, 'column': None},
            expected_key_outputs=['tool_selected', 'metric_name', 'evidence_level', 'clinical_limitation'],
            ground_truth_metric={
                'type': 'tool_execution_evidence_row',
                'metric': item.get('metric'),
                'value': item.get('value'),
                'evidence_level': item.get('evidence_level'),
                'artifact': item.get('artifact'),
            },
        ))
        if limit and len(rows) >= limit:
            break
    return rows



def _record_id(record: dict[str, Any], idx: int) -> str:
    base = record.get('record') or record.get('path') or f'record_{idx:05d}'
    if record.get('window_start_s') is not None:
        base = f"{base}_{record.get('window_start_s')}s"
    return sanitize(str(base))


def _signal(record: dict[str, Any], path_key: str = 'path', sr_key: str = 'sampling_rate') -> dict[str, Any]:
    return {
        'path': record.get(path_key),
        'sampling_rate': record.get(sr_key),
        'column': 'signal',
        'record': record.get('record'),
        'dataset': record.get('dataset'),
        'window_start_s': record.get('window_start_s'),
    }


def _tool_cases_for_record(record: dict[str, Any], idx: int) -> list[dict[str, Any]]:
    """Create realistic user-facing single-signal tasks from a processed waveform window."""
    modality = str(record.get('modality', '')).lower()
    path = record.get('path')
    if not modality or not path:
        return []
    rid = _record_id(record, idx)
    source = str(record.get('manifest_source') or record.get('dataset') or 'processed_public_signal_manifest')
    common = {
        'signal': _signal(record),
        'expected_key_outputs': ['measurement', 'confidence', 'quality_or_limitation'],
    }
    specs: list[tuple[str, str, list[str], dict[str, Any]]] = []
    if modality == 'ecg':
        specs.extend([
            ('hrv', 'Estimate heart rate and HRV from this ECG segment.', ['ECG_assess_quality', 'ECG_compute_hrv'], {'type': 'ecg_hr_hrv_from_window'}),
            ('rhythm', 'Screen this ECG segment for rhythm irregularity or arrhythmia concern.', ['ECG_assess_quality', 'ECG_compute_hrv', 'ECG_screen_arrhythmia'], {'type': 'ecg_arrhythmia_window', 'label': record.get('label'), 'binary_label': record.get('binary_label')}),
        ])
    elif modality == 'ppg':
        specs.extend([
            ('pulse', 'Estimate pulse rate from this PPG segment.', ['PPG_assess_quality'], {'type': 'ppg_pulse_rate'}),
            ('fiducials', 'Detect PPG pulse onset, systolic peak, dicrotic notch, and diastolic peak fiducials.', ['PPG_assess_quality', 'PPG_detect_fiducial_points'], {'type': 'ppg_on_sp_dn_dp_fiducials'}),
            ('prv', 'Compute pulse-rate variability and pulse irregularity from this PPG segment.', ['PPG_assess_quality', 'PPG_compute_prv', 'PPG_screen_pulse_irregularity'], {'type': 'ppg_prv_irregularity'}),
        ])
    elif modality == 'resp':
        specs.extend([
            ('rate', 'Estimate respiratory rate from this breathing segment.', ['RESP_assess_quality', 'RESP_estimate_rate'], {'type': 'resp_rate'}),
            ('breath_peaks', 'Detect inhale and exhale peaks from this respiration segment.', ['RESP_assess_quality', 'RESP_detect_breath_peaks'], {'type': 'resp_inhale_exhale_peaks'}),
            ('event', 'Screen this respiration segment for apnea or hypopnea-like events.', ['RESP_assess_quality', 'RESP_estimate_rate'], {'type': 'resp_event_window', 'label': record.get('respiratory_event_label'), 'event_types': record.get('event_types')}),
        ])
    elif modality == 'spo2':
        specs.extend([
            ('oxygen', 'Summarize oxygen saturation and desaturation burden from this SpO2 segment.', ['SpO2_assess_quality', 'SpO2_summarize'], {'type': 'spo2_desaturation'}),
            ('peaks_troughs', 'Detect SpO2 peaks and troughs from this oximetry segment.', ['SpO2_assess_quality', 'SpO2_detect_peaks_troughs'], {'type': 'spo2_peaks_troughs'}),
            ('apnea_support', 'Use this SpO2 segment as supportive evidence for sleep-disordered breathing risk.', ['SpO2_assess_quality', 'SpO2_summarize', 'SpO2_screen_sleep_apnea_oximetry'], {'type': 'spo2_sleep_apnea_support'}),
        ])
    elif modality == 'eeg':
        specs.extend([
            ('bandpower', 'Compute EEG bandpower features for this epoch.', ['EEG_assess_quality'], {'type': 'eeg_bandpower'}),
            ('sleep', 'Estimate sleep-stage features from this EEG epoch.', ['EEG_assess_quality', 'EEG_estimate_sleep_stage_features'], {'type': 'eeg_sleep_features', 'label': record.get('sleep_stage')}),
        ])
    elif modality == 'acc':
        specs.extend([
            ('activity', 'Classify or summarize physical activity from this accelerometer segment.', ['ACC_assess_quality', 'ACC_summarize_activity', 'ACC_classify_activity_ml'], {'type': 'acc_activity'}),
            ('sleep_wake', 'Estimate rest, wake, or sleep-wake evidence from this accelerometer segment.', ['ACC_assess_quality', 'ACC_summarize_activity', 'ACC_estimate_sleep_wake'], {'type': 'acc_sleep_wake'}),
        ])
    elif modality == 'eda':
        specs.extend([
            ('stress', 'Screen this EDA segment for stress or sympathetic arousal.', ['EDA_assess_quality', 'EDA_extract_tonic_phasic_features', 'EDA_screen_stress_ml'], {'type': 'eda_stress_arousal'}),
            ('features', 'Extract tonic and phasic skin conductance features from this EDA segment.', ['EDA_assess_quality', 'EDA_extract_tonic_phasic_features'], {'type': 'eda_tonic_phasic'}),
        ])
    elif modality == 'abp':
        specs.extend([
            ('hemo', 'Compute arterial pressure pulses and hemodynamic summary from this ABP segment.', ['ABP_assess_quality', 'ABP_detect_fiducial_points', 'ABP_compute_hemodynamics'], {'type': 'abp_hemodynamics'}),
            ('hypotension', 'Screen this ABP segment for hypotension or pressure-event concern.', ['ABP_assess_quality', 'ABP_detect_fiducial_points', 'ABP_classify_pressure_events'], {'type': 'abp_pressure_event'}),
        ])
    elif modality == 'pcg':
        specs.extend([
            ('sounds', 'Detect S1/S2 heart sounds and estimate heart rate from this PCG segment.', ['PCG_assess_quality', 'PCG_detect_heart_sounds', 'PCG_estimate_heart_rate'], {'type': 'pcg_heart_sounds'}),
            ('murmur', 'Screen this PCG segment for murmur or abnormal heart sound evidence.', ['PCG_assess_quality', 'PCG_extract_murmur_features', 'PCG_screen_murmur_proxy'], {'type': 'pcg_murmur_screen'}),
        ])
    elif modality == 'emg':
        specs.extend([
            ('activation', 'Summarize muscle activation and RMS from this EMG segment.', ['EMG_assess_quality', 'EMG_summarize_activation'], {'type': 'emg_activation'}),
            ('neuromuscular', 'Screen this EMG segment for neuromuscular abnormality patterns.', ['EMG_assess_quality', 'EMG_summarize_activation', 'EMG_screen_neuromuscular_abnormality'], {'type': 'emg_neuromuscular_screen'}),
        ])
    elif modality == 'bcg':
        specs.extend([
            ('jpeaks', 'Detect BCG J peaks and estimate heart rate from this mechanical cardiac segment.', ['BCG_assess_quality', 'BCG_detect_j_peaks'], {'type': 'bcg_hr'}),
            ('respiration', 'Estimate respiration from this BCG segment and note motion limitations.', ['BCG_assess_quality', 'BCG_assess_bed_presence_motion', 'BCG_estimate_respiration'], {'type': 'bcg_respiration'}),
        ])
    elif modality == 'scg':
        specs.extend([
            ('jpeaks', 'Detect SCG fiducial/J peaks and estimate heart rate from this segment.', ['SCG_assess_quality', 'SCG_detect_fiducial_points'], {'type': 'scg_hr'}),
            ('respiration', 'Estimate respiration or mechanical modulation from this SCG segment.', ['SCG_assess_quality', 'SCG_estimate_respiration'], {'type': 'scg_respiration'}),
        ])
    out = []
    for task_suffix, question, tools, metric in specs:
        out.append(case(
            f'public_signal_{modality}_{rid}_{task_suffix}',
            'tool_execution',
            question,
            'csv',
            modality,
            tools,
            source,
            **common,
            ground_truth_metric=metric,
        ))
    return out


def public_signal_task_cases(manifest_paths: list[str], max_records_per_manifest: int = 300, max_cases: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        payload = read_json(manifest_path)
        if not payload:
            continue
        records = payload.get('records', []) if isinstance(payload, dict) else payload
        for idx, record in enumerate(records[:max_records_per_manifest]):
            rec = dict(record)
            rec['manifest_source'] = str(manifest_path)
            rows.extend(_tool_cases_for_record(rec, idx))
            if max_cases is not None and len(rows) >= max_cases:
                return rows[:max_cases]
    return rows


def psg_session_task_cases(manifest_path: str | Path, max_records: int = 240, stride: int = 5) -> list[dict[str, Any]]:
    payload = read_json(manifest_path)
    if not payload:
        return []
    records = payload.get('records', []) if isinstance(payload, dict) else payload
    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records[::max(1, stride)][:max_records]):
        rid = _record_id(record, idx)
        signals = [
            {'modality': 'resp', 'path': record.get('resp_path'), 'sampling_rate': record.get('resp_sampling_rate'), 'column': 'signal'},
            {'modality': 'spo2', 'path': record.get('spo2_path'), 'sampling_rate': record.get('spo2_sampling_rate'), 'column': 'signal'},
            {'modality': 'eeg', 'path': record.get('eeg_path'), 'sampling_rate': record.get('eeg_sampling_rate'), 'column': 'signal'},
        ]
        rows.append(case(
            f'public_session_ucddb_{rid}_resp_spo2_sleep',
            'multimodal_session_reasoning',
            'Review this sleep segment using respiration, SpO2, and EEG. Is there evidence of apnea, hypopnea, oxygen desaturation, or sleep-stage uncertainty?',
            'session',
            'eeg+resp+spo2',
            ['RESP_assess_quality', 'RESP_estimate_rate', 'SpO2_assess_quality', 'SpO2_summarize', 'EEG_assess_quality', 'EEG_estimate_sleep_stage_features', 'Multimodal_screen_sleep_apnea_report'],
            str(manifest_path),
            signals=signals,
            expected_key_outputs=['respiratory_rate_bpm', 'desaturation_burden', 'sleep_stage_features', 'modality_agreement', 'limitations'],
            ground_truth_metric={'type': 'ucddb_multimodal_sleep_resp_event', 'respiratory_event_label': record.get('respiratory_event_label'), 'sleep_stage': record.get('sleep_stage'), 'event_types': record.get('event_types')},
        ))
    return rows

def composition(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'artifact': 'BioSignalBenchExpanded',
        'version': 'v1-expanded',
        'num_cases': len(cases),
        'task_counts': dict(sorted(Counter(c['benchmark_task'] for c in cases).items())),
        'input_type_counts': dict(sorted(Counter(c['input_type'] for c in cases).items())),
        'modality_counts': dict(sorted(Counter(str(c['modality']).lower() for c in cases).items())),
        'source_counts': dict(sorted(Counter(str(c.get('source')) for c in cases).items())),
    }


def write_composition_md(summary: dict[str, Any], validation: dict[str, Any], path: str | Path) -> None:
    text = [
        '# Table 27. BioSignalBench v1-Expanded Composition',
        '',
        'A larger stress/scale manifest built from existing BioSignalAgent artifacts. This complements, but does not replace, frozen BioSignalBench v1.',
        '',
        f"Total cases: {summary['num_cases']}",
        f"Validation errors: {validation['num_errors']}",
        '',
        '## Cases By Task',
        markdown_table(['Task', 'Cases'], [[k, v] for k, v in summary['task_counts'].items()]),
        '## Cases By Input Type',
        markdown_table(['Input type', 'Cases'], [[k, v] for k, v in summary['input_type_counts'].items()]),
        '## Cases By Modality',
        markdown_table(['Modality', 'Cases'], [[k, v] for k, v in summary['modality_counts'].items()]),
    ]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('\n'.join(text) + '\n')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-jsonl', default=str(OUTPUT_ROOT / 'biosignalbench_v1_expanded.jsonl'))
    ap.add_argument('--out-summary', default=str(OUTPUT_ROOT / 'biosignalbench_v1_expanded_summary.json'))
    ap.add_argument('--out-validation', default=str(OUTPUT_ROOT / 'biosignalbench_v1_expanded_validation.json'))
    ap.add_argument('--out-md', default=str(OUTPUT_ROOT / 'paper_tables/table27_biosignalbench_expanded_composition.md'))
    ap.add_argument('--trace-dir', default=str(OUTPUT_ROOT / 'traces'))
    ap.add_argument('--trace-limit', type=int, default=400)
    ap.add_argument('--session-limit', type=int, default=220)
    ap.add_argument('--digitization-max-per-manifest', type=int, default=60)
    ap.add_argument('--tool-metric-limit', type=int, default=None)
    ap.add_argument('--public-signal-max-records-per-manifest', type=int, default=400)
    ap.add_argument('--public-signal-max-cases', type=int, default=1800)
    ap.add_argument('--psg-session-max-records', type=int, default=240)
    ap.add_argument('--psg-session-stride', type=int, default=5)
    ap.add_argument('--digitization-manifest', action='append', default=[
        str(DATA_ROOT / 'digitization_benchmark_manifest.json'),
        str(DATA_ROOT / 'digitization_benchmark_more_10s_manifest.json'),
        str(DATA_ROOT / 'digitization_benchmark_one_per_modality_mixed_30s_manifest.json'),
    ])
    ap.add_argument('--sft-jsonl', action='append', default=[
        str(OUTPUT_ROOT / 'biosignal_txagent_planning_sft_expanded_tasks.jsonl'),
        str(OUTPUT_ROOT / 'biosignal_sft_planner_v5_train_split_live_controller.jsonl'),
        str(OUTPUT_ROOT / 'biosignal_sft_planner_v6_train_session_aug.jsonl'),
    ])
    ap.add_argument('--tool-metrics', default=str(OUTPUT_ROOT / 'tool_execution_metrics_index.json'))
    ap.add_argument('--public-signal-manifest', action='append', default=[
        str(DATA_ROOT / 'real_world_manifest.json'),
        str(DATA_ROOT / 'dedicated_common_manifest.json'),
        str(DATA_ROOT / 'dedicated_bcg_manifest.json'),
        str(DATA_ROOT / 'labeled_arrhythmia_manifest.json'),
        str(DATA_ROOT / 'psg_sleep_manifest.json'),
    ])
    args = ap.parse_args()

    rows = []
    rows.extend(planning_cases())
    rows.extend(digitization_cases(args.digitization_manifest, max_per_manifest=args.digitization_max_per_manifest))
    rows.extend(trace_cases(args.trace_dir, args.trace_limit))
    rows.extend(ptbxl_12lead_cases())
    rows.extend(session_cases(args.sft_jsonl, limit=args.session_limit))
    rows.extend(tool_metric_cases(args.tool_metrics, args.tool_metric_limit))
    rows.extend(public_signal_task_cases(args.public_signal_manifest, max_records_per_manifest=args.public_signal_max_records_per_manifest, max_cases=args.public_signal_max_cases))
    rows.extend(psg_session_task_cases(DATA_ROOT / 'psg_sleep_manifest.json', max_records=args.psg_session_max_records, stride=args.psg_session_stride))
    rows.extend(negative_cases())

    seen = set()
    unique = []
    for row in rows:
        cid = row['case_id']
        if cid in seen:
            continue
        seen.add(cid)
        unique.append(row)

    summary = composition(unique)
    validation = validate_bench_cases(unique)
    write_jsonl(args.out_jsonl, unique)
    write_json(args.out_summary, summary)
    write_json(args.out_validation, validation)
    write_composition_md(summary, validation, args.out_md)
    print(json.dumps({'out_jsonl': args.out_jsonl, 'out_summary': args.out_summary, 'out_validation': args.out_validation, 'out_md': args.out_md, **summary, 'num_validation_errors': validation['num_errors']}, indent=2))
    if validation['num_errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
