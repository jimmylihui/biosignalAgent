from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.agent.tool_registry import TOOLS, WORKFLOWS


def load_records(paths: list[str]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        records.extend(payload.get('records', []))
    return records


def scalar_summary(result: dict[str, Any]) -> dict[str, Any]:
    keys = [
        'quality',
        'confidence',
        'heart_rate_bpm',
        'respiratory_rate_bpm',
        'mean_spo2_percent',
        'min_spo2_percent',
        'time_below_90_fraction',
        'plausible_ratio',
        'jump_fraction',
        'dynamic_range',
        'finite_ratio',
        'activity_level',
        'activity_std',
        'mean_level',
        'phasic_std',
        'rms',
        'mean_absolute_value',
        'total_power',
        'interval_cv',
        'regularity_confidence',
        'rr_cv',
        'pause_count',
        'ectopy_proxy_fraction',
        'arrhythmia_risk',
        'apnea_event_count',
        'apnea_index_per_hour',
        'longest_event_s',
        'desaturation_event_count',
        'oxygen_desaturation_index_per_hour',
        'sleep_stage_hint',
        'sleep_wake_hint',
        'method',
        'pulse_amplitude_proxy',
        'pulse_interval_cv',
        'perfusion_level',
        'pulse_variability_risk',
        'median_systolic_value',
        'approx_diastolic_value',
        'pressure_risk',
        'high_frequency_ratio',
        'continuous_sound_fraction',
        'murmur_proxy_score',
        'murmur_risk',
        'arousal_event_count',
        'arousal_rate_per_min',
        'arousal_level',
        'median_frequency_hz',
        'fatigue_proxy',
        'spike_count',
        'spike_rate_per_min',
        'fast_power_ratio',
        'seizure_like_risk',
        'hypopnea_event_count',
        'hypopnea_index_per_hour',
        'reduced_respiration_fraction',
        'time_below_88_fraction',
        'hypoxemia_burden',
        'error',
    ]
    output = {key: result.get(key) for key in keys if key in result}
    for count_key in ['num_peaks', 'num_breaths', 'num_pulses', 'num_sounds', 'num_samples']:
        if count_key in result:
            output[count_key] = result[count_key]
    return output


def audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for record in records:
        modality = record['modality']
        workflow = WORKFLOWS.get(modality, [])
        for tool_name in workflow:
            try:
                result = TOOLS[tool_name](signal_path=record['path'], sampling_rate=float(record['sampling_rate']), column=None)
                ok = isinstance(result, dict) and not result.get('error')
                summary = scalar_summary(result if isinstance(result, dict) else {})
            except Exception as exc:
                ok = False
                summary = {'error': str(exc)}
            rows.append({
                'dataset': record.get('dataset'),
                'record': record.get('record'),
                'modality': modality,
                'tool': tool_name,
                'ok': ok,
                'sampling_rate': record.get('sampling_rate'),
                'source_channel': record.get('source_channel'),
                **summary,
            })
    by_modality: dict[str, dict[str, Any]] = {}
    for row in rows:
        stats = by_modality.setdefault(row['modality'], {'tool_runs': 0, 'ok_runs': 0, 'errors': 0, 'low_confidence': 0})
        stats['tool_runs'] += 1
        if row['ok']:
            stats['ok_runs'] += 1
        else:
            stats['errors'] += 1
        confidence = row.get('confidence')
        if isinstance(confidence, (int, float)) and confidence < 0.5:
            stats['low_confidence'] += 1
    for stats in by_modality.values():
        stats['ok_rate'] = stats['ok_runs'] / stats['tool_runs'] if stats['tool_runs'] else 0.0
    return {'num_records': len(records), 'num_tool_runs': len(rows), 'by_modality': by_modality, 'rows': rows}


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='Audit BioSignalAgent tool outputs across manifests.')
    parser.add_argument('--manifest', action='append', required=True)
    parser.add_argument('--out-json', default='/data1/jiahui/biosignal-agent/outputs/tool_output_audit.json')
    parser.add_argument('--out-csv', default='/data1/jiahui/biosignal-agent/outputs/tool_output_audit.csv')
    args = parser.parse_args()
    report = audit(load_records(args.manifest))
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    write_csv(report['rows'], args.out_csv)
    print(json.dumps({key: report[key] for key in ['num_records', 'num_tool_runs', 'by_modality']}, indent=2))


if __name__ == '__main__':
    main()
