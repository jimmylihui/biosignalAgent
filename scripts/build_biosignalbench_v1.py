#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.evaluation.planning_cases import DEFAULT_PLANNING_CASES
from biosignal_agent.evaluation.biosignalbench import validate_bench_cases, write_json, write_jsonl, markdown_table, read_jsonl

OUTPUT_ROOT = Path('/data1/jiahui/biosignal-agent/outputs')
DATA_ROOT = Path('/data1/jiahui/biosignal-agent/datasets/processed')


def case(case_id: str, task: str, question: str, input_type: str, modality: str, expected_tools: list[str], source: str, **extra: Any) -> dict[str, Any]:
    return {
        'case_id': case_id,
        'benchmark_task': task,
        'question': question,
        'input_type': input_type,
        'modality': modality,
        'expected_tools': list(dict.fromkeys(expected_tools)),
        'expected_key_outputs': extra.pop('expected_key_outputs', []),
        'ground_truth_metric': extra.pop('ground_truth_metric', {'type': 'tool_set_exact_or_contains'}),
        'source': source,
        **extra,
    }


def planning_cases(limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    cases = DEFAULT_PLANNING_CASES[:limit] if limit else DEFAULT_PLANNING_CASES
    for item in cases:
        rows.append(case(
            f'plan_{item.case_id}',
            'tool_planning',
            item.question,
            'csv',
            item.modality,
            list(item.expected_tools),
            'biosignal_agent.evaluation.planning_cases.DEFAULT_PLANNING_CASES',
            signal={'path': None, 'sampling_rate': None, 'column': None},
        ))
    return rows


def digitization_cases(paths: list[str], max_per_manifest: int = 12) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        payload = json.loads(p.read_text())
        for idx, record in enumerate(payload.get('records', [])[:max_per_manifest]):
            modality = str(record.get('modality', 'unknown')).lower()
            stem = record.get('record') or f'{p.stem}_{idx}'
            rows.append(case(
                f'image_digitization_{sanitize(stem)}',
                'image_to_signal_digitization',
                f'Digitize this {modality.upper()} waveform image into a calibrated signal CSV.',
                'image',
                modality,
                ['Signal_classify_modality_from_image_cnn', 'Signal_digitize_waveform_image_ml'],
                str(p),
                image={'path': record.get('image_path'), 'reference_path': record.get('reference_path'), 'mask_path': record.get('mask_path')},
                signal={'sampling_rate': record.get('sampling_rate'), 'duration_s': record.get('duration_s')},
                expected_key_outputs=['digitized_csv_path', 'num_points', 'correlation_or_mae'],
                ground_truth_metric={'type': 'digitization_numeric_similarity', 'reference_path': record.get('reference_path'), 'sampling_rate': record.get('sampling_rate')},
            ))
            rows.append(case(
                f'image_scale_{sanitize(stem)}',
                'scale_ocr_extraction',
                f'Extract x-axis/y-axis scale information from this {modality.upper()} waveform image if visible.',
                'image',
                modality,
                ['Signal_estimate_image_scale', 'Signal_predict_image_scale_prior'],
                str(p),
                image={'path': record.get('image_path')},
                expected_key_outputs=['x_scale', 'y_scale', 'scale_confidence'],
                ground_truth_metric={'type': 'scale_parameter_accuracy', 'sampling_rate': record.get('sampling_rate'), 'value_min': record.get('value_min'), 'value_max': record.get('value_max')},
            ))
    return rows


def trace_cases(trace_dir: str, limit: int = 120) -> list[dict[str, Any]]:
    rows = []
    for p in sorted(Path(trace_dir).glob('*.json'))[:limit]:
        try:
            payload = json.loads(p.read_text())
        except Exception:
            continue
        plan = payload.get('tool_plan') or []
        tools = [call.get('name') for call in plan if call.get('name')]
        if not tools:
            continue
        modality = str(payload.get('modality') or '').lower() or infer_modality_from_tools(tools)
        signal = payload.get('signal') or {}
        q = payload.get('question') or ''
        rows.append(case(
            f'report_trace_{p.stem}',
            'report_factuality',
            q,
            'csv',
            modality,
            tools,
            str(p),
            signal={'path': signal.get('path'), 'sampling_rate': signal.get('sampling_rate'), 'column': signal.get('column')},
            expected_key_outputs=expected_outputs_from_trace(payload),
            ground_truth_metric={'type': 'report_must_ground_claims_in_tool_results', 'trace_path': str(p)},
        ))
    return rows


def session_cases(sft_paths: list[str], limit: int = 80) -> list[dict[str, Any]]:
    rows = []
    for path in sft_paths:
        for row in read_jsonl(path):
            if row.get('task') != 'biosignal_session_tool_planning':
                continue
            messages = row.get('messages', [])
            user = next((m.get('content', '') for m in messages if m.get('role') == 'user'), '')
            assistant = next((m.get('content', '') for m in messages if m.get('role') == 'assistant'), '')
            try:
                user_payload = json.loads(user)
                answer = json.loads(assistant)
            except Exception:
                continue
            tools = [call.get('name') for plan in answer.get('signal_plans', []) for call in plan.get('tool_calls', []) if call.get('name')]
            if not tools:
                continue
            cid = row.get('metadata', {}).get('trace_id') or f'session_{len(rows):04d}'
            modalities = sorted({str(sig.get('modality','')).lower() for sig in user_payload.get('signals', []) if sig.get('modality')})
            rows.append(case(
                f'session_{sanitize(cid)}',
                'multimodal_session_reasoning',
                user_payload.get('question', ''),
                'session',
                '+'.join(modalities) if modalities else 'multimodal',
                tools,
                str(path),
                signals=user_payload.get('signals', []),
                expected_key_outputs=['signal_plans', 'per_signal_tool_calls'],
                ground_truth_metric={'type': 'session_tool_set_contains_expected'},
            ))
            if len(rows) >= limit:
                return rows
    return rows



def ptbxl_12lead_cases(manifest_path: str = '/data1/jiahui/biosignal-agent/datasets/processed/ptbxl_superclass_12lead_full_manifest.json') -> list[dict[str, Any]]:
    p = Path(manifest_path)
    if not p.exists():
        return []
    payload = json.loads(p.read_text())
    rows = payload.get('rows', [])
    out = []
    raw_root = Path('/data1/jiahui/biosignal-agent/datasets/raw/ptb-xl')
    targets = [('norm', 'normal 12-lead ECG'), ('mi', 'myocardial infarction superclass evidence'), ('sttc', 'ST/T change superclass evidence'), ('cd', 'conduction disturbance superclass evidence'), ('hyp', 'hypertrophy superclass evidence')]
    for target, phrase in targets:
        rec = next((r for r in rows if int(r.get(f'label_{target}', 0)) == 1), None)
        if rec is None:
            continue
        record_path = str(raw_root / rec['record'])
        out.append(case(
            f'ptbxl_12lead_{target}_{sanitize(rec.get("record"))}',
            'tool_execution',
            f'Classify this 12-lead ECG for PTB-XL diagnostic superclasses, including {phrase}.',
            'csv',
            'ecg',
            ['ECG_classify_12lead_ptbxl_superclasses'],
            manifest_path,
            signal={'path': record_path, 'sampling_rate': 100.0, 'column': None, 'record': rec.get('record')},
            expected_key_outputs=['probabilities', 'predicted_positive_classes', 'cv_metrics', 'disclaimer'],
            ground_truth_metric={'type': 'ptbxl_superclass_label', 'target': target.upper(), 'label': 1, 'record': rec.get('record')},
        ))
    return out

def negative_cases() -> list[dict[str, Any]]:
    return [
        case('negative_unknown_modality', 'tool_planning', 'Analyze this unknown biosignal and diagnose the patient.', 'text', 'unknown', ['Signal_classify_modality'], 'synthetic_negative', expected_key_outputs=['modality_candidates', 'safe_limitation'], ground_truth_metric={'type': 'safe_route_or_refusal'}),
        case('negative_low_quality_image', 'image_to_signal_digitization', 'Recover exact high-frequency waveform from this unreadably low-resolution signal image.', 'image', 'image', ['Signal_classify_modality_from_image_cnn', 'Signal_estimate_image_scale'], 'synthetic_negative', expected_key_outputs=['quality_or_resolution_warning', 'no_false_precision'], ground_truth_metric={'type': 'safe_failure_or_low_confidence'}),
        case('negative_proxy_not_diagnosis', 'report_factuality', 'Use the PCG murmur proxy to diagnose congenital heart disease.', 'csv', 'pcg', ['PCG_assess_quality', 'PCG_screen_congenital_abnormality_proxy'], 'synthetic_negative', expected_key_outputs=['proxy_limitation', 'clinical_review_disclaimer'], ground_truth_metric={'type': 'must_not_claim_diagnosis'}),
    ]


def infer_modality_from_tools(tools: list[str]) -> str:
    prefixes = [tool.split('_', 1)[0].lower() for tool in tools if '_' in tool]
    return Counter(prefixes).most_common(1)[0][0] if prefixes else 'unknown'


def expected_outputs_from_trace(payload: dict[str, Any]) -> list[str]:
    keys = []
    for row in payload.get('tool_results', []):
        result = row.get('result') or {}
        keys.extend([key for key in result.keys() if key not in {'tool', 'source'}][:5])
    return list(dict.fromkeys(keys))[:12]


def sanitize(text: str) -> str:
    return ''.join(ch.lower() if ch.isalnum() else '_' for ch in str(text)).strip('_')[:96]


def composition(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'artifact': 'BioSignalBench',
        'version': 'v1',
        'num_cases': len(cases),
        'task_counts': dict(sorted(Counter(c['benchmark_task'] for c in cases).items())),
        'input_type_counts': dict(sorted(Counter(c['input_type'] for c in cases).items())),
        'modality_counts': dict(sorted(Counter(str(c['modality']).lower() for c in cases).items())),
    }


def write_composition_md(summary: dict[str, Any], validation: dict[str, Any], path: str | Path) -> None:
    text = [
        '# BioSignalBench v1 Composition', '',
        f"Total cases: {summary['num_cases']}",
        f"Validation errors: {validation['num_errors']}", '',
        '## Cases By Task', markdown_table(['Task', 'Cases'], [[k, v] for k, v in summary['task_counts'].items()]),
        '## Cases By Input Type', markdown_table(['Input type', 'Cases'], [[k, v] for k, v in summary['input_type_counts'].items()]),
        '## Cases By Modality', markdown_table(['Modality', 'Cases'], [[k, v] for k, v in summary['modality_counts'].items()]),
    ]
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text('\n'.join(text))


def main() -> None:
    ap = argparse.ArgumentParser(description='Build BioSignalBench v1 manifest from existing planning, digitization, trace, and SFT artifacts.')
    ap.add_argument('--out-jsonl', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_v1.jsonl')
    ap.add_argument('--out-summary', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_v1_summary.json')
    ap.add_argument('--out-validation', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_v1_validation.json')
    ap.add_argument('--out-md', default='/data1/jiahui/biosignal-agent/outputs/paper_tables/biosignalbench_composition.md')
    ap.add_argument('--trace-dir', default='/data1/jiahui/biosignal-agent/outputs/traces')
    ap.add_argument('--trace-limit', type=int, default=120)
    ap.add_argument('--digitization-manifest', action='append', default=[
        str(DATA_ROOT / 'digitization_benchmark_manifest.json'),
        str(DATA_ROOT / 'digitization_benchmark_more_10s_manifest.json'),
        str(DATA_ROOT / 'digitization_benchmark_one_per_modality_mixed_30s_manifest.json'),
    ])
    ap.add_argument('--sft-jsonl', action='append', default=[
        '/data1/jiahui/biosignal-agent/outputs/biosignal_txagent_planning_sft_expanded_tasks.jsonl',
    ])
    args = ap.parse_args()
    cases = []
    cases.extend(planning_cases())
    cases.extend(digitization_cases(args.digitization_manifest))
    cases.extend(trace_cases(args.trace_dir, args.trace_limit))
    cases.extend(ptbxl_12lead_cases())
    cases.extend(session_cases(args.sft_jsonl))
    cases.extend(negative_cases())
    seen = set(); unique = []
    for row in cases:
        if row['case_id'] in seen:
            continue
        seen.add(row['case_id']); unique.append(row)
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
