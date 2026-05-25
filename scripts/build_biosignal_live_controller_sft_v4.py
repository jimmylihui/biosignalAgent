#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.agent.tool_retriever import ToolRetriever
from biosignal_agent.evaluation.biosignalbench import load_bench_cases, retrieve_tools_for_case, write_json, write_jsonl
from scripts.run_biosignalagent_e2e_controller import PLANNER_SYSTEM, task_hint_for_case


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def compact_case_for_prompt(case: dict[str, Any], retrieved_tools: list[str]) -> dict[str, Any]:
    return {
        'question': case.get('question'),
        'input_type': case.get('input_type'),
        'modality_hint': case.get('modality'),
        'signal': compact_signal(case.get('signal')),
        'image': compact_image(case.get('image')),
        'signals': compact_signals(case.get('signals')),
        'retrieved_tools': retrieved_tools,
        'task_hint': task_hint_for_case(case),
        'instruction': 'Choose the minimal sufficient tools from retrieved_tools only. Do not include tools not needed by the question. Keep arguments compact with null placeholders; do not copy long signal_path or image_path strings. For multimodal sessions, include quality plus core measurement tools for every mentioned modality; 6-10 tool calls are acceptable when the session has several modalities.',
    }


def compact_signal(signal: Any) -> Any:
    if not isinstance(signal, dict):
        return signal
    return {
        'has_path': bool(signal.get('path')),
        'sampling_rate': signal.get('sampling_rate'),
        'column': signal.get('column'),
    }


def compact_image(image: Any) -> Any:
    if not isinstance(image, dict):
        return image
    return {'has_path': bool(image.get('path')), 'filename': Path(str(image.get('path'))).name if image.get('path') else None}


def compact_signals(signals: Any) -> Any:
    if not isinstance(signals, list):
        return signals
    return [
        {
            'modality': sig.get('modality'),
            'has_path': bool(sig.get('path')),
            'sampling_rate': sig.get('sampling_rate'),
            'column': sig.get('column'),
        }
        for sig in signals
        if isinstance(sig, dict)
    ]


def assistant_for_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        'modality': case.get('modality'),
        'tool_calls': [{'name': name, 'arguments': compact_arguments(name, case)} for name in case.get('expected_tools', [])],
        'safety_notes': safety_notes(case),
        'limitations': limitations(case),
    }


def compact_arguments(name: str, case: dict[str, Any]) -> dict[str, Any]:
    if case.get('input_type') == 'image':
        return {'image_path': None, 'sampling_rate': (case.get('signal') or {}).get('sampling_rate')}
    if case.get('input_type') == 'session':
        signal = session_signal_for_tool(name, case.get('signals') or [])
        return {
            'signal_path': None,
            'sampling_rate': signal.get('sampling_rate') if signal else None,
            'column': signal.get('column') if signal else None,
        }
    signal = case.get('signal') or {}
    return {'signal_path': None, 'sampling_rate': signal.get('sampling_rate'), 'column': signal.get('column')}


def session_signal_for_tool(name: str, signals: list[dict[str, Any]]) -> dict[str, Any] | None:
    prefix = name.split('_', 1)[0].lower()
    aliases = {'spo2': 'spo2'}
    wanted = aliases.get(prefix, prefix)
    for signal in signals:
        if str(signal.get('modality', '')).lower() == wanted:
            return signal
    return None


def safety_notes(case: dict[str, Any]) -> list[str]:
    notes = []
    text = f"{case.get('question','')} {' '.join(case.get('expected_tools', []))}".lower()
    if 'proxy' in text or 'screen' in text or 'diagnos' in text:
        notes.append('Proxy or screening outputs are not clinical diagnoses.')
    if case.get('input_type') == 'image':
        notes.append('Report low confidence when image trace, resolution, or axis scale is unclear.')
    return notes


def limitations(case: dict[str, Any]) -> list[str]:
    limits = ['Research use only; not a clinical diagnosis.']
    if case.get('input_type') == 'image':
        limits.append('Image-derived signals and scale estimates depend on visible traces and axis labels.')
    if case.get('benchmark_task') == 'multimodal_session_reasoning':
        limits.append('Session conclusions must remain grounded in per-signal tool outputs.')
    return limits


def oversample_count(case: dict[str, Any], failed_ids: set[str], focus: int) -> int:
    task = case.get('benchmark_task')
    if case.get('case_id') in failed_ids:
        return max(focus, 4)
    if task in {'scale_ocr_extraction', 'tool_execution', 'multimodal_session_reasoning'}:
        return max(focus, 3)
    if task in {'image_to_signal_digitization', 'report_factuality'}:
        return max(2, focus // 2)
    return 1


def build_rows(manifest: str, live_failures: str | None, retrieved_k: int, focus: int) -> list[dict[str, Any]]:
    cases = load_bench_cases(manifest)
    failed_ids = set()
    if live_failures:
        for row in read_jsonl(live_failures):
            if row.get('failure_reason'):
                failed_ids.add(str(row.get('case_id')))
    retriever = ToolRetriever()
    rows = []
    for case in cases:
        retrieved = retrieve_tools_for_case(case, retriever, 'tfidf', retrieved_k, {})
        # Make sure the target tools are available during SFT. If the retriever misses
        # a tool, put it at the end so this trains planning rather than retrieval.
        for tool in case.get('expected_tools') or []:
            if tool not in retrieved:
                retrieved.append(tool)
        repeats = oversample_count(case, failed_ids, focus)
        for rep in range(repeats):
            user = compact_case_for_prompt(case, retrieved)
            if rep > 0:
                user['sft_focus'] = 'hard_case_minimal_tool_selection'
            rows.append({
                'task': 'planner',
                'messages': [
                    {'role': 'system', 'content': PLANNER_SYSTEM},
                    {'role': 'user', 'content': json.dumps(user, sort_keys=True)},
                    {'role': 'assistant', 'content': json.dumps(assistant_for_case(case), sort_keys=True)},
                ],
                'metadata': {
                    'source': manifest,
                    'case_id': case.get('case_id'),
                    'benchmark_task': case.get('benchmark_task'),
                    'repeat': rep,
                    'from_live_failure': case.get('case_id') in failed_ids,
                },
            })
    rows.extend(extra_negative_rows())
    return rows


def extra_negative_rows() -> list[dict[str, Any]]:
    specs = [
        {
            'case_id': 'negative_scale_not_digitize',
            'user': {
                'question': 'Extract x-axis/y-axis scale information from this waveform image if visible.',
                'input_type': 'image',
                'modality_hint': 'ppg',
                'retrieved_tools': ['Signal_classify_modality_from_image_cnn', 'Signal_digitize_waveform_image_ml', 'Signal_estimate_image_scale', 'Signal_predict_image_scale_prior'],
                'task_hint': 'scale_or_axis_extraction: choose scale/OCR/prior tools, not waveform digitization tools.',
                'instruction': 'Choose the minimal sufficient tools from retrieved_tools only.',
            },
            'assistant': {
                'modality': 'ppg',
                'tool_calls': [{'name': 'Signal_estimate_image_scale', 'arguments': {'image_path': None}}, {'name': 'Signal_predict_image_scale_prior', 'arguments': {'image_path': None}}],
                'safety_notes': ['Do not digitize when the user only asks for scale/axis extraction.'],
                'limitations': ['Scale extraction depends on visible axis labels; research use only.'],
            },
        },
        {
            'case_id': 'negative_12lead_not_rpeaks',
            'user': {
                'question': 'Classify this 12-lead ECG into PTB-XL diagnostic superclasses.',
                'input_type': 'csv',
                'modality_hint': 'ecg',
                'retrieved_tools': ['ECG_assess_quality', 'ECG_detect_r_peaks', 'ECG_compute_hrv', 'ECG_classify_12lead_ptbxl_superclasses'],
                'task_hint': '12lead_ecg_superclass_classification: choose ECG_classify_12lead_ptbxl_superclasses.',
                'instruction': 'Choose the minimal sufficient tools from retrieved_tools only.',
            },
            'assistant': {
                'modality': 'ecg',
                'tool_calls': [{'name': 'ECG_classify_12lead_ptbxl_superclasses', 'arguments': {'signal_path': None, 'sampling_rate': None, 'column': None}}],
                'safety_notes': [],
                'limitations': ['Research use only; not a clinical diagnosis.'],
            },
        },
    ]
    return [
        {
            'task': 'planner',
            'messages': [
                {'role': 'system', 'content': PLANNER_SYSTEM},
                {'role': 'user', 'content': json.dumps(spec['user'], sort_keys=True)},
                {'role': 'assistant', 'content': json.dumps(spec['assistant'], sort_keys=True)},
            ],
            'metadata': {'source': 'synthetic_live_controller_negative', 'case_id': spec['case_id']},
        }
        for spec in specs
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description='Build live-controller-aligned planner SFT data from BioSignalBench and live failures.')
    ap.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_v1.jsonl')
    ap.add_argument('--live-failures', default='/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_live_cases.jsonl')
    ap.add_argument('--retrieved-k', type=int, default=20)
    ap.add_argument('--focus', type=int, default=4)
    ap.add_argument('--out-jsonl', default='/data1/jiahui/biosignal-agent/outputs/biosignal_sft_planner_v4_live_controller.jsonl')
    ap.add_argument('--out-summary', default='/data1/jiahui/biosignal-agent/outputs/biosignal_sft_planner_v4_live_controller_summary.json')
    args = ap.parse_args()
    rows = build_rows(args.manifest, args.live_failures, args.retrieved_k, args.focus)
    write_jsonl(args.out_jsonl, rows)
    summary = {
        'artifact': 'BioSignalLiveControllerPlannerSFTData',
        'version': 'v4_live_controller',
        'num_examples': len(rows),
        'manifest': args.manifest,
        'live_failures': args.live_failures,
        'retrieved_k': args.retrieved_k,
        'focus': args.focus,
        'out_jsonl': args.out_jsonl,
    }
    write_json(args.out_summary, summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
