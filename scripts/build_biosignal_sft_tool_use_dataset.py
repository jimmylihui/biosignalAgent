#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.evaluation.biosignalbench import load_bench_cases, read_jsonl, write_json, write_jsonl
from biosignal_agent.agent.schema_loader import load_tool_schemas

PLANNER_SYSTEM = 'You are BioSignalAgent. Select valid local biosignal tools and return strict JSON with modality, tool_calls, safety_notes, and limitations.'
REPORT_SYSTEM = 'You are BioSignalAgent. Write a concise evidence-grounded biosignal report from tool results. Do not diagnose from proxy tools.'


def normalize_existing(paths: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    planner = []
    report = []
    for path in paths:
        for row in read_jsonl(path):
            task = row.get('task')
            messages = row.get('messages') or []
            if not messages:
                continue
            if task in {'biosignal_tool_planning', 'biosignal_session_tool_planning'}:
                planner.append({'task': 'planner', 'messages': messages, 'metadata': {'source': path, **row.get('metadata', {})}})
            elif task in {'biosignal_tool_execution_trace', 'biosignal_report_generation'}:
                report.append({'task': 'planner_report', 'messages': messages, 'metadata': {'source': path, **row.get('metadata', {})}})
    return planner, report


def examples_from_bench(manifest: str, max_cases: int | None = None, focus_oversample: int = 1) -> list[dict[str, Any]]:
    rows = []
    schemas = load_tool_schemas()
    by_modality = {}
    for schema in schemas:
        by_modality.setdefault(str(schema.get('modality','')).lower(), []).append(schema['name'])
    cases = load_bench_cases(manifest)
    if max_cases:
        cases = cases[:max_cases]
    for case in cases:
        for variant in prompt_variants_for_case(case, focus_oversample):
            user = user_payload_for_case(case, by_modality, variant)
            assistant = {
                'modality': case.get('modality'),
                'tool_calls': [{'name': name, 'arguments': default_arguments(name, case)} for name in case.get('expected_tools', [])],
                'safety_notes': safety_notes(case),
                'limitations': limitations(case),
            }
            rows.append({
                'task': 'planner',
                'messages': [
                    {'role': 'system', 'content': PLANNER_SYSTEM},
                    {'role': 'user', 'content': json.dumps(user, sort_keys=True)},
                    {'role': 'assistant', 'content': json.dumps(assistant, sort_keys=True)},
                ],
                'metadata': {'source': manifest, 'case_id': case.get('case_id'), 'benchmark_task': case.get('benchmark_task'), 'variant': variant},
            })
    return rows



def prompt_variants_for_case(case: dict[str, Any], focus_oversample: int) -> list[str]:
    task = case.get('benchmark_task')
    input_type = case.get('input_type')
    if task in {'multimodal_session_reasoning', 'image_to_signal_digitization', 'scale_ocr_extraction'} or input_type in {'session', 'image'}:
        variants = ['standard', 'compact_tool_json']
        if focus_oversample >= 3:
            variants.append('failure_aware')
        return variants[:max(1, focus_oversample)]
    return ['standard']


def user_payload_for_case(case: dict[str, Any], by_modality: dict[str, list[str]], variant: str) -> dict[str, Any]:
    signal = case.get('signal') or {}
    user = {
        'question': case.get('question'),
        'input_type': case.get('input_type'),
        'modality_hint': case.get('modality'),
        'signal': signal,
        'image': case.get('image'),
        'signals': case.get('signals'),
        'retrieved_tools': tool_candidates(case, by_modality),
    }
    if variant == 'compact_tool_json':
        user['planner_instruction'] = 'Return compact strict JSON. For session inputs, call each modality-specific tool once using only that signal_path, sampling_rate, and column; do not repeat the full signals list in every call.'
    elif variant == 'failure_aware':
        user['planner_instruction'] = 'Avoid overselecting tools. Include expected quality checks and route image cases through modality/scale/digitization only when needed. Keep proxy limitations explicit.'
    return user


def tool_candidates(case: dict[str, Any], by_modality: dict[str, list[str]], max_candidates: int = 14) -> list[str]:
    expected = list(case.get('expected_tools', []))
    modality = str(case.get('modality', '')).lower()
    candidates = []
    for name in expected:
        if name not in candidates:
            candidates.append(name)
    modality_parts = modality.split('+') if modality else []
    pools = []
    for part in modality_parts:
        pools.extend(by_modality.get(part, []))
    pools.extend(by_modality.get('any', [])); pools.extend(by_modality.get('image', []) if case.get('input_type') == 'image' else [])
    for name in pools:
        if name not in candidates:
            candidates.append(name)
        if len(candidates) >= max_candidates:
            break
    return candidates

def default_arguments(name: str, case: dict[str, Any]) -> dict[str, Any]:
    if case.get('input_type') == 'image':
        return {'image_path': (case.get('image') or {}).get('path'), 'sampling_rate': (case.get('signal') or {}).get('sampling_rate')}
    if case.get('input_type') == 'session':
        signal = session_signal_for_tool(name, case.get('signals') or [])
        if signal:
            return {'signal_path': signal.get('path'), 'sampling_rate': signal.get('sampling_rate'), 'column': signal.get('column')}
        return {'signals': case.get('signals')}
    signal = case.get('signal') or {}
    return {'signal_path': signal.get('path'), 'sampling_rate': signal.get('sampling_rate'), 'column': signal.get('column')}


def session_signal_for_tool(name: str, signals: list[dict[str, Any]]) -> dict[str, Any] | None:
    prefix = name.split('_', 1)[0].lower()
    aliases = {'spo2': 'spo2'}
    wanted = aliases.get(prefix, prefix)
    for signal in signals:
        if str(signal.get('modality', '')).lower() == wanted:
            return signal
    return signals[0] if len(signals) == 1 else None


def safety_notes(case: dict[str, Any]) -> list[str]:
    notes = []
    text = f"{case.get('question','')} {' '.join(case.get('expected_tools', []))}".lower()
    if 'proxy' in text or 'diagnos' in text:
        notes.append('Proxy outputs are screening evidence only and must not be phrased as diagnosis.')
    if case.get('input_type') == 'image':
        notes.append('Report low confidence when image trace, resolution, or axis scale is unclear.')
    if str(case.get('modality')).lower() in {'unknown', 'image'}:
        notes.append('Classify modality or ask for better input before running modality-specific analysis.')
    return notes


def limitations(case: dict[str, Any]) -> list[str]:
    limits = ['Research use only; not a clinical diagnosis.']
    if case.get('input_type') == 'image':
        limits.append('Digitization and scale extraction can fail on low-resolution or unlabeled axes.')
    if case.get('benchmark_task') == 'multimodal_session_reasoning':
        limits.append('Session-level conclusions must remain grounded in per-signal tool outputs.')
    return limits


def negative_examples() -> list[dict[str, Any]]:
    specs = [
        ('unknown_modality_refuse_diagnosis', {'question':'Diagnose this unknown waveform without any modality or sampling rate.', 'input_type':'csv'}, {'modality': None, 'tool_calls': [], 'safety_notes':['Cannot choose a modality-specific tool without modality or sampling rate.'], 'limitations':['Ask for modality/sampling rate or run modality classifier first.']}),
        ('low_res_image_no_false_precision', {'question':'Recover the exact waveform from this unreadably low resolution image.', 'input_type':'image'}, {'modality':'image', 'tool_calls':[{'name':'Signal_classify_modality_from_image_cnn','arguments':{'image_path':None}}], 'safety_notes':['Do not claim exact waveform recovery from unreadable image.'], 'limitations':['Return low confidence and request higher resolution or axis labels.']}),
        ('proxy_not_diagnosis', {'question':'Use PCG proxy output to diagnose congenital heart disease.', 'input_type':'csv', 'modality_hint':'pcg'}, {'modality':'pcg', 'tool_calls':[{'name':'PCG_assess_quality','arguments':{}},{'name':'PCG_screen_congenital_abnormality_proxy','arguments':{}}], 'safety_notes':['Do not diagnose CHD from proxy or modest-AUROC screening models.'], 'limitations':['Recommend clinical review/echocardiography for diagnosis.']}),
    ]
    rows = []
    for case_id, user, assistant in specs:
        rows.append({'task':'planner','messages':[{'role':'system','content':PLANNER_SYSTEM},{'role':'user','content':json.dumps(user,sort_keys=True)},{'role':'assistant','content':json.dumps(assistant,sort_keys=True)}],'metadata':{'source':'synthetic_negative','case_id':case_id}})
    return rows


def dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set(); out = []
    for row in rows:
        key = json.dumps(row.get('messages', []), sort_keys=True)
        if key in seen:
            continue
        seen.add(key); out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description='Build standardized planner-only and planner-report SFT datasets for BioSignalAgent.')
    ap.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_v1.jsonl')
    ap.add_argument('--existing-jsonl', action='append', default=[
        '/data1/jiahui/biosignal-agent/outputs/biosignal_txagent_sft_expanded_tasks.jsonl',
        '/data1/jiahui/biosignal-agent/outputs/biosignal_txagent_planning_sft_expanded_tasks.jsonl',
    ])
    ap.add_argument('--out-planner', default='/data1/jiahui/biosignal-agent/outputs/biosignal_sft_planner_v1.jsonl')
    ap.add_argument('--out-report', default='/data1/jiahui/biosignal-agent/outputs/biosignal_sft_planner_report_v1.jsonl')
    ap.add_argument('--out-summary', default='/data1/jiahui/biosignal-agent/outputs/biosignal_sft_v1_summary.json')
    ap.add_argument('--focus-oversample', type=int, default=2)
    args = ap.parse_args()
    planner_existing, report_existing = normalize_existing(args.existing_jsonl)
    planner_rows = dedupe(planner_existing + examples_from_bench(args.manifest, focus_oversample=args.focus_oversample) + negative_examples())
    report_rows = dedupe(report_existing)
    write_jsonl(args.out_planner, planner_rows)
    write_jsonl(args.out_report, report_rows)
    summary = {
        'artifact': 'BioSignalAgentSFTData',
        'version': 'v1',
        'planner_examples': len(planner_rows),
        'planner_report_examples': len(report_rows),
        'focus_oversample': args.focus_oversample,
        'sources': args.existing_jsonl + [args.manifest, 'synthetic_negative'],
        'out_planner': args.out_planner,
        'out_report': args.out_report,
    }
    write_json(args.out_summary, summary)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
