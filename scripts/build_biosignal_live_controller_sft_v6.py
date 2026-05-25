#!/usr/bin/env python3
"""Build v6 planner SFT data with train-only multimodal session augmentation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.agent.tool_retriever import ToolRetriever
from biosignal_agent.evaluation.biosignalbench import retrieve_tools_for_case, write_json, write_jsonl
from scripts.build_biosignal_live_controller_sft_v4 import (
    assistant_for_case,
    build_rows,
    compact_case_for_prompt,
)
from scripts.run_biosignalagent_e2e_controller import PLANNER_SYSTEM

CORE_TOOLS = {
    'ecg': ['ECG_detect_r_peaks', 'ECG_compute_hrv'],
    'ppg': ['PPG_detect_peaks'],
    'resp': ['RESP_estimate_rate'],
    'spo2': ['SpO2_summarize'],
    'abp': ['ABP_detect_pulses'],
    'pcg': ['PCG_detect_heart_sounds'],
    'acc': ['ACC_summarize_activity'],
    'eda': ['EDA_summarize'],
    'eeg': ['EEG_compute_bandpower'],
    'emg': ['EMG_summarize_activation'],
    'scg': ['SCG_detect_j_peaks'],
    'bcg': ['BCG_detect_j_peaks'],
}

COMBOS = [
    ('abp+ecg+spo2', ['ecg', 'abp', 'spo2'], 'Analyze ECG, arterial blood pressure, and SpO2 to summarize heart rate and oxygenation.'),
    ('abp+ecg+pcg', ['pcg', 'abp', 'ecg'], 'Analyze PCG heart sounds together with ABP and ECG heart-rate evidence.'),
    ('ecg+ppg+resp+spo2', ['ecg', 'ppg', 'resp', 'spo2'], 'Estimate cardiac rate, respiratory rate, oxygen saturation, and summarize confidence across these signals.'),
    ('acc+eda+ppg', ['ppg', 'acc', 'eda'], 'Analyze pulse, activity, and skin conductance to summarize wearable physiological state.'),
    ('acc+eeg+emg', ['eeg', 'emg', 'acc'], 'Analyze EEG, EMG, and accelerometer signals to summarize neuro-muscular activity and movement confidence.'),
    ('resp+scg', ['scg', 'resp'], 'Analyze SCG cardiac mechanical peaks together with respiration rate evidence.'),
    ('bcg+ecg+resp', ['bcg', 'ecg', 'resp'], 'Analyze BCG, ECG, and respiration to summarize cardiac and breathing status.'),
]

FOLLOWUP_TEMPLATES = [
    'For each modality in this session, select only the core measurement tools needed before reporting; add quality tools only if the question asks about reliability, artifacts, or confidence.',
    'Do not stop after one signal; plan tools for every modality that appears in the session.',
    'Use compact null arguments and avoid redundant preflight tools unless explicitly requested.',
]


def expected_tools(modalities: list[str]) -> list[str]:
    tools: list[str] = []
    for mod in modalities:
        tools.extend(CORE_TOOLS[mod])
    return list(dict.fromkeys(tools))


def synthetic_case(combo_name: str, modalities: list[str], question: str, idx: int) -> dict[str, Any]:
    return {
        'case_id': f'synthetic_train_session_{idx:03d}_{combo_name.replace("+", "_")}',
        'benchmark_task': 'multimodal_session_reasoning',
        'question': question,
        'input_type': 'session',
        'modality': combo_name,
        'expected_tools': expected_tools(modalities),
        'expected_key_outputs': ['signal_plans', 'per_signal_tool_calls'],
        'ground_truth_metric': {'type': 'session_tool_set_contains_expected'},
        'source': 'synthetic_train_only_session_augmentation_v6',
        'signals': [
            {'modality': mod, 'path': None, 'sampling_rate': default_fs(mod), 'column': None}
            for mod in modalities
        ],
    }


def default_fs(modality: str) -> float:
    return {'ecg': 250.0, 'ppg': 125.0, 'resp': 50.0, 'spo2': 1.0, 'abp': 125.0, 'pcg': 1000.0, 'acc': 50.0, 'eda': 4.0, 'eeg': 128.0, 'emg': 1000.0, 'scg': 200.0, 'bcg': 100.0}.get(modality, 100.0)


def synthetic_rows(retrieved_k: int, repeats: int) -> list[dict[str, Any]]:
    retriever = ToolRetriever()
    rows = []
    idx = 0
    for combo_name, modalities, base_question in COMBOS:
        for suffix in [''] + FOLLOWUP_TEMPLATES:
            question = base_question if not suffix else f'{base_question} {suffix}'
            case = synthetic_case(combo_name, modalities, question, idx)
            idx += 1
            retrieved = retrieve_tools_for_case(case, retriever, 'tfidf', retrieved_k, {})
            for tool in case['expected_tools']:
                if tool not in retrieved:
                    retrieved.append(tool)
            for rep in range(repeats):
                user = compact_case_for_prompt(case, retrieved)
                user['sft_focus'] = 'train_only_multimodal_complete_per_modality_bundle'
                if rep > 0:
                    user['session_aug_repeat'] = rep
                rows.append({
                    'task': 'planner',
                    'messages': [
                        {'role': 'system', 'content': PLANNER_SYSTEM},
                        {'role': 'user', 'content': json.dumps(user, sort_keys=True)},
                        {'role': 'assistant', 'content': json.dumps(assistant_for_case(case), sort_keys=True)},
                    ],
                    'metadata': {
                        'source': 'synthetic_train_only_session_augmentation_v6',
                        'case_id': case['case_id'],
                        'benchmark_task': case['benchmark_task'],
                        'repeat': rep,
                    },
                })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_splits_v1/biosignalbench_v1_train.jsonl')
    ap.add_argument('--live-failures', default='/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_live_cases.jsonl')
    ap.add_argument('--retrieved-k', type=int, default=20)
    ap.add_argument('--focus', type=int, default=4)
    ap.add_argument('--session-repeats', type=int, default=3)
    ap.add_argument('--out-jsonl', default='/data1/jiahui/biosignal-agent/outputs/biosignal_sft_planner_v6_train_session_aug.jsonl')
    ap.add_argument('--out-summary', default='/data1/jiahui/biosignal-agent/outputs/biosignal_sft_planner_v6_train_session_aug_summary.json')
    args = ap.parse_args()
    base = build_rows(args.manifest, args.live_failures, args.retrieved_k, args.focus)
    aug = synthetic_rows(args.retrieved_k, args.session_repeats)
    rows = base + aug
    write_jsonl(args.out_jsonl, rows)
    summary = {
        'artifact': 'BioSignalLiveControllerPlannerSFTData',
        'version': 'v6_train_split_session_aug',
        'num_examples': len(rows),
        'base_examples': len(base),
        'synthetic_session_examples': len(aug),
        'manifest': args.manifest,
        'retrieved_k': args.retrieved_k,
        'focus': args.focus,
        'session_repeats': args.session_repeats,
        'out_jsonl': args.out_jsonl,
    }
    write_json(args.out_summary, summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
