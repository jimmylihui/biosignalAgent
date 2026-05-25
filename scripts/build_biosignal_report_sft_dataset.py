#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.evaluation.biosignalbench import load_bench_cases, write_json, write_jsonl

REPORT_SYSTEM = 'You are BioSignalAgent. Write a concise evidence-grounded biosignal report from tool outputs. Do not diagnose from proxy tools.'


def compact_result(value: Any, max_list: int = 8) -> Any:
    if isinstance(value, dict):
        return {k: compact_result(v, max_list=max_list) for k, v in value.items() if k not in {'source'}}
    if isinstance(value, list):
        if len(value) > max_list:
            return value[:max_list] + [f'... ({len(value)} total)']
        return value
    return value


def trace_to_example(trace: dict[str, Any], source: str, compact: bool = True, use_grounded_template: bool = False) -> dict[str, Any]:
    tool_results = trace.get('tool_results') or []
    if compact:
        tool_results = compact_result(tool_results)
    user = {
        'question': trace.get('question'),
        'tool_results': tool_results,
        'disclaimer_required': True,
        'report_requirements': [
            'Only state findings supported by tool_results.',
            'Include numeric values only when present in tool_results.',
            'Mention low confidence, proxy status, or limitations when present.',
            'End with research-use / not-a-clinical-diagnosis disclaimer.',
        ],
    }
    assistant = grounded_template_report(trace) if use_grounded_template else (trace.get('final_report') or deterministic_report(trace))
    return {
        'task': 'report_grounding',
        'messages': [
            {'role': 'system', 'content': REPORT_SYSTEM},
            {'role': 'user', 'content': json.dumps(user, sort_keys=True)},
            {'role': 'assistant', 'content': assistant},
        ],
        'metadata': {'source': source, 'trace_id': trace.get('trace_id'), 'compact_tool_results': compact},
    }



def grounded_template_report(trace: dict[str, Any]) -> str:
    lines=[f"Question: {trace.get('question','')}", 'Tool-grounded findings:']
    for item in trace.get('tool_results') or []:
        tool=item.get('tool') or item.get('name') or 'tool'
        result=item.get('result') or {}
        parts=[]
        for key,val in result.items():
            if key in {'tool','source','r_peak_indices','peak_indices','events','segments'} or val is None:
                continue
            if isinstance(val, list):
                parts.append(f'{key}={len(val)} values' if len(val)>5 else f'{key}={val}')
            elif isinstance(val, dict):
                compact=[]
                for kk,vv in val.items():
                    if isinstance(vv,(str,int,float,bool)) and vv is not None:
                        compact.append(f'{kk}={round(vv,3) if isinstance(vv,float) else vv}')
                    if len(compact)>=4:
                        break
                if compact:
                    parts.append(f'{key}: ' + ', '.join(compact))
            else:
                parts.append(f'{key}={round(val,3) if isinstance(val,float) else val}')
        if not parts:
            parts=['completed; no scalar summary fields returned']
        lines.append(f'- {tool}: ' + '; '.join(parts) + '.')
    lines.append('Interpretation is limited to these tool outputs; research use only, not a clinical diagnosis.')
    return '\n'.join(lines)

def deterministic_report(trace: dict[str, Any]) -> str:
    lines = [f"Question: {trace.get('question', '')}", 'Tool findings:']
    for item in trace.get('tool_results') or []:
        tool = item.get('tool') or item.get('name') or 'tool'
        result = item.get('result') or {}
        bits = []
        for key in ['quality','heart_rate_bpm','respiratory_rate_bpm','mean_rr_ms','sdnn_ms','rmssd_ms','num_peaks','confidence','classification','prediction','risk_level','method']:
            if key in result and result.get(key) is not None:
                val = result[key]
                if isinstance(val, float):
                    val = round(val, 3)
                bits.append(f'{key}={val}')
        if not bits:
            bits = ['completed']
        lines.append(f'- {tool}: ' + ', '.join(bits) + '.')
    lines.append('Research use only; not a clinical diagnosis.')
    return '\n'.join(lines)


def dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen=set(); out=[]
    for row in rows:
        key=json.dumps(row['messages'], sort_keys=True)
        if key in seen:
            continue
        seen.add(key); out.append(row)
    return out


def main() -> None:
    ap=argparse.ArgumentParser(description='Build report-grounding SFT data from BioSignalBench report traces.')
    ap.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_v1.jsonl')
    ap.add_argument('--out-jsonl', default='/data1/jiahui/biosignal-agent/outputs/biosignal_sft_report_grounding_v1.jsonl')
    ap.add_argument('--out-summary', default='/data1/jiahui/biosignal-agent/outputs/biosignal_sft_report_grounding_v1_summary.json')
    ap.add_argument('--include-full-results', action='store_true')
    ap.add_argument('--use-grounded-template', action='store_true')
    args=ap.parse_args()
    rows=[]; missing=[]
    for case in load_bench_cases(args.manifest):
        if case.get('benchmark_task') != 'report_factuality':
            continue
        trace_path=(case.get('ground_truth_metric') or {}).get('trace_path') or case.get('source')
        if not trace_path or not Path(trace_path).exists():
            missing.append(case.get('case_id')); continue
        trace=json.loads(Path(trace_path).read_text())
        rows.append(trace_to_example(trace, trace_path, compact=True, use_grounded_template=args.use_grounded_template))
        if args.include_full_results:
            rows.append(trace_to_example(trace, trace_path, compact=False, use_grounded_template=args.use_grounded_template))
    rows=dedupe(rows)
    write_jsonl(args.out_jsonl, rows)
    summary={'artifact':'BioSignalReportGroundingSFT','num_examples':len(rows),'missing_traces':missing,'out_jsonl':args.out_jsonl}
    write_json(args.out_summary, summary)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
