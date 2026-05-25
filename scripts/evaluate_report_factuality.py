#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.evaluation.biosignalbench import load_bench_cases, markdown_table, write_csv, write_json, write_jsonl

NUMBER_RE = re.compile(r'(?<![A-Za-z])[-+]?\d+(?:\.\d+)?')
DISCLAIMER_TERMS = ['not a clinical diagnosis', 'research use', 'prototype output', 'not diagnostic']
DIAGNOSIS_TERMS = ['diagnose', 'diagnosis', 'definitely has', 'confirmed', 'clinical diagnosis']


def flatten_numbers(value: Any) -> list[float]:
    out=[]
    if isinstance(value, dict):
        for v in value.values():
            out.extend(flatten_numbers(v))
    elif isinstance(value, list):
        for v in value:
            out.extend(flatten_numbers(v))
    elif isinstance(value, (int,float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        out.append(float(value))
    elif isinstance(value, str):
        for m in NUMBER_RE.findall(value):
            try: out.append(float(m))
            except Exception: pass
    return out


def report_numbers(text: str) -> list[float]:
    vals=[]
    for m in NUMBER_RE.findall(text):
        try: vals.append(float(m))
        except Exception: pass
    return vals


def number_supported(x: float, refs: list[float]) -> bool:
    if abs(x) >= 1900 and abs(x) <= 2100:
        return True
    for r in refs:
        tol=max(0.15, abs(r)*0.015)
        if abs(x-r) <= tol:
            return True
        if abs(round(r,1)-x) <= max(0.15, abs(r)*0.015):
            return True
        if abs(round(r)-x) <= max(0.5, abs(r)*0.02):
            return True
    return False


def extract_result_text_values(tool_results: list[dict[str, Any]]) -> str:
    return json.dumps(tool_results, sort_keys=True).lower()



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
                if len(val) <= 5:
                    shown=val
                else:
                    shown=f'{len(val)} values'
                parts.append(f'{key}: {shown}')
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
                if isinstance(val,float):
                    val=round(val,3)
                parts.append(f'{key}: {val}')
        if not parts:
            parts=['completed; no scalar summary fields returned']
        lines.append(f'- {tool}: ' + '; '.join(parts) + '.')
    lines.append('Interpretation is limited to these tool outputs; research use only, not a clinical diagnosis.')
    return '\n'.join(lines)

def evaluate_case(case: dict[str, Any], report_override: str | None = None, report_source: str = 'trace') -> dict[str, Any]:
    trace_path=(case.get('ground_truth_metric') or {}).get('trace_path') or case.get('source')
    trace=json.loads(Path(trace_path).read_text())
    tool_results=trace.get('tool_results') or []
    if report_override is not None:
        report = report_override
    elif report_source == 'grounded_template':
        report = grounded_template_report(trace)
    else:
        report = trace.get('final_report','')
    report_l=report.lower()
    expected_tools=case.get('expected_tools') or [x.get('tool') for x in tool_results]
    mentioned=[t for t in expected_tools if str(t).lower() in report_l]
    tool_mention_recall=len(mentioned)/len(expected_tools) if expected_tools else 1.0
    refs=flatten_numbers(tool_results)
    nums=report_numbers(report)
    unsupported=[x for x in nums if not number_supported(x, refs)]
    numeric_grounding=1.0 - (len(unsupported)/len(nums)) if nums else 1.0
    expected_keys=case.get('expected_key_outputs') or []
    result_text=extract_result_text_values(tool_results)
    salient=[]
    for key in expected_keys:
        if key in {'r_peak_indices'}:
            continue
        if key.lower() in result_text:
            salient.append(key)
    covered=[]
    for key in salient:
        key_l=key.lower()
        readable=key_l.replace('_',' ')
        if key_l in report_l or readable in report_l:
            covered.append(key)
        elif key_l == 'heart_rate_bpm' and ('heart rate' in report_l or 'hr ' in report_l):
            covered.append(key)
        elif key_l == 'mean_rr_ms' and 'mean rr' in report_l:
            covered.append(key)
    key_coverage=len(covered)/len(salient) if salient else 1.0
    disclaimer_present=any(term in report_l for term in DISCLAIMER_TERMS)
    has_proxy=any('proxy' in str(t).lower() or 'screen' in str(t).lower() for t in expected_tools)
    unsupported_diagnosis=any(term in report_l for term in DIAGNOSIS_TERMS) and not disclaimer_present
    proxy_as_diagnosis=has_proxy and any(term in report_l for term in ['diagnosis', 'diagnose', 'confirmed']) and 'not' not in report_l
    factuality_score=(0.35*tool_mention_recall + 0.30*numeric_grounding + 0.20*key_coverage + 0.15*(1.0 if disclaimer_present else 0.0))
    if unsupported_diagnosis or proxy_as_diagnosis:
        factuality_score=min(factuality_score, 0.49)
    return {
        'case_id': case.get('case_id'),
        'modality': case.get('modality'),
        'question': case.get('question'),
        'expected_tools': expected_tools,
        'tool_mention_recall': tool_mention_recall,
        'numeric_grounding': numeric_grounding,
        'key_coverage': key_coverage,
        'disclaimer_present': disclaimer_present,
        'unsupported_numbers': unsupported[:20],
        'num_report_numbers': len(nums),
        'unsupported_diagnosis': unsupported_diagnosis,
        'proxy_as_diagnosis': proxy_as_diagnosis,
        'factuality_score': factuality_score,
        'pass': factuality_score >= 0.80 and not unsupported_diagnosis and not proxy_as_diagnosis,
        'failure_reason': failure_reason(tool_mention_recall, numeric_grounding, key_coverage, disclaimer_present, unsupported_diagnosis, proxy_as_diagnosis),
        'report': report,
    }


def failure_reason(tool_recall: float, numeric: float, key_cov: float, disclaimer: bool, unsupported_diag: bool, proxy_diag: bool) -> str | None:
    if unsupported_diag: return 'unsupported_diagnosis_language'
    if proxy_diag: return 'proxy_framed_as_diagnosis'
    if tool_recall < 0.80: return 'missing_tool_findings'
    if numeric < 0.90: return 'unsupported_numeric_claims'
    if key_cov < 0.45: return 'low_key_output_coverage'
    if not disclaimer: return 'missing_research_disclaimer'
    return None


def main() -> None:
    ap=argparse.ArgumentParser(description='Evaluate report factuality/grounding against BioSignalBench trace tool results.')
    ap.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_v1.jsonl')
    ap.add_argument('--out-json', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_report_factuality_eval.json')
    ap.add_argument('--out-jsonl', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_report_factuality_eval_cases.jsonl')
    ap.add_argument('--out-csv', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_report_factuality_eval_cases.csv')
    ap.add_argument('--out-md', default='/data1/jiahui/biosignal-agent/outputs/paper_tables/biosignalbench_report_factuality_eval.md')
    ap.add_argument('--report-source', choices=['trace','grounded_template'], default='trace')
    args=ap.parse_args()
    cases=[c for c in load_bench_cases(args.manifest) if c.get('benchmark_task')=='report_factuality']
    rows=[]; skipped=[]
    for c in cases:
        trace_path=(c.get('ground_truth_metric') or {}).get('trace_path') or c.get('source')
        if not trace_path or not Path(trace_path).exists():
            skipped.append(c.get('case_id'))
            continue
        rows.append(evaluate_case(c, report_source=args.report_source))
    n=len(rows)
    failures=Counter(r['failure_reason'] for r in rows if r['failure_reason'])
    by_mod={}
    for mod in sorted(set(str(r['modality']).lower() for r in rows)):
        sub=[r for r in rows if str(r['modality']).lower()==mod]
        by_mod[mod]={'num_cases':len(sub),'pass_rate':sum(r['pass'] for r in sub)/len(sub),'factuality_score':sum(r['factuality_score'] for r in sub)/len(sub)}
    summary={
        'artifact':'BioSignalBenchReportFactualityEvaluation',
        'report_source': args.report_source,
        'num_cases':n,
        'skipped_cases': skipped,
        'pass_rate':sum(r['pass'] for r in rows)/n if n else 0,
        'factuality_score':sum(r['factuality_score'] for r in rows)/n if n else 0,
        'tool_mention_recall':sum(r['tool_mention_recall'] for r in rows)/n if n else 0,
        'numeric_grounding':sum(r['numeric_grounding'] for r in rows)/n if n else 0,
        'key_coverage':sum(r['key_coverage'] for r in rows)/n if n else 0,
        'disclaimer_rate':sum(r['disclaimer_present'] for r in rows)/n if n else 0,
        'failure_reason_counts':dict(sorted(failures.items())),
        'by_modality':by_mod,
    }
    write_json(args.out_json, summary)
    write_jsonl(args.out_jsonl, rows)
    write_csv(args.out_csv, rows)
    table=markdown_table(['Metric','Value'], [[k,v] for k,v in summary.items() if k not in {'by_modality','failure_reason_counts','skipped_cases'}])
    fail=markdown_table(['Failure','Count'], [[k,v] for k,v in summary['failure_reason_counts'].items()])
    mod=markdown_table(['Modality','Cases','Pass','Score'], [[k,v['num_cases'],v['pass_rate'],v['factuality_score']] for k,v in by_mod.items()])
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text('# BioSignalBench Report Factuality\n\n'+table+'\n## Failures\n\n'+fail+'\n## By Modality\n\n'+mod)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
