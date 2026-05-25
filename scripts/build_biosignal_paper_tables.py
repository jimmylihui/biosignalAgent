#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.evaluation.biosignalbench import markdown_table, read_json


def main() -> None:
    ap = argparse.ArgumentParser(description='Build paper-ready BioSignalAgent tables from ToolUniverse, BioSignalBench, eval, and ablation artifacts.')
    ap.add_argument('--tool-universe', default='/data1/jiahui/biosignal-agent/outputs/biosignal_tool_universe_v1.json')
    ap.add_argument('--bench-summary', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_v1_summary.json')
    ap.add_argument('--eval-json', action='append', default=[
        '/data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_no_tool.json',
        '/data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_rule.json',
        '/data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_tfidf_toolrag.json',
        '/data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_sft_replay.json',
        '/data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_openrouter_owl_alpha.json',
        '/data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_oracle.json',
        '/data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_sft_lora_toolrag_v2_e6.json',
        '/data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_sft_lora_toolrag_v2_e6_longgen.json',
        '/data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_sft_lora_toolrag_v3_focused_238.json',
    ])
    ap.add_argument('--ablation-summary', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_ablation_summary.json')
    ap.add_argument('--ptbxl-12lead-eval', default='/data1/jiahui/biosignal-agent/outputs/ptbxl_12lead_tool_eval.json')
    ap.add_argument('--ptbxl-12lead-report', default='/data1/jiahui/biosignal-agent/outputs/ptbxl_full_12lead_resnet/ecg_ptbxl_full_12lead_resnet_train_report.json')
    ap.add_argument('--execution-eval-json', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_oracle_execute_all.json')
    ap.add_argument('--e2e-summary-json', default='/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_summary.json')
    ap.add_argument('--report-eval-json', action='append', default=[
        '/data1/jiahui/biosignal-agent/outputs/biosignalbench_report_factuality_eval.json',
        '/data1/jiahui/biosignal-agent/outputs/biosignalbench_report_factuality_eval_grounded_template.json',
        '/data1/jiahui/biosignal-agent/outputs/biosignalbench_report_sft_eval.json',
    ])
    ap.add_argument('--out-dir', default='/data1/jiahui/biosignal-agent/outputs/paper_tables')
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    universe = read_json(args.tool_universe)
    bench = read_json(args.bench_summary)
    ablations = read_json(args.ablation_summary)
    write(out/'table1_tool_universe.md', table_tool_universe(universe))
    write(out/'table2_biosignalbench_composition.md', table_bench_composition(bench))
    write(out/'table3_main_benchmark_comparison.md', table_main_eval(args.eval_json))
    write(out/'table4_ablation.md', table_ablation(ablations))
    write(out/'table5_report_factuality.md', table_report_factuality(args.report_eval_json))
    write(out/'table6_ptbxl_12lead_ecg.md', table_ptbxl_12lead(args.ptbxl_12lead_report, args.ptbxl_12lead_eval))
    write(out/'table7_execution_readiness.md', table_execution_readiness(args.execution_eval_json))
    write(out/'table8_e2e_agent_summary.md', table_e2e_summary(args.e2e_summary_json))
    write(out/'failure_analysis.md', failure_analysis(args.eval_json, ablations))
    print(json.dumps({'out_dir': str(out), 'tables': sorted(p.name for p in out.glob('table*.md')), 'failure_analysis': str(out/'failure_analysis.md')}, indent=2))


def write(path: Path, text: str) -> None:
    path.write_text(text)


def table_tool_universe(universe: dict) -> str:
    s = universe['summary']
    rows = []
    modalities = sorted(s['tool_count_by_modality'])
    for modality in modalities:
        rows.append([modality, s['tool_count_by_modality'][modality]])
    ev = markdown_table(['Evidence Level', 'Tools'], [[k, v] for k, v in s['tool_count_by_evidence_level'].items()])
    kind = markdown_table(['Tool Kind', 'Tools'], [[k, v] for k, v in s['tool_count_by_kind'].items()])
    return '# Table 1. BioSignalToolUniverse v1\n\n' + f"Total frozen tools: {universe['num_tools']}\n\n" + markdown_table(['Modality','Tools'], rows) + '\n## Evidence Breakdown\n\n' + ev + '\n## Implementation Kind Breakdown\n\n' + kind


def table_bench_composition(bench: dict) -> str:
    return '# Table 2. BioSignalBench v1 Composition\n\n' + f"Total cases: {bench['num_cases']}\n\n" + '## By Task\n\n' + markdown_table(['Task','Cases'], [[k,v] for k,v in bench['task_counts'].items()]) + '\n## By Input Type\n\n' + markdown_table(['Input Type','Cases'], [[k,v] for k,v in bench['input_type_counts'].items()]) + '\n## By Modality\n\n' + markdown_table(['Modality','Cases'], [[k,v] for k,v in bench['modality_counts'].items()])


def table_main_eval(paths: list[str]) -> str:
    rows = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        r = read_json(p)
        rows.append([p.stem.replace('biosignalbench_eval_',''), r.get('planner_backend') or r.get('model') or 'sft_lora', r.get('retriever_backend') or ('metadata_aware_toolrag' if 'openrouter' in p.stem else 'tool_candidates_in_prompt'), r.get('num_cases'), r.get('retrieval_accuracy'), r.get('planning_accuracy'), r.get('execution_accuracy'), r.get('tool_f1'), r.get('parse_rate')])
    return '# Table 3. Main BioSignalBench Comparison\n\n' + markdown_table(['Run','Planner','Retriever','Cases','Retrieval Acc','Planning Acc','Execution Acc','Tool F1','Parse Rate'], rows)


def table_ablation(summary: dict) -> str:
    rows = [[r['ablation'], r['planner_backend'], r['retriever_backend'], ','.join(r['flags']), r['num_cases'], r['retrieval_accuracy'], r['planning_accuracy'], r.get('tool_f1'), r['execution_accuracy']] for r in summary.get('runs', [])]
    return '# Table 4. Systematic Ablation\n\n' + markdown_table(['Ablation','Planner','Retriever','Flags','Cases','Retrieval','Planning','Tool F1','Execution'], rows)






def table_e2e_summary(summary_path: str) -> str:
    p=Path(summary_path)
    if not p.exists():
        return '# Table 8. End-to-End BioSignalAgent Summary\n\nNo E2E summary artifact found.\n'
    s=read_json(p)
    rows=[]
    for r in s.get('methods', []):
        rows.append([r.get('method'), r.get('cases'), r.get('planning_accuracy'), r.get('tool_f1'), r.get('parse_rate'), r.get('execution_ready_success'), r.get('report_factuality'), r.get('numeric_task_score'), r.get('overall_hmean')])
    text='# Table 8. End-to-End BioSignalAgent Summary\n\n'
    text+='TxAgent-style rollup over BioSignalBench v1. This is a composite artifact-level summary, not a claim that every case currently performs full automatic execution.\n\n'
    text+=markdown_table(['Method','Cases','Planning','Tool F1','Parse','Exec-ready','Report factuality','Numeric smoke','Overall H-mean'], rows)
    notes=s.get('notes') or []
    if notes:
        text+='\nNotes:\n'
        for note in notes:
            text+=f'- {note}\n'
    return text

def table_execution_readiness(eval_path: str) -> str:
    p=Path(eval_path)
    if not p.exists():
        return '# Table 7. Execution Readiness\n\nNo execution evaluation artifact found.\n'
    r=read_json(p)
    rows=[]
    for task, metrics in sorted((r.get('by_task') or {}).items()):
        rows.append([task, metrics.get('num_cases'), metrics.get('execution_accuracy')])
    focused=[]
    cases_path = Path(str(eval_path).replace('.json', '_cases.jsonl'))
    case_rows=[]
    if cases_path.exists():
        case_rows=[json.loads(line) for line in cases_path.read_text().splitlines() if line.strip()]
    trace_rows=[row for row in case_rows if row.get('benchmark_task') == 'report_factuality' and row.get('case_id') != 'negative_proxy_not_diagnosis']
    ptb_rows=[row for row in case_rows if row.get('benchmark_task') == 'tool_execution']
    if trace_rows:
        focused.append(['trace-backed report tool execution', len(trace_rows), sum(row.get('execution_ok') is True for row in trace_rows) / len(trace_rows)])
    if ptb_rows:
        focused.append(['PTB-XL 12-lead tool execution smoke', len(ptb_rows), sum(row.get('execution_ok') is True for row in ptb_rows) / len(ptb_rows)])
    text='# Table 7. Execution Readiness\n\n'
    text+='Oracle tool-selection execution over BioSignalBench separates executable CSV/tool cases from planning-only, image, and session cases.\n\n'
    text+=markdown_table(['Task','Cases','Execution Accuracy'], rows)
    if focused:
        text+='\n## Execution-Ready Subset\n\n'+markdown_table(['Subset','Cases','Execution Accuracy'], focused)
    text+='\nFailure reason counts:\n\n'+markdown_table(['Reason','Count'], [[k,v] for k,v in (r.get('failure_reason_counts') or {}).items()])
    return text

def table_ptbxl_12lead(report_path: str, eval_path: str) -> str:
    rows=[]
    p=Path(report_path)
    if p.exists():
        report=read_json(p)
        for target, payload in sorted((report.get('targets') or {}).items()):
            cv=payload.get('cv_metrics') or {}
            rows.append([target.upper(), cv.get('eval_records'), cv.get('average_precision'), cv.get('roc_auc'), cv.get('f1'), cv.get('precision'), cv.get('recall'), payload.get('threshold')])
    smoke=[]
    ep=Path(eval_path)
    if ep.exists():
        e=read_json(ep)
        smoke=[['BioSignalBench smoke cases', e.get('num_cases'), e.get('target_recall'), len(e.get('errors') or [])]]
    text='# Table 6. PTB-XL Full 12-Lead ECG Superclass Tool\n\n'
    text+='Validated backend: `ECG_classify_12lead_ptbxl_superclasses`. Metrics are fold-10 evaluation from the full PTB-XL 12-lead training report.\n\n'
    text+=markdown_table(['Target','Eval Records','AP','AUROC','F1','Precision','Recall','Threshold'], rows)
    if smoke:
        text+='\n## BioSignalBench Execution Smoke\n\n'+markdown_table(['Item','Cases','Target Recall','Errors'], smoke)
    return text

def table_report_factuality(paths: list[str]) -> str:
    rows=[]
    for path in paths:
        p=Path(path)
        if not p.exists():
            continue
        r=read_json(p)
        run=p.stem.replace('biosignalbench_report_','').replace('_eval','')
        rows.append([
            run,
            r.get('report_source') or ('sft_lora' if 'sft' in p.stem else 'trace'),
            r.get('num_cases'),
            r.get('pass_rate'),
            r.get('factuality_score'),
            r.get('tool_mention_recall'),
            r.get('numeric_grounding'),
            r.get('key_coverage'),
            r.get('disclaimer_rate'),
        ])
    return '# Table 5. Report Factuality and Grounding\n\n' + markdown_table(['Run','Report Source','Cases','Pass','Score','Tool Recall','Numeric Grounding','Key Coverage','Disclaimer'], rows)

def failure_analysis(eval_paths: list[str], ablations: dict) -> str:
    sections = ['# Failure Analysis Summary', '']
    for path in eval_paths:
        p = Path(path)
        if p.exists():
            r = read_json(p)
            sections.append(f"## {p.stem}")
            sections.append(markdown_table(['Failure Reason','Count'], [[k,v] for k,v in r.get('failure_reason_counts', {}).items()]))
    sections.append('## Ablation Failure Reasons')
    for r in ablations.get('runs', []):
        sections.append(f"### {r['ablation']}")
        sections.append(markdown_table(['Failure Reason','Count'], [[k,v] for k,v in (r.get('failure_reason_counts') or {}).items()]))
    return '\n'.join(sections)

if __name__ == '__main__':
    main()
