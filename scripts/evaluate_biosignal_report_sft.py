#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.evaluation.biosignalbench import load_bench_cases, markdown_table, write_csv, write_json, write_jsonl
from scripts.evaluate_report_factuality import evaluate_case

SYSTEM='You are BioSignalAgent. Write a concise evidence-grounded biosignal report from tool outputs. Do not diagnose from proxy tools.'


def compact_result(value: Any, max_list: int = 8) -> Any:
    if isinstance(value, dict):
        return {k: compact_result(v, max_list=max_list) for k,v in value.items() if k not in {'source'}}
    if isinstance(value, list):
        if len(value) > max_list:
            return value[:max_list] + [f'... ({len(value)} total)']
        return value
    return value


def render_prompt(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return ''.join(f"<{m['role']}>\n{m['content']}\n" for m in messages)+'<assistant>\n'


def prompt_for_case(case: dict[str, Any]) -> tuple[list[dict[str,str]], dict[str, Any]]:
    trace_path=(case.get('ground_truth_metric') or {}).get('trace_path') or case.get('source')
    trace=json.loads(Path(trace_path).read_text())
    user={
        'question': trace.get('question'),
        'tool_results': compact_result(trace.get('tool_results') or []),
        'disclaimer_required': True,
        'report_requirements': [
            'Only state findings supported by tool_results.',
            'Include numeric values only when present in tool_results.',
            'Mention low confidence, proxy status, or limitations when present.',
            'End with research-use / not-a-clinical-diagnosis disclaimer.',
        ],
    }
    return [{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(user,sort_keys=True)}], trace


def main() -> None:
    ap=argparse.ArgumentParser(description='Generate reports with a report-grounding LoRA and evaluate factuality.')
    ap.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_v1.jsonl')
    ap.add_argument('--base-model', default='Qwen/Qwen2.5-0.5B-Instruct')
    ap.add_argument('--adapter', default='/data1/jiahui/biosignal-agent/outputs/biosignal_sft_report_lora_qwen25_05b_grounding_v3/best_adapter')
    ap.add_argument('--max-new-tokens', type=int, default=512)
    ap.add_argument('--max-input-tokens', type=int, default=2048)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--out-json', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_report_sft_eval.json')
    ap.add_argument('--out-jsonl', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_report_sft_eval_cases.jsonl')
    ap.add_argument('--out-csv', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_report_sft_eval_cases.csv')
    ap.add_argument('--out-md', default='/data1/jiahui/biosignal-agent/outputs/paper_tables/biosignalbench_report_sft_eval.md')
    args=ap.parse_args()
    cases=[c for c in load_bench_cases(args.manifest) if c.get('benchmark_task')=='report_factuality']
    cases=[c for c in cases if ((c.get('ground_truth_metric') or {}).get('trace_path') or c.get('source')) and Path(((c.get('ground_truth_metric') or {}).get('trace_path') or c.get('source'))).exists()]
    if args.limit:
        cases=cases[:args.limit]
    tokenizer=AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True, torch_dtype=torch.bfloat16 if device.type=='cuda' else torch.float32)
    model.to(device)
    model=PeftModel.from_pretrained(model,args.adapter)
    model.eval()
    rows=[]
    for idx,case in enumerate(cases,1):
        messages,_=prompt_for_case(case)
        prompt=render_prompt(tokenizer,messages)
        enc=tokenizer(prompt,return_tensors='pt',truncation=True,max_length=args.max_input_tokens).to(model.device)
        with torch.no_grad():
            out=model.generate(**enc,max_new_tokens=args.max_new_tokens,do_sample=False,num_beams=1,use_cache=True,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
        report=tokenizer.decode(out[0][enc['input_ids'].shape[1]:],skip_special_tokens=True).strip()
        row=evaluate_case(case, report_override=report)
        row['generated_report']=report
        rows.append(row)
        if idx % 10 == 0:
            print(f'evaluated {idx}/{len(cases)}', flush=True)
    n=len(rows)
    summary={
        'artifact':'BioSignalBenchReportSFTGenerationEvaluation',
        'adapter':args.adapter,
        'base_model':args.base_model,
        'num_cases':n,
        'pass_rate':sum(r['pass'] for r in rows)/n if n else 0,
        'factuality_score':sum(r['factuality_score'] for r in rows)/n if n else 0,
        'tool_mention_recall':sum(r['tool_mention_recall'] for r in rows)/n if n else 0,
        'numeric_grounding':sum(r['numeric_grounding'] for r in rows)/n if n else 0,
        'key_coverage':sum(r['key_coverage'] for r in rows)/n if n else 0,
        'disclaimer_rate':sum(r['disclaimer_present'] for r in rows)/n if n else 0,
        'failure_reason_counts':{},
    }
    from collections import Counter
    summary['failure_reason_counts']=dict(sorted(Counter(r['failure_reason'] for r in rows if r['failure_reason']).items()))
    write_json(args.out_json, summary); write_jsonl(args.out_jsonl, rows); write_csv(args.out_csv, rows)
    table=markdown_table(['Metric','Value'], [[k,v] for k,v in summary.items() if k!='failure_reason_counts'])
    fail=markdown_table(['Failure','Count'], [[k,v] for k,v in summary['failure_reason_counts'].items()])
    Path(args.out_md).parent.mkdir(parents=True,exist_ok=True); Path(args.out_md).write_text('# BioSignalBench Report SFT Evaluation\n\n'+table+'\n## Failures\n\n'+fail)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
