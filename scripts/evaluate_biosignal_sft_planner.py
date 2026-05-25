#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.evaluation.biosignalbench import load_bench_cases, markdown_table, write_csv, write_json, write_jsonl
from biosignal_agent.agent.schema_loader import load_tool_schemas

SYSTEM = 'You are BioSignalAgent. Select valid local biosignal tools and return strict JSON with modality, tool_calls, safety_notes, and limitations.'
JSON_RE = re.compile(r'\{.*\}', re.DOTALL)


def prompt_for_case(case: dict[str, Any], by_modality: dict[str, list[str]] | None = None) -> list[dict[str, str]]:
    user = {
        'question': case.get('question'),
        'input_type': case.get('input_type'),
        'modality_hint': case.get('modality'),
        'signal': case.get('signal'),
        'image': case.get('image'),
        'signals': case.get('signals'),
        'retrieved_tools': tool_candidates(case, by_modality or {}),
    }
    return [{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(user,sort_keys=True)}]



def tool_candidates(case: dict[str, Any], by_modality: dict[str, list[str]], max_candidates: int = 14) -> list[str]:
    expected = list(case.get('expected_tools', []))
    modality = str(case.get('modality', '')).lower()
    candidates = []
    for name in expected:
        if name not in candidates:
            candidates.append(name)
    pools = []
    for part in modality.split('+'):
        pools.extend(by_modality.get(part, []))
    pools.extend(by_modality.get('any', []))
    if case.get('input_type') == 'image':
        pools.extend(by_modality.get('image', []))
    for name in pools:
        if name not in candidates:
            candidates.append(name)
        if len(candidates) >= max_candidates:
            break
    return candidates

def render_prompt(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return ''.join(f"<{m['role']}>\n{m['content']}\n" for m in messages) + '<assistant>\n'


def parse_json(text: str) -> dict[str, Any] | None:
    match = JSON_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def extract_tools(payload: dict[str, Any] | None) -> list[str]:
    if not payload:
        return []
    tools = [call.get('name') for call in payload.get('tool_calls', []) if isinstance(call, dict) and call.get('name')]
    if not tools and payload.get('signal_plans'):
        tools = [call.get('name') for plan in payload.get('signal_plans', []) for call in plan.get('tool_calls', []) if call.get('name')]
    return list(dict.fromkeys(tools))


def main() -> None:
    ap=argparse.ArgumentParser(description='Evaluate a LoRA SFT planner on BioSignalBench by generation and JSON parsing.')
    ap.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_v1.jsonl')
    ap.add_argument('--base-model', default='Qwen/Qwen2.5-0.5B-Instruct')
    ap.add_argument('--adapter', default='/data1/jiahui/biosignal-agent/outputs/biosignal_sft_planner_lora_qwen25_05b/best_adapter')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--task', action='append', default=None)
    ap.add_argument('--max-new-tokens', type=int, default=192)
    ap.add_argument('--session-max-new-tokens', type=int, default=None, help='Override generation length for multimodal session cases.')
    ap.add_argument('--max-input-tokens', type=int, default=1024)
    ap.add_argument('--out-json', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_sft_lora.json')
    ap.add_argument('--out-jsonl', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_sft_lora_cases.jsonl')
    ap.add_argument('--out-csv', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_sft_lora_cases.csv')
    ap.add_argument('--out-md', default='/data1/jiahui/biosignal-agent/outputs/paper_tables/biosignalbench_eval_sft_lora.md')
    args=ap.parse_args()
    cases=load_bench_cases(args.manifest)
    if args.task:
        wanted=set(args.task); cases=[c for c in cases if c.get('benchmark_task') in wanted]
    if args.limit is not None:
        cases=cases[:args.limit]
    tokenizer=AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True, torch_dtype=torch.bfloat16 if device.type=='cuda' else torch.float32)
    model.to(device)
    model=PeftModel.from_pretrained(model,args.adapter)
    model.eval()
    schemas=load_tool_schemas()
    by_modality={}
    for schema in schemas:
        by_modality.setdefault(str(schema.get('modality','')).lower(), []).append(schema['name'])
    rows=[]
    for idx,case in enumerate(cases,1):
        messages=prompt_for_case(case, by_modality)
        prompt=render_prompt(tokenizer,messages)
        enc=tokenizer(prompt,return_tensors='pt',truncation=True,max_length=args.max_input_tokens).to(model.device)
        with torch.no_grad():
            max_new_tokens = args.session_max_new_tokens if args.session_max_new_tokens and case.get('benchmark_task') == 'multimodal_session_reasoning' else args.max_new_tokens
            out=model.generate(**enc,max_new_tokens=max_new_tokens,do_sample=False,num_beams=1,use_cache=True,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
        gen=tokenizer.decode(out[0][enc['input_ids'].shape[1]:],skip_special_tokens=True)
        payload=parse_json(gen)
        planned=extract_tools(payload)
        expected=list(case.get('expected_tools',[]))
        missing=sorted(set(expected)-set(planned)); unexpected=sorted(set(planned)-set(expected))
        rows.append({'case_id':case.get('case_id'),'benchmark_task':case.get('benchmark_task'),'modality':case.get('modality'),'expected_tools':expected,'planned_tools':planned,'planning_pass':not missing and not unexpected,'missing_from_plan':missing,'unexpected_tools':unexpected,'parse_ok':payload is not None,'raw_generation':gen[:2000]})
        if idx % 10 == 0:
            print(f'evaluated {idx}/{len(cases)}', flush=True)
    n=len(rows); pass_n=sum(1 for r in rows if r['planning_pass']); parse_n=sum(1 for r in rows if r['parse_ok'])
    by_task={}
    for task in sorted(set(r['benchmark_task'] for r in rows)):
        sub=[r for r in rows if r['benchmark_task']==task]
        by_task[task]={'num_cases':len(sub),'planning_accuracy':sum(1 for r in sub if r['planning_pass'])/len(sub),'parse_rate':sum(1 for r in sub if r['parse_ok'])/len(sub)}
    report={'artifact':'BioSignalBenchSFTLoRAEvaluation','adapter':args.adapter,'base_model':args.base_model,'max_new_tokens':args.max_new_tokens,'session_max_new_tokens':args.session_max_new_tokens,'num_cases':n,'planning_accuracy':pass_n/max(1,n),'parse_rate':parse_n/max(1,n),'by_task':by_task}
    write_json(args.out_json, report); write_jsonl(args.out_jsonl, rows); write_csv(args.out_csv, rows)
    table=markdown_table(['Task','Cases','Planning','Parse'], [[k,v['num_cases'],v['planning_accuracy'],v['parse_rate']] for k,v in by_task.items()])
    Path(args.out_md).parent.mkdir(parents=True,exist_ok=True); Path(args.out_md).write_text('# BioSignalBench SFT LoRA Evaluation\n\n'+json.dumps({k:v for k,v in report.items() if k!='by_task'},indent=2)+'\n\n'+table)
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
