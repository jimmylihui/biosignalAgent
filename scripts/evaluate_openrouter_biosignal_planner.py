#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.agent.schema_loader import load_tool_schemas
from biosignal_agent.agent.tool_retriever import ToolRetriever
from biosignal_agent.evaluation.biosignalbench import load_bench_cases, markdown_table, retrieve_tools_for_case, tool_set_scores, write_json, write_jsonl

BASE_URL = 'https://openrouter.ai/api/v1/chat/completions'
DEFAULT_KEY_FILE = '/home/myid/jl57095/TwinMarket/openrouter_caption_with_P_wave.py'


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with p.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with p.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def load_openrouter_keys(path: str | Path) -> list[str]:
    mod = ast.parse(Path(path).read_text(errors='ignore'))
    keys: list[str] = []
    for node in mod.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == 'candidate_keys':
                keys = ast.literal_eval(node.value)
    deduped = []
    seen = set()
    for key in keys:
        if not isinstance(key, str):
            continue
        key = key.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find('{')
    end = text.rfind('}')
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


def planned_tools_from_payload(payload: dict[str, Any] | None) -> list[str]:
    if not isinstance(payload, dict):
        return []
    calls = payload.get('tool_calls') or payload.get('tools') or []
    tools = []
    if isinstance(calls, list):
        for call in calls:
            if isinstance(call, str):
                tools.append(call)
            elif isinstance(call, dict):
                name = call.get('name') or call.get('tool') or call.get('function')
                if isinstance(name, dict):
                    name = name.get('name')
                if name:
                    tools.append(str(name))
    return list(dict.fromkeys(tools))


def build_prompt(case: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, str]]:
    candidate_payload = [
        {
            'name': schema.get('name'),
            'modality': schema.get('modality'),
            'description': schema.get('description', '')[:260],
            'returns': schema.get('returns', [])[:8],
        }
        for schema in candidates
    ]
    case_payload = {
        'case_id': case.get('case_id'),
        'benchmark_task': case.get('benchmark_task'),
        'input_type': case.get('input_type'),
        'modality_hint': case.get('modality'),
        'question': case.get('question'),
        'signal_metadata_available': bool((case.get('signal') or {}).get('path')),
    }
    system = (
        'You are the BioSignalAgent tool planner. Choose only tools from the provided candidate list. '
        'Return strict JSON only with keys: modality, tool_calls, limitations. '
        'tool_calls must be an array of {"name": tool_name, "arguments": {}}. '
        'Do not invent tools. Select the minimal sufficient set of tools; do not include extra downstream tools unless the user question asks for them. Include quality and routing tools when needed. '
        'For proxy/screening tasks, add a limitation that the output is research-use only and not a clinical diagnosis.'
    )
    user = json.dumps({'case': case_payload, 'candidate_tools': candidate_payload}, ensure_ascii=False)
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


async def call_openrouter(client: httpx.AsyncClient, keys: list[str], model: str, messages: list[dict[str, str]], max_tokens: int, temperature: float, key_start: int, max_key_attempts: int) -> tuple[str, str | None, int | None]:
    payload = {'model': model, 'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens}
    last_error = None
    attempts = len(keys) if max_key_attempts <= 0 else min(len(keys), max_key_attempts)
    for offset in range(attempts):
        idx = (key_start + offset) % len(keys)
        key = keys[idx]
        try:
            resp = await client.post(
                BASE_URL,
                json=payload,
                headers={
                    'Authorization': f'Bearer {key}',
                    'HTTP-Referer': 'https://github.com/biosignal-agent',
                    'X-Title': 'BioSignalAgent Benchmark',
                    'Content-Type': 'application/json',
                },
            )
            data = resp.json() if resp.content else {}
            if resp.status_code == 200 and data.get('choices'):
                content = data['choices'][0].get('message', {}).get('content', '')
                return str(content), None, idx
            err = data.get('error') or data
            last_error = f'status_{resp.status_code}:{str(err)[:220]}'
        except Exception as exc:
            last_error = f'{type(exc).__name__}:{str(exc)[:220]}'
    return '', last_error or 'unknown_openrouter_error', None


async def evaluate_async(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    keys = load_openrouter_keys(args.key_file)
    if not keys:
        raise SystemExit('No OpenRouter keys found in key file.')
    cases = load_bench_cases(args.manifest)
    if args.task:
        wanted = set(args.task)
        cases = [case for case in cases if case.get('benchmark_task') in wanted]
    if args.limit is not None:
        cases = cases[:args.limit]

    cached = {row.get('case_id'): row for row in read_jsonl(args.cache_jsonl) if row.get('case_id')} if args.cache_jsonl else {}
    schemas_by_name = {schema['name']: schema for schema in load_tool_schemas()}
    retriever = ToolRetriever()
    sem = asyncio.Semaphore(args.concurrency)
    out_rows: list[dict[str, Any] | None] = [None] * len(cases)

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        async def one(i: int, case: dict[str, Any]) -> None:
            cid = case.get('case_id')
            if cid in cached and not args.refresh_cache:
                out_rows[i] = cached[cid]
                return
            async with sem:
                names = retrieve_tools_for_case(case, retriever, 'tfidf', args.candidate_k, {})
                candidates = [schemas_by_name[name] for name in names if name in schemas_by_name]
                raw, error, key_index = await call_openrouter(
                    client, keys, args.model, build_prompt(case, candidates), args.max_tokens, args.temperature, (i * 37) % len(keys), args.max_key_attempts
                )
                parsed = extract_json(raw)
                planned = planned_tools_from_payload(parsed)
                candidate_names = [schema['name'] for schema in candidates]
                planned = [tool for tool in planned if tool in schemas_by_name]
                expected = list(case.get('expected_tools') or [])
                precision, recall, f1 = tool_set_scores(expected, planned)
                missing = sorted(set(expected) - set(planned))
                unexpected = sorted(set(planned) - set(expected))
                out_rows[i] = {
                    'case_id': cid,
                    'benchmark_task': case.get('benchmark_task'),
                    'input_type': case.get('input_type'),
                    'modality': str(case.get('modality', '')).lower(),
                    'question': case.get('question'),
                    'model': args.model,
                    'candidate_tools': candidate_names,
                    'expected_tools': expected,
                    'planned_tools': planned,
                    'parse_ok': parsed is not None,
                    'planning_pass': not missing and not unexpected,
                    'tool_precision': precision,
                    'tool_recall': recall,
                    'tool_f1': f1,
                    'missing_from_plan': missing,
                    'unexpected_tools': unexpected,
                    'error': error,
                    'key_index': key_index,
                    'raw_generation': raw[:4000],
                }
        await asyncio.gather(*(one(i, case) for i, case in enumerate(cases)))
    rows = [row for row in out_rows if row is not None]
    return summarize(rows, args), rows


def mean(vals: list[float | bool]) -> float:
    values = [float(v) for v in vals]
    return sum(values) / len(values) if values else 0.0


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_task = {}
    for task in sorted({row['benchmark_task'] for row in rows}):
        sub = [row for row in rows if row['benchmark_task'] == task]
        by_task[task] = {
            'num_cases': len(sub),
            'parse_rate': mean([row['parse_ok'] for row in sub]),
            'planning_accuracy': mean([row['planning_pass'] for row in sub]),
            'tool_f1': mean([row['tool_f1'] for row in sub]),
        }
    failures = Counter()
    for row in rows:
        if row.get('error'):
            failures['api_error'] += 1
        elif not row.get('parse_ok'):
            failures['parse_failed'] += 1
        elif row.get('missing_from_plan'):
            failures['missing_expected_tools'] += 1
        elif row.get('unexpected_tools'):
            failures['unexpected_tools'] += 1
    return {
        'artifact': 'BioSignalBenchOpenRouterPlannerEvaluation',
        'model': args.model,
        'num_cases': len(rows),
        'candidate_k': args.candidate_k,
        'concurrency': args.concurrency,
        'max_key_attempts': args.max_key_attempts,
        'parse_rate': mean([row['parse_ok'] for row in rows]),
        'planning_accuracy': mean([row['planning_pass'] for row in rows]),
        'tool_precision': mean([row['tool_precision'] for row in rows]),
        'tool_recall': mean([row['tool_recall'] for row in rows]),
        'tool_f1': mean([row['tool_f1'] for row in rows]),
        'failure_reason_counts': dict(sorted(failures.items())),
        'by_task': by_task,
    }


def write_markdown(summary: dict[str, Any], path: str | Path) -> None:
    rows = [[summary['model'], summary['num_cases'], summary['parse_rate'], summary['planning_accuracy'], summary['tool_precision'], summary['tool_recall'], summary['tool_f1']]]
    task_rows = [[task, vals['num_cases'], vals['parse_rate'], vals['planning_accuracy'], vals['tool_f1']] for task, vals in summary.get('by_task', {}).items()]
    fail_rows = [[k, v] for k, v in summary.get('failure_reason_counts', {}).items()]
    text = '# OpenRouter Planner Baseline\n\n'
    text += markdown_table(['Model', 'Cases', 'Parse', 'Planning', 'Precision', 'Recall', 'Tool F1'], rows)
    text += '\n## By Task\n\n'
    text += markdown_table(['Task', 'Cases', 'Parse', 'Planning', 'Tool F1'], task_rows)
    text += '\n## Failures\n\n'
    text += markdown_table(['Failure', 'Count'], fail_rows)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser(description='Evaluate OpenRouter planner baseline on BioSignalBench.')
    ap.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_v1.jsonl')
    ap.add_argument('--key-file', default=DEFAULT_KEY_FILE)
    ap.add_argument('--model', default='openrouter/owl-alpha')
    ap.add_argument('--candidate-k', type=int, default=20)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--task', action='append', default=None)
    ap.add_argument('--concurrency', type=int, default=32)
    ap.add_argument('--timeout', type=float, default=90)
    ap.add_argument('--max-tokens', type=int, default=384)
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--max-key-attempts', type=int, default=0, help='Keys to try per request; 0 means all loaded keys.')
    ap.add_argument('--refresh-cache', action='store_true')
    ap.add_argument('--cache-jsonl', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_openrouter_owl_alpha_cases.jsonl')
    ap.add_argument('--out-json', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_openrouter_owl_alpha.json')
    ap.add_argument('--out-jsonl', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_openrouter_owl_alpha_cases.jsonl')
    ap.add_argument('--out-csv', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_openrouter_owl_alpha_cases.csv')
    ap.add_argument('--out-md', default='/data1/jiahui/biosignal-agent/outputs/paper_tables/table11_openrouter_owl_alpha_planner.md')
    args = ap.parse_args()
    summary, rows = asyncio.run(evaluate_async(args))
    write_json(args.out_json, summary)
    write_jsonl(args.out_jsonl, rows)
    write_csv(args.out_csv, rows)
    write_markdown(summary, args.out_md)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
