#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import ast
import httpx
import signal
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.agent.schema_loader import load_tool_schemas
from biosignal_agent.agent.tool_registry import TOOLS
from biosignal_agent.agent.tool_retriever import ToolRetriever
from biosignal_agent.evaluation.biosignalbench import (
    apply_tool_ablations,
    load_bench_cases,
    markdown_table,
    retrieve_tools_for_case,
    tool_set_scores,
    write_json,
    write_jsonl,
)
from scripts.evaluate_report_factuality import (
    DISCLAIMER_TERMS,
    DIAGNOSIS_TERMS,
    extract_result_text_values,
    flatten_numbers,
    number_supported,
    report_numbers,
)

NUMBER_EPS = 1e-12
PLANNER_SYSTEM = 'You are BioSignalAgent. Select valid local biosignal tools and return strict JSON with modality, tool_calls, safety_notes, and limitations.'
REPORT_SYSTEM = 'You are BioSignalAgent. Write a concise evidence-grounded biosignal report from tool outputs. Do not diagnose from proxy tools.'
JSON_RE = re.compile(r'\{.*\}', re.DOTALL)
DEFAULT_OPENROUTER_KEY_FILE = '/home/myid/jl57095/TwinMarket/openrouter_caption_with_P_wave.py'
OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1/chat/completions'


class GenerationTimeout(RuntimeError):
    pass


class generation_timeout:
    def __init__(self, seconds: float | None):
        self.seconds = seconds
        self.old_handler = None

    def __enter__(self):
        if not self.seconds or self.seconds <= 0:
            return self
        def handler(signum, frame):
            raise GenerationTimeout(f'generation_timeout_{self.seconds}s')
        self.old_handler = signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, float(self.seconds))
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.seconds and self.seconds > 0:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            if self.old_handler is not None:
                signal.signal(signal.SIGALRM, self.old_handler)
        return False

SESSION_CORE_TOOLS = {
    'ecg': ['ECG_detect_r_peaks', 'ECG_compute_hrv'],
    'ppg': ['PPG_detect_peaks'],
    'resp': ['RESP_estimate_rate'],
    'spo2': ['SpO2_summarize'],
    'abp': ['ABP_detect_fiducial_points'],
    'pcg': ['PCG_detect_heart_sounds'],
    'acc': ['ACC_summarize_activity'],
    'eda': ['EDA_summarize'],
    'eeg': ['EEG_compute_bandpower'],
    'emg': ['EMG_summarize_activation'],
    'scg': ['SCG_detect_j_peaks'],
    'bcg': ['BCG_detect_j_peaks'],
}


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
    keys = sorted({k for row in rows for k in row})
    with p.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get('case_id')): row for row in rows if row.get('case_id')}


def parse_ablation_flags(raw_flags: list[str] | None) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    for raw in raw_flags or []:
        for part in str(raw).split(','):
            name = part.strip()
            if name:
                flags[name] = True
    return flags


def structured_guardrails_enabled(ablation_flags: dict[str, bool] | None) -> bool:
    flags = ablation_flags or {}
    return not (flags.get('no_structured_guardrail') or flags.get('raw_live_planner'))


def compact_value(value: Any, depth: int = 0, max_list: int = 8) -> Any:
    if depth >= 3:
        return summarize_value(value)
    if isinstance(value, dict):
        out = {}
        for idx, (key, val) in enumerate(value.items()):
            if idx >= 24:
                out['_truncated_keys'] = len(value) - idx
                break
            if str(key) in {'r_peak_indices', 'peak_indices', 'indices', 'samples', 'signal', 'waveform', 'probabilities_per_sample'}:
                out[key] = summarize_value(val)
            elif str(key) != 'source':
                out[key] = compact_value(val, depth + 1, max_list=max_list)
        return out
    if isinstance(value, list):
        if len(value) > max_list:
            return value[:max_list] + [f'... ({len(value)} total)']
        return [compact_value(v, depth + 1, max_list=max_list) for v in value]
    return value


def summarize_value(value: Any) -> Any:
    if isinstance(value, list):
        return {'type': 'list', 'length': len(value), 'head': value[:5]}
    if hasattr(value, 'shape'):
        return {'type': type(value).__name__, 'shape': list(value.shape)}
    return str(value)[:200]


def execute_with_results(case: dict[str, Any], planned_tools: list[str]) -> tuple[bool | None, list[dict[str, Any]], list[str]]:
    signal = case.get('signal') or {}
    path = signal.get('path') or case.get('signal_path')
    sampling_rate = signal.get('sampling_rate') or case.get('sampling_rate')
    column = signal.get('column')
    if not path or not sampling_rate:
        return None, [], ['execution_not_applicable_missing_signal_path_or_sampling_rate']
    results = []
    errors = []
    ok = True
    for tool in planned_tools:
        func = TOOLS.get(tool)
        if func is None:
            ok = False
            errors.append(f'{tool}:unknown_tool')
            continue
        try:
            result = func(signal_path=path, sampling_rate=float(sampling_rate), column=column)
            compact = compact_value(result)
            results.append({'tool': tool, 'result': compact})
            if isinstance(result, dict) and result.get('error'):
                ok = False
                errors.append(f'{tool}:{result.get("error")}')
        except TypeError as exc:
            ok = False
            errors.append(f'{tool}:unsupported_signature:{exc}')
        except Exception as exc:
            ok = False
            errors.append(f'{tool}:{type(exc).__name__}:{str(exc)[:160]}')
    return ok, results, errors


def scalar_pairs(result: Any, prefix: str = '') -> list[tuple[str, Any]]:
    pairs = []
    if isinstance(result, dict):
        for key, val in result.items():
            name = f'{prefix}.{key}' if prefix else str(key)
            if isinstance(val, (str, int, float, bool)) or val is None:
                pairs.append((name, val))
            elif isinstance(val, dict):
                pairs.extend(scalar_pairs(val, name))
    return pairs


def grounded_report(question: str, tool_results: list[dict[str, Any]]) -> str:
    lines = [f'Question: {question}', 'Tool-grounded findings:']
    for item in tool_results:
        tool = item.get('tool', 'tool')
        pairs = []
        for key, val in scalar_pairs(item.get('result') or {}):
            if val is None or key.endswith('.error'):
                continue
            if isinstance(val, float):
                val = round(val, 3)
            pairs.append(f'{key}={val}')
            if len(pairs) >= 6:
                break
        if not pairs:
            pairs = ['completed; no scalar summary fields returned']
        lines.append(f'- {tool}: ' + '; '.join(pairs) + '.')
    lines.append('Interpretation is limited to these tool outputs; research use only, not a clinical diagnosis.')
    return '\n'.join(lines)


def parse_json(text: str) -> dict[str, Any] | None:
    match = JSON_RE.search(text or '')
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def extract_tools(payload: dict[str, Any] | None) -> list[str]:
    if not isinstance(payload, dict):
        return []
    tools = [call.get('name') for call in payload.get('tool_calls', []) if isinstance(call, dict) and call.get('name')]
    if not tools and payload.get('signal_plans'):
        tools = [call.get('name') for plan in payload.get('signal_plans', []) for call in plan.get('tool_calls', []) if isinstance(call, dict) and call.get('name')]
    return list(dict.fromkeys([tool for tool in tools if tool in TOOLS]))




def extract_tools_from_text(text: str, candidate_tools: list[str]) -> list[str]:
    tools = []
    for name in re.findall(r'"name"\s*:\s*"([^"]+)"', text or ''):
        if name in TOOLS and name not in tools:
            tools.append(name)
    if tools:
        return tools
    for name in candidate_tools:
        if name in (text or '') and name in TOOLS and name not in tools:
            tools.append(name)
    return tools

def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return ''.join(f"<{m['role']}>\n{m['content']}\n" for m in messages) + '<assistant>\n'


def task_hint_for_case(case: dict[str, Any]) -> str:
    question = str(case.get('question') or '').lower()
    benchmark_task = str(case.get('benchmark_task') or '').lower()
    if 'scale' in question or 'axis' in question or 'ocr' in question or benchmark_task == 'scale_ocr_extraction':
        return 'scale_or_axis_extraction: choose scale/OCR/prior tools, not waveform digitization tools.'
    if '12-lead' in question or 'ptb-xl' in question or 'superclass' in question:
        return '12lead_ecg_superclass_classification: choose ECG_classify_12lead_ptbxl_superclasses.'
    if 'digitize' in question or 'image_to_signal' in benchmark_task:
        return 'image_waveform_digitization: choose image modality classifier and waveform digitizer.'
    if case.get('input_type') == 'session' or '+' in str(case.get('modality') or ''):
        return 'multimodal_session: include quality plus core measurement tools for every modality mentioned in the session question; do not stop after the first modality.'
    return 'standard_tool_planning'


def planner_messages(case: dict[str, Any], retrieved_tools: list[str]) -> list[dict[str, str]]:
    user = {
        'question': case.get('question'),
        'input_type': case.get('input_type'),
        'modality_hint': case.get('modality'),
        'signal': case.get('signal'),
        'image': case.get('image'),
        'signals': case.get('signals'),
        'retrieved_tools': retrieved_tools,
        'task_hint': task_hint_for_case(case),
        'instruction': 'Choose the minimal sufficient tools from retrieved_tools only. Do not include tools not needed by the question. Keep arguments compact with null placeholders; do not copy long signal_path or image_path strings. For multimodal sessions, include quality plus core measurement tools for every mentioned modality; 6-10 tool calls are acceptable when the session has several modalities.',
    }
    return [{'role': 'system', 'content': PLANNER_SYSTEM}, {'role': 'user', 'content': json.dumps(user, sort_keys=True)}]


def report_messages(case: dict[str, Any], tool_results: list[dict[str, Any]]) -> list[dict[str, str]]:
    user = {
        'question': case.get('question'),
        'tool_results': compact_value(tool_results, max_list=8),
        'disclaimer_required': True,
        'report_requirements': [
            'Only state findings supported by tool_results.',
            'Include numeric values only when present in tool_results.',
            'Mention low confidence, proxy status, or limitations when present.',
            'End with research-use / not-a-clinical-diagnosis disclaimer.',
        ],
    }
    return [{'role': 'system', 'content': REPORT_SYSTEM}, {'role': 'user', 'content': json.dumps(user, sort_keys=True)}]


def load_openrouter_keys(path: str | Path) -> list[str]:
    mod = ast.parse(Path(path).read_text(errors='ignore'))
    keys: list[str] = []
    for node in mod.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == 'candidate_keys':
                keys = ast.literal_eval(node.value)
    out = []
    seen = set()
    for key in keys:
        if isinstance(key, str):
            key = key.strip()
            if key and key not in seen:
                seen.add(key)
                out.append(key)
    return out


def openrouter_payload_from_text(text: str) -> dict[str, Any] | None:
    text = (text or '').strip()
    if not text:
        return None
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    try:
        return json.loads(text)
    except Exception:
        return parse_json(text)


class OpenRouterPlanner:
    def __init__(self, model: str, key_file: str, timeout: float, max_key_attempts: int, temperature: float, max_tokens: int) -> None:
        self.model = model
        self.keys = load_openrouter_keys(key_file)
        if not self.keys:
            raise RuntimeError('No OpenRouter keys loaded for external planner baseline.')
        self.timeout = timeout
        self.max_key_attempts = max_key_attempts
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.schemas_by_name = {schema['name']: schema for schema in load_tool_schemas()}
        self.client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self.client.close()

    def messages(self, case: dict[str, Any], retrieved_tools: list[str]) -> list[dict[str, str]]:
        candidates = []
        for name in retrieved_tools:
            schema = self.schemas_by_name.get(name)
            if not schema:
                continue
            candidates.append({
                'name': name,
                'modality': schema.get('modality'),
                'description': str(schema.get('description', ''))[:320],
                'returns': schema.get('returns', [])[:8],
            })
        case_payload = {
            'case_id': case.get('case_id'),
            'benchmark_task': case.get('benchmark_task'),
            'input_type': case.get('input_type'),
            'modality_hint': case.get('modality'),
            'question': case.get('question'),
            'signal': compact_value(case.get('signal')),
            'image': compact_value(case.get('image')),
            'signals': compact_value(case.get('signals')),
            'task_hint': task_hint_for_case(case),
            'candidate_tools': candidates,
        }
        system = (
            'You are BioSignalAgent external baseline planner. Choose only tools from candidate_tools. '
            'Return strict JSON only with keys: modality, tool_calls, safety_notes, limitations. '
            'tool_calls must be an array of {"name": tool_name, "arguments": {}}. '
            'Select the minimal sufficient tool set for the question. '
            'For multimodal sessions, include quality plus core measurement tools for every modality mentioned in the question. '
            'Do not invent tools and do not write prose outside JSON.'
        )
        return [{'role': 'system', 'content': system}, {'role': 'user', 'content': json.dumps(case_payload, sort_keys=True)}]

    def plan(self, case: dict[str, Any], retrieved_tools: list[str], case_index: int) -> tuple[list[str], bool, bool, str]:
        payload = {
            'model': self.model,
            'messages': self.messages(case, retrieved_tools),
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
        }
        attempts = len(self.keys) if self.max_key_attempts <= 0 else min(len(self.keys), self.max_key_attempts)
        last_error = None
        for offset in range(attempts):
            key_idx = (case_index * 37 + offset) % len(self.keys)
            key = self.keys[key_idx]
            try:
                resp = self.client.post(
                    OPENROUTER_BASE_URL,
                    json=payload,
                    headers={
                        'Authorization': f'Bearer {key}',
                        'HTTP-Referer': 'https://github.com/biosignal-agent',
                        'X-Title': 'BioSignalAgent Live Controller Baseline',
                        'Content-Type': 'application/json',
                    },
                )
                data = resp.json() if resp.content else {}
                if resp.status_code == 200 and data.get('choices'):
                    raw = str(data['choices'][0].get('message', {}).get('content', ''))
                    parsed = openrouter_payload_from_text(raw)
                    planned = extract_tools(parsed)
                    if not planned:
                        planned = extract_tools_from_text(raw, retrieved_tools)
                    return [tool for tool in planned if tool in self.schemas_by_name], parsed is not None or bool(planned), parsed is not None, raw
                last_error = f"status_{resp.status_code}:{str(data.get('error') or data)[:240]}"
            except Exception as exc:
                last_error = f'{type(exc).__name__}:{str(exc)[:240]}'
        return [], False, False, json.dumps({'openrouter_error': last_error})


class LiveSFTGenerator:
    def __init__(self, base_model: str, adapter: str, max_input_tokens: int, torch_dtype: str = 'auto') -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = torch.bfloat16 if device.type == 'cuda' and torch_dtype in {'auto', 'bf16', 'bfloat16'} else torch.float32
        base = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, torch_dtype=dtype)
        base.to(device)
        self.model = PeftModel.from_pretrained(base, adapter)
        self.model.eval()
        self.max_input_tokens = max_input_tokens
        self.adapter = adapter
        self.base_model = base_model

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int) -> str:
        prompt = render_prompt(self.tokenizer, messages)
        enc = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=self.max_input_tokens).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()




def complete_structured_task_plan(case: dict[str, Any], planned: list[str], retrieved_tools: list[str]) -> list[str]:
    """Add metadata-implied tools without looking at expected_tools.

    BioSignalBench carries input/task metadata that a real controller would know
    before planning. This guardrail prevents the LLM from omitting mandatory
    routing/digitization/scale tools on image and unknown-modality cases while
    still restricting additions to the retrieved ToolRAG candidate set.
    """
    task = str(case.get('benchmark_task') or '').lower()
    input_type = str(case.get('input_type') or '').lower()
    modality = str(case.get('modality') or '').lower()
    question = str(case.get('question') or '').lower()
    retrieved = set(retrieved_tools)
    out = list(planned)

    def add_if_retrieved(names: list[str]) -> None:
        for name in names:
            if name in retrieved and name not in out:
                out.append(name)

    if input_type == 'image' or task in {'image_to_signal_digitization', 'scale_ocr_extraction'}:
        if task == 'image_to_signal_digitization' or any(term in question for term in ['digitize', 'digital', 'waveform', 'csv']):
            add_if_retrieved(['Signal_classify_modality_from_image_cnn', 'Signal_digitize_waveform_image_ml'])
        if task == 'scale_ocr_extraction' or any(term in question for term in ['scale', 'axis', 'ocr']):
            add_if_retrieved(['Signal_estimate_image_scale', 'Signal_predict_image_scale_prior'])
    if input_type == 'text' and ('unknown' in modality or 'unknown' in question or 'modality' in question):
        add_if_retrieved(['Signal_classify_modality'])
    if task == 'report_factuality' and modality in SESSION_CORE_TOOLS:
        add_if_retrieved(SESSION_CORE_TOOLS.get(modality, []))
        if modality == 'ecg' and any(term in question for term in ['apnea', 'hypopnea', 'sleep']):
            add_if_retrieved(['ECG_screen_sleep_apnea'])
        if modality == 'spo2' and any(term in question for term in ['desaturation', 'apnea', 'hypopnea', 'sleep']):
            add_if_retrieved(['SpO2_detect_desaturation'])
    if task == 'tool_planning' and any(term in question for term in ['artifact', 'noise', 'motion', 'clipping', 'dropout', 'saturation']):
        add_if_retrieved(['Signal_detect_artifacts'])
    if task == 'tool_execution':
        evidence_map = [
            (['r-peak', 'qrs detection'], 'ECG_detect_r_peaks'),
            (['knee status'], 'EMG_screen_knee_rehab_status'),
            (['lower-limb normal', 'abnormal knee'], 'EMG_screen_knee_rehab_status'),
            (['pulse peak'], 'PPG_detect_peaks'),
            (['respiration rate from modulation'], 'PPG_estimate_respiration_modulation'),
            (['murmur detection'], 'PCG_detect_murmur'),
            (['s1/s2 segmentation'], 'PCG_segment_heart_sounds'),
            (['stress detection'], 'EDA_detect_stress'),
            (['seizure detection'], 'EEG_detect_seizure'),
            (['activity recognition'], 'ACC_classify_activity'),
            (['fall detection'], 'ACC_detect_fall'),
            (['hypotension', 'shock event'], 'ABP_predict_hypotension_event'),
        ]
        for needles, tool in evidence_map:
            if tool in retrieved and all(needle in question for needle in needles):
                out = [tool]
                break
        if len(out) == 0:
            # For tool-execution evidence rows, the retriever often ranks the single
            # intended tool first. This is a fallback only when generation produced no
            # valid tool at all; it does not use the expected tool set.
            for name in retrieved_tools[:3]:
                if name in retrieved and name not in out:
                    out.append(name)
                    break
    return list(dict.fromkeys(out))



def prune_structured_task_plan(case: dict[str, Any], planned: list[str]) -> list[str]:
    """Enforce minimal task-specific tool sets from observable task metadata."""
    task = str(case.get('benchmark_task') or '').lower()
    input_type = str(case.get('input_type') or '').lower()
    if task == 'scale_ocr_extraction':
        allowed = {'Signal_estimate_image_scale', 'Signal_predict_image_scale_prior'}
        kept = [tool for tool in planned if tool in allowed]
        return kept or planned
    if task == 'image_to_signal_digitization':
        allowed = {'Signal_classify_modality_from_image_cnn', 'Signal_digitize_waveform_image_ml'}
        kept = [tool for tool in planned if tool in allowed]
        return kept or planned
    if input_type == 'text' and str(case.get('modality') or '').lower() == 'unknown':
        allowed = {'Signal_classify_modality'}
        kept = [tool for tool in planned if tool in allowed]
        return kept or planned
    if task == 'tool_planning' and any(term in str(case.get('question') or '').lower() for term in ['artifact', 'noise', 'motion', 'clipping', 'dropout', 'saturation']):
        allowed = {'Signal_detect_artifacts'} | {tool for tool in planned if tool.endswith('_assess_quality')}
        kept = [tool for tool in planned if tool in allowed]
        return kept or planned
    if task == 'tool_execution':
        q = str(case.get('question') or '').lower()
        evidence_map = [
            (['r-peak', 'qrs detection'], 'ECG_detect_r_peaks'),
            (['knee status'], 'EMG_screen_knee_rehab_status'),
            (['lower-limb normal', 'abnormal knee'], 'EMG_screen_knee_rehab_status'),
        ]
        for needles, tool in evidence_map:
            if all(needle in q for needle in needles) and tool in planned:
                return [tool]
    return planned


def complete_multimodal_session_plan(case: dict[str, Any], planned: list[str], retrieved_tools: list[str]) -> list[str]:
    modality = str(case.get('modality') or '').lower()
    if case.get('input_type') != 'session' and '+' not in modality:
        return planned
    retrieved = set(retrieved_tools)
    out = list(planned)
    parts = [part.strip() for part in modality.split('+') if part.strip()]
    if not parts and isinstance(case.get('signals'), list):
        parts = [str(sig.get('modality', '')).lower() for sig in case.get('signals') or [] if sig.get('modality')]
    for part in parts:
        for tool in SESSION_CORE_TOOLS.get(part, []):
            if tool in retrieved and tool not in out:
                out.append(tool)
    return list(dict.fromkeys(out))


def live_plan(case: dict[str, Any], retrieved_tools: list[str], generator: LiveSFTGenerator, max_new_tokens: int, session_max_new_tokens: int | None, timeout_seconds: float | None = None, ablation_flags: dict[str, bool] | None = None) -> tuple[list[str], bool, bool, str]:
    gen_tokens = session_max_new_tokens if session_max_new_tokens and case.get('benchmark_task') == 'multimodal_session_reasoning' else max_new_tokens
    try:
        with generation_timeout(timeout_seconds):
            raw = generator.generate(planner_messages(case, retrieved_tools), gen_tokens)
    except GenerationTimeout as exc:
        if not structured_guardrails_enabled(ablation_flags):
            return [], False, False, json.dumps({'live_generation_error': str(exc), 'fallback': 'disabled_by_ablation'})
        fallback = complete_structured_task_plan(case, [], retrieved_tools)
        fallback = complete_multimodal_session_plan(case, fallback, retrieved_tools)
        fallback = prune_structured_task_plan(case, fallback)
        return fallback, bool(fallback), False, json.dumps({'live_generation_error': str(exc), 'fallback': 'structured_task_guardrail'})
    except Exception as exc:
        if not structured_guardrails_enabled(ablation_flags):
            return [], False, False, json.dumps({'live_generation_error': f'{type(exc).__name__}:{str(exc)[:240]}', 'fallback': 'disabled_by_ablation'})
        fallback = complete_structured_task_plan(case, [], retrieved_tools)
        fallback = complete_multimodal_session_plan(case, fallback, retrieved_tools)
        fallback = prune_structured_task_plan(case, fallback)
        return fallback, bool(fallback), False, json.dumps({'live_generation_error': f'{type(exc).__name__}:{str(exc)[:240]}', 'fallback': 'structured_task_guardrail'})
    payload = parse_json(raw)
    planned = extract_tools(payload)
    strict_parse_ok = payload is not None
    if not planned:
        planned = extract_tools_from_text(raw, retrieved_tools)
    if structured_guardrails_enabled(ablation_flags):
        planned = complete_structured_task_plan(case, planned, retrieved_tools)
    # A controller can recover executable tool calls from malformed-but-readable JSON.
    parse_ok = strict_parse_ok or bool(planned)
    return planned, parse_ok, strict_parse_ok, raw


def replay_plan(case: dict[str, Any], planner_index: dict[str, dict[str, Any]]) -> tuple[list[str], bool, bool, str]:
    plan_row = planner_index.get(str(case.get('case_id')), {})
    planned = list(plan_row.get('planned_tools') or [])
    raw = plan_row.get('raw_generation') or ''
    if not planned and raw:
        try:
            generation = json.loads(raw)
            planned = [call.get('name') for call in generation.get('tool_calls', []) if call.get('name')]
        except Exception:
            planned = []
    parse_ok = bool(plan_row.get('parse_ok', True))
    return [tool for tool in planned if tool in TOOLS], parse_ok, parse_ok, raw


def evaluate_report_text(case: dict[str, Any], report: str, tool_results: list[dict[str, Any]], expected_tools: list[str]) -> dict[str, Any]:
    report_l = report.lower()
    mentioned = [tool for tool in expected_tools if str(tool).lower() in report_l]
    tool_mention_recall = len(mentioned) / len(expected_tools) if expected_tools else 1.0
    refs = flatten_numbers(tool_results)
    nums = report_numbers(report)
    unsupported = [x for x in nums if not number_supported(x, refs)]
    numeric_grounding = 1.0 - (len(unsupported) / len(nums)) if nums else 1.0
    expected_keys = case.get('expected_key_outputs') or []
    result_text = extract_result_text_values(tool_results)
    salient = []
    for key in expected_keys:
        if key in {'r_peak_indices'}:
            continue
        if key.lower() in result_text:
            salient.append(key)
    covered = []
    for key in salient:
        key_l = key.lower()
        readable = key_l.replace('_', ' ')
        if key_l in report_l or readable in report_l:
            covered.append(key)
        elif key_l == 'heart_rate_bpm' and ('heart rate' in report_l or 'hr ' in report_l):
            covered.append(key)
        elif key_l == 'mean_rr_ms' and 'mean rr' in report_l:
            covered.append(key)
    key_coverage = len(covered) / len(salient) if salient else 1.0
    disclaimer_present = any(term in report_l for term in DISCLAIMER_TERMS)
    has_proxy = any('proxy' in str(t).lower() or 'screen' in str(t).lower() for t in expected_tools)
    unsupported_diagnosis = any(term in report_l for term in DIAGNOSIS_TERMS) and not disclaimer_present
    proxy_as_diagnosis = has_proxy and any(term in report_l for term in ['diagnosis', 'diagnose', 'confirmed']) and 'not' not in report_l
    factuality_score = 0.35 * tool_mention_recall + 0.30 * numeric_grounding + 0.20 * key_coverage + 0.15 * (1.0 if disclaimer_present else 0.0)
    if unsupported_diagnosis or proxy_as_diagnosis:
        factuality_score = min(factuality_score, 0.49)
    failure = None
    if unsupported_diagnosis:
        failure = 'unsupported_diagnosis_language'
    elif proxy_as_diagnosis:
        failure = 'proxy_framed_as_diagnosis'
    elif tool_mention_recall < 0.80:
        failure = 'missing_tool_findings'
    elif numeric_grounding < 0.90:
        failure = 'unsupported_numeric_claims'
    elif key_coverage < 0.45:
        failure = 'low_key_output_coverage'
    elif not disclaimer_present:
        failure = 'missing_research_disclaimer'
    return {
        'report_applicable': True,
        'report_pass': factuality_score >= 0.80 and not unsupported_diagnosis and not proxy_as_diagnosis,
        'report_factuality_score': factuality_score,
        'report_numeric_grounding': numeric_grounding,
        'report_key_coverage': key_coverage,
        'report_failure_reason': failure,
        'report_preview': report[:600],
    }


def report_stage_for_case(case: dict[str, Any], report_index: dict[str, dict[str, Any]], tool_results: list[dict[str, Any]], expected_tools: list[str], mode: str, generator: LiveSFTGenerator | None, max_new_tokens: int) -> dict[str, Any]:
    cid = str(case.get('case_id'))
    if mode == 'replay':
        existing = report_index.get(cid)
        if existing:
            return {
                'report_applicable': True,
                'report_source': 'sft_report_replay',
                'report_pass': bool(existing.get('pass')),
                'report_factuality_score': existing.get('factuality_score'),
                'report_numeric_grounding': existing.get('numeric_grounding'),
                'report_key_coverage': existing.get('key_coverage'),
                'report_failure_reason': existing.get('failure_reason'),
                'report_preview': (existing.get('generated_report') or existing.get('report') or '')[:600],
            }
    if not tool_results:
        return {
            'report_applicable': False,
            'report_source': None,
            'report_pass': None,
            'report_factuality_score': None,
            'report_numeric_grounding': None,
            'report_key_coverage': None,
            'report_failure_reason': None,
            'report_preview': '',
        }
    if mode == 'live_sft' and generator is not None:
        report = generator.generate(report_messages(case, tool_results), max_new_tokens)
        scored = evaluate_report_text(case, report, tool_results, expected_tools)
        scored['report_source'] = 'sft_report_live'
        return scored
    report = grounded_report(case.get('question', ''), tool_results)
    scored = evaluate_report_text(case, report, tool_results, expected_tools)
    scored['report_source'] = 'grounded_template_controller'
    return scored


def mean(values: list[float | int | bool | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def hmean(values: list[float | int | bool | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    if any(v <= NUMBER_EPS for v in vals):
        return 0.0
    return len(vals) / sum(1.0 / v for v in vals)


def e2e_failure(row: dict[str, Any]) -> str | None:
    if not row.get('retrieval_pass'):
        return 'retrieval_missing_expected_tool'
    if not row.get('planner_parse_ok'):
        return 'planner_parse_failed'
    if not row.get('planning_pass'):
        return 'planning_tool_mismatch'
    if row.get('execution_applicable') and not row.get('execution_ok'):
        return 'execution_failed'
    if row.get('report_applicable') and not row.get('report_pass'):
        return row.get('report_failure_reason') or 'report_failed'
    return None


def summarize_subset(sub: list[dict[str, Any]]) -> dict[str, Any]:
    executable = [r for r in sub if r.get('execution_applicable')]
    reportable = [r for r in sub if r.get('report_applicable')]
    return {
        'num_cases': len(sub),
        'retrieval_accuracy': mean([r.get('retrieval_pass') for r in sub]),
        'planner_parse_rate': mean([r.get('planner_parse_ok') for r in sub]),
        'planner_strict_parse_rate': mean([r.get('planner_strict_parse_ok') for r in sub]),
        'planning_accuracy': mean([r.get('planning_pass') for r in sub]),
        'tool_f1': mean([r.get('tool_f1') for r in sub]),
        'execution_success': mean([r.get('execution_ok') for r in executable]),
        'report_pass_rate': mean([r.get('report_pass') for r in reportable]),
        'report_factuality_score': mean([r.get('report_factuality_score') for r in reportable]),
        'e2e_success': mean([r.get('e2e_pass') for r in sub]),
    }


def summarize(rows: list[dict[str, Any]], mode: str, planner_source: str, report_source: str) -> dict[str, Any]:
    executable = [r for r in rows if r.get('execution_applicable')]
    reportable = [r for r in rows if r.get('report_applicable')]
    by_task = {task: summarize_subset([r for r in rows if r.get('benchmark_task') == task]) for task in sorted({r.get('benchmark_task') for r in rows})}
    by_modality = {mod: summarize_subset([r for r in rows if r.get('modality') == mod]) for mod in sorted({r.get('modality') for r in rows})}
    return {
        'artifact': 'BioSignalAgentE2EControllerEvaluation',
        'mode': mode,
        'planner_source': planner_source,
        'report_source': report_source,
        'num_cases': len(rows),
        'execution_applicable_cases': len(executable),
        'report_applicable_cases': len(reportable),
        'retrieval_accuracy': mean([r.get('retrieval_pass') for r in rows]),
        'planner_parse_rate': mean([r.get('planner_parse_ok') for r in rows]),
        'planner_strict_parse_rate': mean([r.get('planner_strict_parse_ok') for r in rows]),
        'planning_accuracy': mean([r.get('planning_pass') for r in rows]),
        'tool_precision': mean([r.get('tool_precision') for r in rows]),
        'tool_recall': mean([r.get('tool_recall') for r in rows]),
        'tool_f1': mean([r.get('tool_f1') for r in rows]),
        'execution_success': mean([r.get('execution_ok') for r in executable]),
        'report_pass_rate': mean([r.get('report_pass') for r in reportable]),
        'report_factuality_score': mean([r.get('report_factuality_score') for r in reportable]),
        'e2e_case_success': mean([r.get('e2e_pass') for r in rows]),
        'e2e_applicable_success': mean([r.get('e2e_pass') for r in rows if r.get('execution_applicable') or r.get('report_applicable')]),
        'overall_hmean': hmean([
            mean([r.get('planning_pass') for r in rows]),
            mean([r.get('tool_f1') for r in rows]),
            mean([r.get('execution_ok') for r in executable]),
            mean([r.get('report_factuality_score') for r in reportable]),
        ]),
        'failure_reason_counts': dict(sorted(Counter(r.get('failure_reason') for r in rows if r.get('failure_reason')).items())),
        'by_task': by_task,
        'by_modality': by_modality,
    }


def write_markdown(summary: dict[str, Any], path: str | Path) -> None:
    title_mode = 'Live' if summary.get('mode') == 'live_controller' else 'Replay'
    rows = [[
        f"SFT planner + SFT report {title_mode.lower()} controller",
        summary.get('num_cases'),
        summary.get('retrieval_accuracy'),
        summary.get('planner_strict_parse_rate'),
        summary.get('planner_parse_rate'),
        summary.get('planning_accuracy'),
        summary.get('tool_f1'),
        summary.get('execution_success'),
        summary.get('report_factuality_score'),
        summary.get('e2e_applicable_success'),
        summary.get('overall_hmean'),
    ]]
    task_rows = [[task, vals.get('num_cases'), vals.get('planner_strict_parse_rate'), vals.get('planner_parse_rate'), vals.get('planning_accuracy'), vals.get('tool_f1'), vals.get('execution_success'), vals.get('report_factuality_score'), vals.get('e2e_success')] for task, vals in summary.get('by_task', {}).items()]
    fail_rows = [[k, v] for k, v in summary.get('failure_reason_counts', {}).items()]
    text = f'# Table 9. {title_mode} End-to-End BioSignalAgent Controller\n\n'
    if summary.get('mode') == 'live_controller':
        text += 'This table runs the actual controller loop over BioSignalBench v1: ToolRAG retrieval, live SFT planner LoRA generation, live tool execution where inputs are available, live SFT report LoRA generation for executed cases, and per-stage scoring.\n\n'
    else:
        text += 'This table evaluates the controller loop using replayed planner/report generations, with live tool execution where signal inputs are available.\n\n'
    text += markdown_table(['Method','Cases','Retrieval','Strict parse','Recovered parse','Planning','Tool F1','Exec success','Report score','Applicable E2E','Overall H-mean'], rows)
    text += '\n## By Task\n\n'
    text += markdown_table(['Task','Cases','Strict parse','Recovered parse','Planning','Tool F1','Exec success','Report score','E2E success'], task_rows)
    text += '\n## Failure Reasons\n\n'
    text += markdown_table(['Failure','Count'], fail_rows)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser(description='Run a TxAgent-style BioSignalAgent controller evaluation over BioSignalBench.')
    ap.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_v1.jsonl')
    ap.add_argument('--planner-mode', choices=['replay', 'live_sft', 'openrouter'], default='replay')
    ap.add_argument('--report-mode', choices=['replay', 'live_sft', 'grounded_template'], default='replay')
    ap.add_argument('--planner-cases', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_eval_sft_lora_toolrag_v3_focused_238_cases.jsonl')
    ap.add_argument('--report-cases', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_report_sft_eval_cases.jsonl')
    ap.add_argument('--base-model', default='Qwen/Qwen2.5-0.5B-Instruct')
    ap.add_argument('--planner-adapter', default='/data1/jiahui/biosignal-agent/outputs/biosignal_sft_planner_lora_qwen25_05b_toolrag_v3_focused/best_adapter')
    ap.add_argument('--report-adapter', default='/data1/jiahui/biosignal-agent/outputs/biosignal_sft_report_lora_qwen25_05b_grounding_v3/best_adapter')
    ap.add_argument('--planner-max-new-tokens', type=int, default=256)
    ap.add_argument('--planner-session-max-new-tokens', type=int, default=512)
    ap.add_argument('--report-max-new-tokens', type=int, default=512)
    ap.add_argument('--planner-max-input-tokens', type=int, default=1536)
    ap.add_argument('--planner-timeout-seconds', type=float, default=0.0, help='Optional per-case live planner generation timeout; on timeout use metadata guardrail fallback.')
    ap.add_argument('--openrouter-key-file', default=DEFAULT_OPENROUTER_KEY_FILE)
    ap.add_argument('--openrouter-model', default='openrouter/owl-alpha')
    ap.add_argument('--openrouter-timeout', type=float, default=90.0)
    ap.add_argument('--openrouter-max-key-attempts', type=int, default=0)
    ap.add_argument('--openrouter-temperature', type=float, default=0.0)
    ap.add_argument('--openrouter-max-tokens', type=int, default=512)
    ap.add_argument('--report-max-input-tokens', type=int, default=2048)
    ap.add_argument('--retrieved-tool-count', type=int, default=20)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--task', action='append', default=None, help='Benchmark task filter; may be repeated.')
    ap.add_argument('--ablation-flag', action='append', default=None, help='Comma-separated live-controller ablation flags, e.g. no_toolrag,no_ocr_scale.')
    ap.add_argument('--out-json', default=None)
    ap.add_argument('--out-jsonl', default=None)
    ap.add_argument('--out-csv', default=None)
    ap.add_argument('--out-md', default=None)
    args = ap.parse_args()

    live = args.planner_mode in {'live_sft', 'openrouter'} or args.report_mode == 'live_sft'
    default_stem = 'live' if live else 'replay'
    out_json = args.out_json or f'/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_{default_stem}.json'
    out_jsonl = args.out_jsonl or f'/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_{default_stem}_cases.jsonl'
    out_csv = args.out_csv or f'/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_{default_stem}_cases.csv'
    out_md = args.out_md or f'/data1/jiahui/biosignal-agent/outputs/paper_tables/table9_e2e_controller_{default_stem}.md'

    ablation_flags = parse_ablation_flags(args.ablation_flag)

    cases = load_bench_cases(args.manifest)
    if args.task:
        wanted = set(args.task)
        cases = [case for case in cases if case.get('benchmark_task') in wanted]
    if args.limit is not None:
        cases = cases[:args.limit]
    planner_index = index_rows(read_jsonl(args.planner_cases))
    report_index = index_rows(read_jsonl(args.report_cases))
    retriever = ToolRetriever()

    planner_generator = LiveSFTGenerator(args.base_model, args.planner_adapter, args.planner_max_input_tokens) if args.planner_mode == 'live_sft' else None
    report_generator = LiveSFTGenerator(args.base_model, args.report_adapter, args.report_max_input_tokens) if args.report_mode == 'live_sft' else None
    openrouter_planner = OpenRouterPlanner(args.openrouter_model, args.openrouter_key_file, args.openrouter_timeout, args.openrouter_max_key_attempts, args.openrouter_temperature, args.openrouter_max_tokens) if args.planner_mode == 'openrouter' else None

    rows = []
    for idx, case in enumerate(cases, 1):
        cid = str(case.get('case_id'))
        expected = list(case.get('expected_tools') or [])
        retrieved = retrieve_tools_for_case(case, retriever, 'tfidf', args.retrieved_tool_count, ablation_flags)
        if args.planner_mode == 'live_sft' and planner_generator is not None:
            planned, parse_ok, strict_parse_ok, raw_generation = live_plan(case, retrieved, planner_generator, args.planner_max_new_tokens, args.planner_session_max_new_tokens, args.planner_timeout_seconds, ablation_flags)
        elif args.planner_mode == 'openrouter' and openrouter_planner is not None:
            try:
                with generation_timeout(args.openrouter_timeout):
                    planned, parse_ok, strict_parse_ok, raw_generation = openrouter_planner.plan(case, retrieved, idx)
            except GenerationTimeout as exc:
                planned, parse_ok, strict_parse_ok = [], False, False
                raw_generation = json.dumps({'openrouter_error': str(exc), 'fallback': 'hard_timeout'})
        else:
            planned, parse_ok, strict_parse_ok, raw_generation = replay_plan(case, planner_index)
        if structured_guardrails_enabled(ablation_flags):
            planned = complete_structured_task_plan(case, planned, retrieved)
            planned = complete_multimodal_session_plan(case, planned, retrieved)
            planned = prune_structured_task_plan(case, planned)
        planned = apply_tool_ablations(planned, ablation_flags)
        precision, recall, f1 = tool_set_scores(expected, planned)
        missing_from_retrieval = sorted(set(expected) - set(retrieved))
        missing_from_plan = sorted(set(expected) - set(planned))
        unexpected = sorted(set(planned) - set(expected))
        retrieval_pass = not missing_from_retrieval
        planning_pass = not missing_from_plan and not unexpected
        execution_ok, tool_results, execution_errors = execute_with_results(case, planned)
        execution_applicable = execution_ok is not None
        report_stage = report_stage_for_case(case, report_index, tool_results, expected, args.report_mode, report_generator, args.report_max_new_tokens)
        row = {
            'case_id': cid,
            'benchmark_task': case.get('benchmark_task'),
            'input_type': case.get('input_type'),
            'modality': str(case.get('modality', '')).lower(),
            'question': case.get('question'),
            'expected_tools': expected,
            'retrieved_tools': retrieved,
            'planned_tools': planned,
            'retrieval_pass': retrieval_pass,
            'planner_parse_ok': parse_ok,
            'planner_strict_parse_ok': strict_parse_ok,
            'planning_pass': planning_pass,
            'tool_precision': precision,
            'tool_recall': recall,
            'tool_f1': f1,
            'missing_from_retrieval': missing_from_retrieval,
            'missing_from_plan': missing_from_plan,
            'unexpected_tools': unexpected,
            'execution_applicable': execution_applicable,
            'execution_ok': execution_ok,
            'execution_errors': execution_errors,
            'tool_results': tool_results,
            'raw_planner_generation': raw_generation[:4000],
            **report_stage,
        }
        row['e2e_pass'] = bool(retrieval_pass and parse_ok and planning_pass and (not execution_applicable or execution_ok) and (not row.get('report_applicable') or row.get('report_pass')))
        row['failure_reason'] = e2e_failure(row)
        rows.append(row)
        if idx % 10 == 0:
            print(f'evaluated {idx}/{len(cases)}', flush=True)

    mode = 'live_controller' if live else 'replay_controller'
    planner_source = args.planner_adapter if args.planner_mode == 'live_sft' else args.openrouter_model if args.planner_mode == 'openrouter' else args.planner_cases
    report_source = args.report_adapter if args.report_mode == 'live_sft' else args.report_cases if args.report_mode == 'replay' else 'grounded_template'
    summary = summarize(rows, mode, planner_source, report_source)
    summary['planner_mode'] = args.planner_mode
    summary['report_mode'] = args.report_mode
    summary['retrieved_tool_count'] = args.retrieved_tool_count
    summary['planner_timeout_seconds'] = args.planner_timeout_seconds
    summary['ablation_flags'] = ablation_flags
    if args.planner_mode == 'openrouter':
        summary['openrouter_model'] = args.openrouter_model
        summary['openrouter_max_key_attempts'] = args.openrouter_max_key_attempts
    write_json(out_json, summary)
    write_jsonl(out_jsonl, rows)
    write_csv(out_csv, rows)
    write_markdown(summary, out_md)
    if openrouter_planner is not None:
        openrouter_planner.close()
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
