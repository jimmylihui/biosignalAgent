from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .openrouter_client import DEFAULT_MODEL, chat_completion
from .planning_agent import PlanningBioSignalAgent
from .schema_loader import load_tool_schemas
from .tool_registry import TOOLS


@dataclass
class OpenRouterBioSignalAgent:
    model: str = DEFAULT_MODEL
    fallback_to_rules: bool = True

    def plan(self, question: str, signal_path: str, sampling_rate: float, column: str | None = None, fallback_modality: str | None = None) -> dict:
        schemas = load_tool_schemas()
        schema_brief = [
            {
                'name': schema['name'],
                'description': schema['description'],
                'modality': schema['modality'],
                'parameters': list(schema['parameters'].keys()),
            }
            for schema in schemas
        ]
        system = (
            'You are BioSignalAgent, a tool-planning assistant for ECG, PPG, and BCG waveforms. '
            'Choose only tools from the provided list. Return strict JSON only, with keys modality and tool_calls. '
            'Each tool call must have name and arguments. Include signal_path, sampling_rate, and column in arguments. '
            'Do not make clinical diagnosis.'
        )
        user = {
            'question': question,
            'signal_path': signal_path,
            'sampling_rate': sampling_rate,
            'column': column,
            'fallback_modality': fallback_modality,
            'available_tools': schema_brief,
        }
        try:
            text = chat_completion([
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': json.dumps(user, ensure_ascii=True)},
            ], model=self.model, temperature=0.0)
            plan = parse_json_object(text)
            return self._normalize_plan(plan, signal_path, sampling_rate, column)
        except Exception as exc:
            if not self.fallback_to_rules:
                raise
            rule_agent = PlanningBioSignalAgent()
            tool_names = rule_agent.plan(question, fallback_modality)
            modality = rule_agent.infer_modality(question, fallback_modality)
            return {
                'modality': modality,
                'tool_calls': [
                    {'name': name, 'arguments': {'signal_path': signal_path, 'sampling_rate': sampling_rate, 'column': column}}
                    for name in tool_names
                ],
                'planner': 'rule_fallback',
                'fallback_reason': str(exc),
            }

    def run(self, question: str, signal_path: str, sampling_rate: float, column: str | None = None, fallback_modality: str | None = None) -> dict:
        plan = self.plan(question, signal_path, sampling_rate, column, fallback_modality)
        tool_results = []
        for call in plan['tool_calls']:
            name = call['name']
            args = dict(call.get('arguments') or {})
            if name not in TOOLS:
                tool_results.append({'tool': name, 'error': 'unknown tool'})
                continue
            args.setdefault('signal_path', signal_path)
            args.setdefault('sampling_rate', sampling_rate)
            args.setdefault('column', column)
            result = TOOLS[name](**args)
            tool_results.append({'tool': name, 'arguments': args, 'result': result})
        final_report = self.generate_report(question, plan, tool_results)
        return {
            'question': question,
            'model': self.model,
            'planner': plan.get('planner', 'openrouter'),
            'modality': plan.get('modality'),
            'tool_plan': plan['tool_calls'],
            'tool_results': tool_results,
            'final_report': final_report,
            'disclaimer': 'Prototype output for research use only; not a clinical diagnosis.',
        }

    def generate_report(self, question: str, plan: dict, tool_results: list[dict]) -> str:
        system = (
            'You are BioSignalAgent. Write a concise research-use report from tool results. '
            'Mention signal quality, estimates, confidence, limitations, and that this is not a clinical diagnosis. '
            'Do not infer disease beyond the tool results.'
        )
        payload = {'question': question, 'plan': plan, 'tool_results': tool_results}
        try:
            return chat_completion([
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': json.dumps(payload, ensure_ascii=True)},
            ], model=self.model, temperature=0.2).strip()
        except Exception as exc:
            lines = [f'Question: {question}', 'Tool findings:']
            for item in tool_results:
                result = item.get('result', {})
                lines.append(f"- {item['tool']}: {json.dumps(result, ensure_ascii=True)}")
            lines.append(f'LLM report generation failed: {exc}')
            lines.append('Prototype output for research use only; not a clinical diagnosis.')
            return '\n'.join(lines)

    def _normalize_plan(self, plan: dict, signal_path: str, sampling_rate: float, column: str | None) -> dict:
        calls = plan.get('tool_calls') or []
        normalized = []
        for call in calls:
            name = call.get('name')
            if name not in TOOLS:
                continue
            args = dict(call.get('arguments') or {})
            args['signal_path'] = signal_path
            args['sampling_rate'] = sampling_rate
            args['column'] = column
            normalized.append({'name': name, 'arguments': args})
        if not normalized:
            raise ValueError('LLM returned no valid tool calls.')
        modality = str(plan.get('modality') or '').lower()
        quality_tool = f'{modality.upper()}_assess_quality' if modality in {'ecg', 'ppg', 'bcg'} else None
        if quality_tool in TOOLS and all(call['name'] != quality_tool for call in normalized):
            normalized.insert(0, {'name': quality_tool, 'arguments': {'signal_path': signal_path, 'sampling_rate': sampling_rate, 'column': column}})
        return {'modality': plan.get('modality'), 'tool_calls': normalized, 'planner': 'openrouter'}


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith('```'):
        stripped = re.sub(r'^```(?:json)?\s*', '', stripped)
        stripped = re.sub(r'\s*```$', '', stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', stripped, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))
