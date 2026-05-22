from __future__ import annotations

import ast
import json
import os
import time
from collections.abc import Iterable
from pathlib import Path

import httpx

SOURCE_CONFIG = Path('/home/myid/jl57095/TwinMarket/openrouter_caption_with_P_wave.py')
DEFAULT_MODEL = 'openrouter/owl-alpha'
DEFAULT_TIMEOUT = 120
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / '.env'


def _load_dotenv() -> None:
    if not ENV_FILE.exists():
        return
    for raw_line in ENV_FILE.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class OpenRouterConfigError(RuntimeError):
    pass


OPENROUTER_SETTING_NAMES = {'API_KEY', 'API_KEYS', 'candidate_keys', 'BASE_URL', 'MODEL', 'HTTP_REFERER', 'APP_TITLE'}


def _literal_assignments(path: Path) -> dict:
    if not path.exists():
        raise OpenRouterConfigError(f'OpenRouter source config not found: {path}')
    tree = ast.parse(path.read_text(errors='replace'), filename=str(path))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in OPENROUTER_SETTING_NAMES:
                    try:
                        values[target.id] = ast.literal_eval(node.value)
                    except Exception:
                        if (
                            isinstance(node.value, ast.Call)
                            and isinstance(node.value.func, ast.Attribute)
                            and node.value.func.attr == 'getenv'
                            and len(node.value.args) >= 1
                        ):
                            env_name = ast.literal_eval(node.value.args[0])
                            default = ast.literal_eval(node.value.args[1]) if len(node.value.args) >= 2 else None
                            values[target.id] = os.getenv(env_name, default)
    return values


def _extend_keys(keys: list[str], value) -> None:
    if value is None:
        return
    if isinstance(value, str):
        keys.append(value)
        return
    if isinstance(value, dict):
        value = value.values()
    if isinstance(value, Iterable):
        for item in value:
            if isinstance(item, str):
                keys.append(item)


def _candidate_keys(values: dict) -> list[str]:
    keys = []
    _extend_keys(keys, values.get('API_KEY'))
    _extend_keys(keys, values.get('API_KEYS'))
    _extend_keys(keys, values.get('candidate_keys'))
    env_candidates = os.getenv('OPENROUTER_CANDIDATE_KEYS', '')
    if env_candidates:
        keys.extend(key.strip() for key in env_candidates.split(','))
    env_key = os.getenv('OPENROUTER_API_KEY')
    if env_key:
        keys.append(env_key)
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


def load_openrouter_settings(model: str | None = None) -> dict:
    _load_dotenv()
    values = _literal_assignments(SOURCE_CONFIG)
    api_keys = _candidate_keys(values)
    if not api_keys:
        raise OpenRouterConfigError('No OpenRouter API key found in source config or environment.')
    return {
        'api_keys': api_keys,
        'base_url': values.get('BASE_URL', 'https://openrouter.ai/api/v1/chat/completions'),
        'model': model or DEFAULT_MODEL,
        'http_referer': values.get('HTTP_REFERER', 'https://github.com/biosignal-agent'),
        'app_title': values.get('APP_TITLE', 'BioSignalAgent'),
    }


def chat_completion(messages: list[dict], model: str | None = None, temperature: float = 0.0, timeout: int = DEFAULT_TIMEOUT, retry_max: int = 3, retry_delay: float = 8.0) -> str:
    settings = load_openrouter_settings(model)
    payload = {'model': settings['model'], 'messages': messages, 'temperature': temperature}
    last_error = None
    with httpx.Client(timeout=timeout) as client:
        for attempt in range(retry_max):
            for key in settings['api_keys']:
                headers = {
                    'Authorization': f'Bearer {key}',
                    'HTTP-Referer': settings['http_referer'],
                    'X-Title': settings['app_title'],
                    'Content-Type': 'application/json',
                }
                try:
                    response = client.post(settings['base_url'], json=payload, headers=headers)
                    response.raise_for_status()
                    return extract_text(response.json())
                except Exception as exc:
                    last_error = exc
            if attempt < retry_max - 1:
                time.sleep(retry_delay * (attempt + 1))
    raise RuntimeError(f'OpenRouter request failed: {last_error}')


def extract_text(data: dict) -> str:
    if 'error' in data:
        raise RuntimeError(json.dumps(data['error'], ensure_ascii=True))
    choices = data.get('choices') or []
    if not choices:
        raise RuntimeError('OpenRouter response had no choices.')
    message = choices[0].get('message') or {}
    content = message.get('content', '')
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                parts.append(item.get('text', ''))
        return '\n'.join(parts)
    return str(content)
