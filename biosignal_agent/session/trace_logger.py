from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

DEFAULT_TRACE_DIR = Path('/data1/jiahui/biosignal-agent/outputs/traces')


def save_trace(trace: dict[str, Any], trace_dir: str | Path = DEFAULT_TRACE_DIR) -> Path:
    out_dir = Path(trace_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    trace_id = trace.get('trace_id') or f'{timestamp}_{uuid4().hex[:8]}'
    trace = {**trace, 'trace_id': trace_id, 'created_at': trace.get('created_at') or timestamp}
    path = out_dir / f'{trace_id}.json'
    path.write_text(json.dumps(trace, indent=2, ensure_ascii=True))
    return path
