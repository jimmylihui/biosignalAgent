from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_tool_schemas() -> list[dict]:
    schema_path = Path(__file__).resolve().parents[1] / "tools" / "schemas.json"
    return json.loads(schema_path.read_text())


def find_tool_schemas(query: str, top_k: int = 5) -> list[dict]:
    """Tiny keyword retriever until we replace this with embeddings/ToolRAG."""
    terms = {token.lower() for token in query.replace("_", " ").split()}
    scored = []
    for schema in load_tool_schemas():
        text = f"{schema['name']} {schema['description']} {schema['modality']}".lower()
        score = sum(1 for term in terms if term in text)
        scored.append((score, schema["name"], schema))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [schema for score, _, schema in scored[:top_k] if score > 0]
