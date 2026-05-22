from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .schema_loader import load_tool_schemas

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
MODALITY_HINTS = {
    "ecg": {"ecg", "ekg", "electrocardiogram", "qrs", "r", "rr", "hrv"},
    "ppg": {"ppg", "photoplethysmography", "pulse", "pleth"},
    "bcg": {"bcg", "ballistocardiogram", "ballistocardiography", "j", "jk"},
    "scg": {"scg", "seismocardiogram", "seismocardiography", "mechanical", "cardiac", "j", "jk"},
    "resp": {"resp", "respiration", "respiratory", "breath", "breathing"},
    "spo2": {"spo2", "oxygen", "saturation", "oximetry", "desaturation"},
    "abp": {"abp", "arterial", "blood", "pressure", "systolic", "diastolic"},
    "pcg": {"pcg", "phonocardiogram", "heart", "sound", "sounds", "s1", "s2"},
    "acc": {"acc", "accelerometer", "acceleration", "activity", "motion"},
    "eda": {"eda", "gsr", "electrodermal", "skin", "conductance", "stress"},
    "eeg": {"eeg", "electroencephalogram", "brain", "alpha", "beta", "theta", "delta", "bandpower"},
    "emg": {"emg", "electromyography", "muscle", "activation", "rms"},
}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text.replace("-", "_"))]


@dataclass
class ToolRetriever:
    """Small ToolRAG-like retriever over local tool schemas.

    It uses TF-IDF lexical scoring plus modality boosts. This keeps the framework
    offline and deterministic until we add an embedding model.
    """

    def __post_init__(self) -> None:
        self.schemas = load_tool_schemas()
        self.docs = [self._schema_text(schema) for schema in self.schemas]
        self.doc_tokens = [tokenize(doc) for doc in self.docs]
        self.doc_freq = Counter()
        for tokens in self.doc_tokens:
            self.doc_freq.update(set(tokens))
        self.num_docs = len(self.doc_tokens)

    def retrieve(self, query: str, top_k: int = 5, modality: str | None = None) -> list[dict]:
        query_tokens = tokenize(query)
        if modality is None:
            modality = self._infer_modality(query_tokens)
        candidates = list(zip(self.schemas, self.doc_tokens))
        if modality:
            modality_lower = modality.lower()
            candidates = [
                (schema, tokens)
                for schema, tokens in candidates
                if schema.get("modality", "").lower() == modality_lower
            ] or candidates
        scores = []
        for schema, tokens in candidates:
            score = self._score(query_tokens, tokens)
            if any(token in schema["name"].lower() for token in query_tokens):
                score += 0.5
            scores.append((score, schema["name"], schema))
        scores.sort(key=lambda item: (-item[0], item[1]))
        selected = [schema for score, _, schema in scores[:top_k] if score > 0]
        if modality and len(selected) < min(top_k, len(candidates)):
            selected_names = {schema["name"] for schema in selected}
            for _, _, schema in scores:
                if schema["name"] not in selected_names:
                    selected.append(schema)
                    selected_names.add(schema["name"])
                if len(selected) >= top_k:
                    break
        return selected or self._fallback_by_modality(modality, top_k)

    def _score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        doc_counts = Counter(doc_tokens)
        score = 0.0
        for token in query_tokens:
            if token not in doc_counts:
                continue
            tf = 1.0 + math.log(doc_counts[token])
            idf = math.log((1 + self.num_docs) / (1 + self.doc_freq[token])) + 1.0
            score += tf * idf
        return score

    def _schema_text(self, schema: dict) -> str:
        returns = " ".join(schema.get("returns", []))
        params = " ".join(schema.get("parameters", {}).keys())
        return f"{schema['name']} {schema.get('description', '')} {schema.get('modality', '')} {returns} {params}"

    def _infer_modality(self, tokens: list[str]) -> str | None:
        token_set = set(tokens)
        scores = {modality: len(token_set & hints) for modality, hints in MODALITY_HINTS.items()}
        modality, score = max(scores.items(), key=lambda item: item[1])
        return modality if score > 0 else None

    def _fallback_by_modality(self, modality: str | None, top_k: int) -> list[dict]:
        if modality is None:
            return self.schemas[:top_k]
        return [schema for schema in self.schemas if schema.get("modality", "").lower() == modality.lower()][:top_k]
