from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_MODALITIES = {"ecg", "ppg", "bcg"}


@dataclass
class SignalInput:
    modality: str
    path: str
    sampling_rate: float
    column: str | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        self.modality = self.modality.lower()
        if self.modality not in SUPPORTED_MODALITIES:
            raise ValueError(f"Unsupported modality {self.modality}. Expected one of {sorted(SUPPORTED_MODALITIES)}")
        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "path": self.path,
            "sampling_rate": self.sampling_rate,
            "column": self.column,
            "label": self.label,
        }


@dataclass
class BioSignalSession:
    question: str
    signals: list[SignalInput] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question cannot be empty")
        if not self.signals:
            raise ValueError("at least one signal is required")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BioSignalSession":
        return cls(
            question=payload["question"],
            signals=[SignalInput(**signal) for signal in payload.get("signals", [])],
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "BioSignalSession":
        import json
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict[str, Any]:
        return {"question": self.question, "signals": [signal.to_dict() for signal in self.signals]}
