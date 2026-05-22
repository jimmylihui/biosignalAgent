from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {
    "modality",
    "task",
    "current_tools",
    "existing_work",
    "candidate_libraries",
    "candidate_datasets",
    "source_urls",
    "wrapper_priority",
    "next_wrapper",
}


def validate(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    entries = payload.get("entries", [])
    errors = []
    seen = set()
    for idx, entry in enumerate(entries):
        missing = sorted(REQUIRED_FIELDS - set(entry))
        if missing:
            errors.append({"index": idx, "task": entry.get("task"), "error": f"missing fields: {missing}"})
        key = (entry.get("modality"), entry.get("task"))
        if key in seen:
            errors.append({"index": idx, "task": entry.get("task"), "error": "duplicate modality/task"})
        seen.add(key)
        priority = entry.get("wrapper_priority")
        if not isinstance(priority, int) or priority < 1 or priority > 5:
            errors.append({"index": idx, "task": entry.get("task"), "error": "wrapper_priority must be an integer from 1 to 5"})
        for field in ["current_tools", "existing_work", "candidate_libraries", "candidate_datasets", "source_urls"]:
            if not isinstance(entry.get(field), list) or not entry.get(field):
                errors.append({"index": idx, "task": entry.get("task"), "error": f"{field} must be a non-empty list"})
        for url in entry.get("source_urls", []):
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                errors.append({"index": idx, "task": entry.get("task"), "error": f"invalid source url: {url}"})
    modality_counts = Counter(entry.get("modality") for entry in entries)
    priority_counts = Counter(entry.get("wrapper_priority") for entry in entries)
    return {
        "path": str(path),
        "num_entries": len(entries),
        "num_errors": len(errors),
        "modality_counts": dict(sorted(modality_counts.items())),
        "priority_counts": dict(sorted(priority_counts.items())),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate BioSignalAgent existing-work source catalog.")
    parser.add_argument("--catalog", default="biosignal_agent/tools/source_catalog.json")
    args = parser.parse_args()
    report = validate(args.catalog)
    print(json.dumps(report, indent=2))
    if report["num_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
