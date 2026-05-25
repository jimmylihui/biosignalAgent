#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.evaluation.biosignalbench import build_tool_universe, markdown_table, validate_tool_universe, write_json


def write_summary_markdown(universe: dict, validation: dict, path: str | Path) -> None:
    rows = [[k, v] for k, v in universe['summary']['tool_count_by_modality'].items()]
    evidence = [[k, v] for k, v in universe['summary']['tool_count_by_evidence_level'].items()]
    kinds = [[k, v] for k, v in universe['summary']['tool_count_by_kind'].items()]
    text = [
        '# BioSignalToolUniverse v1 Summary',
        '',
        f"Total frozen tools: {universe['num_tools']}",
        f"Validation errors: {validation['num_errors']}",
        '',
        '## Tool Count By Modality',
        markdown_table(['Modality', 'Tools'], rows),
        '## Tool Count By Evidence Level',
        markdown_table(['Evidence level', 'Tools'], evidence),
        '## Tool Count By Tool Kind',
        markdown_table(['Tool kind', 'Tools'], kinds),
        '## Missing Source Metadata',
        '\n'.join(f"- `{name}`" for name in universe['summary'].get('tools_missing_source_metadata', [])) or 'None',
        '',
    ]
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text('\n'.join(text))


def main() -> None:
    ap = argparse.ArgumentParser(description='Build frozen BioSignalToolUniverse v1 from schemas and source catalog.')
    ap.add_argument('--schemas', default='biosignal_agent/tools/schemas.json')
    ap.add_argument('--source-catalog', default='biosignal_agent/tools/source_catalog.json')
    ap.add_argument('--out-json', default='/data1/jiahui/biosignal-agent/outputs/biosignal_tool_universe_v1.json')
    ap.add_argument('--out-validation', default='/data1/jiahui/biosignal-agent/outputs/biosignal_tool_universe_v1_validation.json')
    ap.add_argument('--out-md', default='/data1/jiahui/biosignal-agent/outputs/paper_tables/tool_universe_summary.md')
    args = ap.parse_args()
    universe = build_tool_universe(args.schemas, args.source_catalog, version='v1')
    validation = validate_tool_universe(universe)
    write_json(args.out_json, universe)
    write_json(args.out_validation, validation)
    write_summary_markdown(universe, validation, args.out_md)
    print(json.dumps({'out_json': args.out_json, 'out_validation': args.out_validation, 'out_md': args.out_md, **validation}, indent=2))
    if validation['num_errors']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
