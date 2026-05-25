#!/usr/bin/env python3
"""Build a comparison table for BioSignalAgent split-protocol controller runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)


def fmt(v: Any) -> str:
    if v is None:
        return ''
    if isinstance(v, float):
        return f'{v:.3f}'
    return str(v)


def row(label: str, split: str, path: str, note: str) -> list[Any]:
    d = read_json(path)
    return [
        label,
        split,
        d.get('num_cases'),
        d.get('planner_strict_parse_rate'),
        d.get('planning_accuracy'),
        d.get('tool_f1'),
        d.get('execution_success'),
        d.get('report_factuality_score'),
        d.get('overall_hmean'),
        note,
    ]


def table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for r in rows:
        out.append('| ' + ' | '.join(fmt(x) for x in r) + ' |')
    return '\n'.join(out) + '\n'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-md', default='/data1/jiahui/biosignal-agent/outputs/paper_tables/table18_split_protocol_controller_comparison.md')
    args = ap.parse_args()
    rows = [
        row('OpenRouter owl-alpha cached planner + SFT report', 'held-out', '/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_openrouter_owl_alpha_heldout.json', 'external free LLM planner generations scored in same controller protocol'),
        row('Live SFT controller v4', 'held-out stress', '/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_live_v4_heldout.json', 'v4 trained before split freeze; stress subset only'),
        row('Live SFT controller v5', 'dev', '/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_live_v5_dev.json', 'trained only on train split'),
        row('Live SFT controller v5', 'held-out', '/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_live_v5_heldout.json', 'trained only on train split; no session completion guardrail'),
        row('Live SFT controller v5 + session guardrail', 'held-out', '/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_live_v5_heldout_guarded.json', 'trained only on train split; session bundle completion from retrieved tools'),
        row('Live SFT controller v6 + session guardrail', 'held-out', '/data1/jiahui/biosignal-agent/outputs/biosignalagent_e2e_controller_live_v6_heldout_guarded.json', 'train split plus synthetic train-only session augmentation'),
    ]
    text = '# Table 18. Split-Protocol Live Controller Comparison\n\n'
    text += 'This table distinguishes the strongest current v4 controller from the cleaner v5 split-protocol controller. v5 is trained only on the frozen train manifest and evaluated on dev/held-out manifests.\n\n'
    text += table(['Method','Split','Cases','Strict parse','Planning','Tool F1','Exec success','Report score','Overall H-mean','Note'], rows)
    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(out)


if __name__ == '__main__':
    main()
