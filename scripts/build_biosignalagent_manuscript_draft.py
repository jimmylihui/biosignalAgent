#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

OUTPUTS = Path('/data1/jiahui/biosignal-agent/outputs')
DOCS = Path('/data1/jiahui/biosignal-agent/code/biosignal-agent/docs')
PAPER_DIR = OUTPUTS / 'paper_tables'

def load(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}

def fmt(value: Any) -> str:
    if value is None:
        return 'NA'
    if isinstance(value, float):
        return f'{value:.3f}'
    return str(value)

def main() -> None:
    ap = argparse.ArgumentParser(description='Build a manuscript-ready BioSignalAgent methods/results draft from frozen artifacts.')
    ap.add_argument('--out-md', default=str(DOCS / 'biosignalagent_manuscript_results_draft.md'))
    ap.add_argument('--outline-md', default=str(DOCS / 'biosignalagent_paper_outline.md'))
    args = ap.parse_args()

    universe = load(OUTPUTS / 'biosignal_tool_universe_v1.json')
    bench = load(OUTPUTS / 'biosignalbench_v1_summary.json')
    split = load(OUTPUTS / 'biosignalbench_splits_v1/summary.json')
    expanded = load(OUTPUTS / 'biosignalbench_v1_expanded_summary.json')
    expanded_split = load(OUTPUTS / 'biosignalbench_v1_expanded_splits/summary.json')
    clean = load(OUTPUTS / 'biosignalagent_e2e_controller_live_v5_heldout_guarded.json')
    expanded_guarded = load(OUTPUTS / 'biosignalagent_e2e_controller_live_v5_expanded_heldout_timeout_guarded.json')
    expanded_raw = load(OUTPUTS / 'biosignalagent_e2e_controller_live_v5_expanded_heldout_raw_no_guardrail.json')
    openrouter = load(OUTPUTS / 'biosignalagent_e2e_controller_openrouter_owl_alpha_heldout.json')
    tool_metrics = load(OUTPUTS / 'tool_execution_metrics_index.json')
    gap = load(OUTPUTS / 'txagent_comparability_gap_matrix.json')

    heldout_n = split.get('splits', {}).get('heldout', {}).get('num_cases')
    train_n = split.get('splits', {}).get('train', {}).get('num_cases')
    dev_n = split.get('splits', {}).get('dev', {}).get('num_cases')
    exp_heldout_n = expanded_split.get('splits', {}).get('heldout', {}).get('num_cases')
    exp_train_n = expanded_split.get('splits', {}).get('train', {}).get('num_cases')
    exp_dev_n = expanded_split.get('splits', {}).get('dev', {}).get('num_cases')
    open_gaps = [r for r in gap.get('rows', []) if str(r.get('status', '')).startswith('partially')]

    lines: list[str] = []
    add = lines.append
    add('# BioSignalAgent Manuscript Results Draft')
    add('')
    add('This draft is generated from frozen paper artifacts. It is a writing scaffold, not a new experiment.')
    add('')
    add('## Abstract-Style Result Summary')
    add('')
    add(f"BioSignalAgent instantiates a TxAgent-style tool-use agent for biosignals with BioSignalToolUniverse v1 ({universe.get('num_tools', 'NA')} frozen tools) and BioSignalBench v1 ({bench.get('num_cases', 'NA')} cases). On the clean held-out split ({heldout_n} cases), the selected live SFT planner/controller reaches planning accuracy {fmt(clean.get('planning_accuracy'))}, Tool F1 {fmt(clean.get('tool_f1'))}, execution success {fmt(clean.get('execution_success'))}, report factuality {fmt(clean.get('report_factuality_score'))}, and overall H-mean {fmt(clean.get('overall_hmean'))}. On the expanded held-out stress split ({exp_heldout_n} cases), raw live SFT reaches planning {fmt(expanded_raw.get('planning_accuracy'))} and Tool F1 {fmt(expanded_raw.get('tool_f1'))}; the timeout/metadata-guarded controller reaches planning {fmt(expanded_guarded.get('planning_accuracy'))}, Tool F1 {fmt(expanded_guarded.get('tool_f1'))}, and overall {fmt(expanded_guarded.get('overall_hmean'))}.")
    add('')
    add('## Methods Draft')
    add('')
    add('### BioSignalToolUniverse v1')
    add('')
    add(f"We freeze BioSignalToolUniverse v1 as {universe.get('num_tools', 'NA')} executable tool schemas covering ECG, PPG, PCG, SCG, BCG, EMG, EDA, EEG, RESP, SpO2, ABP, ACC, image digitization, and multimodal/session reasoning. Each tool includes structured metadata for modality, task, evidence level, data source, metric, known failure modes, and clinical limitations. Proxy tools are retained for realistic routing but explicitly labeled and not framed as diagnostic validators.")
    add('')
    add('Recommended artifact: `biosignal_tool_universe_v1.json`; recommended table: `table1_tool_universe.md`.')
    add('')
    add('### BioSignalBench v1')
    add('')
    add(f"BioSignalBench v1 contains {bench.get('num_cases', 'NA')} cases split into train/dev/held-out = {train_n}/{dev_n}/{heldout_n}. It covers modality routing, image-to-signal digitization, scale/OCR extraction, tool planning, executable tool evaluation, report factuality, and multimodal session reasoning. BioSignalBench v1-expanded contains {expanded.get('num_cases', 'NA')} cases with train/dev/held-out = {exp_train_n}/{exp_dev_n}/{exp_heldout_n}; it is a stress complement, not a replacement for the clean split.")
    add('')
    add('Recommended tables: `table2_biosignalbench_composition.md`, `table27_biosignalbench_expanded_composition.md`.')
    add('')
    add('### Agent Controller')
    add('')
    add('The controller uses ToolRAG retrieval, SFT planner generation, executable local biosignal tools, and grounded report generation/scoring. The final live controller includes timeout-safe generation plus metadata-derived guardrails for image digitization, scale/OCR extraction, unknown-modality routing, multimodal session completion, and minimal task-specific pruning.')
    add('')
    add('## Results Draft')
    add('')
    add('### Main Clean Held-Out Result')
    add('')
    for label, value in [
        ('Cases', clean.get('num_cases', heldout_n)),
        ('Retrieval accuracy', clean.get('retrieval_accuracy')),
        ('Strict/recovered planner parse', f"{fmt(clean.get('planner_strict_parse_rate'))}/{fmt(clean.get('planner_parse_rate'))}"),
        ('Planning accuracy', clean.get('planning_accuracy')),
        ('Tool F1', clean.get('tool_f1')),
        ('Execution success', clean.get('execution_success')),
        ('Report factuality score', clean.get('report_factuality_score')),
        ('Overall H-mean', clean.get('overall_hmean')),
        ('Failure reasons', json.dumps(clean.get('failure_reason_counts', {}), sort_keys=True)),
    ]:
        add(f'- {label}: {fmt(value) if not isinstance(value, str) else value}')
    add('')
    add('Use this as the primary paper result because it follows the frozen split protocol. Recommended table: `table18_split_protocol_controller_comparison.md`.')
    add('')
    add('### Expanded Stress Result and Guardrail Ablation')
    add('')
    add(f"- Expanded held-out cases: {expanded_guarded.get('num_cases', exp_heldout_n)}")
    add(f"- Raw live SFT planning / Tool F1 / overall: {fmt(expanded_raw.get('planning_accuracy'))} / {fmt(expanded_raw.get('tool_f1'))} / {fmt(expanded_raw.get('overall_hmean'))}")
    add(f"- Guarded live SFT planning / Tool F1 / overall: {fmt(expanded_guarded.get('planning_accuracy'))} / {fmt(expanded_guarded.get('tool_f1'))} / {fmt(expanded_guarded.get('overall_hmean'))}")
    add(f"- Raw failure reasons: {json.dumps(expanded_raw.get('failure_reason_counts', {}), sort_keys=True)}")
    add(f"- Guarded failure reasons: {json.dumps(expanded_guarded.get('failure_reason_counts', {}), sort_keys=True)}")
    add('')
    add('Interpretation: retrieval is not the bottleneck on the expanded split. Raw live SFT often emits plausible but non-exact tool sets, especially for report-factuality and multimodal cases. The metadata guardrail converts these into exact executable plans, so raw and guarded rows should be reported separately. Recommended table: `table28_expanded_heldout_baseline_comparison.md`.')
    add('')
    add('### External LLM Baseline')
    add('')
    add(f"- OpenRouter/owl-alpha clean held-out planning: {fmt(openrouter.get('planning_accuracy'))}")
    add(f"- OpenRouter/owl-alpha clean held-out Tool F1: {fmt(openrouter.get('tool_f1'))}")
    add(f"- OpenRouter/owl-alpha clean held-out overall: {fmt(openrouter.get('overall_hmean'))}")
    add(f"- OpenRouter failure reasons: {json.dumps(openrouter.get('failure_reason_counts', {}), sort_keys=True)}")
    add('')
    add('Expanded OpenRouter runs were not used as a final artifact because synchronous external API calls were unstable during SSL read. Treat this as an external-baseline limitation unless a stable paid endpoint is added.')
    add('')
    add('### Module Ablation')
    add('')
    add('The held-out ablation table isolates controller components. Removing ToolRAG gives the largest drop; removing quality gates, image digitization, OCR/scale, modality classification, or DL tools produces targeted degradations. Recommended table: `table23_live_controller_ablation_v5_guarded_heldout.md`.')
    add('')
    add('### Tool Execution Evidence')
    add('')
    add(f"The unified tool execution index contains {tool_metrics.get('num_rows', 'NA')} metric rows across major modalities. Strong validated examples include ECG R-peak detection, PTB-XL 12-lead classification, PPG pulse detection, PCG S1/S2 segmentation, EDA stress detection, ACC activity recognition, and ABP hypotension/shock prediction. Weaker/proxy rows remain explicitly labeled. Recommended table: `table26_tool_execution_metrics_index.md`.")
    add('')
    add('## Discussion Draft')
    add('')
    add('The main empirical pattern is that BioSignalAgent has strong retrieval coverage and executable tool plumbing, while exact structured planning remains the central modeling challenge. The raw-vs-guarded expanded result makes this explicit: the SFT planner often selects plausible tools but misses benchmark minimality, and metadata-aware controller repair closes that gap. This is defensible for a tool-use system, but the manuscript should avoid presenting guardrail-repaired plans as pure LLM generation.')
    add('')
    add('## Limitations To State Explicitly')
    add('')
    for limitation in [
        f'Clean held-out size is {heldout_n}; expanded held-out is useful stress evidence but not the primary frozen split.',
        'Many tools are proxy or benchmark-limited and should not be described as clinical diagnostic systems.',
        'The final expanded perfect planning row depends on metadata-derived guardrails; raw SFT performance should be reported alongside it.',
        'External LLM baseline coverage is limited to a free OpenRouter model on the clean split because expanded synchronous calls were unstable.',
        'Several bottom-level signal models remain below modality-specific SOTA and need larger external validation before clinical claims.',
    ]:
        add(f'- {limitation}')
    add('')
    add('## Current TxAgent-Comparability Gaps')
    add('')
    if open_gaps:
        for row in open_gaps:
            add(f"- {row.get('gap')}: {row.get('remaining')}")
    else:
        add('- No partially closed gaps found in the generated gap matrix.')
    add('')
    add('## Paper Table Checklist')
    add('')
    for name in [
        'table1_tool_universe.md',
        'table2_biosignalbench_composition.md',
        'table18_split_protocol_controller_comparison.md',
        'table23_live_controller_ablation_v5_guarded_heldout.md',
        'table24_final_failure_analysis.md',
        'table25_paper_artifact_index.md',
        'table26_tool_execution_metrics_index.md',
        'table27_biosignalbench_expanded_composition.md',
        'table28_expanded_heldout_baseline_comparison.md',
        'table30_txagent_gap_matrix.md',
    ]:
        add(f"- `{name}`: {'present' if (PAPER_DIR / name).exists() else 'missing'}")
    add('')

    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines))

    outline = Path(args.outline_md)
    outline.write_text('\n'.join([
        '# BioSignalAgent Paper Outline', '',
        '## Working Title', '',
        'BioSignalAgent: A Tool-Using Agent and Benchmark for Multimodal Biosignal Reasoning', '',
        '## Core Claim', '',
        'BioSignalAgent adapts the TxAgent-style ToolUniverse + tool-use SFT + benchmark evaluation recipe to biosignals. The artifact now includes a frozen BioSignalToolUniverse, BioSignalBench, live controller, clean split protocol, external LLM baseline, module ablations, failure analysis, expanded stress testing, and a unified tool-execution metric index.', '',
        '## Main Numbers', '',
        f"- BioSignalToolUniverse v1: `{universe.get('num_tools', 'NA')}` frozen tools.",
        f"- BioSignalBench v1: `{bench.get('num_cases', 'NA')}` cases; train/dev/held-out = `{train_n}/{dev_n}/{heldout_n}`.",
        f"- BioSignalBench v1-expanded: `{expanded.get('num_cases', 'NA')}` cases; train/dev/held-out = `{exp_train_n}/{exp_dev_n}/{exp_heldout_n}`.",
        f"- Clean held-out live v5+guardrail: planning `{fmt(clean.get('planning_accuracy'))}`, Tool F1 `{fmt(clean.get('tool_f1'))}`, execution `{fmt(clean.get('execution_success'))}`, report score `{fmt(clean.get('report_factuality_score'))}`, overall `{fmt(clean.get('overall_hmean'))}`.",
        f"- Expanded raw live SFT: planning `{fmt(expanded_raw.get('planning_accuracy'))}`, Tool F1 `{fmt(expanded_raw.get('tool_f1'))}`, overall `{fmt(expanded_raw.get('overall_hmean'))}`.",
        f"- Expanded timeout/metadata-guarded live SFT: planning `{fmt(expanded_guarded.get('planning_accuracy'))}`, Tool F1 `{fmt(expanded_guarded.get('tool_f1'))}`, overall `{fmt(expanded_guarded.get('overall_hmean'))}`.",
        f"- Tool execution index: `{tool_metrics.get('num_rows', 'NA')}` metric rows.", '',
        '## Paper Structure', '',
        '1. Introduction: biosignal reasoning needs tool-use agents with explicit modality routing, scale/image handling, executable signal tools, and grounded reporting.',
        '2. BioSignalToolUniverse v1: frozen schemas, evidence metadata, proxy labeling, and source catalog.',
        '3. BioSignalBench v1: benchmark tasks, JSON schema, clean split protocol, expanded stress split.',
        '4. BioSignalAgent: ToolRAG retrieval, SFT planner, timeout-safe live controller, metadata guardrails, tool execution, grounded reporting.',
        '5. Experiments: clean held-out, expanded stress, raw-vs-guarded ablation, module ablations, external LLM baseline, failure analysis.',
        '6. Tool evidence: modality-level execution metrics and validated/proxy limitations.',
        '7. Discussion and limitations: guardrails vs raw generation, small clean split, proxy tools, external API instability, non-clinical framing.', '',
        '## Recommended Primary Tables', '',
        '- ToolUniverse summary: `table1_tool_universe.md`',
        '- BioSignalBench composition: `table2_biosignalbench_composition.md`',
        '- Main split-protocol controller comparison: `table18_split_protocol_controller_comparison.md`',
        '- Held-out ablation: `table23_live_controller_ablation_v5_guarded_heldout.md`',
        '- Failure analysis: `table24_final_failure_analysis.md`',
        '- Tool execution evidence: `table26_tool_execution_metrics_index.md`',
        '- Expanded benchmark composition: `table27_biosignalbench_expanded_composition.md`',
        '- Expanded baseline/raw-vs-guarded comparison: `table28_expanded_heldout_baseline_comparison.md`',
        '- TxAgent comparability matrix: `table30_txagent_gap_matrix.md`', '',
        '## Next Work Before Submission', '',
        '1. Add stronger stable external LLM baselines if API/budget allows.',
        '2. Add qualitative examples for success/failure cases in an appendix.',
        '3. Convert `docs/biosignalagent_manuscript_results_draft.md` into full prose.',
        '4. Keep proxy/clinical limitations explicit in every report and table caption.', ''
    ]))
    print(json.dumps({'out_md': str(out), 'outline_md': str(outline)}, indent=2))

if __name__ == '__main__':
    main()
