#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biosignal_agent.evaluation.biosignalbench import load_bench_cases, markdown_table, write_csv, write_json, write_jsonl
from biosignal_agent.tools.ecg_tools import ECG_classify_12lead_ptbxl_superclasses


def main() -> None:
    ap=argparse.ArgumentParser(description='Evaluate ECG_classify_12lead_ptbxl_superclasses on BioSignalBench PTB-XL 12-lead cases.')
    ap.add_argument('--manifest', default='/data1/jiahui/biosignal-agent/outputs/biosignalbench_v1.jsonl')
    ap.add_argument('--out-json', default='/data1/jiahui/biosignal-agent/outputs/ptbxl_12lead_tool_eval.json')
    ap.add_argument('--out-jsonl', default='/data1/jiahui/biosignal-agent/outputs/ptbxl_12lead_tool_eval_cases.jsonl')
    ap.add_argument('--out-csv', default='/data1/jiahui/biosignal-agent/outputs/ptbxl_12lead_tool_eval_cases.csv')
    ap.add_argument('--out-md', default='/data1/jiahui/biosignal-agent/outputs/paper_tables/ptbxl_12lead_tool_eval.md')
    args=ap.parse_args()
    cases=[c for c in load_bench_cases(args.manifest) if c.get('benchmark_task')=='tool_execution' and 'ECG_classify_12lead_ptbxl_superclasses' in c.get('expected_tools',[])]
    rows=[]
    for c in cases:
        target=(c.get('ground_truth_metric') or {}).get('target')
        path=(c.get('signal') or {}).get('path')
        res=ECG_classify_12lead_ptbxl_superclasses(path)
        positives=res.get('predicted_positive_classes') or []
        probs=res.get('probabilities') or {}
        ok=target in positives if target else not bool(res.get('error'))
        rows.append({
            'case_id':c.get('case_id'),
            'target':target,
            'signal_path':path,
            'target_probability':probs.get(target),
            'predicted_positive_classes':positives,
            'label_correct':ok,
            'error':res.get('error'),
            'method':res.get('method'),
        })
    n=len(rows)
    summary={'artifact':'PTBXL12LeadToolEvaluation','num_cases':n,'target_recall':sum(r['label_correct'] for r in rows)/n if n else 0,'errors':[r for r in rows if r.get('error')]}
    write_json(args.out_json, summary); write_jsonl(args.out_jsonl, rows); write_csv(args.out_csv, rows)
    table=markdown_table(['Case','Target','Prob','Predicted','Correct'], [[r['case_id'],r['target'],r['target_probability'],','.join(r['predicted_positive_classes']),r['label_correct']] for r in rows])
    Path(args.out_md).parent.mkdir(parents=True,exist_ok=True); Path(args.out_md).write_text('# PTB-XL 12-Lead Tool Evaluation\n\n'+json.dumps(summary,indent=2)+'\n\n'+table)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
