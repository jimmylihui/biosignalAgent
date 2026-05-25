
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biosignal_agent.tools.bcg_tools import BCG_screen_arrhythmia
from scripts.evaluate_bcg_figshare_hr import OUT_DIR, evaluate_subject, figshare_files

RAW_INFO = Path('/data1/jiahui/biosignal-agent/datasets/raw/dedicated_bcg_figshare/Overall_info.xlsx')
DEFAULT_HR_EVAL = OUT_DIR / 'bcg_figshare_hr_eval_46x60s.json'


def parse_af_label(text: str) -> int:
    text = str(text).lower()
    has_af = 'atrial fibrillation' in text or re.search(r'\baf\b', text) is not None
    sinus_majority = 'sinus rhythm' in text and ('persistent atrial fibrillation' not in text)
    return int(has_af and not sinus_majority)


def load_labels() -> dict[str, dict]:
    df = pd.read_excel(RAW_INFO)
    labels = {}
    for _, row in df.iterrows():
        record = f"Sub{int(row['Idx']):02d}"
        conclusion = str(row.get('Conclusion', ''))
        labels[record] = {
            'af_label': parse_af_label(conclusion),
            'sex': row.get('Sex'),
            'age': int(row.get('Age')) if pd.notna(row.get('Age')) else None,
            'conclusion_excerpt': conclusion[:220].replace('\n', ' '),
        }
    return labels


def ensure_segments(subjects: int, seconds: float) -> list[str]:
    files = {f['name']: f for f in figshare_files()}
    records = []
    for i in range(1, subjects + 1):
        record = f'Sub{i:02d}'
        segment_path = OUT_DIR / f'{record.lower()}_bcg_{int(seconds)}s.csv'
        if not segment_path.exists():
            evaluate_subject(record, files, seconds)
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects', type=int, default=46)
    parser.add_argument('--seconds', type=float, default=60.0)
    parser.add_argument('--output', type=Path, default=OUT_DIR / 'bcg_figshare_af_proxy_eval_46x60s.json')
    args = parser.parse_args()
    labels = load_labels()
    records = ensure_segments(args.subjects, args.seconds)
    rows = []
    for record in records:
        segment_path = OUT_DIR / f'{record.lower()}_bcg_{int(args.seconds)}s.csv'
        if record not in labels or not segment_path.exists():
            continue
        result = BCG_screen_arrhythmia(str(segment_path), 125.0)
        rows.append({
            'record': record,
            'af_label': labels[record]['af_label'],
            'score': result.get('irregularity_score'),
            'risk': result.get('arrhythmia_risk'),
            'confidence': result.get('confidence'),
            'interval_cv': result.get('interval_cv'),
            'rmssd_ms': result.get('rmssd_ms'),
            'pnn50': result.get('pnn50'),
            'age': labels[record]['age'],
            'conclusion_excerpt': labels[record]['conclusion_excerpt'],
        })
    valid = [r for r in rows if r.get('score') is not None]
    y = np.asarray([r['af_label'] for r in valid], dtype=int)
    s = np.asarray([r['score'] for r in valid], dtype=float)
    pred = (s >= 0.65).astype(int)
    summary = {
        'dataset': 'figshare_bed_bcg_2025',
        'label_source': 'Overall_info.xlsx conclusion text; persistent atrial fibrillation parsed as AF-positive',
        'subjects_evaluated': len(valid),
        'af_positive': int(y.sum()) if len(y) else 0,
        'af_negative': int(len(y) - y.sum()) if len(y) else 0,
        'seconds_per_subject': args.seconds,
        'auroc': float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None,
        'average_precision': float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else None,
        'balanced_accuracy_at_0p65': float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) == 2 else None,
        'macro_f1_at_0p65': float(f1_score(y, pred, average='macro')) if len(np.unique(y)) == 2 else None,
        'accuracy_at_0p65': float(accuracy_score(y, pred)) if len(y) else None,
        'rows': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2))
    print('wrote', args.output)


if __name__ == '__main__':
    main()
