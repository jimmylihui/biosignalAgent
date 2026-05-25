from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_MANIFESTS = [
    '/data1/jiahui/biosignal-agent/datasets/processed/real_world_manifest.json',
    '/data1/jiahui/biosignal-agent/datasets/processed/dedicated_common_manifest.json',
    '/data1/jiahui/biosignal-agent/datasets/processed/dedicated_bcg_manifest.json',
    '/data1/jiahui/biosignal-agent/datasets/processed/pcg_murmur_manifest.json',
    '/data1/jiahui/biosignal-agent/datasets/processed/ppg_af_manifest.json',
    '/data1/jiahui/biosignal-agent/datasets/processed/acc_activity_manifest.json',
    '/data1/jiahui/biosignal-agent/datasets/processed/chbmit_seizure_manifest.json',
]


def load_records(paths: list[str]) -> list[dict[str, Any]]:
    records = []
    seen = set()
    for path in paths:
        payload_path = Path(path)
        if not payload_path.exists():
            continue
        payload = json.loads(payload_path.read_text())
        for row in payload.get('records', []):
            modality = row.get('modality')
            signal_path = row.get('path')
            sampling_rate = row.get('sampling_rate')
            if not modality or not signal_path or sampling_rate is None:
                continue
            key = (signal_path, modality)
            if key in seen:
                continue
            seen.add(key)
            records.append({
                'dataset': row.get('dataset'),
                'record': row.get('record'),
                'modality': str(modality).lower(),
                'path': signal_path,
                'sampling_rate': float(sampling_rate),
                'source_channel': row.get('source_channel'),
            })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare manifest for signal modality classifier baseline.')
    parser.add_argument('--manifest', action='append', default=None)
    parser.add_argument('--out-json', default='/data1/jiahui/biosignal-agent/datasets/processed/modality_classifier_manifest.json')
    args = parser.parse_args()
    records = load_records(args.manifest or DEFAULT_MANIFESTS)
    report = {
        'dataset': 'biosignal_modality_classifier',
        'records': records,
        'num_records': len(records),
        'modality_counts': dict(Counter(row['modality'] for row in records)),
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ['num_records', 'modality_counts']}, indent=2))


if __name__ == '__main__':
    main()
