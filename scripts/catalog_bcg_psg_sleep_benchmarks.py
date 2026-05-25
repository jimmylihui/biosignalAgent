from __future__ import annotations

import json
from pathlib import Path

import requests

OUT = Path('/data1/jiahui/biosignal-agent/datasets/processed/bcg_sleep_benchmark_catalog.json')

BENCHMARKS = [
    {
        'id': 'bcg_sleepstaging_2025_eaai',
        'name': 'Automatic sleep staging based on single-channel BCG signals and multiple scales temporal feature analysis',
        'task': '5-class sleep staging from single-channel BCG HRV/RRV multi-scale features',
        'subjects_or_recordings': '10 recordings / 10,614 30s segments reported',
        'labels': 'PSG sleep stages reported in paper',
        'bcg_available': 'paper claims GitHub source/data, but repository currently 404 from GitHub API/clone',
        'url': 'https://www.sciencedirect.com/science/article/abs/pii/S0952197625012503',
        'code_url': 'https://github.com/ZJSRU-ICLaboratory/BCG-Sleepstaging',
        'public_download_status': 'unavailable_or_broken_link',
        'use_for_training': False,
        'reason': 'Cannot reproduce without accessible data/code; keep as SOTA reference only until repository is restored.',
    },
    {
        'id': 'keio_bed_leg_bcg_sleep_stage_2020',
        'name': 'Sleep Stage Estimation from Bed Leg Ballistocardiogram Sensors',
        'task': 'BCG HRV-derived sleep stage estimation',
        'subjects_or_recordings': '99 BCG/PPG HRV validation datasets; 100 sleep-stage datasets from 25 subjects reported',
        'labels': 'PSG sleep stages reported',
        'bcg_available': 'not found as public raw dataset',
        'url': 'https://www.mdpi.com/1424-8220/20/19/5688',
        'public_download_status': 'paper_only',
        'use_for_training': False,
        'reason': 'Useful method reference; raw synchronized BCG+PSG labels are not public in the article.',
    },
    {
        'id': 'frontiers_bcg_audio_psg_2026',
        'name': 'Contactless BCG/audio sleep quality with PSG reference',
        'task': 'REM/NREM or sleep-stage classification with BCG/audio and PSG reference',
        'subjects_or_recordings': 'study-specific cohort reported',
        'labels': 'PSG RemLogic scored stages',
        'bcg_available': 'not found as public raw dataset',
        'url': 'https://www.frontiersin.org/journals/network-physiology/articles/10.3389/fnetp.2026.1779111/full',
        'public_download_status': 'paper_only',
        'use_for_training': False,
        'reason': 'Current public article describes PSG labels but does not expose raw BCG+stage files.',
    },
    {
        'id': 'mendeley_bcg_sleep_8yzmk4dd7p2',
        'name': 'Ballistocardiography sleep dataset',
        'task': 'BCG sleeping-person recordings / subject ID classification',
        'subjects_or_recordings': '20 individuals',
        'labels': 'subject ID labels, not PSG sleep-stage labels',
        'bcg_available': 'public Mendeley data',
        'url': 'https://data.mendeley.com/datasets/8yzmk4dd7p/2',
        'public_download_status': 'public_but_not_sleep_stage_labeled',
        'use_for_training': False,
        'reason': 'Do not use for PSG sleep staging; labels are person IDs rather than W/N1/N2/N3/REM.',
    },
    {
        'id': 'long_term_natural_sleep_bcg_2024',
        'name': 'Long-term natural sleep BCG dataset with reference sensor signals',
        'task': 'Long-term BCG vital signs/reference sensor signals in natural sleep',
        'subjects_or_recordings': 'long-term natural sleep environment reported',
        'labels': 'reference sensors; PSG sleep stages not identified as public labels',
        'bcg_available': 'public/associated dataset likely useful for HR/resp but not PSG staging',
        'url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11455873/',
        'public_download_status': 'not_psg_stage_benchmark',
        'use_for_training': False,
        'reason': 'Potential for BCG HR/resp validation; not a direct PSG-labeled sleep-stage benchmark.',
    },
    {
        'id': 'bed_based_bcg_ecg_ppg_bp_2021',
        'name': 'Bed-Based Ballistocardiography ECG/PPG/BP dataset',
        'task': 'BCG cardiovascular parameters and BP tracking',
        'subjects_or_recordings': 'public time-aligned ECG, PPG, BCG, BP',
        'labels': 'ECG/PPG/BP references, not PSG sleep stages',
        'bcg_available': 'public',
        'url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC7795624/',
        'public_download_status': 'public_but_not_sleep_stage_labeled',
        'use_for_training': False,
        'reason': 'Good for HR/BP tools, not for sleep staging labels.',
    },
]


def probe(url: str) -> dict:
    try:
        r = requests.get(url, timeout=20, allow_redirects=True, headers={'User-Agent': 'biosignal-agent-benchmark-catalog'})
        return {'http_status': int(r.status_code), 'final_url': r.url, 'reachable': bool(r.ok)}
    except Exception as exc:
        return {'http_status': None, 'final_url': url, 'reachable': False, 'error': str(exc)}


def main() -> None:
    enriched = []
    for item in BENCHMARKS:
        row = dict(item)
        row['url_probe'] = probe(row['url'])
        if row.get('code_url'):
            row['code_url_probe'] = probe(row['code_url'])
            api = row['code_url'].replace('https://github.com/', 'https://api.github.com/repos/')
            row['github_api_probe'] = probe(api)
        enriched.append(row)
    summary = {
        'purpose': 'BCG+PSG sleep-stage benchmark availability audit',
        'usable_public_psg_bcg_sleep_stage_benchmarks': [r['id'] for r in enriched if r['use_for_training']],
        'num_candidates': len(enriched),
        'num_usable_for_training': sum(1 for r in enriched if r['use_for_training']),
        'recommendation': 'No directly usable public PSG-labeled BCG sleep-stage benchmark was confirmed. Use current BCG sleep tool as feature/proxy only, or train when user supplies synchronized BCG CSV plus 30s PSG hypnogram labels.',
        'benchmarks': enriched,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'benchmarks'}, indent=2))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
