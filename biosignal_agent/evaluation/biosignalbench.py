
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biosignal_agent.agent.planning_agent import PlanningBioSignalAgent
from biosignal_agent.agent.tool_registry import TOOLS, WORKFLOWS
from biosignal_agent.agent.tool_retriever import ToolRetriever

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path('/data1/jiahui/biosignal-agent/outputs')
TOKEN_RE = re.compile(r'[A-Za-z0-9_]+')


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2) + '\n')


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with p.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})



def tool_hierarchy_metadata(tool_name: str, modality: str, task: str = '', description: str = '') -> dict[str, Any]:
    """Infer the physiological computation layer and dependencies for a tool.

    Level 1 tools operate directly on raw signal/image data. Level 2 tools build
    physiological representations from primitive events/features. Level 3 tools
    perform task-level screening, classification, or report-level reasoning.
    """
    name = tool_name.lower()
    modality_l = str(modality or '').lower()
    text = f'{name} {task} {description}'.lower()
    primitive_terms = [
        'detect_r_peaks', 'detect_peaks', 'detect_j_peaks', 'detect_pulses', 'detect_breath_peaks',
        'detect_heart_sounds', 'segment_s1_s2', 'delineate_waves',
        'detect_fiducial_points', 'compute_bandpower', 'detect_bursts',
        'detect_desaturation', 'detect_peaks_troughs', 'detect_apnea', 'detect_hypopnea',
        'digitize_waveform', 'classify_modality', 'detect_artifacts',
        'read_image_text_ocr',
    ]
    representation_terms = [
        'compute_hrv', 'compute_prv', 'estimate_heart_rate', 'estimate_respiration',
        'compute_cardiac_time_intervals', 'compute_hemodynamics',
        'extract_tonic_phasic_features', 'extract_oximetry_features',
        'extract_actigraphy_features', 'summarize_activity', 'summarize_activation',
        'summarize_event_burden', 'summarize', 'assess_quality',
        'assess_perfusion_variability', 'assess_hypoxemia_burden',
        'assess_bed_presence_motion', 'assess_sensor_placement',
        'estimate_sleep_features', 'estimate_sleep_stage_features',
        'extract_murmur_features', 'measure_morphology_intervals',
        'analyze_qt_interval', 'estimate_rate', 'estimate_image_scale',
        'predict_image_scale', 'extract_spectrogram', 'render_spectrogram',
    ]
    screening_terms = [
        'screen_', 'classify_', 'detect_afib', 'classify_rhythm',
        'classify_beats', 'estimate_sleep_wake', 'estimate_drowsiness',
        'estimate_fatigue', 'predict_movement_intent', 'monitor_',
        'route_task_recommendation', 'estimate_bp_proxy', 'estimate_spo2',
        'estimate_exercise_intensity', 'assess_stress', 'assess_vascular_health',
    ]
    if tool_name == 'ABP_detect_pulses':
        level = 'representation'
    elif tool_name.startswith('Multimodal_'):
        level = 'screening'
    elif any(term in name for term in screening_terms):
        level = 'screening'
    elif any(term in name for term in representation_terms):
        level = 'representation'
    elif any(term in name for term in primitive_terms):
        level = 'primitive'
    else:
        level = 'representation' if any(term in text for term in ['feature', 'quality', 'summary']) else 'screening'
    deps = dependency_tools_for(tool_name, modality_l, level)
    consumes, produces = io_semantics_for(tool_name, modality_l, level)
    return {'tool_level': level, 'depends_on': deps, 'consumes': consumes, 'produces': produces}


def dependency_tools_for(tool_name: str, modality: str, level: str) -> list[str]:
    n = tool_name.lower()
    deps: list[str] = []
    if level == 'primitive':
        return deps
    if tool_name == 'ABP_detect_pulses':
        return ['ABP_detect_fiducial_points']
    if modality == 'ecg':
        if any(term in n for term in ['compute_hrv', 'estimate_heart_rate', 'screen_arrhythmia', 'detect_afib', 'screen_sleep_apnea', 'classify_rhythm', 'classify_beats']):
            pass
        if any(term in n for term in ['screen_arrhythmia', 'detect_afib', 'screen_sleep_apnea']):
            deps.append('ECG_compute_hrv')
        if any(term in n for term in ['morphology', 'qt', 'conduction', 'ischemia', 'st']):
            deps.append('ECG_delineate_waves_dl')
    elif modality == 'ppg':
        if any(term in n for term in ['compute_prv', 'estimate_heart_rate', 'screen_pulse_irregularity', 'detect_afib', 'respiration_modulation']):
            pass
        if any(term in n for term in ['compute_prv', 'screen_pulse_irregularity', 'detect_afib', 'assess_stress']):
            deps.append('PPG_compute_prv')
        if 'fiducial' in n or 'vascular' in n or 'bp_proxy' in n:
            deps.append('PPG_detect_fiducial_points')
    elif modality == 'bcg':
        if any(term in n for term in ['compute_hrv', 'screen_arrhythmia', 'estimate_bp_proxy']):
            deps.append('BCG_detect_j_peaks')
        if 'estimate_respiration' in n or 'sleep' in n:
            deps.append('BCG_assess_bed_presence_motion')
    elif modality == 'scg':
        if any(term in n for term in ['cardiac_time', 'contractility', 'mechanical_abnormality']):
            deps.append('SCG_detect_fiducial_points')
        if 'j_peak' in n or 'j_peaks' in n:
            pass
    elif modality == 'resp':
        if 'estimate_rate' in n:
            deps.append('RESP_detect_breath_peaks')
        if any(term in n for term in ['detect_apnea', 'detect_hypopnea', 'sleep_apnea', 'event_burden', 'rate_pattern']):
            deps.append('RESP_estimate_rate')
    elif modality == 'spo2':
        if any(term in n for term in ['summarize', 'sleep_apnea', 'hypoxemia', 'oximetry']):
            pass
    elif modality == 'pcg':
        if any(term in n for term in ['estimate_heart_rate', 'rhythm_irregularity', 'murmur', 'valve', 'congenital', 's3_s4', 'heart_function']):
            deps.append('PCG_detect_heart_sounds')
        if any(term in n for term in ['murmur', 'valve', 'congenital']):
            deps.append('PCG_extract_murmur_features')
    elif modality == 'eda':
        if any(term in n for term in ['stress', 'affective', 'arousal', 'summarize']):
            deps.append('EDA_extract_tonic_phasic_features')
    elif modality == 'eeg':
        if any(term in n for term in ['sleep', 'drowsiness', 'seizure', 'artifact']):
            pass
    elif modality == 'emg':
        if any(term in n for term in ['activation', 'fatigue', 'gesture', 'action', 'gait', 'rehab', 'neuromuscular', 'intent']):
            pass
    elif modality == 'abp':
        if any(term in n for term in ['hemodynamics', 'pressure', 'hypotensive', 'shock']):
            deps.append('ABP_detect_fiducial_points')
        if any(term in n for term in ['pressure', 'hypotensive', 'shock']):
            deps.append('ABP_compute_hemodynamics')
    elif modality == 'acc':
        if any(term in n for term in ['activity', 'fall', 'sleep_wake']):
            deps.append('ACC_extract_actigraphy_features')
    if tool_name.startswith('Multimodal_'):
        if 'sleep_apnea' in n:
            deps.extend([])
        if 'ecg_ppg' in n or 'pat' in n:
            deps.extend([])
    return list(dict.fromkeys(dep for dep in deps if dep != tool_name))


def io_semantics_for(tool_name: str, modality: str, level: str) -> tuple[list[str], list[str]]:
    n = tool_name.lower()
    consumes = ['raw_signal'] if level == 'primitive' else ['primitive_events_or_features']
    produces = []
    if 'quality' in n:
        produces.append('signal_quality')
    if any(term in n for term in ['r_peaks', 'j_peaks', 'detect_peaks', 'detect_pulses']) and not (modality == 'spo2' and 'detect_peaks_troughs' in n):
        produces.append('beat_or_pulse_events')
    if modality == 'abp' and 'fiducial' in n:
        produces.extend(['systolic_onset', 'systolic_peak', 'dicrotic_notch', 'diastolic_peak', 'diastolic_phase_endpoint'])
    if modality == 'ppg' and 'fiducial' in n:
        produces.extend(['pulse_onset', 'systolic_peak', 'dicrotic_notch', 'diastolic_peak'])
    if modality == 'pcg' and any(term in n for term in ['detect_heart_sounds', 'segment_s1_s2']):
        produces.extend(['s1_point', 's2_point'])
    if modality == 'scg' and 'fiducial' in n:
        produces.extend(['mc_point', 'im_point', 'ao_point', 'ac_point', 'mo_point'])
    if modality == 'ecg' and ('delineate' in n or 'fiducial' in n):
        produces.extend(['p_wave_peak', 'qrs_complex_peak', 't_wave_peak'])
    if 'hrv' in n:
        produces.extend(['rr_intervals', 'hrv_features'])
    if 'prv' in n:
        produces.extend(['pulse_intervals', 'prv_features'])
    if 'heart_rate' in n or 'estimate_heart_rate' in n:
        produces.append('heart_rate_bpm')
    if modality == 'resp' and 'detect_breath_peaks' in n:
        produces.extend(['inhale_peak', 'exhale_peak'])
    if 'resp' in n or 'apnea' in n or 'hypopnea' in n:
        produces.append('respiratory_features')
    if modality == 'spo2' and 'detect_peaks_troughs' in n:
        produces.extend(['spo2_peak', 'spo2_trough'])
    if 'spo2' in n or 'desaturation' in n or 'oximetry' in n:
        produces.append('oximetry_features')
    if 'bandpower' in n or modality == 'eeg':
        produces.append('eeg_spectral_features')
    if 'stress' in n or modality == 'eda':
        produces.append('autonomic_features')
    if 'murmur' in n or modality == 'pcg':
        produces.append('heart_sound_features')
    if 'digitize' in n or 'image' in n:
        consumes = ['waveform_image']
        produces.append('digitized_signal_or_image_metadata')
    if level == 'screening':
        produces.append('screening_or_task_label')
    if not produces:
        produces.append(f'{modality}_features' if modality else 'features')
    return list(dict.fromkeys(consumes)), list(dict.fromkeys(produces))

def tool_kind(tool_name: str, description: str = '') -> str:
    text = f'{tool_name} {description}'.lower()
    if any(term in text for term in ['_dl', '_cnn', '_ml', 'deep', 'model', 'classifier', 'classify']):
        return 'deep_or_ml'
    if 'proxy' in text or any(term in text for term in ['screen_', 'estimate_', 'risk']):
        return 'proxy_or_rule'
    return 'deterministic_signal_processing'


def evidence_level(tool_name: str, description: str = '', source_entry: dict[str, Any] | None = None) -> str:
    text = f'{tool_name} {description} {json.dumps(source_entry or {})}'.lower()
    if 'proxy' in text:
        return 'proxy'
    if any(term in text for term in ['trained', 'benchmark', 'cv', 'auroc', 'f1', 'mae', 'mit-bih', 'ptb-xl', 'circor', 'springer', 'bmd-hs', 'wesad']):
        return 'benchmarked'
    if any(term in text for term in ['quality', 'detect', 'compute', 'summarize', 'estimate']):
        return 'algorithmic'
    return 'experimental'


def limitation_for(tool_name: str, modality: str, evidence: str) -> str:
    if evidence == 'proxy':
        return 'Research screening/proxy output only; not a clinical diagnosis.'
    if modality.lower() == 'image':
        return 'Image pipeline depends on visible trace, scale/OCR quality, and rendering assumptions.'
    if evidence == 'benchmarked':
        return 'Validated only on listed benchmark/data sources; out-of-distribution signals require quality review.'
    return 'Algorithmic research output; verify signal quality and task assumptions before interpretation.'


def failure_modes_for(tool_name: str, modality: str) -> list[str]:
    modes = ['low_signal_quality', 'artifact_or_motion', 'out_of_distribution_input']
    lower = tool_name.lower()
    if 'digitize' in lower or modality.lower() == 'image':
        modes.extend(['unclear_trace', 'missing_axis_scale', 'low_resolution_image'])
    if 'peak' in lower or 'hrv' in lower:
        modes.extend(['missed_peaks', 'false_peaks'])
    if 'proxy' in lower or 'screen' in lower:
        modes.append('proxy_not_diagnostic')
    return sorted(set(modes))


def source_entries_by_tool(source_catalog: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in source_catalog.get('entries', []):
        for tool in entry.get('current_tools', []):
            out[tool].append(entry)
    return out


def build_tool_universe(schema_path: str | Path, source_catalog_path: str | Path, version: str = 'v1') -> dict[str, Any]:
    schemas = read_json(schema_path)
    source_catalog = read_json(source_catalog_path)
    by_tool = source_entries_by_tool(source_catalog)
    seen = set()
    tools = []
    for schema in schemas:
        name = schema['name']
        if name in seen:
            continue
        seen.add(name)
        entries = by_tool.get(name, [])
        primary = entries[0] if entries else None
        ev = evidence_level(name, schema.get('description', ''), primary)
        kind = tool_kind(name, schema.get('description', ''))
        datasets = sorted({dataset for entry in entries for dataset in entry.get('candidate_datasets', [])})
        urls = sorted({url for entry in entries for url in entry.get('source_urls', [])})
        metrics = extract_metric_snippets(entries)
        task = primary.get('task') if primary else infer_task_from_schema(schema)
        hierarchy = tool_hierarchy_metadata(name, str(schema.get('modality', '')), task, schema.get('description', ''))
        tools.append({
            'name': name,
            'version': version,
            'frozen': True,
            'modality': schema.get('modality'),
            'task': task,
            'description': schema.get('description', ''),
            'parameters': schema.get('parameters', {}),
            'returns': schema.get('returns', []),
            'tool_kind': kind,
            'tool_level': hierarchy['tool_level'],
            'depends_on': hierarchy['depends_on'],
            'consumes': hierarchy['consumes'],
            'produces': hierarchy['produces'],
            'evidence_level': ev,
            'datasets': datasets,
            'metrics': metrics,
            'source_urls': urls,
            'source_catalog_tasks': [entry.get('task') for entry in entries],
            'failure_modes': failure_modes_for(name, str(schema.get('modality', ''))),
            'clinical_limitation': limitation_for(name, str(schema.get('modality', '')), ev),
        })
    modality_counts = Counter(tool['modality'] for tool in tools)
    evidence_counts = Counter(tool['evidence_level'] for tool in tools)
    kind_counts = Counter(tool['tool_kind'] for tool in tools)
    level_counts = Counter(tool['tool_level'] for tool in tools)
    return {
        'artifact': 'BioSignalToolUniverse',
        'version': version,
        'frozen': True,
        'schema_source': str(schema_path),
        'source_catalog_source': str(source_catalog_path),
        'num_tools': len(tools),
        'summary': {
            'tool_count_by_modality': dict(sorted(modality_counts.items(), key=lambda x: str(x[0]))),
            'tool_count_by_evidence_level': dict(sorted(evidence_counts.items())),
            'tool_count_by_kind': dict(sorted(kind_counts.items())),
            'tool_count_by_level': dict(sorted(level_counts.items())),
            'tools_missing_source_metadata': sorted([tool['name'] for tool in tools if not tool['source_catalog_tasks']]),
        },
        'tools': tools,
    }


def infer_task_from_schema(schema: dict[str, Any]) -> str:
    desc = schema.get('description') or ''
    return desc.split('.')[0][:120] if desc else schema['name']


def extract_metric_snippets(entries: list[dict[str, Any]]) -> list[str]:
    snippets = []
    metric_terms = ['f1', 'auroc', 'accuracy', 'mae', 'recall', 'specificity', 'macro', 'weighted', 'cv']
    for entry in entries:
        for text in entry.get('existing_work', []):
            lower = str(text).lower()
            if any(term in lower for term in metric_terms):
                snippets.append(str(text))
    return snippets[:8]


def validate_tool_universe(universe: dict[str, Any]) -> dict[str, Any]:
    errors = []
    names = set()
    required = {'name', 'version', 'frozen', 'modality', 'task', 'parameters', 'returns', 'tool_level', 'depends_on', 'consumes', 'produces', 'evidence_level', 'failure_modes', 'clinical_limitation'}
    for idx, tool in enumerate(universe.get('tools', [])):
        missing = sorted(required - set(tool))
        if missing:
            errors.append({'index': idx, 'tool': tool.get('name'), 'error': f'missing fields: {missing}'})
        if tool.get('name') in names:
            errors.append({'index': idx, 'tool': tool.get('name'), 'error': 'duplicate tool name'})
        names.add(tool.get('name'))
        if not tool.get('source_catalog_tasks'):
            errors.append({'index': idx, 'tool': tool.get('name'), 'error': 'missing source catalog metadata'})
        if not tool.get('clinical_limitation'):
            errors.append({'index': idx, 'tool': tool.get('name'), 'error': 'missing clinical limitation'})
        if not tool.get('failure_modes'):
            errors.append({'index': idx, 'tool': tool.get('name'), 'error': 'missing failure modes'})
        if tool.get('tool_level') not in {'primitive', 'representation', 'screening'}:
            errors.append({'index': idx, 'tool': tool.get('name'), 'error': 'invalid tool_level'})
        if not isinstance(tool.get('depends_on'), list):
            errors.append({'index': idx, 'tool': tool.get('name'), 'error': 'depends_on must be list'})
        if not isinstance(tool.get('consumes'), list) or not tool.get('consumes'):
            errors.append({'index': idx, 'tool': tool.get('name'), 'error': 'consumes must be non-empty list'})
        if not isinstance(tool.get('produces'), list) or not tool.get('produces'):
            errors.append({'index': idx, 'tool': tool.get('name'), 'error': 'produces must be non-empty list'})
    return {
        'artifact': universe.get('artifact'),
        'version': universe.get('version'),
        'num_tools': len(universe.get('tools', [])),
        'num_errors': len(errors),
        'errors': errors,
        'summary': universe.get('summary', {}),
    }


def load_bench_cases(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def validate_bench_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    errors = []
    ids = set()
    required = {'case_id', 'benchmark_task', 'question', 'input_type', 'modality', 'expected_tools', 'ground_truth_metric', 'source'}
    for idx, case in enumerate(cases):
        missing = sorted(required - set(case))
        if missing:
            errors.append({'index': idx, 'case_id': case.get('case_id'), 'error': f'missing fields: {missing}'})
        if case.get('case_id') in ids:
            errors.append({'index': idx, 'case_id': case.get('case_id'), 'error': 'duplicate case_id'})
        ids.add(case.get('case_id'))
        if not isinstance(case.get('expected_tools'), list) or not case.get('expected_tools'):
            errors.append({'index': idx, 'case_id': case.get('case_id'), 'error': 'expected_tools must be non-empty list'})
        for tool in case.get('expected_tools', []):
            if tool not in TOOLS:
                errors.append({'index': idx, 'case_id': case.get('case_id'), 'error': f'unknown expected tool: {tool}'})
        if case.get('input_type') not in {'csv', 'image', 'session', 'text'}:
            errors.append({'index': idx, 'case_id': case.get('case_id'), 'error': 'invalid input_type'})
    task_counts = Counter(case.get('benchmark_task') for case in cases)
    input_counts = Counter(case.get('input_type') for case in cases)
    modality_counts = Counter(str(case.get('modality')).lower() for case in cases)
    return {
        'num_cases': len(cases),
        'num_errors': len(errors),
        'task_counts': dict(sorted(task_counts.items())),
        'input_type_counts': dict(sorted(input_counts.items())),
        'modality_counts': dict(sorted(modality_counts.items())),
        'errors': errors,
    }


def tokenize(text: str) -> set[str]:
    return {tok.lower() for tok in TOKEN_RE.findall(text.replace('-', '_'))}


class TraceReplayPlanner:
    def __init__(self, sft_paths: list[str | Path]):
        self.index: dict[str, list[str]] = {}
        for path in sft_paths:
            for row in read_jsonl(path):
                if row.get('task') not in {'biosignal_tool_planning', 'biosignal_session_tool_planning'}:
                    continue
                messages = row.get('messages', [])
                user = next((msg.get('content', '') for msg in messages if msg.get('role') == 'user'), '')
                assistant = next((msg.get('content', '') for msg in messages if msg.get('role') == 'assistant'), '')
                try:
                    user_payload = json.loads(user)
                    answer = json.loads(assistant)
                except Exception:
                    continue
                question = str(user_payload.get('question', '')).strip().lower()
                tools = [call.get('name') for call in answer.get('tool_calls', []) if call.get('name')]
                if not tools and answer.get('signal_plans'):
                    tools = [call.get('name') for plan in answer.get('signal_plans', []) for call in plan.get('tool_calls', []) if call.get('name')]
                if question and tools:
                    self.index.setdefault(question, tools)

    def plan(self, question: str, fallback_modality: str | None = None) -> list[str]:
        exact = self.index.get(question.strip().lower())
        if exact:
            return list(dict.fromkeys(exact))
        q_tokens = tokenize(question)
        best_score = 0
        best_tools: list[str] = []
        for known, tools in self.index.items():
            score = len(q_tokens & tokenize(known))
            if score > best_score:
                best_score = score
                best_tools = tools
        return list(dict.fromkeys(best_tools)) if best_score >= 3 else []


def evaluate_benchmark_cases(
    cases: list[dict[str, Any]],
    planner_backend: str = 'rule',
    retriever_backend: str = 'tfidf',
    retrieved_tool_count: int = 7,
    execute: bool = False,
    sft_paths: list[str | Path] | None = None,
    ablation_flags: dict[str, bool] | None = None,
) -> dict[str, Any]:
    ablation_flags = ablation_flags or {}
    retriever = ToolRetriever()
    rule_agent = PlanningBioSignalAgent()
    replay = TraceReplayPlanner(sft_paths or []) if planner_backend in {'sft_replay', 'sft_planner', 'sft_report'} else None
    rows = []
    for case in cases:
        expected = list(case.get('expected_tools', []))
        question = case.get('question', '')
        modality = str(case.get('modality', '')).lower()
        retrieved = retrieve_tools_for_case(case, retriever, retriever_backend, retrieved_tool_count, ablation_flags)
        planned, planner_error = plan_tools_for_case(case, rule_agent, replay, planner_backend, ablation_flags, retrieved)
        retrieval_missing = sorted(set(expected) - set(retrieved))
        plan_missing = sorted(set(expected) - set(planned))
        unexpected = sorted(set(planned) - set(expected))
        tool_precision, tool_recall, tool_f1 = tool_set_scores(expected, planned)
        execution_ok = None
        execution_errors: list[str] = []
        if execute and case.get('input_type') == 'csv':
            execution_ok, execution_errors = execute_case(case, planned)
        elif execute:
            execution_ok = None
            execution_errors = ['execution_skipped_for_non_csv_or_session_case']
        row = {
            'case_id': case.get('case_id'),
            'benchmark_task': case.get('benchmark_task'),
            'input_type': case.get('input_type'),
            'modality': modality,
            'question': question,
            'expected_tools': expected,
            'retrieved_tools': retrieved,
            'planned_tools': planned,
            'retrieval_pass': not retrieval_missing,
            'planning_pass': not plan_missing and not unexpected,
            'tool_precision': tool_precision,
            'tool_recall': tool_recall,
            'tool_f1': tool_f1,
            'execution_ok': execution_ok,
            'missing_from_retrieval': retrieval_missing,
            'missing_from_plan': plan_missing,
            'unexpected_tools': unexpected,
            'execution_errors': execution_errors,
            'planner_error': planner_error,
            'failure_reason': failure_reason(case, retrieval_missing, plan_missing, unexpected, execution_ok, execution_errors, planner_error, ablation_flags),
        }
        rows.append(row)
    return summarize_eval(rows, planner_backend, retriever_backend, retrieved_tool_count, execute, ablation_flags)


def retrieve_tools_for_case(case: dict[str, Any], retriever: ToolRetriever, backend: str, top_k: int, ablation_flags: dict[str, bool]) -> list[str]:
    if backend in {'none', 'disabled'} or ablation_flags.get('no_toolrag'):
        return []
    if backend == 'oracle':
        return list(case.get('expected_tools', []))
    schemas = retriever.retrieve(case.get('question', ''), top_k=top_k, modality=str(case.get('modality', '')).lower())
    lexical_names = [schema['name'] for schema in schemas]
    # BioSignalBench cases carry structured input metadata that a ToolUniverse
    # retriever should use: image cases need image tools even when the signal
    # modality is ECG/PPG/etc., and session cases need per-modality tools.
    # Put metadata priors first: these are high-confidence routing signals,
    # while lexical TF-IDF remains useful for task-specific long-tail tools.
    names = metadata_prior_tools_for_case(case) + lexical_names
    return apply_tool_ablations(list(dict.fromkeys(names)), ablation_flags)


def metadata_prior_tools_for_case(case: dict[str, Any]) -> list[str]:
    input_type = str(case.get('input_type', '')).lower()
    task = str(case.get('benchmark_task', '')).lower()
    question = str(case.get('question', '')).lower()
    modality = str(case.get('modality', '')).lower()
    tools: list[str] = []
    if input_type == 'image' or task in {'image_to_signal_digitization', 'scale_ocr_extraction'}:
        if any(term in question or term in task for term in ['digitize', 'digital', 'waveform', 'csv', 'image_to_signal']):
            tools.extend(['Signal_classify_modality_from_image_cnn', 'Signal_digitize_waveform_image_ml'])
        if any(term in question or term in task for term in ['scale', 'axis', 'ocr']):
            tools.extend(['Signal_estimate_image_scale', 'Signal_predict_image_scale_prior'])
    if input_type == 'text' and ('unknown' in modality or 'modality' in question):
        tools.append('Signal_classify_modality')
    if modality in WORKFLOWS:
        tools.extend(session_prior_tools(modality, question))
    if input_type == 'session' or '+' in modality:
        parts = [part.strip() for part in modality.split('+') if part.strip()]
        for part in parts:
            tools.extend(session_prior_tools(part, question))
        if any(term in question for term in ['pat', 'pulse arrival', 'blood pressure', 'bp']):
            tools.append('Multimodal_estimate_ecg_ppg_pat_bp_proxy')
        if any(term in question for term in ['apnea', 'sleep', 'desaturation', 'spo2']):
            tools.append('Multimodal_screen_sleep_apnea_report')
    return [tool for tool in dict.fromkeys(tools) if tool in TOOLS]


def session_prior_tools(modality: str, question: str) -> list[str]:
    base = WORKFLOWS.get(modality, [])
    if not base:
        return []
    selected: list[str] = []
    core_by_modality = {
        'ecg': ['ECG_assess_quality', 'ECG_compute_hrv'],
        'ppg': ['PPG_assess_quality'],
        'bcg': ['BCG_assess_quality', 'BCG_detect_j_peaks'],
        'scg': ['SCG_assess_quality'],
        'resp': ['RESP_assess_quality', 'RESP_estimate_rate'],
        'spo2': ['SpO2_assess_quality', 'SpO2_summarize'],
        'abp': ['ABP_assess_quality', 'ABP_detect_fiducial_points'],
        'pcg': ['PCG_assess_quality', 'PCG_detect_heart_sounds'],
        'acc': ['ACC_assess_quality', 'ACC_summarize_activity'],
        'eda': ['EDA_assess_quality', 'EDA_summarize'],
        'eeg': ['EEG_assess_quality'],
        'emg': ['EMG_assess_quality', 'EMG_summarize_activation'],
    }
    selected.extend([tool for tool in core_by_modality.get(modality, []) if tool in base])
    quality = next((tool for tool in base if tool.endswith('_assess_quality')), None)
    if quality:
        selected.append(quality)
    if 'quality' in question or 'confidence' in question:
        selected.extend([tool for tool in base if tool.endswith('_assess_quality')])
    if any(term in question for term in ['heart rate', 'hr ', 'hrv', 'beat', 'peak', 'confidence']):
        selected.extend([tool for tool in base if any(key in tool.lower() for key in ['detect_r_peaks', 'detect_peaks', 'detect_j_peaks', 'heart_rate', 'compute_hrv'])])
    if any(term in question for term in ['resp', 'breath', 'apnea', 'sleep']):
        selected.extend([tool for tool in base if any(key in tool.lower() for key in ['resp', 'apnea', 'sleep', 'desaturation'])])
    if any(term in question for term in ['pressure', 'bp', 'hemodynamic', 'hypotension', 'shock']):
        selected.extend([tool for tool in base if any(key in tool.lower() for key in ['pressure', 'bp', 'hemodynamic', 'hypotensive', 'shock'])])
    if any(term in question for term in ['murmur', 'sound', 's1', 's2', 'valve']):
        selected.extend([tool for tool in base if any(key in tool.lower() for key in ['heart_sound', 'murmur', 's1_s2', 'valve'])])
    if any(term in question for term in ['arrhythmia', 'afib', 'rhythm', 'irregular']):
        selected.extend([tool for tool in base if any(key in tool.lower() for key in ['arrhythmia', 'afib', 'rhythm', 'irregular'])])
    if any(term in question for term in ['stress', 'arousal', 'activity', 'sleep stage', 'seizure', 'gesture', 'fatigue']):
        q_tokens = tokenize(question)
        selected.extend([tool for tool in base if q_tokens & tokenize(tool)])
    return list(dict.fromkeys(selected))


def plan_tools_for_case(case: dict[str, Any], rule_agent: PlanningBioSignalAgent, replay: TraceReplayPlanner | None, backend: str, ablation_flags: dict[str, bool], retrieved_tools: list[str] | None = None) -> tuple[list[str], str | None]:
    if backend in {'none', 'no_tool_llm'} or ablation_flags.get('no_sft') and backend.startswith('sft'):
        return [], None
    if backend == 'oracle':
        return apply_tool_ablations(list(case.get('expected_tools', [])), ablation_flags), None
    if backend == 'toolrag':
        return apply_tool_ablations(list(retrieved_tools or []), ablation_flags), None
    if backend in {'sft_replay', 'sft_planner', 'sft_report'} and replay is not None:
        tools = replay.plan(case.get('question', ''), str(case.get('modality', '')).lower())
        if tools:
            return apply_tool_ablations(tools, ablation_flags), None
        fallback = normalized_rule_modality(case.get('modality'))
        if fallback is None:
            return [], 'sft_replay_miss_no_rule_modality'
        try:
            return apply_tool_ablations(rule_agent.plan(case.get('question', ''), fallback), ablation_flags), 'sft_replay_miss_rule_fallback'
        except Exception as exc:
            return [], f'sft_replay_miss_rule_error:{exc}'
    try:
        fallback = normalized_rule_modality(case.get('modality'))
        return apply_tool_ablations(rule_agent.plan(case.get('question', ''), fallback), ablation_flags), None
    except Exception as exc:
        return [], str(exc)


def normalized_rule_modality(modality: Any) -> str | None:
    text = str(modality or '').lower()
    if text in WORKFLOWS:
        return text
    for part in text.split('+'):
        if part in WORKFLOWS:
            return part
    return None


def apply_tool_ablations(tools: list[str], flags: dict[str, bool]) -> list[str]:
    out = []
    for tool in tools:
        lower = tool.lower()
        if flags.get('no_modality_classifier') and 'classify_modality' in lower:
            continue
        if flags.get('no_ocr_scale') and ('scale' in lower or 'ocr' in lower):
            continue
        if flags.get('no_image_digitization') and 'digitize' in lower:
            continue
        if flags.get('no_quality_gate') and 'assess_quality' in lower:
            continue
        if flags.get('no_dl_tools') and any(term in lower for term in ['_dl', '_cnn', '_ml', 'classify']):
            continue
        out.append(tool)
    return list(dict.fromkeys(out))


def execute_case(case: dict[str, Any], planned: list[str]) -> tuple[bool, list[str]]:
    signal = case.get('signal') or {}
    path = signal.get('path') or case.get('signal_path')
    sampling_rate = signal.get('sampling_rate') or case.get('sampling_rate')
    column = signal.get('column')
    errors = []
    if not path or not sampling_rate:
        return False, ['missing_signal_path_or_sampling_rate']
    ok = True
    for tool in planned:
        func = TOOLS.get(tool)
        if func is None:
            ok = False
            errors.append(f'{tool}:unknown_tool')
            continue
        try:
            result = func(signal_path=path, sampling_rate=float(sampling_rate), column=column)
            if isinstance(result, dict) and result.get('error'):
                ok = False
                errors.append(f'{tool}:{result.get("error")}')
        except TypeError as exc:
            ok = False
            errors.append(f'{tool}:unsupported_signature:{exc}')
        except Exception as exc:
            ok = False
            errors.append(f'{tool}:{type(exc).__name__}:{str(exc)[:120]}')
    return ok, errors


def failure_reason(case: dict[str, Any], retrieval_missing: list[str], plan_missing: list[str], unexpected: list[str], execution_ok: bool | None, execution_errors: list[str], planner_error: str | None, ablation_flags: dict[str, bool] | None = None) -> str | None:
    ablation_flags = ablation_flags or {}
    if planner_error:
        return 'planner_error'
    if retrieval_missing:
        return categorize_missing_tools(retrieval_missing, case, 'retrieval', ablation_flags)
    if plan_missing:
        return categorize_missing_tools(plan_missing, case, 'planning', ablation_flags)
    if unexpected:
        return 'planning_unexpected_tools'
    if execution_ok is False:
        if execution_errors and any('unsupported_signature' in err for err in execution_errors):
            return 'execution_signature_mismatch'
        if execution_errors and any('missing_signal_path' in err for err in execution_errors):
            return 'execution_missing_input'
        return 'execution_failed'
    if execution_errors:
        return 'execution_skipped_or_partial'
    return None


def categorize_missing_tools(missing: list[str], case: dict[str, Any], stage: str, ablation_flags: dict[str, bool]) -> str:
    lower = ' '.join(missing).lower()
    input_type = str(case.get('input_type', '')).lower()
    task = str(case.get('benchmark_task', '')).lower()
    if ablation_flags.get('no_toolrag') and stage == 'retrieval':
        return 'ablation_toolrag_disabled'
    if ablation_flags.get('no_modality_classifier') and 'classify_modality' in lower:
        return 'ablation_modality_classifier_disabled'
    if ablation_flags.get('no_ocr_scale') and ('scale' in lower or 'ocr' in lower):
        return 'ablation_ocr_scale_disabled'
    if ablation_flags.get('no_image_digitization') and 'digitize' in lower:
        return 'ablation_image_digitization_disabled'
    if ablation_flags.get('no_quality_gate') and 'assess_quality' in lower:
        return 'ablation_quality_gate_disabled'
    if ablation_flags.get('no_dl_tools') and any(term in lower for term in ['_dl', '_cnn', '_ml', 'classify']):
        return 'ablation_dl_tools_disabled'
    if 'classify_modality' in lower:
        return f'{stage}_modality_router_missing'
    if 'scale' in lower or 'ocr' in lower or 'axis' in lower:
        return f'{stage}_scale_ocr_missing'
    if 'digitize' in lower or task == 'image_to_signal_digitization' or input_type == 'image':
        return f'{stage}_image_digitization_missing'
    if 'assess_quality' in lower or 'quality' in lower:
        return f'{stage}_quality_gate_missing'
    if 'proxy' in lower or 'screen' in lower:
        return f'{stage}_screening_or_proxy_tool_missing'
    if '+' in str(case.get('modality', '')) or input_type == 'session':
        return f'{stage}_multimodal_tool_missing'
    return f'{stage}_expected_tool_missing'


def summarize_eval(rows: list[dict[str, Any]], planner_backend: str, retriever_backend: str, top_k: int, execute: bool, ablation_flags: dict[str, bool]) -> dict[str, Any]:
    n = len(rows)
    executable = [row for row in rows if row.get('execution_ok') is not None]
    failure_counts = Counter(row.get('failure_reason') for row in rows if row.get('failure_reason'))
    task_counts = Counter(row.get('benchmark_task') for row in rows)
    input_counts = Counter(row.get('input_type') for row in rows)
    modality_counts = Counter(row.get('modality') for row in rows)
    by_task = {}
    for task in sorted(task_counts):
        subset = [row for row in rows if row.get('benchmark_task') == task]
        by_task[task] = {
            'num_cases': len(subset),
            'retrieval_accuracy': mean_bool(row['retrieval_pass'] for row in subset),
            'planning_accuracy': mean_bool(row['planning_pass'] for row in subset),
            'execution_accuracy': mean_bool(row['execution_ok'] for row in subset if row['execution_ok'] is not None) if any(row['execution_ok'] is not None for row in subset) else None,
            'tool_precision': mean_float(row['tool_precision'] for row in subset),
            'tool_recall': mean_float(row['tool_recall'] for row in subset),
            'tool_f1': mean_float(row['tool_f1'] for row in subset),
        }
    by_input_type = {}
    for input_type in sorted(input_counts):
        subset = [row for row in rows if row.get('input_type') == input_type]
        by_input_type[input_type] = {
            'num_cases': len(subset),
            'retrieval_accuracy': mean_bool(row['retrieval_pass'] for row in subset),
            'planning_accuracy': mean_bool(row['planning_pass'] for row in subset),
            'execution_accuracy': mean_bool(row['execution_ok'] for row in subset if row['execution_ok'] is not None) if any(row['execution_ok'] is not None for row in subset) else None,
            'tool_precision': mean_float(row['tool_precision'] for row in subset),
            'tool_recall': mean_float(row['tool_recall'] for row in subset),
            'tool_f1': mean_float(row['tool_f1'] for row in subset),
        }
    by_modality = {}
    for modality in sorted(modality_counts):
        subset = [row for row in rows if row.get('modality') == modality]
        by_modality[modality] = {
            'num_cases': len(subset),
            'retrieval_accuracy': mean_bool(row['retrieval_pass'] for row in subset),
            'planning_accuracy': mean_bool(row['planning_pass'] for row in subset),
            'execution_accuracy': mean_bool(row['execution_ok'] for row in subset if row['execution_ok'] is not None) if any(row['execution_ok'] is not None for row in subset) else None,
            'tool_precision': mean_float(row['tool_precision'] for row in subset),
            'tool_recall': mean_float(row['tool_recall'] for row in subset),
            'tool_f1': mean_float(row['tool_f1'] for row in subset),
        }
    return {
        'artifact': 'BioSignalBenchEvaluation',
        'planner_backend': planner_backend,
        'retriever_backend': retriever_backend,
        'retrieved_tool_count': top_k,
        'execute': execute,
        'ablation_flags': ablation_flags,
        'num_cases': n,
        'retrieval_accuracy': mean_bool(row['retrieval_pass'] for row in rows),
        'planning_accuracy': mean_bool(row['planning_pass'] for row in rows),
        'tool_precision': mean_float(row['tool_precision'] for row in rows),
        'tool_recall': mean_float(row['tool_recall'] for row in rows),
        'tool_f1': mean_float(row['tool_f1'] for row in rows),
        'execution_accuracy': mean_bool(row['execution_ok'] for row in executable) if executable else None,
        'failure_reason_counts': dict(sorted(failure_counts.items())),
        'by_task': by_task,
        'by_input_type': by_input_type,
        'by_modality': by_modality,
        'cases': rows,
    }


def tool_set_scores(expected: list[str], planned: list[str]) -> tuple[float, float, float]:
    expected_set = set(expected)
    planned_set = set(planned)
    if not expected_set and not planned_set:
        return 1.0, 1.0, 1.0
    if not planned_set:
        return 0.0, 0.0, 0.0
    true_positive = len(expected_set & planned_set)
    precision = true_positive / len(planned_set) if planned_set else 0.0
    recall = true_positive / len(expected_set) if expected_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def mean_bool(values: Any) -> float:
    vals = [bool(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def mean_float(values: Any) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f'{value:.3f}'
        if value is None:
            return ''
        return str(value)
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join(fmt(value) for value in row) + ' |')
    return '\n'.join(lines) + '\n'
