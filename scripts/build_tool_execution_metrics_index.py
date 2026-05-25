#!/usr/bin/env python3
"""Build a paper-ready index of BioSignalAgent tool execution metrics.

The controller tables measure routing/planning/reporting. This script collects the
best available numeric evidence for the underlying signal tools from existing
training/evaluation artifacts, preserving whether the evidence is validated,
proxy, or smoke-test only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

OUTPUTS = Path('/data1/jiahui/biosignal-agent/outputs')
PAPER_DIR = OUTPUTS / 'paper_tables'


def load_json(path: str | Path) -> Any | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open(newline='') as f:
        return list(csv.DictReader(f))


def fnum(x: Any) -> float | None:
    try:
        if x is None or x == '':
            return None
        return float(x)
    except Exception:
        return None


def fmt(x: Any) -> str:
    if x is None or x == '':
        return 'NA'
    if isinstance(x, float):
        if abs(x) >= 100:
            return f'{x:.1f}'
        return f'{x:.3f}'
    return str(x)


def add(rows: list[dict[str, Any]], *, modality: str, task: str, tool: str, backend: str,
        dataset: str, n: Any, metric: str, value: Any, evidence: str,
        artifact: str, note: str = '') -> None:
    rows.append({
        'modality': modality,
        'task': task,
        'tool': tool,
        'backend': backend,
        'dataset': dataset,
        'n': n,
        'metric': metric,
        'value': value,
        'evidence_level': evidence,
        'artifact': artifact,
        'note': note,
    })


def collect() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # ECG: R peaks on MIT-BIH. Prefer the deep model tool eval; fall back to CSV summaries.
    rpeak = load_json(OUTPUTS / 'mitdb_rpeak_tool_eval_5min.json')
    if rpeak:
        m = rpeak.get('aggregate', {})
        add(rows, modality='ECG', task='R-peak/QRS detection', tool='ECG_detect_r_peaks',
            backend='deep R-peak segmentation CNN', dataset='MIT-BIH Arrhythmia, first 300s/record', n=rpeak.get('num_records'),
            metric='F1 / precision / recall', value=f"{fmt(m.get('f1'))} / {fmt(m.get('precision'))} / {fmt(m.get('recall'))}", evidence='validated',
            artifact=str(OUTPUTS / 'mitdb_rpeak_tool_eval_5min.json'),
            note='Reference beat annotations; 100 ms tolerance.')
    else:
        mit = load_csv_rows(OUTPUTS / 'mitdb_pantompkins_60s_all_summary.csv') or load_csv_rows(OUTPUTS / 'mitdb_nk2_60s_all_summary.csv')
        if mit:
            f1s = [fnum(r.get('f1')) for r in mit]
            add(rows, modality='ECG', task='R-peak/QRS detection', tool='ECG_detect_r_peaks',
                backend='rule-based detector', dataset='MIT-BIH Arrhythmia, 60s windows', n=len(mit),
                metric='mean F1', value=mean([x for x in f1s if x is not None]), evidence='validated',
                artifact=str(OUTPUTS / 'mitdb_pantompkins_60s_all_summary.csv'),
                note='Reference beat annotations; 100 ms tolerance.')

    # ECG: 12-lead PTB-XL superclass classifier.
    ptb = load_json(OUTPUTS / 'ptbxl_full_12lead_resnet/ecg_ptbxl_full_12lead_resnet_train_report.json')
    if ptb:
        aps = []
        aucs = []
        f1s = []
        n_eval = None
        for name, target in sorted(ptb.get('targets', {}).items()):
            m = target.get('cv_metrics', {})
            aps.append(m.get('average_precision'))
            aucs.append(m.get('roc_auc'))
            f1s.append(m.get('f1'))
            n_eval = n_eval or m.get('eval_records')
        add(rows, modality='ECG', task='12-lead superclass diagnosis', tool='ECG_classify_12lead_ptbxl_superclasses',
            backend='12-lead ResNet8', dataset='PTB-XL fold 10', n=n_eval or ptb.get('num_records'),
            metric='macro AP / AUROC / F1',
            value=' / '.join(fmt(mean([x for x in vals if x is not None])) for vals in [aps, aucs, f1s]),
            evidence='validated', artifact=str(OUTPUTS / 'ptbxl_full_12lead_resnet/ecg_ptbxl_full_12lead_resnet_train_report.json'),
            note='Targets: CD, HYP, MI, NORM, STTC.')

    arr = load_json(OUTPUTS / 'ecg_arrhythmia_aami_multidb_cnn_train_report.json') or load_json(OUTPUTS / 'labeled_arrhythmia_eval_after_rpeak_threshold095.json') or load_json(OUTPUTS / 'labeled_arrhythmia_eval.json')
    if arr:
        m = arr.get('cv_metrics') or arr.get('metrics') or arr.get('test_metrics') or arr.get('val_metrics') or arr
        add(rows, modality='ECG', task='arrhythmia / abnormal rhythm screening', tool='ECG_screen_arrhythmia',
            backend=arr.get('model', 'CNN/proxy backend'), dataset='MIT-BIH/INCART/SVDB or labeled arrhythmia eval',
            n=arr.get('num_beats') or arr.get('num_windows') or arr.get('num_examples') or arr.get('num_records'), metric='binary F1 / macro F1', value=f"{fmt(m.get('binary_f1') or m.get('f1'))} / {fmt(m.get('macro_f1'))}",
            evidence='validated' if arr.get('cv_metrics') else 'proxy', artifact=str(OUTPUTS / 'ecg_arrhythmia_aami_multidb_cnn_train_report.json'),
            note='Uses latest available arrhythmia report if present; proxy fallback is not diagnostic.')

    apn = load_json(OUTPUTS / 'ecg_apnea_rri_amp_edr_cnn_transformer_lstm_apnea_only_report.json') or load_json(OUTPUTS / 'apnea_ecg_eval.json')
    if apn:
        m = apn.get('cv_metrics') or apn.get('best_metrics') or apn.get('test_metrics') or apn.get('metrics') or apn
        add(rows, modality='ECG', task='sleep apnea screening', tool='ECG_screen_sleep_apnea',
            backend=apn.get('model', 'RRI/EDR CNN-Transformer-LSTM or proxy'), dataset=apn.get('dataset', 'Apnea-ECG/UCDB/SLPDB eval'),
            n=apn.get('num_windows') or apn.get('num_records'), metric='F1/AUROC',
            value='/'.join(fmt(m.get(k)) for k in ['f1', 'roc_auc'] if m.get(k) is not None) or fmt(m.get('accuracy')),
            evidence='validated' if m.get('roc_auc') is not None else 'proxy', artifact=str(OUTPUTS / 'ecg_apnea_rri_amp_edr_cnn_transformer_lstm_apnea_only_report.json'),
            note='Long-context ECG-derived respiration/RRI evidence when available.')

    qt = load_json(OUTPUTS / 'ecg_delineation_qtdb_cached90_event_eval.json')
    if qt:
        res = qt.get('results', {})
        add(rows, modality='ECG', task='P/QRS/T delineation and interval morphology', tool='ECG_measure_morphology_intervals',
            backend='QTDB U-Net event delineator', dataset='QTDB manual delineation held-out records', n=qt.get('num_windows'),
            metric='macro event F1', value=res.get('macro_event_f1'), evidence='validated',
            artifact=str(OUTPUTS / 'ecg_delineation_qtdb_cached90_event_eval.json'),
            note='T-wave performance remains the weakest component.')

    # PPG.
    ppg_peak = load_json(OUTPUTS / 'ppg_peak_unet_capnobase_report.json')
    if ppg_peak:
        m = ppg_peak.get('val_metrics', {})
        add(rows, modality='PPG', task='pulse peak detection', tool='PPG_detect_peaks', backend=ppg_peak.get('model', '1D U-Net'),
            dataset='CapnoBase direct pleth peak labels', n=ppg_peak.get('val_records'), metric='F1 / median timing error ms',
            value=f"{fmt(m.get('mean_f1'))} / {fmt(m.get('median_abs_timing_error_ms'))}", evidence='validated',
            artifact=str(OUTPUTS / 'ppg_peak_unet_capnobase_report.json'), note='Uses direct PPG labels, not MIMIC proxy labels.')
    ppg_resp = load_json(OUTPUTS / 'ppg_respiration_candidate_selector_multidb_report.json')
    if ppg_resp:
        m = ppg_resp.get('best_cv_metrics', {})
        add(rows, modality='PPG', task='respiration rate from modulation', tool='PPG_estimate_respiration_modulation',
            backend=ppg_resp.get('best_model', 'candidate selector'), dataset='BIDMC + CapnoBase + Aeration', n=ppg_resp.get('num_records'),
            metric='MAE bpm', value=m.get('mae_bpm'), evidence='validated', artifact=str(OUTPUTS / 'ppg_respiration_candidate_selector_multidb_report.json'),
            note=f"Current rule MAE {fmt(ppg_resp.get('current_tool_mae_bpm'))} bpm.")
    for p, task, tool in [
        ('ppg_af_interval_attention_bilstm_report.json', 'AF / pulse irregularity screening', 'PPG_screen_afib'),
        ('ppg_signal_quality_capnobase_classifier_report.json', 'signal quality assessment', 'PPG_assess_quality'),
    ]:
        d = load_json(OUTPUTS / p)
        if d:
            m = d.get('cv_metrics_record_level') or d.get('cv_metrics') or d.get('test_metrics') or d.get('val_metrics') or d.get('metrics') or d.get('best_cv_metrics') or d
            add(rows, modality='PPG', task=task, tool=tool, backend=d.get('model', 'trained classifier'), dataset=d.get('dataset', 'PPG benchmark'),
                n=d.get('num_records') or d.get('num_windows') or d.get('training_windows'), metric='macro F1/AUROC',
                value='/'.join(fmt(m.get(k)) for k in ['macro_f1', 'f1', 'auroc', 'roc_auc'] if m.get(k) is not None), evidence='validated', artifact=str(OUTPUTS / p))

    # PCG.
    pcg = load_json(OUTPUTS / 'pcg_murmur_sota_stack_balanced_e6_s1_report.json') or load_json(OUTPUTS / 'pcg_murmur_patient_multiloc_multitask_cnn_e12_report.json')
    if pcg:
        m = pcg.get('cv_metrics') or pcg.get('patient_metrics') or pcg.get('val_metrics') or pcg.get('metrics') or pcg
        add(rows, modality='PCG', task='murmur detection', tool='PCG_detect_murmur', backend=pcg.get('model', 'spectrogram CNN / SOTA stack'),
            dataset=pcg.get('dataset', 'CirCor 2022'), n=pcg.get('num_patients') or pcg.get('num_records'), metric='F1/AUROC/weighted accuracy',
            value='/'.join(fmt(m.get(k)) for k in ['macro_f1', 'ovr_auroc', 'weighted_murmur_accuracy'] if m.get(k) is not None), evidence='validated',
            artifact=str(OUTPUTS / 'pcg_murmur_sota_stack_balanced_e6_s1_report.json'))
    pcg_seg = load_json(OUTPUTS / 'pcg_springer_segmentation_tcn_postprocess_report.json') or load_json(OUTPUTS / 'pcg_springer_segmentation_tcn_report.json')
    if pcg_seg:
        best_seg = pcg_seg.get('best') or pcg_seg
        s1 = (best_seg.get('s1_summary') or {}).get('micro_f1')
        s2 = (best_seg.get('s2_summary') or {}).get('micro_f1')
        add(rows, modality='PCG', task='S1/S2 segmentation and HR', tool='PCG_segment_heart_sounds', backend='Springer segmentation TCN',
            dataset='PhysioNet/CinC heart sound segmentation', n=pcg_seg.get('num_val_records') or pcg_seg.get('num_records'), metric='S1/S2 F1 / HR MAE bpm',
            value=f"{fmt(s1)}/{fmt(s2)} / {fmt(best_seg.get('heart_rate_mae_bpm'))}", evidence='validated',
            artifact=str(OUTPUTS / 'pcg_springer_segmentation_tcn_postprocess_report.json'))
    valve = load_json(OUTPUTS / 'pcg_bmdhs_valve_spectrogram_cnn_fold4_report.json') or load_json(OUTPUTS / 'pcg_bmdhs_valve_spectrogram_cnn_report.json')
    if valve:
        bm = valve.get('best_metrics')
        if isinstance(bm, dict):
            vals = [x for x in bm.values() if isinstance(x, dict)]
            m = {
                'macro_f1': mean([x.get('f1') for x in vals if x.get('f1') is not None]),
                'auroc': mean([x.get('roc_auc') for x in vals if x.get('roc_auc') is not None]),
            }
        else:
            m = valve.get('test_metrics') or valve.get('val_metrics') or valve.get('metrics') or valve
        add(rows, modality='PCG', task='valve disease screening', tool='PCG_screen_valve_disease_proxy', backend='BMD-HS spectrogram CNN',
            dataset=valve.get('dataset', 'BMD-HS'), n=valve.get('num_val_records') or valve.get('num_records') or valve.get('num_windows'), metric='macro F1/AUROC',
            value='/'.join(fmt(m.get(k)) for k in ['macro_f1', 'f1', 'auroc', 'roc_auc'] if m.get(k) is not None), evidence='validated/proxy', artifact=str(OUTPUTS / 'pcg_bmdhs_valve_spectrogram_cnn_fold4_report.json'))

    # SCG.
    for p, task, tool, metric_keys in [
        ('scg_cebs_hybrid_baseline_unet_b001_b020_report.json', 'heartbeat/J-peak detection', 'SCG_detect_heartbeats', ['f1', 'count_hr_mae_bpm', 'interval_hr_mae_bpm']),
        ('scg_vhd_ao_scg_free_cnn_postprocess_report.json', 'AO timing estimation', 'SCG_estimate_aortic_opening', ['f1', 'mae_ms', 'heart_rate_mae_bpm']),
        ('scg_rhc_hf_sota_report.json', 'heart failure/hemodynamic monitoring', 'SCG_monitor_heart_function', ['auroc', 'macro_f1', 'balanced_accuracy']),
        ('scg_arrhythmia_related_mscardio_encoder_downstream_report.json', 'arrhythmia-related SCG screening', 'SCG_screen_arrhythmia_related', ['auroc', 'macro_f1', 'accuracy']),
    ]:
        d = load_json(OUTPUTS / p)
        if d:
            if p == 'scg_rhc_hf_sota_report.json':
                selected = (d.get('feature_model', {}).get('elevated_pcwp', {}) or {}).get('selected', {})
                m = {'auroc': selected.get('auroc'), 'macro_f1': selected.get('best_f1'), 'balanced_accuracy': selected.get('positive_rate')}
            elif p == 'scg_arrhythmia_related_mscardio_encoder_downstream_report.json':
                oof = d.get('oof', {})
                m = {'auroc': oof.get('auroc'), 'macro_f1': oof.get('f1_at_0_5'), 'accuracy': oof.get('accuracy')}
            else:
                m = d.get('test_hybrid') or d.get('test') or d.get('cv_metrics') or d.get('overall') or d.get('raw_cnn') or d.get('feature_model') or d.get('test_metrics') or d.get('val_metrics') or d.get('metrics') or d
            if isinstance(m, dict) and 'ao_100ms' in m:
                m = {'f1': m['ao_100ms'].get('f1'), 'mae_ms': m['ao_100ms'].get('timing_mae_ms'), 'heart_rate_mae_bpm': m.get('hr_mae_bpm')}
            if isinstance(m, dict) and 'metrics' in m:
                m = m['metrics']
            add(rows, modality='SCG', task=task, tool=tool, backend=d.get('model', 'SCG trained backend'), dataset=d.get('dataset', 'SCG benchmark'),
                n=d.get('num_records') or d.get('dataset_summary', {}).get('num_records'), metric='/'.join(metric_keys),
                value='/'.join(fmt(m.get(k)) for k in metric_keys if isinstance(m, dict) and m.get(k) is not None), evidence='validated/proxy', artifact=str(OUTPUTS / p))

    # EMG summary already normalized.
    emg = load_json(OUTPUTS / 'emg_tool_performance_summary.json')
    if isinstance(emg, list):
        for item in emg:
            add(rows, modality='EMG', task=item.get('note', item.get('tool', 'EMG task')), tool=item.get('tool', 'EMG_tool'),
                backend='feature ensemble / TCN / CNN best available', dataset=item.get('dataset', 'EMG benchmark'), n=item.get('n_windows'),
                metric='accuracy/macro F1 or task score',
                value='/'.join(fmt(item.get(k)) for k in ['accuracy', 'macro_f1', 'calibrated_macro_f1', 'feature_calibrated_top1', 'balanced_accuracy', 'f1', 'auroc'] if item.get(k) is not None),
                evidence='validated', artifact=item.get('report', str(OUTPUTS / 'emg_tool_performance_summary.json')), note=item.get('note', ''))

    # EDA/EEG/ACC/RESP/SpO2/ABP/BCG.
    eda = load_json(OUTPUTS / 'eda_wesad/eda_wesad_training_summary.json')
    if eda:
        m = eda.get('tasks', {}).get('binary', {}).get('feature', {}).get('overall', {})
        add(rows, modality='EDA', task='stress detection', tool='EDA_detect_stress', backend='WESAD feature ensemble', dataset='WESAD wrist EDA subject-grouped CV',
            n=eda.get('tasks', {}).get('binary', {}).get('feature', {}).get('num_windows'), metric='macro F1/AUROC', value=f"{fmt(m.get('macro_f1'))}/{fmt(m.get('auroc'))}", evidence='validated', artifact=str(OUTPUTS / 'eda_wesad/eda_wesad_training_summary.json'))
    eeg = load_json(OUTPUTS / 'eeg_seizure/eeg_chbmit_chb01_seizure_feature_report.json')
    if eeg:
        m = eeg.get('overall') or eeg.get('metrics') or eeg.get('test_metrics') or eeg
        add(rows, modality='EEG', task='seizure detection', tool='EEG_detect_seizure', backend='CHB-MIT feature model', dataset='CHB-MIT chb01', n=eeg.get('num_windows'),
            metric='F1/AUROC', value='/'.join(fmt(m.get(k)) for k in ['macro_f1', 'f1', 'auroc', 'roc_auc'] if m.get(k) is not None), evidence='validated/proxy', artifact=str(OUTPUTS / 'eeg_seizure/eeg_chbmit_chb01_seizure_feature_report.json'))
    acc = load_json(OUTPUTS / 'acc_activity/acc_uci_har_triaxial_activity_report.json')
    if acc:
        m = acc.get('overall') or acc.get('metrics') or acc.get('test_metrics') or acc
        add(rows, modality='ACC', task='activity recognition', tool='ACC_classify_activity', backend='UCI-HAR tri-axial classifier', dataset='UCI HAR', n=acc.get('test_windows') or acc.get('num_windows'),
            metric='accuracy/macro F1', value='/'.join(fmt(m.get(k)) for k in ['accuracy', 'macro_f1'] if m.get(k) is not None), evidence='validated', artifact=str(OUTPUTS / 'acc_activity/acc_uci_har_triaxial_activity_report.json'))
    fall = load_json(OUTPUTS / 'acc_fall/acc_fall_eval.json')
    if fall:
        m = fall.get('metrics') or fall
        add(rows, modality='ACC', task='fall detection', tool='ACC_detect_fall', backend='fall feature rule/model', dataset='fall benchmark', n=fall.get('num_windows') or fall.get('num_records'),
            metric='F1', value=m.get('f1'), evidence='validated/proxy', artifact=str(OUTPUTS / 'acc_fall/acc_fall_eval.json'))
    resp = load_json(OUTPUTS / 'resp_spo2/resp_spo2_ucddb_event_report.json') or load_json(OUTPUTS / 'ucddb_resp_spo2_eval.json')
    if resp:
        m = resp.get('fusion', {}).get('overall') or resp.get('metrics') or resp.get('event_metrics') or resp
        add(rows, modality='RESP+SpO2', task='apnea/event detection', tool='RESP_SpO2_screen_apnea_events', backend='UCDB event feature model', dataset='UCDDB RESP+SpO2', n=resp.get('num_windows') or resp.get('num_events'),
            metric='F1/AUROC', value='/'.join(fmt(m.get(k)) for k in ['macro_f1', 'f1', 'auroc'] if m.get(k) is not None), evidence='validated', artifact=str(OUTPUTS / 'resp_spo2/resp_spo2_ucddb_event_report.json'))
    spo2 = load_json(OUTPUTS / 'spo2_ucddb_event_feature_model_report.json')
    if spo2:
        m = spo2.get('model_metrics') or spo2.get('metrics') or spo2
        add(rows, modality='SpO2', task='desaturation/event detection', tool='SpO2_detect_desaturation_events', backend='UCDB event feature model', dataset='UCDDB SpO2', n=spo2.get('num_windows') or spo2.get('num_events'),
            metric='F1/AUROC', value='/'.join(fmt(m.get(k)) for k in ['macro_f1', 'f1', 'auroc'] if m.get(k) is not None), evidence='validated', artifact=str(OUTPUTS / 'spo2_ucddb_event_feature_model_report.json'))
    abp = load_json(OUTPUTS / 'abp_challenge2009/abp_challenge2009_ensemble_ranker_report.json')
    if abp:
        m = abp.get('metrics', {})
        add(rows, modality='ABP', task='hypotension/shock event prediction', tool='ABP_predict_hypotension_event', backend='Challenge 2009 ensemble ranker', dataset='Computers in Cardiology Challenge 2009', n=abp.get('num_records'),
            metric='LOO AUROC / official-style score', value=f"{fmt(m.get('loo_auroc'))} / {fmt(abp.get('event_scores', {}).get('event2_best_valid_toph', {}).get('score'))}", evidence='validated', artifact=str(OUTPUTS / 'abp_challenge2009/abp_challenge2009_ensemble_ranker_report.json'))
    bcg = load_json(OUTPUTS / 'bcg_figshare_hr_eval.json') or load_json(OUTPUTS / 'bcg_figshare_hr_summary.json')
    if bcg:
        add(rows, modality='BCG', task='heart-rate estimation', tool='BCG_estimate_heart_rate', backend='J-peak/rate estimator', dataset=bcg.get('dataset', 'BCG benchmark'), n=bcg.get('num_records'),
            metric='MAE bpm', value=bcg.get('mae_bpm') or bcg.get('heart_rate_mae_bpm'), evidence='validated/proxy', artifact='BCG HR eval output')

    # Image-to-signal/digitization.
    dig = load_json(OUTPUTS / 'scg_fast_grid_summary.json') or load_json(OUTPUTS / 'larger_digitization_direct_hr_grid.json')
    if dig:
        summary = dig.get('summary', [])
        for item in summary:
            add(rows, modality=item.get('modality', 'image'), task='low-res image-to-signal HR recovery', tool='Signal_digitize_waveform_image_ml', backend=item.get('best_candidate', 'grid/direct HR detector'), dataset='digitization benchmark', n=item.get('num_records'),
                metric='best HR MAE bpm', value=item.get('best_direct_grid_hr_mae_bpm') or item.get('best_grid_hr_mae_bpm'), evidence='validated', artifact=str(OUTPUTS / 'scg_fast_grid_summary.json'))

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['modality', 'task', 'tool', 'backend', 'dataset', 'n', 'metric', 'value', 'evidence_level', 'artifact', 'note']
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['# Table 26. Tool Execution Metric Index', '']
    lines.append('Best available numeric evidence for BioSignalToolUniverse v1 execution tools. `validated/proxy` rows are usable for agent/tool-selection evaluation but should not be framed as clinical diagnostic validation.')
    lines.append('')
    headers = ['Modality', 'Task', 'Tool', 'Dataset', 'N', 'Metric', 'Value', 'Evidence']
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
    for r in rows:
        lines.append('| ' + ' | '.join(str(x).replace('|', '/') for x in [
            r['modality'], r['task'], r['tool'], r['dataset'], fmt(r['n']), r['metric'], fmt(r['value']), r['evidence_level']
        ]) + ' |')
    lines.append('')
    lines.append(f'Machine-readable artifacts: `{OUTPUTS / "tool_execution_metrics_index.json"}` and `{OUTPUTS / "tool_execution_metrics_index.csv"}`.')
    path.write_text('\n'.join(lines) + '\n')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-json', type=Path, default=OUTPUTS / 'tool_execution_metrics_index.json')
    ap.add_argument('--out-csv', type=Path, default=OUTPUTS / 'tool_execution_metrics_index.csv')
    ap.add_argument('--out-md', type=Path, default=PAPER_DIR / 'table26_tool_execution_metrics_index.md')
    args = ap.parse_args()
    rows = collect()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({'num_rows': len(rows), 'rows': rows}, indent=2))
    write_csv(args.out_csv, rows)
    write_md(args.out_md, rows)
    print(json.dumps({'rows': len(rows), 'out_json': str(args.out_json), 'out_md': str(args.out_md)}, indent=2))


if __name__ == '__main__':
    main()
