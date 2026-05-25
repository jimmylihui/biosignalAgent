from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.agent.planning_agent import PlanningBioSignalAgent
from biosignal_agent.tools.digitize_tools import Signal_digitize_waveform_image_ml, Signal_estimate_image_scale, Signal_predict_image_scale_prior, estimate_digitization_resolution_risk
from biosignal_agent.tools.image_modality_tools import Signal_classify_modality_from_image
from biosignal_agent.tools.image_cnn_tools import Signal_classify_modality_from_image_cnn
from biosignal_agent.tools.modality_tools import Signal_classify_modality

QUESTION_BY_MODALITY = {
    "ecg": "Does this uploaded ECG image show abnormal rhythm? Detect R peaks, estimate heart rate and HRV, and give a concise screening report.",
    "ppg": "Does this uploaded PPG image show irregular pulse or atrial fibrillation-like pulse variability? Estimate pulse rate and screen irregularity.",
    "bcg": "Estimate breathing rate from this uploaded BCG image and summarize the respiration modulation.",
    "scg": "Estimate respiratory rate from this uploaded SCG image and summarize the breathing modulation.",
    "resp": "Detect apnea-like breathing pauses from this uploaded respiration image and estimate respiratory rate.",
    "spo2": "Summarize this uploaded SpO2 image and detect oxygen desaturation burden.",
    "abp": "Analyze this uploaded arterial blood pressure image, detect pulses, and compute MAP and pulse pressure.",
    "pcg": "Classify this uploaded PCG image for murmur or abnormal heart sound using spectrogram features.",
    "acc": "Screen this uploaded accelerometer image for fall or impact events and summarize activity.",
    "eda": "Estimate stress level from this uploaded EDA image using tonic phasic activity and arousal events.",
    "eeg": "Screen this uploaded EEG image for seizure-like spikes or epileptiform abnormal activity.",
    "emg": "Estimate EMG muscle fatigue from this uploaded EMG image using activation and median frequency features.",
}


def compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload.get(key) for key in ["tool", "num_points", "pixel_coverage", "mask_pixel_fraction", "confidence", "method", "error"] if key in payload}


def choose_planner_fallback(agent: PlanningBioSignalAgent, question: str, classifier_modality: str | None) -> str | None:
    try:
        agent.infer_modality(question, None)
        return None
    except Exception:
        return classifier_modality


def run_record(record: dict[str, Any], out_dir: Path, model_path: str, image_model_path: str, cnn_image_model_path: str, probability_threshold: float, trace_method: str) -> dict[str, Any]:
    modality = str(record["modality"]).lower()
    question = QUESTION_BY_MODALITY.get(modality, f"Analyze this uploaded {modality.upper()} signal image and provide a concise research-use report.")
    record_id = str(record.get("record") or Path(record["image_path"]).stem)
    record_dir = out_dir / "records"
    record_dir.mkdir(parents=True, exist_ok=True)
    out_csv = record_dir / f"{record_id}_digitized.csv"
    out_json = record_dir / f"{record_id}_report.json"
    metadata_sampling_rate = record.get("sampling_rate")
    metadata_duration_s = record.get("duration_s")
    sampling_rate: float | None = float(metadata_sampling_rate) if metadata_sampling_rate is not None else None

    row: dict[str, Any] = {
        "record": record_id,
        "expected_modality": modality,
        "variant": record.get("variant"),
        "image_path": record["image_path"],
        "question": question,
        "sampling_rate": sampling_rate,
        "scale_source": "metadata_sampling_rate" if sampling_rate is not None else "unknown",
        "digitized_csv": str(out_csv),
        "report_json": str(out_json),
    }

    try:
        image_classifier = Signal_classify_modality_from_image(
            record["image_path"],
            crop_left=int(record.get("crop_left") or 0),
            crop_right=int(record.get("crop_right") or 0),
            crop_top=int(record.get("crop_top") or 0),
            crop_bottom=int(record.get("crop_bottom") or 0),
            model_path=image_model_path,
        )
    except Exception as exc:
        image_classifier = {"error": str(exc), "predicted_modality": None, "confidence": 0.0}
    image_predicted_modality = image_classifier.get("predicted_modality")
    row.update({
        "image_classified_modality": image_predicted_modality,
        "image_classifier_confidence": image_classifier.get("confidence"),
        "image_classifier_correct": image_predicted_modality == modality,
        "image_classifier_error": image_classifier.get("error"),
    })

    try:
        cnn_image_classifier = Signal_classify_modality_from_image_cnn(
            record["image_path"],
            crop_left=int(record.get("crop_left") or 0),
            crop_right=int(record.get("crop_right") or 0),
            crop_top=int(record.get("crop_top") or 0),
            crop_bottom=int(record.get("crop_bottom") or 0),
            model_path=cnn_image_model_path,
        )
    except Exception as exc:
        cnn_image_classifier = {"error": str(exc), "predicted_modality": None, "confidence": 0.0}
    cnn_image_predicted_modality = cnn_image_classifier.get("predicted_modality")
    row.update({
        "cnn_image_classified_modality": cnn_image_predicted_modality,
        "cnn_image_classifier_confidence": cnn_image_classifier.get("confidence"),
        "cnn_image_classifier_correct": cnn_image_predicted_modality == modality,
        "cnn_image_classifier_error": cnn_image_classifier.get("error"),
    })

    scale = Signal_estimate_image_scale(
        image_path=record["image_path"],
        crop_left=int(record.get("crop_left") or 0),
        crop_right=int(record.get("crop_right") or 0),
        crop_top=int(record.get("crop_top") or 0),
        crop_bottom=int(record.get("crop_bottom") or 0),
        duration_s=float(metadata_duration_s) if metadata_duration_s is not None else None,
        use_ocr=True,
    )
    if sampling_rate is None and scale.get("sampling_rate") is not None:
        sampling_rate = float(scale["sampling_rate"])
        row["sampling_rate"] = sampling_rate
        row["scale_source"] = "axis_ocr_or_metadata_duration"
    row.update({
        "scale_duration_s": scale.get("duration_s"),
        "scale_sampling_rate": scale.get("sampling_rate"),
        "scale_ocr_status": scale.get("ocr_status"),
        "scale_confidence": scale.get("confidence"),
        "scale_missing": json.dumps(scale.get("missing_for_physical_scale", [])),
    })
    scale_model_decision = None
    if sampling_rate is None:
        model_modality = str(cnn_image_predicted_modality or image_predicted_modality or modality).lower()
        scale_model_decision = Signal_predict_image_scale_prior(
            image_path=record["image_path"],
            modality=model_modality,
            top_k=3,
            model_scope="auto",
        )
        candidates = scale_model_decision.get("duration_candidates") or []
        best = candidates[0] if candidates else {}
        if best.get("sampling_rate_hz_if_used") is not None and not scale_model_decision.get("requires_confirmation", True):
            sampling_rate = float(best["sampling_rate_hz_if_used"])
            row["sampling_rate"] = sampling_rate
            row["scale_source"] = "model_scale_decision"
            row["scale_duration_s"] = best.get("duration_s")
        else:
            row.update({
                "pipeline_ok": False,
                "failure_stage": "scale_model_requires_confirmation",
                "digitizer_error": "scale model did not approve an unconfirmed x-axis scale",
                "classifier_error": "not_run_without_model_approved_sampling_rate",
                "scale_model_confidence": scale_model_decision.get("confidence"),
                "scale_model_requires_confirmation": scale_model_decision.get("requires_confirmation"),
                "scale_model_candidates": json.dumps(candidates),
                "agent_plan": "[]",
                "num_tool_calls": 0,
                "num_findings": 0,
                "tool_errors": json.dumps([{"tool": "scale_model", "error": "requires_confirmation"}]),
                "findings": "[]",
            })
            out_json.write_text(json.dumps({
                "record": record,
                "question": question,
                "scale_estimation": scale,
                "scale_model_decision": scale_model_decision,
                "image_modality_classification": image_classifier,
                "cnn_image_modality_classification": cnn_image_classifier,
                "row": row,
                "disclaimer": "No digitization or signal-tool report was run because the scale model required confirmation for x-axis calibration.",
            }, indent=2))
            return row

    digitized = Signal_digitize_waveform_image_ml(
        image_path=record["image_path"],
        sampling_rate=sampling_rate,
        out_csv=str(out_csv),
        value_min=float(record["value_min"]) if record.get("value_min") is not None else None,
        value_max=float(record["value_max"]) if record.get("value_max") is not None else None,
        crop_left=int(record.get("crop_left") or 0),
        crop_right=int(record.get("crop_right") or 0),
        crop_top=int(record.get("crop_top") or 0),
        crop_bottom=int(record.get("crop_bottom") or 0),
        model_path=model_path,
        probability_threshold=probability_threshold,
        smooth_window=3,
        trace_method=trace_method,
    )
    row.update({
        "digitizer_error": digitized.get("error"),
        "digitizer_confidence": digitized.get("confidence"),
        "pixel_coverage": digitized.get("pixel_coverage"),
        "mask_pixel_fraction": digitized.get("mask_pixel_fraction"),
        "num_points": digitized.get("num_points"),
    })
    if digitized.get("error"):
        row.update({"pipeline_ok": False, "failure_stage": "digitize"})
        out_json.write_text(json.dumps({"record": record, "digitization": digitized, "row": row}, indent=2))
        return row

    try:
        classifier = Signal_classify_modality(str(out_csv), sampling_rate=sampling_rate, column="signal")
    except Exception as exc:
        classifier = {"error": str(exc), "predicted_modality": None, "confidence": 0.0}
    predicted_modality = classifier.get("predicted_modality")
    row.update({
        "classified_modality": predicted_modality,
        "classifier_confidence": classifier.get("confidence"),
        "classifier_correct": predicted_modality == modality,
        "classifier_error": classifier.get("error"),
    })

    try:
        risk = estimate_digitization_resolution_risk(
            image_width=int(record.get("width") or record.get("num_points") or 0) or None,
            duration_s=record.get("duration_s"),
            modality=modality,
        )
    except Exception as exc:
        risk = {"error": str(exc)}

    agent = PlanningBioSignalAgent()
    fallback = modality
    try:
        report = agent.run(question, str(out_csv), sampling_rate=sampling_rate, column="signal", fallback_modality=fallback)
        tool_errors = [call for call in report.get("tool_calls", []) if isinstance(call.get("result"), dict) and call["result"].get("error")]
        pipeline_ok = not tool_errors
        row.update({
            "pipeline_ok": pipeline_ok,
            "failure_stage": "tool_execution" if tool_errors else "",
            "agent_modality": report.get("modality"),
            "agent_plan": json.dumps(report.get("plan", [])),
            "num_tool_calls": len(report.get("tool_calls", [])),
            "num_findings": len(report.get("findings", [])),
            "tool_errors": json.dumps([{ "tool": item.get("tool"), "error": item.get("result", {}).get("error") } for item in tool_errors]),
            "findings": json.dumps(report.get("findings", [])),
        })
    except Exception as exc:
        report = {"error": str(exc)}
        row.update({
            "pipeline_ok": False,
            "failure_stage": "agent_run",
            "agent_modality": None,
            "agent_plan": "[]",
            "num_tool_calls": 0,
            "num_findings": 0,
            "tool_errors": json.dumps([{ "tool": "agent", "error": str(exc) }]),
            "findings": "[]",
        })

    out_json.write_text(json.dumps({
        "record": record,
        "question": question,
        "digitization": digitized,
        "image_modality_classification": image_classifier,
        "cnn_image_modality_classification": cnn_image_classifier,
        "modality_classification": classifier,
        "scale_estimation": scale,
        "scale_model_decision": scale_model_decision,
        "resolution_risk": risk,
        "agent_report": report,
        "row": row,
    }, indent=2))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch test image+question -> digitize -> route -> plan/tools -> report pipeline.")
    parser.add_argument("--manifest", default="/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark_highres_aligned_manifest.json")
    parser.add_argument("--out-dir", default="/data1/jiahui/biosignal-agent/outputs/image_question_pipeline_batch")
    parser.add_argument("--model-path", default="/data1/jiahui/biosignal-agent/outputs/waveform_digitization_pixel_model_highres.joblib")
    parser.add_argument("--image-model-path", default="/data1/jiahui/biosignal-agent/outputs/image_modality_classifier_model.joblib")
    parser.add_argument("--cnn-image-model-path", default="/data1/jiahui/biosignal-agent/outputs/image_modality_classifier_cnn_80e.pt")
    parser.add_argument("--probability-threshold", type=float, default=0.5)
    parser.add_argument("--trace-method", default="median", choices=["median", "path", "momentum", "full", "lazy", "fragmented"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-modality", action="append", default=None)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    records = manifest.get("records", [])
    if args.include_modality:
        wanted = {item.lower() for item in args.include_modality}
        records = [record for record in records if str(record.get("modality", "")).lower() in wanted]
    if args.limit is not None:
        records = records[: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_record(record, out_dir, args.model_path, args.image_model_path, args.cnn_image_model_path, args.probability_threshold, args.trace_method) for record in records]

    out_csv = out_dir / "summary.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    modality_counts = Counter(row["expected_modality"] for row in rows)
    ok_counts = Counter(row["expected_modality"] for row in rows if row.get("pipeline_ok"))
    classifier_counts = Counter(row["expected_modality"] for row in rows if row.get("classifier_correct"))
    image_classifier_counts = Counter(row["expected_modality"] for row in rows if row.get("image_classifier_correct"))
    cnn_image_classifier_counts = Counter(row["expected_modality"] for row in rows if row.get("cnn_image_classifier_correct"))
    failure_stage_counts = Counter(row.get("failure_stage") or "ok" for row in rows)
    by_modality = {}
    for modality in sorted(modality_counts):
        total = modality_counts[modality]
        by_modality[modality] = {
            "records": total,
            "pipeline_ok": ok_counts[modality],
            "pipeline_success_rate": ok_counts[modality] / total if total else 0.0,
            "classifier_correct": classifier_counts[modality],
            "classifier_accuracy": classifier_counts[modality] / total if total else 0.0,
            "image_classifier_correct": image_classifier_counts[modality],
            "image_classifier_accuracy": image_classifier_counts[modality] / total if total else 0.0,
            "cnn_image_classifier_correct": cnn_image_classifier_counts[modality],
            "cnn_image_classifier_accuracy": cnn_image_classifier_counts[modality] / total if total else 0.0,
        }

    report = {
        "manifest": args.manifest,
        "model_path": args.model_path,
        "image_model_path": args.image_model_path,
        "cnn_image_model_path": args.cnn_image_model_path,
        "trace_method": args.trace_method,
        "num_records": len(rows),
        "pipeline_success_rate": sum(1 for row in rows if row.get("pipeline_ok")) / len(rows) if rows else 0.0,
        "classifier_accuracy": sum(1 for row in rows if row.get("classifier_correct")) / len(rows) if rows else 0.0,
        "image_classifier_accuracy": sum(1 for row in rows if row.get("image_classifier_correct")) / len(rows) if rows else 0.0,
        "cnn_image_classifier_accuracy": sum(1 for row in rows if row.get("cnn_image_classifier_correct")) / len(rows) if rows else 0.0,
        "failure_stage_counts": dict(failure_stage_counts),
        "by_modality": by_modality,
        "out_csv": str(out_csv),
    }
    out_json = out_dir / "summary.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
