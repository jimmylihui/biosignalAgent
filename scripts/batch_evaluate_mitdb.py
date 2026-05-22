from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import wfdb

MITDB_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]


def run_command(args: list[str]) -> None:
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--records", nargs="*", default=MITDB_RECORDS)
    parser.add_argument("--raw-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/raw/mitdb"))
    parser.add_argument("--processed-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/processed"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/outputs"))
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--method-tag", default="pantompkins")
    args = parser.parse_args()

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []

    for record in args.records:
        print(f"=== MIT-BIH {record} ===", flush=True)
        csv_path = args.processed_dir / f"mitdb_{record}_mlii_{args.seconds}s.csv"
        report_path = args.outputs_dir / f"mitdb_{record}_mlii_{args.seconds}s_report_{args.method_tag}.json"
        metrics_path = args.outputs_dir / f"mitdb_{record}_mlii_{args.seconds}s_peak_metrics_{args.method_tag}.json"
        try:
            if not (args.skip_existing and metrics_path.exists()):
                run_command([sys.executable, "scripts/prepare_mitdb_record.py", "--record", record, "--seconds", str(args.seconds), "--raw-dir", str(args.raw_dir), "--out-csv", str(csv_path)])
                header = wfdb.rdheader(str(args.raw_dir / record))
                with report_path.open("w") as handle:
                    subprocess.run([sys.executable, "examples/run_basic_report.py", "--csv", str(csv_path), "--modality", "ecg", "--sampling-rate", str(float(header.fs))], check=True, stdout=handle)
                run_command([sys.executable, "scripts/evaluate_mitdb_peaks.py", "--record", record, "--seconds", str(args.seconds), "--report", str(report_path), "--raw-dir", str(args.raw_dir), "--out", str(metrics_path)])
            metrics = json.loads(metrics_path.read_text())
            rows.append(metrics)
            print({key: round(metrics[key], 4) if isinstance(metrics[key], float) else metrics[key] for key in ["detected_peaks", "reference_beats", "precision", "recall", "f1"]}, flush=True)
        except Exception as exc:
            failures.append({"record": record, "error": str(exc)})
            print(f"FAILED {record}: {exc}", flush=True)

    summary_path = args.summary or args.outputs_dir / f"mitdb_{args.method_tag}_{args.seconds}s_all_summary.csv"
    if rows:
        frame = pd.DataFrame(rows).sort_values("record")
        frame.to_csv(summary_path, index=False)
        print(f"summary: {summary_path}")
        print(frame[["record", "detected_peaks", "reference_beats", "precision", "recall", "f1"]].to_string(index=False))
        print("macro avg:", frame[["precision", "recall", "f1"]].mean().to_dict())
    if failures:
        failure_path = args.outputs_dir / f"mitdb_{args.method_tag}_{args.seconds}s_failures.json"
        failure_path.write_text(json.dumps(failures, indent=2))
        print(f"failures: {failure_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
