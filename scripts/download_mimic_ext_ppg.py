from __future__ import annotations

import argparse
import json
from pathlib import Path

import wfdb
from requests import HTTPError


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or probe MIMIC-III-Ext-PPG from PhysioNet.")
    parser.add_argument("--db", default="mimic-iii-ext-ppg")
    parser.add_argument("--out-dir", default="/data1/jiahui/biosignal-agent/datasets/raw/mimic-iii-ext-ppg")
    parser.add_argument("--max-records", type=int, default=0, help="0 means all records after access is confirmed.")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    report = {"db": args.db, "out_dir": str(out_dir), "status": None, "records": None, "error": None}
    try:
        records = wfdb.get_record_list(args.db)
        report["records"] = len(records)
        report["first_records"] = records[:10]
        report["status"] = "accessible"
        if args.download:
            out_dir.mkdir(parents=True, exist_ok=True)
            selected = records if args.max_records <= 0 else records[: args.max_records]
            wfdb.dl_database(args.db, dl_dir=str(out_dir), records=selected, keep_subdirs=True, overwrite=False)
            report["downloaded_records"] = len(selected)
    except Exception as exc:
        report["status"] = "blocked"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["next_step"] = "MIMIC-III-Ext-PPG requires PhysioNet credentialed access/login in this environment. Authenticate with PhysioNet or download externally, then rerun with --download."
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
