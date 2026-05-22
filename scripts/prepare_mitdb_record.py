from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import wfdb


def download_record(record: str, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = [f"{record}.hea", f"{record}.dat", f"{record}.atr"]
    wfdb.dl_files("mitdb", str(raw_dir), files=files, keep_subdirs=False, overwrite=False)


def export_csv(record: str, raw_dir: Path, out_csv: Path, seconds: int | None) -> tuple[float, list[str], str]:
    header = wfdb.rdheader(str(raw_dir / record))
    sampto = int(seconds * header.fs) if seconds is not None else None
    wfdb_record = wfdb.rdrecord(str(raw_dir / record), sampto=sampto)
    lead = "MLII" if "MLII" in wfdb_record.sig_name else wfdb_record.sig_name[0]
    lead_idx = wfdb_record.sig_name.index(lead)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"signal": wfdb_record.p_signal[:, lead_idx]}).to_csv(out_csv, index=False)
    return float(wfdb_record.fs), list(wfdb_record.sig_name), lead


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", default="100")
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--raw-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/raw/mitdb"))
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    download_record(args.record, args.raw_dir)
    out_csv = args.out_csv or Path(f"/data1/jiahui/biosignal-agent/datasets/processed/mitdb_{args.record}_mlii_{args.seconds}s.csv")
    fs, sig_names, lead = export_csv(args.record, args.raw_dir, out_csv, args.seconds)
    print({"record": args.record, "fs": fs, "sig_names": sig_names, "exported_lead": lead, "csv": str(out_csv)})


if __name__ == "__main__":
    main()
