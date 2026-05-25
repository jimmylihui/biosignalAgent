from __future__ import annotations

import argparse
import re
from pathlib import Path

import requests

BASE_URL = "https://physionet.org/files/respiratory-heartrate-dataset/1.0.0/Processed_Dataset/"


def list_processed_files() -> list[str]:
    response = requests.get(BASE_URL, timeout=30)
    response.raise_for_status()
    return sorted(set(re.findall(r'href="([^"]+\.csv)"', response.text)))


def download_file(name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    if out_path.exists() and out_path.stat().st_size > 1024:
        return out_path
    url = BASE_URL + name
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        tmp.replace(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("/data1/jiahui/biosignal-agent/datasets/raw/respiratory_heartrate_aeration/Processed_Dataset"))
    parser.add_argument("--trial", default="PEEP", choices=["PEEP", "PEEP_BH", "FEM", "all"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    names = list_processed_files()
    if args.trial != "all":
        names = [name for name in names if name.endswith(f"_{args.trial}.csv")]
    if args.limit is not None:
        names = names[: args.limit]
    print(f"Downloading {len(names)} files to {args.out_dir}")
    for i, name in enumerate(names, start=1):
        path = download_file(name, args.out_dir)
        print(f"{i}/{len(names)} {name} {path.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
