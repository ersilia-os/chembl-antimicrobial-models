"""
Step 17 (ECBD) — Download per-assay activity CSVs from the ECBD website.

For each (pathogen, assay EOS ID) pair in config/ecbd_pathogen_assays.csv,
fetches the activity CSV from https://ecbd.eu and saves it to
data/raw/ecbd/<pathogen>/<eos>.csv.

CSV columns: eos, control, compound_name, smiles, inchikey, concentration, activity, value

Usage:
    python scripts/17_download_datasets_ecbd.py --pathogen saureus
    python scripts/17_download_datasets_ecbd.py --all
"""

import argparse
import os
import re

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(ROOT, "..")

BASE_URL = "https://ecbd.eu"
CONFIG = os.path.join(REPO_ROOT, "config", "ecbd_pathogen_assays.csv")
OUT_DIR = os.path.join(REPO_ROOT, "data", "raw", "ecbd")


def get_epid(session: requests.Session, eos: str) -> str:
    resp = session.get(f"{BASE_URL}/assays/{eos}", timeout=30)
    resp.raise_for_status()
    match = re.search(r':endpoint_id="(\d+)"', resp.text)
    if not match:
        raise ValueError(f"Could not find endpoint_id in page for {eos}")
    return match.group(1)


def download_assay(session: requests.Session, eos: str, out_path: str) -> int:
    epid = get_epid(session, eos)
    resp = session.get(
        f"{BASE_URL}/assays/export_data",
        params={"epid": epid, "format": "csv"},
        timeout=120,
    )
    resp.raise_for_status()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(resp.text)
    return resp.text.count("\n") - 1  # approximate row count


def run(pathogens: list[str]) -> None:
    df = pd.read_csv(CONFIG)
    subset = df[df["pathogen"].isin(pathogens)]

    session = requests.Session()
    session.verify = False  # ECBD has a self-signed cert issue on this machine
    requests.packages.urllib3.disable_warnings()

    for _, row in subset.iterrows():
        pathogen, eos = row["pathogen"], row["eos"]
        out_path = os.path.join(OUT_DIR, pathogen, f"{eos}.csv")
        if os.path.exists(out_path):
            print(f"  skip {eos} (already exists)")
            continue
        print(f"  {pathogen}/{eos} ... ", end="", flush=True)
        try:
            n = download_assay(session, eos, out_path)
            print(f"{n} rows")
        except Exception as e:
            print(f"ERROR: {e}")


def main() -> None:
    all_pathogens = pd.read_csv(CONFIG)["pathogen"].unique().tolist()

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pathogen", choices=all_pathogens)
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()

    pathogens = all_pathogens if args.all else [args.pathogen]
    print(f"Downloading ECBD assays for: {pathogens}")
    run(pathogens)


if __name__ == "__main__":
    main()
