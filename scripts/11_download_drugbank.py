"""
Step 10 — Download DrugBank SMILES.

Downloads the DrugBank SMILES reference file from:
  https://github.com/ersilia-os/sars-cov-2-chemspace/blob/main/data/drugbank_smiles.csv

Usage:
    python scripts/10_download_drugbank.py
    python scripts/10_download_drugbank.py --output path/to/file.csv
    python scripts/10_download_drugbank.py --only_smiles
"""

import argparse
import io
import os
import urllib.request

import pandas as pd

ROOT      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))

URL = (
    "https://raw.githubusercontent.com/ersilia-os/sars-cov-2-chemspace"
    "/main/data/drugbank_smiles.csv"
)
DEFAULT_OUT = os.path.join(REPO_ROOT, "data", "processed", "10_drugbank_smiles.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download DrugBank SMILES.")
    parser.add_argument("--output", default=DEFAULT_OUT, help="Output CSV path")
    parser.add_argument("--only_smiles", action="store_true",
                        help="Write a single 'smiles' column sorted alphabetically")
    args = parser.parse_args()

    print(f"Downloading DrugBank SMILES from {URL}")
    with urllib.request.urlopen(URL) as response:
        df = pd.read_csv(io.StringIO(response.read().decode()))

    if args.only_smiles:
        smiles_col = next(c for c in df.columns if c.lower() == "smiles")
        df = pd.DataFrame({"smiles": sorted(df[smiles_col].dropna().unique())})

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows → {args.output}")


if __name__ == "__main__":
    main()
