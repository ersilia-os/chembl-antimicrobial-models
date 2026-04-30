"""
Step 10 — Download DrugBank SMILES.

Downloads the DrugBank SMILES reference file from:
  https://github.com/ersilia-os/sars-cov-2-chemspace/blob/main/data/drugbank_smiles.csv

and saves it to output/results/10_drugbank_smiles.csv.

Usage:
    python scripts/10_download_drugbank.py
"""

import os
import urllib.request

ROOT      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))

URL = (
    "https://raw.githubusercontent.com/ersilia-os/sars-cov-2-chemspace"
    "/main/data/drugbank_smiles.csv"
)
OUT_PATH = os.path.join(REPO_ROOT, "output", "results", "10_drugbank_smiles.csv")


def main() -> None:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    print(f"Downloading DrugBank SMILES from {URL}")
    urllib.request.urlretrieve(URL, OUT_PATH)
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
