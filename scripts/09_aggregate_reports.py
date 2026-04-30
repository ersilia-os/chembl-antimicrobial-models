"""
Step 09 — Aggregate per-dataset CV reports into a single file.

Reads all CSVs from output/results/08_reports/ and concatenates them into
output/results/09_reports.csv.

Usage:
    python scripts/09_aggregate_reports.py
"""

import glob
import os

import pandas as pd
from tqdm import tqdm

ROOT      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))

IN_DIR   = os.path.join(REPO_ROOT, "output", "results", "08_reports")
OUT_PATH = os.path.join(REPO_ROOT, "output", "results", "09_reports.csv")


def main() -> None:
    files = sorted(glob.glob(os.path.join(IN_DIR, "**", "*.csv"), recursive=True))
    if not files:
        print(f"No report CSVs found in {IN_DIR}")
        return

    df = pd.concat(
        (pd.read_csv(f) for f in tqdm(files, desc="Aggregating reports", unit="dataset")),
        ignore_index=True,
    )
    df.to_csv(OUT_PATH, index=False)
    print(f"{len(df)} rows from {len(files)} datasets → {OUT_PATH}")


if __name__ == "__main__":
    main()
