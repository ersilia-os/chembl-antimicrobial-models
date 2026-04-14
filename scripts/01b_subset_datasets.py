"""
Step 01b — Select representative datasets.

Reads data/processed/01_chembl_datasets_all.csv and randomly samples datasets
across a grid of compound count ranges × activity ratio ranges, ignoring pathogen.

  Compound ranges:   100–1000 (×2), 1000–10000 (×2), 10000–50000 (×1)
  Ratio ranges:      0.01–0.3 (×1), 0.3–0.5 (×1)

Total: 10 datasets (2×2 + 2×2 + 1×2).

Produces data/processed/01b_selected_datasets.csv.

Usage:
    python scripts/01b_select_datasets.py [--seed 42]
"""

import argparse
import os
from itertools import product

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(ROOT, "..")

# (min_compounds, max_compounds, n_samples)
COMPOUND_BINS = [
    (100,   1_000,  2),
    (1_000, 10_000, 2),
    (10_000, 50_000, 1),
]

# (min_ratio, max_ratio)
RATIO_BINS = [
    (0.01, 0.3),
    (0.3,  0.5),
]


def main(seed: int) -> None:
    input_path = os.path.join(REPO_ROOT, "data", "processed", "01_chembl_datasets_all.csv")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path).dropna(subset=["compounds", "ratio"])

    selected = []
    for (c_min, c_max, n), (r_min, r_max) in product(COMPOUND_BINS, RATIO_BINS):
        cell = df[
            (df["compounds"] >= c_min) & (df["compounds"] < c_max) &
            (df["ratio"] >= r_min) & (df["ratio"] < r_max)
        ]
        if cell.empty:
            print(f"[WARN] No datasets in compounds=[{c_min},{c_max}) ratio=[{r_min},{r_max})")
            continue
        n_draw = min(n, len(cell))
        selected.append(cell.sample(n=n_draw, random_state=seed))

    result = pd.concat(selected).reset_index(drop=True)
    print(result[["pathogen", "name", "compounds", "ratio"]].to_string())
    out_path = os.path.join(REPO_ROOT, "data", "processed", "01b_selected_datasets.csv")
    result.to_csv(out_path, index=False)
    print(f"\nSaved {len(result)} datasets to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Randomly select representative datasets across compound and ratio bins."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    args = parser.parse_args()
    main(args.seed)
