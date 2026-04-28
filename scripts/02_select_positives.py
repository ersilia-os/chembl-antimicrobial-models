"""
Step 02 — Extract active compounds from all datasets.

Reads data/processed/01_chembl_datasets_all.csv and, for each dataset, opens the
corresponding zip archive to extract SMILES with bin == 1.

Zip file mapping:
  Labels A, B, M  ->  data/raw/<pathogen>/19_final_datasets.zip
                       file inside: {name}.csv  (columns: smiles, bin)
  Label G         ->  data/raw/<pathogen>/20_general_datasets.zip
                       file inside: ORG_{activity_type}_{unit}_{cutoff}.csv.gz
                                    (columns: compound_chembl_id, bin, smiles)

Produces output/results/02_selected_positives.csv with one row per unique SMILES, sorted
alphabetically, with columns:
  - smiles    : unique SMILES string
  - n_active  : number of datasets in which the compound was active
  - found_in  : semicolon-separated list of pathogen|dataset tags
  - split     : integer split index (0 = first --split_size compounds, 1 = next, …)

Usage:
    python scripts/02_select_positives.py
    python scripts/02_select_positives.py --datasets path/to/other.csv
    python scripts/02_select_positives.py --split_size 1000
"""

import argparse
import io
import os
import zipfile
from collections import defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(ROOT, "..")


def read_dataset(row: pd.Series) -> pd.Series | None:
    pathogen = row["pathogen"]
    label = row["label"]
    name = row["name"]

    if label in {"A", "B", "M"}:
        zip_path = os.path.join(REPO_ROOT, "data", "raw", pathogen, "19_final_datasets.zip")
        inner_name = f"{name}.csv"
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(inner_name) as f:
                df = pd.read_csv(f)

    elif label == "G":
        zip_path = os.path.join(REPO_ROOT, "data", "raw", pathogen, "20_general_datasets.zip")
        inner_name = f"ORG_{row['activity_type']}_{row['unit']}_{row['cutoff']}.csv.gz"
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(inner_name) as f:
                df = pd.read_csv(io.BytesIO(f.read()), compression="gzip")

    else:
        return None

    actives = df[df["bin"] == 1]["smiles"].dropna()
    return actives


def main(datasets_path: str, split_size: int) -> None:
    meta = pd.read_csv(datasets_path)
    actives: dict[str, set] = defaultdict(set)

    for _, row in meta.iterrows():
        smiles_series = read_dataset(row)
        if smiles_series is None or smiles_series.empty:
            continue
        tag = f"{row['pathogen']}|{row['name']}"
        for smi in smiles_series.unique():
            actives[smi].add(tag)

    sorted_smiles = sorted(actives.keys())
    result = pd.DataFrame({
        "smiles": sorted_smiles,
        "n_active": [len(actives[s]) for s in sorted_smiles],
        "found_in": [";".join(sorted(actives[s])) for s in sorted_smiles],
        "split": [i // split_size for i in range(len(sorted_smiles))],
    })

    out_path = os.path.join(REPO_ROOT, "output", "results", "02_selected_positives.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"Saved {len(result):,} unique active SMILES to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract active compounds from all ChEMBL datasets."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=os.path.join(REPO_ROOT, "data", "processed", "01_chembl_datasets_all.csv"),
        help="Path to the datasets CSV (default: data/processed/01_chembl_datasets_all.csv).",
    )
    parser.add_argument(
        "--split_size",
        type=int,
        default=500,
        help="Number of compounds per split (default: 500).",
    )
    args = parser.parse_args()
    main(args.datasets, args.split_size)
