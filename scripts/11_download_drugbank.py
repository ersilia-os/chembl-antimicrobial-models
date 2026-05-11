"""
Step 11 — Download DrugBank SMILES.

Downloads the DrugBank SMILES reference file from:
  https://github.com/ersilia-os/sars-cov-2-chemspace/blob/main/data/drugbank_smiles.csv

Saves a single 'smiles' column sorted alphabetically (default). Use
--keep_all_columns to retain all original columns.

Filters applied before saving (each dropped with a warning):
  - Invalid SMILES (RDKit cannot parse)
  - Inorganic compounds (no carbon atom)
  - Heavy molecules (MW > 1000 Da)

Usage:
    python scripts/11_download_drugbank.py
    python scripts/11_download_drugbank.py --output path/to/file.csv
    python scripts/11_download_drugbank.py --keep_all_columns
"""

import argparse
import io
import os
import urllib.request

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

ROOT      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))

URL = (
    "https://raw.githubusercontent.com/ersilia-os/sars-cov-2-chemspace"
    "/main/data/drugbank_smiles.csv"
)
DEFAULT_OUT = os.path.join(REPO_ROOT, "data", "processed", "11_drugbank_smiles.csv")


MW_CAP = 1000.0


def _filter_compounds(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    mols = [
        Chem.MolFromSmiles(s)
        for s in tqdm(df[smiles_col], desc="Validating SMILES", unit="cpd")
    ]
    n_invalid = sum(m is None for m in mols)
    keep = []
    n_inorganic = n_heavy = 0
    for mol in mols:
        if mol is None:
            keep.append(False)
            continue
        if not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
            keep.append(False)
            n_inorganic += 1
        elif Descriptors.MolWt(mol) > MW_CAP:
            keep.append(False)
            n_heavy += 1
        else:
            keep.append(True)
    n_total = len(mols)
    n_kept = sum(keep)
    print(f"  Invalid SMILES  : {n_invalid}")
    print(f"  Inorganic (no C): {n_inorganic}")
    print(f"  MW > {MW_CAP:.0f} Da    : {n_heavy}")
    print(f"  Total filtered  : {n_total - n_kept} / {n_total} -> {n_kept} compounds kept")
    return df[keep].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download DrugBank SMILES.")
    parser.add_argument("--output", default=DEFAULT_OUT, help="Output CSV path")
    parser.add_argument("--keep_all_columns", action="store_true",
                        help="Retain all original columns instead of writing only 'smiles'")
    args = parser.parse_args()

    print(f"Downloading DrugBank SMILES from {URL}")
    with urllib.request.urlopen(URL) as response:
        df = pd.read_csv(io.StringIO(response.read().decode()))

    smiles_col = next(c for c in df.columns if c.lower() == "smiles")
    df = df.dropna(subset=[smiles_col])
    df = _filter_compounds(df, smiles_col)

    if not args.keep_all_columns:
        df = pd.DataFrame({"smiles": sorted(df[smiles_col].unique())})

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows → {args.output}")


if __name__ == "__main__":
    main()
