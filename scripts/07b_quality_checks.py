"""
Script 07b — Per-dataset duplication quality report for outputs of script 07.

Audits the post-script-07 datasets in output/07_datasets/{pathogen}/*.csv to
verify the InChIKey-level deduplication. ChEMBL and PubChem datasets are
distinguished by filename: purely-numeric basenames are PubChem AIDs, anything
else is ChEMBL.

Per dataset, reports:
  rows                    — total rows (decoys included)
  unparsable              — rows whose SMILES RDKit cannot parse
  unique_compounds        — distinct InChIKeys among parsable rows
  dup_compounds           — InChIKeys appearing in >1 row
  dup_via_diff_smiles     — of dup_compounds, those duplicated via >1 distinct SMILES string
  redundant_rows          — rows - unparsable - unique_compounds
  conflict_compounds      — InChIKeys carrying both bin=0 and bin=1
  decoy_active_collisions — decoy rows whose InChIKey is labelled active in any other
                            dataset of the same pathogen (should be 0; non-zero means
                            script 07's pathogen-wide active filter let one slip)

Output:
  output/07b_quality_checks/dup_report.csv

Usage:
    python scripts/07b_quality_checks.py
"""

import argparse
import glob
import importlib.util
import os
import re
import sys

import pandas as pd
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "..", "src"))

REPO_ROOT    = os.path.abspath(os.path.join(root, ".."))
DATASETS_DIR = os.path.join(REPO_ROOT, "output", "07_datasets")
OUT_DIR      = DATASETS_DIR
OUT_PATH     = os.path.join(OUT_DIR, "07_dup_report.csv")
os.makedirs(OUT_DIR, exist_ok=True)


def _load_smiles_to_inchikey():
    """Import smiles_to_inchikey from 07a_prepare_datasets.py (digit-leading filename)."""
    spec = importlib.util.spec_from_file_location(
        "_prep07", os.path.join(root, "07a_prepare_datasets.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.smiles_to_inchikey


smiles_to_inchikey = _load_smiles_to_inchikey()


def _source_of(basename: str) -> str:
    """PubChem AIDs are purely numeric basenames; everything else is ChEMBL."""
    return "pubchem" if re.fullmatch(r"\d+", basename) else "chembl"


def build_pathogen_active_iks(datasets_dir: str) -> dict:
    """For each pathogen, the union of InChIKeys labelled active (bin=1, non-decoy)
    across all its datasets. Used to detect decoys that collide with a known active
    in a sibling dataset of the same pathogen."""
    actives: dict[str, set[str]] = {}
    for path in sorted(glob.glob(os.path.join(datasets_dir, "*", "*.csv"))):
        if "metadata" in os.path.basename(path):
            continue
        pathogen = os.path.basename(os.path.dirname(path))
        df = pd.read_csv(path)
        df = df.dropna(subset=["smiles"]).copy()
        df["smiles"] = df["smiles"].astype(str)
        if "decoy" in df.columns:
            df = df[~df["decoy"]]
        s = actives.setdefault(pathogen, set())
        for smi in df.loc[df["bin"] == 1, "smiles"]:
            _, ik = smiles_to_inchikey(smi)
            if ik is not None:
                s.add(ik)
    return actives


def audit_dataset(path: str, pathogen_active_iks: set) -> dict:
    pathogen = os.path.basename(os.path.dirname(path))
    name     = os.path.basename(path)[:-4]
    source   = _source_of(name)

    df = pd.read_csv(path)
    df = df.dropna(subset=["smiles"]).copy()
    df["smiles"] = df["smiles"].astype(str)

    df["ik"] = df["smiles"].map(lambda s: smiles_to_inchikey(s)[1])

    n_rows       = len(df)
    n_unparsable = int(df["ik"].isna().sum())
    valid        = df.dropna(subset=["ik"])

    g                       = valid.groupby("ik")
    n_compounds             = g.ngroups
    dup_compounds           = int((g.size() > 1).sum())
    dup_compounds_multi_smi = int((g["smiles"].nunique() > 1).sum())
    redundant_rows          = int(n_rows - n_unparsable - n_compounds)
    conflict_compounds      = int((g["bin"].nunique() > 1).sum())

    if "decoy" in df.columns:
        decoy_iks = set(df.loc[df["decoy"] & df["ik"].notna(), "ik"])
        decoy_active_collisions = len(decoy_iks & pathogen_active_iks)
    else:
        decoy_active_collisions = 0

    return {
        "pathogen":                pathogen,
        "source":                  source,
        "name":                    name,
        "rows":                    n_rows,
        "unparsable":              n_unparsable,
        "unique_compounds":        n_compounds,
        "dup_compounds":           dup_compounds,
        "dup_via_diff_smiles":     dup_compounds_multi_smi,
        "redundant_rows":          redundant_rows,
        "conflict_compounds":      conflict_compounds,
        "decoy_active_collisions": decoy_active_collisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_dir", default=DATASETS_DIR)
    parser.add_argument("--output",       default=OUT_PATH)
    args = parser.parse_args()

    files = sorted(
        f for f in glob.glob(os.path.join(args.datasets_dir, "*", "*.csv"))
        if "metadata" not in os.path.basename(f)
    )
    if not files:
        print(f"No datasets found under {args.datasets_dir}")
        return

    print("Building per-pathogen active InChIKey sets...")
    pathogen_actives = build_pathogen_active_iks(args.datasets_dir)
    print(f"  {sum(len(s) for s in pathogen_actives.values()):,} actives across "
          f"{len(pathogen_actives)} pathogens")

    rows = []
    for f in files:
        pathogen = os.path.basename(os.path.dirname(f))
        row = audit_dataset(f, pathogen_actives.get(pathogen, set()))
        rows.append(row)
        print(f"  [{row['pathogen']}/{row['name']}] rows={row['rows']} "
              f"dup={row['dup_compounds']} conflict={row['conflict_compounds']} "
              f"decoy_active_collisions={row['decoy_active_collisions']}",
              flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print(f"\n=== SAVED {args.output} ({len(out)} datasets) ===")


if __name__ == "__main__":
    main()
