"""
Step 03 — Extract active compounds from ChEMBL and PubChem datasets.

Reads processed dataset metadata from:
  - data/processed/chembl/01_chembl_datasets_all.csv
  - data/processed/pubchem/02_pubchem_datasets_all.csv

For each ChEMBL dataset, opens the corresponding zip archive and extracts
SMILES with bin == 1.

Zip file mapping:
  Labels A, B, M  ->  data/raw/chembl/<pathogen>/19_final_datasets.zip
                       file inside: {name}.csv  (columns: smiles, bin)
  Label G         ->  data/raw/chembl/<pathogen>/20_general_datasets.zip
                       file inside: ORG_{activity_type}_{unit}_{cutoff}.csv.gz
                                    (columns: compound_chembl_id, bin, smiles)

For each PubChem dataset, opens the corresponding raw assay CSV under
data/raw/pubchem/<pathogen>/<aid>.csv and extracts active SMILES from the
compound-level activity labels.

Raw SMILES are deduplicated by InChIKey: all SMILES that map to the same
InChIKey are collapsed into one row. SMILES that RDKit cannot parse are
discarded.

Produces output/results/03_selected_positives.csv with one row per unique
compound (InChIKey), sorted by canonical_smiles, with columns:
  - canonical_smiles : RDKit canonical SMILES (primary identifier)
  - smiles           : semicolon-separated list of all raw SMILES for this compound
  - inchikey         : InChIKey (deduplication key; null for edge-case molecules)
  - n_active         : number of datasets in which the compound was active
  - found_in         : semicolon-separated list of source|pathogen|dataset tags
  - split            : integer split index (0 = first --split_size compounds, 1 = next, …)

Usage:
    python scripts/03_select_positives.py
    python scripts/03_select_positives.py --chembl_datasets path/to/chembl.csv
    python scripts/03_select_positives.py --pubchem_datasets path/to/pubchem.csv
    python scripts/03_select_positives.py --split_size 1000
"""

import argparse
import io
import os
import zipfile
from collections import defaultdict

import pandas as pd
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(ROOT, "..")
CHEMBL_DATASETS = os.path.join(REPO_ROOT, "data", "processed", "chembl", "01_chembl_datasets_all.csv")
PUBCHEM_DATASETS = os.path.join(REPO_ROOT, "data", "processed", "pubchem", "02_pubchem_datasets_all.csv")
OUTPUT_PATH = os.path.join(REPO_ROOT, "output", "results", "03_selected_positives.csv")


def read_chembl_dataset(row: pd.Series) -> pd.Series | None:
    pathogen = row["pathogen"]
    label = row["label"]
    name = row["name"]

    if label in {"A", "B", "M"}:
        zip_path = os.path.join(REPO_ROOT, "data", "raw", "chembl", pathogen, "19_final_datasets.zip")
        inner_name = f"{name}.csv"
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(inner_name) as f:
                df = pd.read_csv(f)

    elif label == "G":
        zip_path = os.path.join(REPO_ROOT, "data", "raw", "chembl", pathogen, "20_general_datasets.zip")
        inner_name = f"ORG_{row['activity_type']}_{row['unit']}_{row['cutoff']}.csv.gz"
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(inner_name) as f:
                df = pd.read_csv(io.BytesIO(f.read()), compression="gzip")

    else:
        return None

    actives = df[df["bin"] == 1]["smiles"].dropna()
    return actives


def read_pubchem_dataset(row: pd.Series) -> pd.Series | None:
    assay_path = os.path.join(REPO_ROOT, "data", "raw", "pubchem", row["pathogen"], f"{row['name']}.csv")
    if not os.path.exists(assay_path):
        return None

    df = pd.read_csv(assay_path)
    if "smiles" not in df.columns:
        return None

    if "activity" in df.columns:
        return df.loc[pd.to_numeric(df["activity"], errors="coerce") == 1, "smiles"].dropna()
    elif "bin" in df.columns:
        return df.loc[df["bin"] == 1, "smiles"].dropna()
    else:
        return None


def collect_actives(meta: pd.DataFrame, reader, source: str) -> dict[str, set[str]]:
    actives: dict[str, set[str]] = defaultdict(set)

    for _, row in meta.iterrows():
        smiles_series = reader(row)
        if smiles_series is None or smiles_series.empty:
            continue
        tag = f"{source}|{row['pathogen']}|{row['name']}"
        for smi in smiles_series.unique():
            actives[smi].add(tag)
    return actives


def merge_actives(*active_maps: dict[str, set[str]]) -> dict[str, set[str]]:
    merged: dict[str, set[str]] = defaultdict(set)
    for active_map in active_maps:
        for smiles, tags in active_map.items():
            merged[smiles].update(tags)
    return merged


def canonicalize_actives(actives: dict[str, set[str]]) -> pd.DataFrame:
    # group_key -> {canonical_smiles, inchikey, raw_smiles set, found_in set}
    groups: dict[str, dict] = {}
    n_discarded = 0

    for smi, tags in actives.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_discarded += 1
            continue

        canonical = Chem.MolToSmiles(mol) or smi
        inchikey = MolToInchiKey(mol) or None

        # fall back to canonical_smiles as grouping key for edge-case molecules
        group_key = inchikey if inchikey else canonical

        if group_key not in groups:
            groups[group_key] = {
                "canonical_smiles": canonical,
                "inchikey": inchikey,
                "raw_smiles": set(),
                "found_in": set(),
            }
        groups[group_key]["raw_smiles"].add(smi)
        groups[group_key]["found_in"].update(tags)

    print(f"  Raw SMILES collected : {len(actives):,}")
    print(f"  Discarded (bad parse): {n_discarded:,}")
    print(f"  Unique compounds     : {len(groups):,}")

    rows = []
    for g in groups.values():
        rows.append({
            "canonical_smiles": g["canonical_smiles"],
            "smiles": ";".join(sorted(g["raw_smiles"])),
            "inchikey": g["inchikey"],
            "n_active": len(g["found_in"]),
            "found_in": ";".join(sorted(g["found_in"])),
        })

    return pd.DataFrame(rows)


def main(chembl_datasets_path: str, pubchem_datasets_path: str, split_size: int) -> None:
    chembl_meta = pd.read_csv(chembl_datasets_path)
    pubchem_meta = pd.read_csv(pubchem_datasets_path)

    chembl_actives = collect_actives(chembl_meta, read_chembl_dataset, source="chembl")
    pubchem_actives = collect_actives(pubchem_meta, read_pubchem_dataset, source="pubchem")
    actives = merge_actives(chembl_actives, pubchem_actives)

    result = canonicalize_actives(actives)
    result = result.sort_values("canonical_smiles").reset_index(drop=True)
    result["split"] = result.index // split_size
    result = result[["canonical_smiles", "smiles", "inchikey", "n_active", "found_in", "split"]]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract active compounds from all ChEMBL and PubChem datasets."
    )
    parser.add_argument(
        "--chembl_datasets",
        type=str,
        default=CHEMBL_DATASETS,
        help="Path to the ChEMBL datasets CSV (default: data/processed/chembl/01_chembl_datasets_all.csv).",
    )
    parser.add_argument(
        "--pubchem_datasets",
        type=str,
        default=PUBCHEM_DATASETS,
        help="Path to the PubChem datasets CSV (default: data/processed/pubchem/02_pubchem_datasets_all.csv).",
    )
    parser.add_argument(
        "--split_size",
        type=int,
        default=500,
        help="Number of compounds per split (default: 500).",
    )
    args = parser.parse_args()
    main(args.chembl_datasets, args.pubchem_datasets, args.split_size)
