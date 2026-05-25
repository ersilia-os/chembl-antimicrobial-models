"""
Step 03 — Extract active compounds from ChEMBL and PubChem datasets.

Reads processed dataset metadata from:
  - data/processed/chembl/01_chembl_datasets_all.csv
  - data/processed/pubchem/02_pubchem_datasets_organism.csv

For each ChEMBL dataset, opens the corresponding zip archive and extracts
SMILES with bin == 1.

Zip file mapping:
  Labels A, B, M  ->  data/raw/chembl/<pathogen>/19_final_datasets.zip
                       file inside: {name}.csv  (columns: smiles, bin)
  Label G         ->  data/raw/chembl/<pathogen>/20_general_no_pubchem_datasets_middle.zip
                       (preferred) or 20_general_datasets_middle.zip
                       file inside: ORG_{activity_type}_{unit}_{cutoff}.csv.gz
                                    (columns: compound_chembl_id, bin, smiles)
  Label G (aggregate) -> data/raw/chembl/<pathogen>/G_ORG_DR.csv or G_ORG_SP.csv
                       (columns: smiles, bin)

For each PubChem dataset, opens the corresponding raw assay CSV under
data/raw/pubchem/<pathogen>/<name>.csv and extracts active SMILES from the
compound-level activity labels.

Raw SMILES are deduplicated by InChIKey: all SMILES that map to the same
InChIKey are collapsed into one row. SMILES that RDKit cannot parse are
discarded. SMILES that parse but cannot be assigned an InChIKey are kept
as their own entry (no merging).

Produces output/03_select_positives/03_selected_positives.csv with one row per unique
compound, sorted by canonical_smiles, with columns:
  - canonical_smiles : RDKit canonical SMILES (primary identifier downstream)
  - smiles           : semicolon-separated list of all raw SMILES that mapped
                       to this compound (may be a single entry)
  - inchikey         : InChIKey used for deduplication (null for edge-case molecules)
  - n_active         : number of datasets in which the compound was active
  - found_in         : semicolon-separated list of source|pathogen|dataset tags
                       (aggregated across all raw SMILES in the group)
  - split            : integer split index (0 = first --split_size compounds, 1 = next, …)

Usage:
    python scripts/03_select_positives.py
"""

import io
import os
import sys
import zipfile
from collections import defaultdict

import pandas as pd
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(ROOT, "..")
sys.path.append(os.path.join(ROOT, "..", "src"))

from default import (
    CHEMBL_ZIP_FINAL,
    CHEMBL_ZIP_GENERAL,
    CHEMBL_ZIP_GENERAL_NO_PUBCHEM,
    COL_BIN,
    COL_CANONICAL_SMILES,
    COL_FOUND_IN,
    COL_INCHIKEY,
    COL_SMILES,
    SPLIT_SIZE,
)

CHEMBL_DATASETS = os.path.join(REPO_ROOT, "data", "processed", "chembl", "01_chembl_datasets_all.csv")
PUBCHEM_DATASETS = os.path.join(REPO_ROOT, "data", "processed", "pubchem", "02_pubchem_datasets_organism.csv")
output_dir = os.path.join(REPO_ROOT, "output", "03_select_positives")
OUTPUT_PATH = os.path.join(output_dir, "03_selected_positives.csv")
SMILES_OUTPUT_PATH = os.path.join(output_dir, "selected_positive_smiles.csv")
os.makedirs(output_dir, exist_ok=True)

def read_chembl_dataset(row: pd.Series) -> pd.Series | None:
    pathogen = row["pathogen"]
    label = row["label"]
    name = row["name"]

    if label in {"A", "B", "M"}:
        zip_path = os.path.join(REPO_ROOT, "data", "raw", "chembl", pathogen, CHEMBL_ZIP_FINAL)
        inner_name = f"{name}.csv"
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(inner_name) as f:
                df = pd.read_csv(f)

    elif label == "G" and row.get("assay_type") == "general_aggregate":
        csv_path = os.path.join(REPO_ROOT, "data", "raw", "chembl", pathogen, f"{name}.csv")
        if not os.path.exists(csv_path):
            return None
        df = pd.read_csv(csv_path)

    elif label == "G":
        base_dir = os.path.join(REPO_ROOT, "data", "raw", "chembl", pathogen)
        no_pubchem = os.path.join(base_dir, CHEMBL_ZIP_GENERAL_NO_PUBCHEM)
        zip_path = no_pubchem if os.path.exists(no_pubchem) else os.path.join(base_dir, CHEMBL_ZIP_GENERAL)
        inner_name = f"ORG_{row['activity_type']}_{row['unit']}_{row['cutoff']}.csv.gz"
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(inner_name) as f:
                df = pd.read_csv(io.BytesIO(f.read()), compression="gzip")

    else:
        return None

    actives = df[df[COL_BIN] == 1][COL_SMILES].dropna()
    return actives


def read_pubchem_dataset(row: pd.Series) -> pd.Series | None:
    assay_path = os.path.join(REPO_ROOT, "data", "raw", "pubchem", row["pathogen_code"], f"{int(row['aid'])}.csv")
    if not os.path.exists(assay_path):
        return None
    df = pd.read_csv(assay_path)
    if COL_SMILES not in df.columns or COL_BIN not in df.columns:
        return None
    return df.loc[df[COL_BIN] == 1, COL_SMILES].dropna()


def collect_actives(meta: pd.DataFrame, reader, source: str, pathogen_col: str = "pathogen", name_col: str = "name") -> dict[str, set[str]]:
    actives: dict[str, set[str]] = defaultdict(set)

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc=source, unit="dataset"):
        smiles_series = reader(row)
        if smiles_series is None or smiles_series.empty:
            continue
        tag = f"{source}|{row[pathogen_col]}|{row[name_col]}"
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

    for smi, tags in tqdm(actives.items(), desc="canonicalizing", unit="smi"):
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
                COL_CANONICAL_SMILES: canonical,
                COL_INCHIKEY: inchikey,
                "raw_smiles": set(),
                COL_FOUND_IN: set(),
            }
        groups[group_key]["raw_smiles"].add(smi)
        groups[group_key][COL_FOUND_IN].update(tags)

    print(f"  Raw SMILES collected : {len(actives):,}")
    print(f"  Discarded (bad parse): {n_discarded:,}")
    print(f"  Unique compounds     : {len(groups):,}")

    rows = []
    for g in groups.values():
        rows.append({
            COL_CANONICAL_SMILES: g[COL_CANONICAL_SMILES],
            COL_SMILES: ";".join(sorted(g["raw_smiles"])),
            COL_INCHIKEY: g[COL_INCHIKEY],
            "n_active": len(g[COL_FOUND_IN]),
            COL_FOUND_IN: ";".join(sorted(g[COL_FOUND_IN])),
        })

    return pd.DataFrame(rows)


def main(split_size: int = SPLIT_SIZE) -> None:
    for path in (CHEMBL_DATASETS, PUBCHEM_DATASETS):
        if not os.path.exists(path):
            print(f"[ERROR] Expected input not found: {path}", file=sys.stderr)
            sys.exit(1)

    chembl_meta = pd.read_csv(CHEMBL_DATASETS)
    pubchem_meta = pd.read_csv(PUBCHEM_DATASETS)

    chembl_actives = collect_actives(chembl_meta, read_chembl_dataset, source="chembl")
    pubchem_actives = collect_actives(pubchem_meta, read_pubchem_dataset, source="pubchem", pathogen_col="pathogen_code", name_col="aid")
    actives = merge_actives(chembl_actives, pubchem_actives)

    result = canonicalize_actives(actives)
    result = result.sort_values(COL_CANONICAL_SMILES).reset_index(drop=True)
    result["split"] = result.index // split_size
    result = result[[COL_CANONICAL_SMILES, COL_SMILES, COL_INCHIKEY, "n_active", COL_FOUND_IN, "split"]]

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")

    result[[COL_CANONICAL_SMILES]].rename(columns={COL_CANONICAL_SMILES: COL_SMILES}).to_csv(SMILES_OUTPUT_PATH, index=False)
    print(f"Saved SMILES-only input to {SMILES_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
