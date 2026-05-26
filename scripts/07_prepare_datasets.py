"""
Step 07 — Prepare datasets for model training.

(1) Loads and normalises dataset metadata from:
      - data/processed/chembl/01_chembl_datasets_all.csv
        → adds 'inactives' (compounds − positives)
      - data/processed/pubchem/02_pubchem_datasets_organism.csv
        → drops 'cids', 'inconclusive', 'unspecified'; renames pathogen_code,
          aid, actives; uppercases 'target_type'; adds source = 'pubchem'
    Both tables are concatenated and share a common 'target_type' column.
(2) Flags ChEMBL datasets superseded by a PubChem assay: if a PubChem row has
    a chembl_id and that ID appears in a ChEMBL dataset name, the ChEMBL dataset
    is marked keep=False. All other datasets are keep=True.
(3) Loads decoy data from output/06_decoys/06_eos3e6s_v1.csv and builds a
    dict mapping each canonical SMILES to up to N_DECOYS randomly sampled decoys.
(4) Loads output/03_select_positives/03_selected_positives.csv to build a raw SMILES →
    canonical SMILES lookup, so decoys can be assigned to raw SMILES in datasets.
(5) Extracts raw compound CSVs (keep=True only) from per-pathogen zip archives
    (ChEMBL) or flat CSVs (PubChem); writes to
    output/07_datasets/{pathogen}/{name}.csv with columns smiles, bin.
    Raises ValueError if any bin value outside {0, 1} is found.
(6) For datasets with ratio > 0.5, augments with decoy compounds to bring
    the active ratio down to ~0.1. Before sampling, decoy candidates that appear
    as actives (bin=1) in any dataset of the same pathogen are excluded from the
    pool. Augmented datasets gain a 'decoy' column (False for original rows, True
    for added decoys). Datasets below the threshold are not modified and will not
    contain a 'decoy' column.
(7) Saves output/07_datasets/07_datasets_metadata.csv — the normalised combined
    metadata (all rows, including keep=False) with additional columns: 'decoys'
    (number of decoy rows added, 0 for non-augmented datasets), 'final_ratio'
    (ratio after augmentation), and 'final_compounds' (total rows after
    augmentation); sorted by pathogen.

Usage:
    python scripts/07_prepare_datasets.py
"""

import gzip
import io
import os
import random
import sys
import zipfile

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
sys.path.append(os.path.join(ROOT, "..", "src"))

from default import (
    CHEMBL_ZIP_FINAL,
    CHEMBL_ZIP_GENERAL,
    CHEMBL_ZIP_GENERAL_HIGH,
    CHEMBL_ZIP_GENERAL_NO_PUBCHEM,
    CHEMBL_ZIP_GENERAL_NO_PUBCHEM_HIGH,
    HIGH_RATIO_THRESHOLD,
    N_DECOYS,
    RANDOM_SEED,
    TARGET_RATIO,
)

CHEMBL_METADATA_PATH = os.path.join(REPO_ROOT, "data", "processed", "chembl", "01_chembl_datasets_all.csv")
PUBCHEM_METADATA_PATH = os.path.join(REPO_ROOT, "data", "processed", "pubchem", "02_pubchem_datasets_organism.csv")
DECOYS_PATH           = os.path.join(REPO_ROOT, "output", "06_decoys", "06_eos3e6s_v1.csv")
POSITIVES_PATH        = os.path.join(REPO_ROOT, "output", "03_select_positives", "03_selected_positives.csv")
RAW_DIR               = os.path.join(REPO_ROOT, "data", "raw")
OUT_DIR               = os.path.join(REPO_ROOT, "output", "07_datasets")
META_OUT_PATH         = os.path.join(REPO_ROOT, "output", "07_datasets", "07_datasets_metadata.csv")


def _validate_bin(df: pd.DataFrame, name: str) -> None:
    invalid = df.loc[~df["bin"].isin([0, 1]), "bin"].unique()
    if len(invalid) > 0:
        raise ValueError(f"Dataset {name!r} contains invalid bin values: {invalid.tolist()}")


# ---------------------------------------------------------------------------
# Step 1 — load metadata
# ---------------------------------------------------------------------------

def load_metadata(chembl_path: str, pubchem_path: str) -> pd.DataFrame:
    chembl = pd.read_csv(chembl_path)
    chembl["inactives"] = chembl["compounds"] - chembl["positives"]


    pubchem = pd.read_csv(pubchem_path)
    pubchem = pubchem.drop(columns=["cids", "inconclusive", "unspecified"])
    pubchem = pubchem.rename(columns={"pathogen_code": "pathogen", "aid": "name", "actives": "positives"})
    pubchem["source"] = "pubchem"
    pubchem["target_type"] = pubchem["target_type"].str.upper()

    df = pd.concat([chembl, pubchem], ignore_index=True)
    if "auroc" in df.columns:
        df = df.rename(columns={"auroc": "auroc_baseline"})
    print(f"Loaded metadata: {len(df)} datasets across {df['pathogen'].nunique()} pathogens "
          f"({len(chembl)} ChEMBL, {len(pubchem)} PubChem)")
    return df


def flag_chembl_duplicates(metadata: pd.DataFrame) -> pd.DataFrame:
    chembl_to_pubchem = {
        str(row["chembl_id"]): row
        for _, row in metadata[metadata["source"] == "pubchem"].iterrows()
        if pd.notna(row["chembl_id"])
    }
    metadata["keep"] = True
    if not chembl_to_pubchem:
        return metadata

    chembl_rows = metadata[metadata["source"] == "chembl"]
    leading_ids = chembl_rows["name"].str.extract(r"^(CHEMBL\d+)", expand=False)
    flagged_indices = chembl_rows.index[leading_ids.isin(chembl_to_pubchem)]
    metadata.loc[flagged_indices, "keep"] = False

    print(f"Flagged {len(flagged_indices)} ChEMBL dataset(s) as keep=False (superseded by PubChem assays):")
    for idx in flagged_indices:
        crow = metadata.loc[idx]
        prow = chembl_to_pubchem[leading_ids.loc[idx]]
        pubchem_info = f"AID {int(prow['name'])} ({int(prow['compounds'])} cpds, {int(prow['positives'])} actives)"
        print(f"  [{crow['pathogen']}] {crow['name']} ({int(crow['compounds'])} cpds, {int(crow['positives'])} actives)  →  PubChem {pubchem_info}")
    return metadata


# ---------------------------------------------------------------------------
# Step 2 — load decoys
# ---------------------------------------------------------------------------

def load_decoys(decoys_path: str) -> dict[str, list[str]]:
    print("Loading decoy data...")
    df = pd.read_csv(decoys_path)
    decoy_cols = [c for c in df.columns if c not in ("key", "input")]
    result = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading decoys", unit="cpd"):
        candidates = row[decoy_cols].dropna().tolist()
        if not candidates:
            continue
        result[row["input"]] = (
            candidates if len(candidates) <= N_DECOYS
            else random.sample(candidates, N_DECOYS)
        )
    counts = [len(v) for v in result.values()]
    print(f"Loaded decoys: {len(result)} compounds with {min(counts)}–{max(counts)} decoys each")
    return result


# ---------------------------------------------------------------------------
# Step 3 — load raw → canonical SMILES mapping
# ---------------------------------------------------------------------------

def load_raw_to_canonical(positives_path: str) -> dict[str, str]:
    df = pd.read_csv(positives_path, usecols=["canonical_smiles", "smiles"])
    mapping = {}
    for _, row in df.iterrows():
        canonical = row["canonical_smiles"]
        for raw in str(row["smiles"]).split(";"):
            mapping[raw.strip()] = canonical
    print(f"Loaded raw→canonical mapping: {len(mapping):,} raw SMILES entries")
    return mapping


# ---------------------------------------------------------------------------
# Step 4 — extract datasets
# ---------------------------------------------------------------------------

def _chembl_inner_filename(row: pd.Series) -> str:
    if row["label"] in ("A", "B", "M"):
        return f"{row['name']}.csv"
    return f"ORG_{row['activity_type']}_{row['unit']}_{row['cutoff']}.csv.gz"


def _chembl_zip_path(row: pd.Series) -> str:
    base = os.path.join(RAW_DIR, "chembl", row["pathogen"])
    if row["label"] in ("A", "B", "M"):
        return os.path.join(base, CHEMBL_ZIP_FINAL)
    if row["pathogen"] == "pfalciparum":
        no_pubchem = os.path.join(base, CHEMBL_ZIP_GENERAL_NO_PUBCHEM_HIGH)
        return no_pubchem if os.path.exists(no_pubchem) else os.path.join(base, CHEMBL_ZIP_GENERAL_HIGH)
    no_pubchem = os.path.join(base, CHEMBL_ZIP_GENERAL_NO_PUBCHEM)
    return no_pubchem if os.path.exists(no_pubchem) else os.path.join(base, CHEMBL_ZIP_GENERAL)


def _extract_chembl(row: pd.Series) -> pd.DataFrame:
    if row.get("assay_type") == "general_aggregate":
        csv_path = os.path.join(RAW_DIR, "chembl", row["pathogen"], f"{row['name']}.csv")
        df = pd.read_csv(csv_path)[["smiles", "bin"]]
        _validate_bin(df, row["name"])
        return df

    inner = _chembl_inner_filename(row)
    with zipfile.ZipFile(_chembl_zip_path(row)) as zf:
        with zf.open(inner) as f:
            raw = gzip.open(f).read() if inner.endswith(".gz") else f.read()
    df = pd.read_csv(io.BytesIO(raw))[["smiles", "bin"]]
    _validate_bin(df, row["name"])
    return df


def _extract_pubchem(row: pd.Series) -> pd.DataFrame | None:
    assay_path = os.path.join(RAW_DIR, "pubchem", row["pathogen"], f"{int(row['name'])}.csv")
    if not os.path.exists(assay_path):
        return None
    df = pd.read_csv(assay_path)
    if "smiles" not in df.columns or "bin" not in df.columns:
        return None
    df = df[["smiles", "bin"]].dropna(subset=["smiles"])
    _validate_bin(df, row["name"])
    return df.reset_index(drop=True)


def extract_datasets(metadata: pd.DataFrame) -> None:
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Extracting datasets", unit="dataset"):
        if not row["keep"]:
            continue
        out_path = os.path.join(OUT_DIR, row["pathogen"], f"{row['name']}.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if row["source"] == "chembl":
            df = _extract_chembl(row)
        else:
            df = _extract_pubchem(row)
            if df is None:
                print(f"  [WARN] PubChem dataset not found or missing columns: {row['name']}")
                continue

        df.to_csv(out_path, index=False)

    print(f"Datasets written to {OUT_DIR}")


# ---------------------------------------------------------------------------
# Step 5 — augment high-ratio datasets with decoys
# ---------------------------------------------------------------------------

def augment_datasets(
    metadata: pd.DataFrame,
    decoys: dict[str, list[str]],
    raw_to_canonical: dict[str, str],
) -> pd.DataFrame:
    meta = metadata.copy()
    meta["decoys"] = 0
    meta["final_ratio"] = meta["ratio"]

    # Build per-pathogen set of all active SMILES across all extracted datasets,
    # so decoys that are known actives in any assay of the same pathogen are excluded.
    print("Building per-pathogen active SMILES sets for decoy filtering...")
    pathogen_actives: dict[str, set[str]] = {}
    for pathogen, group in meta[meta["keep"]].groupby("pathogen"):
        active_set: set[str] = set()
        for name in group["name"]:
            fpath = os.path.join(OUT_DIR, pathogen, f"{name}.csv")
            if os.path.exists(fpath):
                df_tmp = pd.read_csv(fpath, usecols=["smiles", "bin"])
                active_set.update(df_tmp.loc[df_tmp["bin"] == 1, "smiles"])
        pathogen_actives[pathogen] = active_set
    total_actives = sum(len(s) for s in pathogen_actives.values())
    print(f"  {total_actives:,} unique active SMILES across {len(pathogen_actives)} pathogens")

    high_mask = (meta["ratio"] > HIGH_RATIO_THRESHOLD) & meta["keep"]
    print(f"Augmenting {high_mask.sum()} datasets with ratio > {HIGH_RATIO_THRESHOLD}")

    for idx, row in tqdm(meta[high_mask].iterrows(), total=high_mask.sum(), desc="Augmenting datasets", unit="dataset"):
        out_path = os.path.join(OUT_DIR, row["pathogen"], f"{row['name']}.csv")
        df = pd.read_csv(out_path)
        df["decoy"] = False

        n_pos = int((df["bin"] == 1).sum())
        n_total = len(df)
        n_needed = 10 * n_pos - n_total

        pathogen_active_smiles = pathogen_actives.get(row["pathogen"], set())
        pool = list({
            decoy_smi
            for raw_smi in df.loc[df["bin"] == 1, "smiles"]
            for canon_smi in [raw_to_canonical.get(raw_smi, raw_smi)]
            if canon_smi in decoys
            for decoy_smi in decoys[canon_smi]
            if decoy_smi not in pathogen_active_smiles
        })

        if not pool:
            print(f"  [WARN] {row['name']}: no decoys available, skipping augmentation")
            df.drop(columns=["decoy"]).to_csv(out_path, index=False)
            continue

        n_sample = min(n_needed, len(pool))

        # canonical SMILES already present in the dataset — used to filter decoys
        existing_canonical = set()
        for smi in df["smiles"]:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                existing_canonical.add(Chem.MolToSmiles(mol))

        random.shuffle(pool)
        selected = []
        n_invalid = 0
        n_duplicate = 0
        for smi in pool:
            if len(selected) >= n_sample:
                break
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                n_invalid += 1
                continue
            canon = Chem.MolToSmiles(mol)
            if canon in existing_canonical:
                n_duplicate += 1
                continue
            selected.append(smi)
            existing_canonical.add(canon)

        if n_duplicate:
            print(f"  [INFO] {row['name']}: skipped {n_duplicate} decoys already present in dataset")

        if n_invalid:
            print(f"  [WARN] {row['name']}: dropped {n_invalid} invalid decoy SMILES")
        n_sample = len(selected)

        achieved = round(n_pos / (n_total + n_sample), 3)
        if n_sample < n_needed:
            print(
                f"  [WARN] {row['name']}: pool has {len(pool)} decoys, "
                f"needed {n_needed}; achieved ratio {achieved} (target {TARGET_RATIO})"
            )

        new_rows = pd.DataFrame({
            "smiles": selected,
            "bin": 0,
            "decoy": True,
        })
        pd.concat([df, new_rows], ignore_index=True).to_csv(out_path, index=False)

        meta.at[idx, "decoys"] = n_sample
        meta.at[idx, "final_ratio"] = achieved

    for idx, row in meta.iterrows():
        path = os.path.join(OUT_DIR, row["pathogen"], f"{row['name']}.csv")
        if os.path.exists(path):
            meta.at[idx, "final_compounds"] = len(pd.read_csv(path))
        else:
            meta.at[idx, "final_compounds"] = row["compounds"] + row["decoys"]
    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    metadata = load_metadata(CHEMBL_METADATA_PATH, PUBCHEM_METADATA_PATH)
    metadata = flag_chembl_duplicates(metadata)
    decoys = load_decoys(DECOYS_PATH)
    raw_to_canonical = load_raw_to_canonical(POSITIVES_PATH)
    extract_datasets(metadata)
    enriched = augment_datasets(metadata, decoys, raw_to_canonical)
    enriched = enriched.sort_values("pathogen").reset_index(drop=True)
    n_dropped = (~enriched["keep"]).sum()
    enriched = enriched[enriched["keep"]].drop(columns=["keep"]).reset_index(drop=True)
    if n_dropped:
        print(f"Dropped {n_dropped} keep=False dataset(s) from metadata")
    os.makedirs(os.path.dirname(META_OUT_PATH), exist_ok=True)
    enriched.to_csv(META_OUT_PATH, index=False)
    print(f"Saved enriched metadata ({len(enriched)} datasets) to {META_OUT_PATH}")


if __name__ == "__main__":
    main()
