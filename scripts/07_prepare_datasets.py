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
(3) Loads decoy data from output/06_decoys/06_eos3e6s_v1.csv and builds a dict
    mapping each input compound's InChIKey to up to N_DECOYS randomly sampled decoys.
(4) Extracts raw compound CSVs (keep=True only) from per-pathogen zip archives
    (ChEMBL) or flat CSVs (PubChem); writes to
    output/07_datasets/{pathogen}/{name}.csv with columns smiles, bin.
    Raises ValueError if any bin value outside {0, 1} is found. Each dataset is
    deduplicated at the InChIKey level (dedup_by_inchikey): one row per molecule,
    SMILES stored as RDKit canonical SMILES, and active wins (bin=1) on an
    active/inactive label conflict. SMILES RDKit cannot parse are kept as-is.
    (ChEMBL is already deduplicated upstream so this is largely a no-op there;
    PubChem carries genuine duplicates and label conflicts that this resolves.)
(5) For datasets with ratio > 0.5, augments with decoy compounds to bring
    the active ratio down to ~0.1. Before sampling, decoy candidates whose
    InChIKey matches a compound already in the dataset, or an active (bin=1) in
    any dataset of the same pathogen, are excluded; selected decoys are stored as
    canonical SMILES. Augmented datasets gain a 'decoy' column (False for original
    rows, True for added decoys). Datasets below the threshold are not modified and
    will not contain a 'decoy' column.
(6) Saves output/07_datasets/07_datasets_metadata.csv — the normalised combined
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
RAW_DIR               = os.path.join(REPO_ROOT, "data", "raw")
OUT_DIR               = os.path.join(REPO_ROOT, "output", "07_datasets")
META_OUT_PATH         = os.path.join(REPO_ROOT, "output", "07_datasets", "07_datasets_metadata.csv")


def _validate_bin(df: pd.DataFrame, name: str) -> None:
    invalid = df.loc[~df["bin"].isin([0, 1]), "bin"].unique()
    if len(invalid) > 0:
        raise ValueError(f"Dataset {name!r} contains invalid bin values: {invalid.tolist()}")


# Memoise SMILES -> (canonical SMILES, InChIKey) for the whole run. The same SMILES
# recurs heavily (duplicate strings within a dataset, shared compounds across datasets,
# and again between extraction, decoy loading and augmentation), and InChIKey generation
# is the dominant cost — caching avoids recomputing it. Keys/values are strings only
# (no RDKit mol objects are retained).
_IK_CACHE: dict[str, tuple] = {}


def smiles_to_inchikey(smi: str) -> tuple:
    """Return (canonical_smiles, inchikey) for a SMILES, or (None, None) if unparsable.

    Result is memoised in the module-level _IK_CACHE.
    """
    s = str(smi)
    cached = _IK_CACHE.get(s)
    if cached is not None:
        return cached
    mol = Chem.MolFromSmiles(s)
    result = (None, None) if mol is None else (Chem.MolToSmiles(mol), Chem.MolToInchiKey(mol))
    _IK_CACHE[s] = result
    return result


def dedup_by_inchikey(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Collapse rows to one per InChIKey.

    Parsable SMILES are grouped by InChIKey; each group becomes a single row whose
    SMILES is the RDKit canonical SMILES and whose bin is the max over the group
    (active wins on an active/inactive label conflict). Rows whose SMILES RDKit
    cannot parse are kept verbatim — never merged, never dropped.
    """
    canon_by_ik: dict[str, str] = {}   # inchikey -> canonical SMILES (first seen)
    bins_by_ik: dict[str, set] = {}    # inchikey -> set of bins observed
    order: list[str] = []              # inchikeys in first-appearance order
    unparsable: list[dict] = []        # rows kept as-is

    for smi, b in zip(df["smiles"], df["bin"]):
        canon, ik = smiles_to_inchikey(smi)
        if ik is None:
            unparsable.append({"smiles": smi, "bin": int(b)})
            continue
        if ik not in bins_by_ik:
            bins_by_ik[ik] = set()
            canon_by_ik[ik] = canon
            order.append(ik)
        bins_by_ik[ik].add(int(b))

    rows = [{"smiles": canon_by_ik[ik], "bin": max(bins_by_ik[ik])} for ik in order]
    n_conflicts = sum(1 for ik in order if len(bins_by_ik[ik]) > 1)
    n_collapsed = len(df) - len(order) - len(unparsable)
    if n_collapsed or n_conflicts or unparsable:
        print(f"  [{name}] IK-dedup: {len(df)} -> {len(rows) + len(unparsable)} rows "
              f"({n_collapsed} duplicate rows collapsed, {n_conflicts} active/inactive "
              f"conflicts resolved active-first, {len(unparsable)} unparsable kept as-is)")
    return pd.DataFrame(rows + unparsable, columns=["smiles", "bin"]).reset_index(drop=True)


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

    chembl_named = metadata[
        (metadata["source"] == "chembl") & metadata["name"].str.startswith("CHEMBL")
    ]
    leading_ids = chembl_named["name"].str.split("_", n=1).str[0]
    flagged_indices = chembl_named.index[leading_ids.isin(chembl_to_pubchem)]
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
    """Map each input compound's InChIKey to up to N_DECOYS sampled decoy SMILES.

    Keying on InChIKey (rather than a canonical-SMILES string) makes the decoy
    lookup robust to notation/canonicalisation differences between this script and
    whatever produced the decoy file. On an InChIKey collision the first list wins.
    """
    print("Loading decoy data...")
    df = pd.read_csv(decoys_path)
    decoy_cols = [c for c in df.columns if c not in ("key", "input")]
    result: dict[str, list[str]] = {}
    n_unparsable = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading decoys", unit="cpd"):
        _, ik = smiles_to_inchikey(row["input"])
        if ik is None:
            n_unparsable += 1
            continue
        candidates = row[decoy_cols].dropna().tolist()
        if not candidates:
            continue
        sampled = candidates if len(candidates) <= N_DECOYS else random.sample(candidates, N_DECOYS)
        result.setdefault(ik, sampled)
    counts = [len(v) for v in result.values()]
    msg = f"Loaded decoys: {len(result)} InChIKeys with {min(counts)}–{max(counts)} decoys each"
    if n_unparsable:
        msg += f" ({n_unparsable} unparsable inputs skipped)"
    print(msg)
    return result


# ---------------------------------------------------------------------------
# Step 3 — extract datasets
# ---------------------------------------------------------------------------

def _chembl_inner_filename(row: pd.Series) -> str:
    if row["label"] in ("A", "B", "M"):
        return f"{row['name']}.csv"
    return f"{row['name']}.csv.gz"


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

        df = dedup_by_inchikey(df, row["name"])
        df.to_csv(out_path, index=False)

    print(f"Datasets written to {OUT_DIR}")


# ---------------------------------------------------------------------------
# Step 5 — augment high-ratio datasets with decoys
# ---------------------------------------------------------------------------

def augment_datasets(
    metadata: pd.DataFrame,
    decoys: dict[str, list[str]],
) -> pd.DataFrame:
    meta = metadata.copy()

    # The extracted CSVs are now InChIKey-deduplicated, so recompute compounds/positives/
    # ratio (and inactives) from the actual rows. This mainly affects PubChem datasets and
    # also sets the ratio used by the high-ratio augmentation gate below.
    print("Recomputing compound counts from deduped datasets...")
    for idx, row in meta[meta["keep"]].iterrows():
        fpath = os.path.join(OUT_DIR, row["pathogen"], f"{row['name']}.csv")
        if not os.path.exists(fpath):
            continue
        df_tmp = pd.read_csv(fpath, usecols=["smiles", "bin"])
        n_comp = len(df_tmp)
        n_pos = int((df_tmp["bin"] == 1).sum())
        meta.at[idx, "compounds"] = n_comp
        meta.at[idx, "positives"] = n_pos
        meta.at[idx, "inactives"] = n_comp - n_pos
        meta.at[idx, "ratio"] = round(n_pos / n_comp, 3) if n_comp else 0.0

    meta["decoys"] = 0
    meta["final_ratio"] = meta["ratio"]

    # Build per-pathogen set of active-compound InChIKeys across all extracted datasets,
    # so decoys that are known actives in any assay of the same pathogen are excluded.
    print("Building per-pathogen active InChIKey sets for decoy filtering...")
    pathogen_actives: dict[str, set[str]] = {}
    for pathogen, group in meta[meta["keep"]].groupby("pathogen"):
        active_set: set[str] = set()
        for name in group["name"]:
            fpath = os.path.join(OUT_DIR, pathogen, f"{name}.csv")
            if os.path.exists(fpath):
                df_tmp = pd.read_csv(fpath, usecols=["smiles", "bin"])
                for smi in df_tmp.loc[df_tmp["bin"] == 1, "smiles"]:
                    _, ik = smiles_to_inchikey(smi)
                    if ik is not None:
                        active_set.add(ik)
        pathogen_actives[pathogen] = active_set
    total_actives = sum(len(s) for s in pathogen_actives.values())
    print(f"  {total_actives:,} unique active InChIKeys across {len(pathogen_actives)} pathogens")

    high_mask = (meta["ratio"] > HIGH_RATIO_THRESHOLD) & meta["keep"]
    print(f"Augmenting {high_mask.sum()} datasets with ratio > {HIGH_RATIO_THRESHOLD}")

    for idx, row in tqdm(meta[high_mask].iterrows(), total=high_mask.sum(), desc="Augmenting datasets", unit="dataset"):
        out_path = os.path.join(OUT_DIR, row["pathogen"], f"{row['name']}.csv")
        df = pd.read_csv(out_path)
        df["decoy"] = False

        n_pos = int((df["bin"] == 1).sum())
        n_total = len(df)
        n_needed = int(n_pos / TARGET_RATIO) - n_total

        pathogen_active_iks = pathogen_actives.get(row["pathogen"], set())
        pool = list({
            decoy_smi
            for active_smi in df.loc[df["bin"] == 1, "smiles"]
            for active_ik in [smiles_to_inchikey(active_smi)[1]]
            if active_ik is not None
            if active_ik in decoys
            for decoy_smi in decoys[active_ik]
            for decoy_ik in [smiles_to_inchikey(decoy_smi)[1]]
            if decoy_ik is not None
            if decoy_ik not in pathogen_active_iks
        })

        if not pool:
            print(f"  [WARN] {row['name']}: no decoys available, skipping augmentation")
            df.drop(columns=["decoy"]).to_csv(out_path, index=False)
            continue

        n_sample = min(n_needed, len(pool))

        # InChIKeys already present in the dataset — used to filter decoys
        existing_iks = set()
        for smi in df["smiles"]:
            _, ik = smiles_to_inchikey(smi)
            if ik is not None:
                existing_iks.add(ik)

        random.shuffle(pool)
        selected = []
        n_invalid = 0
        n_duplicate = 0
        for smi in pool:
            if len(selected) >= n_sample:
                break
            canon, ik = smiles_to_inchikey(smi)
            if ik is None:
                n_invalid += 1
                continue
            if ik in existing_iks:
                n_duplicate += 1
                continue
            selected.append(canon)  # store canonical SMILES, IK-unique
            existing_iks.add(ik)

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
    extract_datasets(metadata)
    enriched = augment_datasets(metadata, decoys)
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
