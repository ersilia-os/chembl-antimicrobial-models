"""
Step 07 — Prepare datasets for model training.

(1) Loads and combines dataset metadata from:
      - data/processed/chembl/01_chembl_datasets_all.csv   (stage4 pooled datasets)
      - data/processed/pubchem/02_pubchem_datasets_organism.csv  (step-08 pooled organism datasets)
    Both tables already share `pathogen`, `source`, `name`, `compounds`, `positives`,
    `target_type`. `inactives` (= compounds − positives) is added; `auroc` (ChEMBL only)
    is renamed `auroc_baseline`.
(2) Extracts the compound table for every dataset:
      - ChEMBL  : data/raw/chembl/<pathogen>/<name>.csv.gz
      - PubChem : data/raw/pubchem/<pathogen>/<name>.csv
    Keeps [inchikey, smiles, bin] and validates bin ∈ {0, 1}. The upstream repos already
    curate and deduplicate their datasets, so this step does NOT re-deduplicate or re-binarise —
    it passes the rows through unchanged and only WARNS if it finds duplicate InChIKeys or
    active/inactive label conflicts (which would indicate an upstream regression). Writes to
    output/07_datasets/<pathogen>/<name>.csv, overwriting, so the step is idempotent
    (augmentation below always starts from the unaugmented file).
(3) Balances every dataset whose active ratio exceeds HIGH_RATIO_THRESHOLD (0.5) down to a
    ratio of 0.5 by adding real measured negatives ("proven negatives") — compounds that are
    inactive (bin=0) in another dataset of the same pathogen. A candidate is excluded if its
    InChIKey is already in the dataset, or if it is a proven active (bin=1) in ANY dataset of
    the same pathogen (label conflict). The pool spans both ChEMBL and PubChem datasets of the
    pathogen. If the proven-negative pool is exhausted before reaching 0.5, the shortfall is
    topped up with decoys from output/06_decoys/06_eos3e6s_v1.csv (drawn from the dataset's own
    actives, same exclusions; loaded lazily, only if a shortfall occurs). Added rows get bin=0
    and added_negative=True.
(4) Saves output/07_datasets/07_datasets_metadata.csv — the combined metadata with added
    columns: added_negatives (real negatives added), added_decoys (decoy fallback added),
    final_ratio (after balancing) and final_compounds; sorted by pathogen.

InChIKey note: extraction and pool building trust the upstream `inchikey` column rather than
recomputing it — both source pipelines emit standard InChIKeys. RDKit is only used on the
decoy fallback path (decoy SMILES have no precomputed InChIKey).

Usage:
    python scripts/07a_prepare_datasets.py
"""

import os
import random
import sys

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
sys.path.append(os.path.join(ROOT, "..", "src"))

from default import (  # noqa: E402
    HIGH_RATIO_THRESHOLD,
    N_DECOYS,
    RANDOM_SEED,
)

CHEMBL_METADATA_PATH = os.path.join(REPO_ROOT, "data", "processed", "chembl", "01_chembl_datasets_all.csv")
PUBCHEM_METADATA_PATH = os.path.join(REPO_ROOT, "data", "processed", "pubchem", "02_pubchem_datasets_organism.csv")
DECOYS_PATH           = os.path.join(REPO_ROOT, "output", "06_decoys", "06_eos3e6s_v1.csv")
RAW_DIR               = os.path.join(REPO_ROOT, "data", "raw")
OUT_DIR               = os.path.join(REPO_ROOT, "output", "07_datasets")
META_OUT_PATH         = os.path.join(REPO_ROOT, "output", "07_datasets", "07_datasets_metadata.csv")

# Balance datasets above this ratio down to exactly this ratio (0.5 = perfectly balanced).
BALANCE_RATIO = HIGH_RATIO_THRESHOLD

DATASET_COLS = ["inchikey", "smiles", "bin"]


def _validate_bin(df: pd.DataFrame, name: str) -> None:
    invalid = df.loc[~df["bin"].isin([0, 1]), "bin"].unique()
    if len(invalid) > 0:
        raise ValueError(f"Dataset {name!r} contains invalid bin values: {invalid.tolist()}")


# RDKit SMILES -> (canonical SMILES, InChIKey), memoised. Used only on the decoy fallback path.
_IK_CACHE: dict[str, tuple] = {}


def smiles_to_inchikey(smi: str) -> tuple:
    """Return (canonical_smiles, inchikey) for a SMILES, or (None, None) if unparsable."""
    s = str(smi)
    cached = _IK_CACHE.get(s)
    if cached is not None:
        return cached
    mol = Chem.MolFromSmiles(s)
    result = (None, None) if mol is None else (Chem.MolToSmiles(mol), Chem.MolToInchiKey(mol))
    _IK_CACHE[s] = result
    return result


def assert_no_duplicates(df: pd.DataFrame, name: str) -> None:
    """Verify the dataset has no duplicate InChIKeys or active/inactive label conflicts.

    We trust the upstream repos' curation and do NOT re-deduplicate or re-binarise — this
    only WARNS if a problem is found (an upstream regression), leaving the data unchanged.
    """
    nn = df.dropna(subset=["inchikey"])
    n_dup = len(nn) - nn["inchikey"].nunique()
    n_conflicts = int((nn.groupby("inchikey")["bin"].nunique() > 1).sum()) if len(nn) else 0
    n_no_ik = int(df["inchikey"].isna().sum())
    if n_dup or n_conflicts or n_no_ik:
        print(f"  [WARN] {name}: {n_dup} duplicate-InChIKey rows, {n_conflicts} active/inactive "
              f"label conflicts, {n_no_ik} rows without an InChIKey — left AS-IS "
              f"(upstream curation trusted; no dedup/re-binarisation applied)")


# ---------------------------------------------------------------------------
# Step 1 — load metadata
# ---------------------------------------------------------------------------

def load_metadata(chembl_path: str, pubchem_path: str) -> pd.DataFrame:
    chembl = pd.read_csv(chembl_path)
    pubchem = pd.read_csv(pubchem_path)

    df = pd.concat([chembl, pubchem], ignore_index=True)
    df["inactives"] = df["compounds"] - df["positives"]
    if "auroc" in df.columns:
        df = df.rename(columns={"auroc": "auroc_baseline"})
    print(f"Loaded metadata: {len(df)} datasets across {df['pathogen'].nunique()} pathogens "
          f"({len(chembl)} ChEMBL, {len(pubchem)} PubChem)")
    return df


# ---------------------------------------------------------------------------
# Step 2 — extract per-dataset compound tables
# ---------------------------------------------------------------------------

def _extract(row: pd.Series) -> pd.DataFrame | None:
    if row["source"] == "chembl":
        path = os.path.join(RAW_DIR, "chembl", row["pathogen"], f"{row['name']}.csv.gz")
    else:
        path = os.path.join(RAW_DIR, "pubchem", row["pathogen"], f"{row['name']}.csv")
    if not os.path.exists(path):
        print(f"  [WARN] source file not found: {os.path.relpath(path, REPO_ROOT)}")
        return None
    df = pd.read_csv(path)  # pandas infers gzip from the .gz suffix
    missing = [c for c in DATASET_COLS if c not in df.columns]
    if missing:
        print(f"  [WARN] {row['name']}: missing columns {missing}")
        return None
    df = df[DATASET_COLS].dropna(subset=["smiles"])
    _validate_bin(df, row["name"])
    return df.reset_index(drop=True)


def extract_datasets(metadata: pd.DataFrame) -> None:
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Extracting datasets", unit="dataset"):
        out_path = os.path.join(OUT_DIR, row["pathogen"], f"{row['name']}.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df = _extract(row)
        if df is None:
            continue
        assert_no_duplicates(df, row["name"])
        # Write the upstream rows through unchanged (overwrite each run) so augmentation
        # always starts from the unaugmented file — makes the whole step idempotent.
        df.to_csv(out_path, index=False)
    print(f"Datasets written to {OUT_DIR}")


# ---------------------------------------------------------------------------
# Step 3 — decoy fallback pool (loaded lazily, only if a shortfall occurs)
# ---------------------------------------------------------------------------

def load_decoys(decoys_path: str) -> dict[str, list[str]]:
    """Map each input compound's InChIKey to up to N_DECOYS decoy SMILES.

    Used only as a fallback when the proven-negative pool cannot balance a dataset to 0.5.
    Returns {} (no fallback) if the decoy file is absent.
    """
    if not os.path.exists(decoys_path):
        print(f"[WARN] decoy file not found ({decoys_path}); no decoy fallback available.")
        return {}
    print("Loading decoy data (fallback)...")
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
    if result:
        counts = [len(v) for v in result.values()]
        msg = f"Loaded decoys: {len(result)} InChIKeys with {min(counts)}–{max(counts)} decoys each"
        if n_unparsable:
            msg += f" ({n_unparsable} unparsable inputs skipped)"
        print(msg)
    return result


# ---------------------------------------------------------------------------
# Step 4 — balance high-ratio datasets with proven negatives (decoy fallback)
# ---------------------------------------------------------------------------

def _build_pathogen_pools(meta: pd.DataFrame) -> tuple[dict, dict]:
    """Per pathogen, return (active InChIKeys, proven-negative {InChIKey: SMILES}).

    A proven negative is a compound seen as bin=0 in some dataset of the pathogen and never
    as bin=1 in any dataset of the pathogen (no conflict). Pools span ChEMBL and PubChem.
    Uses the upstream inchikey column (no RDKit).
    """
    print("Building per-pathogen active / proven-negative pools...")
    pathogen_actives: dict[str, set[str]] = {}
    pathogen_negatives: dict[str, dict[str, str]] = {}
    for pathogen, group in meta.groupby("pathogen"):
        actives: set[str] = set()
        negatives: dict[str, str] = {}
        for name in group["name"]:
            fpath = os.path.join(OUT_DIR, pathogen, f"{name}.csv")
            if not os.path.exists(fpath):
                continue
            df_tmp = pd.read_csv(fpath, usecols=DATASET_COLS).dropna(subset=["inchikey"])
            actives.update(df_tmp.loc[df_tmp["bin"] == 1, "inchikey"])
            for ik, smi in zip(df_tmp.loc[df_tmp["bin"] == 0, "inchikey"],
                               df_tmp.loc[df_tmp["bin"] == 0, "smiles"]):
                negatives.setdefault(ik, smi)
        negatives = {ik: s for ik, s in negatives.items() if ik not in actives}
        pathogen_actives[pathogen] = actives
        pathogen_negatives[pathogen] = negatives
    n_neg = sum(len(v) for v in pathogen_negatives.values())
    print(f"  {sum(len(s) for s in pathogen_actives.values()):,} active and {n_neg:,} "
          f"proven-negative InChIKeys across {len(pathogen_actives)} pathogens")
    return pathogen_actives, pathogen_negatives


def _decoy_rows(df: pd.DataFrame, decoys: dict, pathogen_actives: set, existing_iks: set) -> list[tuple]:
    """Return (inchikey, canonical_smiles) decoy candidates for a dataset's actives,
    excluding pathogen actives and InChIKeys already present. Sorted for reproducibility."""
    out: dict[str, str] = {}
    for active_ik in df.loc[df["bin"] == 1, "inchikey"].dropna().unique():
        for decoy_smi in decoys.get(active_ik, []):
            canon, ik = smiles_to_inchikey(decoy_smi)
            if ik is None or ik in pathogen_actives or ik in existing_iks:
                continue
            out.setdefault(ik, canon)
    return sorted(out.items())


def augment_datasets(metadata: pd.DataFrame, decoys_path: str) -> pd.DataFrame:
    meta = metadata.copy()

    # Recompute counts from the deduped files (mainly matters for PubChem and sets the
    # ratio used by the balancing gate below).
    print("Recomputing compound counts from deduped datasets...")
    for idx, row in meta.iterrows():
        fpath = os.path.join(OUT_DIR, row["pathogen"], f"{row['name']}.csv")
        if not os.path.exists(fpath):
            continue
        b = pd.read_csv(fpath, usecols=["bin"])["bin"]
        n_comp = len(b)
        n_pos = int((b == 1).sum())
        meta.at[idx, "compounds"] = n_comp
        meta.at[idx, "positives"] = n_pos
        meta.at[idx, "inactives"] = n_comp - n_pos
        meta.at[idx, "ratio"] = round(n_pos / n_comp, 3) if n_comp else 0.0

    meta["added_negatives"] = 0
    meta["added_decoys"] = 0
    meta["final_ratio"] = meta["ratio"]

    pathogen_actives, pathogen_negatives = _build_pathogen_pools(meta)

    # Decoy fallback is loaded lazily on first shortfall.
    _decoy_cache: dict = {}

    def get_decoys() -> dict:
        if "map" not in _decoy_cache:
            _decoy_cache["map"] = load_decoys(decoys_path)
        return _decoy_cache["map"]

    high_mask = meta["ratio"] > BALANCE_RATIO
    print(f"Balancing {int(high_mask.sum())} datasets with ratio > {BALANCE_RATIO} down to {BALANCE_RATIO}")

    for idx, row in tqdm(meta[high_mask].iterrows(), total=int(high_mask.sum()), desc="Balancing datasets", unit="dataset"):
        out_path = os.path.join(OUT_DIR, row["pathogen"], f"{row['name']}.csv")
        df = pd.read_csv(out_path)

        n_pos = int((df["bin"] == 1).sum())
        n_total = len(df)
        n_needed = int(round(n_pos / BALANCE_RATIO)) - n_total
        if n_needed <= 0:
            continue

        existing_iks = set(df["inchikey"].dropna())

        # 1) proven negatives from the pathogen pool, not already in the dataset
        neg_pool = sorted(
            (ik, smi) for ik, smi in pathogen_negatives.get(row["pathogen"], {}).items()
            if ik not in existing_iks
        )
        random.shuffle(neg_pool)
        selected = neg_pool[:n_needed]
        existing_iks.update(ik for ik, _ in selected)
        n_neg = len(selected)

        # 2) decoy fallback for any remaining shortfall
        n_decoy = 0
        if n_needed - n_neg > 0:
            decoys = get_decoys()
            if decoys:
                pool = _decoy_rows(df, decoys, pathogen_actives.get(row["pathogen"], set()), existing_iks)
                random.shuffle(pool)
                decoy_sel = pool[: n_needed - n_neg]
                selected += decoy_sel
                n_decoy = len(decoy_sel)

        if not selected:
            print(f"  [WARN] {row['name']}: no proven negatives or decoys available, left unbalanced")
            continue

        achieved = round(n_pos / (n_total + len(selected)), 3)
        if len(selected) < n_needed:
            print(f"  [WARN] {row['name']}: needed {n_needed}, added {n_neg} negatives "
                  f"+ {n_decoy} decoys; achieved ratio {achieved} (target {BALANCE_RATIO})")

        df["added_negative"] = False
        new_rows = pd.DataFrame({
            "inchikey": [ik for ik, _ in selected],
            "smiles": [smi for _, smi in selected],
            "bin": 0,
            "added_negative": True,
        })
        pd.concat([df, new_rows], ignore_index=True).to_csv(out_path, index=False)

        meta.at[idx, "added_negatives"] = n_neg
        meta.at[idx, "added_decoys"] = n_decoy
        meta.at[idx, "final_ratio"] = achieved

    for idx, row in meta.iterrows():
        path = os.path.join(OUT_DIR, row["pathogen"], f"{row['name']}.csv")
        if os.path.exists(path):
            meta.at[idx, "final_compounds"] = len(pd.read_csv(path, usecols=["bin"]))
        else:
            meta.at[idx, "final_compounds"] = row["compounds"] + row["added_negatives"] + row["added_decoys"]
    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    metadata = load_metadata(CHEMBL_METADATA_PATH, PUBCHEM_METADATA_PATH)
    extract_datasets(metadata)
    enriched = augment_datasets(metadata, DECOYS_PATH)
    enriched = enriched.sort_values("pathogen").reset_index(drop=True)
    os.makedirs(os.path.dirname(META_OUT_PATH), exist_ok=True)
    enriched.to_csv(META_OUT_PATH, index=False)
    print(f"Saved enriched metadata ({len(enriched)} datasets) to {META_OUT_PATH}")


if __name__ == "__main__":
    main()
