"""
Step 06 — Prepare datasets for model training.

(1) Loads dataset metadata from data/processed/01_chembl_datasets_all.csv.
(2) Loads decoy data from output/results/05_eos3e6s_v1.csv and builds a
    dict mapping each input SMILES to up to 10 randomly sampled decoys.
(3) Extracts raw compound CSVs from per-pathogen zip archives and writes
    them to output/results/06_datasets/{pathogen}/{name}.csv.
    Output columns are normalised to smiles, bin for all datasets.
(4) For datasets with ratio > 0.5, augments with decoy compounds to bring
    the active ratio down to 0.1. Augmented datasets gain a 'decoy' column
    (False for original rows, True for added decoys). Datasets below the
    threshold are not modified and will not contain a 'decoy' column.
(5) Saves output/results/06_datasets_metadata.csv — a copy of the input
    metadata with three additional columns: 'decoys' (number of decoy rows
    added, 0 for non-augmented datasets), 'final_ratio' (ratio after
    augmentation, equals 'ratio' for non-augmented datasets), and
    'final_compounds' (total rows after augmentation, equals 'compounds'
    for non-augmented datasets).

Usage:
    python scripts/06_prepare_datasets.py
    python scripts/06_prepare_datasets.py --metadata path/to/other.csv
"""

import argparse
import gzip
import io
import os
import random
import zipfile

import pandas as pd
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))

METADATA_PATH = os.path.join(REPO_ROOT, "data", "processed", "01_chembl_datasets_all.csv")
DECOYS_PATH   = os.path.join(REPO_ROOT, "output", "results", "05_eos3e6s_v1.csv")
RAW_DIR       = os.path.join(REPO_ROOT, "data", "raw")
OUT_DIR       = os.path.join(REPO_ROOT, "output", "results", "06_datasets")
N_DECOYS      = 20


# ---------------------------------------------------------------------------
# Step 1 — load metadata
# ---------------------------------------------------------------------------

def load_metadata(metadata_path: str) -> pd.DataFrame:
    df = pd.read_csv(metadata_path)
    print(f"Loaded metadata: {len(df)} datasets across {df['pathogen'].nunique()} pathogens")
    return df


# ---------------------------------------------------------------------------
# Step 2 — load decoys
# ---------------------------------------------------------------------------

def load_decoys() -> dict[str, list[str]]:
    print("Loading decoy data (this may take a moment)...")
    df = pd.read_csv(DECOYS_PATH)
    decoy_cols = [c for c in df.columns if c not in ("key", "input")]
    result = {}
    for _, row in df.iterrows():
        candidates = row[decoy_cols].dropna().tolist()
        if not candidates:
            continue
        result[row["input"]] = (
            candidates if len(candidates) <= N_DECOYS
            else random.sample(candidates, N_DECOYS)
        )
    print(f"Loaded decoys: {len(result)} compounds with 1–{N_DECOYS} decoys each")
    return result


# ---------------------------------------------------------------------------
# Step 3 — extract datasets
# ---------------------------------------------------------------------------

def _inner_filename(row: pd.Series) -> str:
    if row["label"] in ("A", "B", "M"):
        return f"{row['name']}.csv"
    return f"ORG_{row['activity_type']}_{row['unit']}_{row['cutoff']}.csv.gz"


def _zip_path(row: pd.Series) -> str:
    zip_name = (
        "19_final_datasets.zip" if row["label"] in ("A", "B", "M")
        else "20_general_datasets.zip"
    )
    return os.path.join(RAW_DIR, row["pathogen"], zip_name)


def extract_datasets(metadata: pd.DataFrame) -> None:
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Extracting datasets", unit="dataset"):
        out_path = os.path.join(OUT_DIR, row["pathogen"], f"{row['name']}.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        inner = _inner_filename(row)
        with zipfile.ZipFile(_zip_path(row)) as zf:
            with zf.open(inner) as f:
                raw = gzip.open(f).read() if inner.endswith(".gz") else f.read()

        pd.read_csv(io.BytesIO(raw))[["smiles", "bin"]].to_csv(out_path, index=False)

    print(f"Datasets written to {OUT_DIR}")


# ---------------------------------------------------------------------------
# Step 4 — augment high-ratio datasets with decoys
# ---------------------------------------------------------------------------

TARGET_RATIO = 0.1
HIGH_RATIO_THRESHOLD = 0.5


def augment_datasets(metadata: pd.DataFrame, decoys: dict[str, list[str]]) -> pd.DataFrame:
    meta = metadata.copy()
    meta["decoys"] = 0
    meta["final_ratio"] = meta["ratio"]

    high_mask = meta["ratio"] > HIGH_RATIO_THRESHOLD
    print(f"Augmenting {high_mask.sum()} datasets with ratio > {HIGH_RATIO_THRESHOLD}")

    for idx, row in tqdm(meta[high_mask].iterrows(), total=high_mask.sum(), desc="Augmenting datasets", unit="dataset"):
        out_path = os.path.join(OUT_DIR, row["pathogen"], f"{row['name']}.csv")
        df = pd.read_csv(out_path)
        df["decoy"] = False

        n_pos = int((df["bin"] == 1).sum())
        n_total = len(df)
        n_needed = 10 * n_pos - n_total

        pool = list({
            smi
            for active_smi in df.loc[df["bin"] == 1, "smiles"]
            if active_smi in decoys
            for smi in decoys[active_smi]
        })

        if not pool:
            print(f"  [WARN] {row['name']}: no decoys available, skipping augmentation")
            df.drop(columns=["decoy"]).to_csv(out_path, index=False)
            continue

        n_sample = min(n_needed, len(pool))
        achieved = round(n_pos / (n_total + n_sample), 3)
        if n_sample < n_needed:
            print(
                f"  [WARN] {row['name']}: pool has {len(pool)} decoys, "
                f"needed {n_needed}; achieved ratio {achieved} (target {TARGET_RATIO})"
            )

        new_rows = pd.DataFrame({
            "smiles": random.sample(pool, n_sample),
            "bin": 0,
            "decoy": True,
        })
        pd.concat([df, new_rows], ignore_index=True).to_csv(out_path, index=False)

        meta.at[idx, "decoys"] = n_sample
        meta.at[idx, "final_ratio"] = achieved

    meta["final_compounds"] = meta["compounds"] + meta["decoys"]
    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

META_OUT_PATH = os.path.join(REPO_ROOT, "output", "results", "06_datasets_metadata.csv")


def print_sbatch_command(n_datasets: int) -> None:
    max_idx = n_datasets - 1
    script_path = os.path.join(ROOT, "07_run_models.sh")
    print(
        f"\nSetup complete. Submit the array job with:\n"
        f"    sbatch --chdir={REPO_ROOT} --array=0-{max_idx}%20 {script_path}"
    )


def main(metadata_path: str) -> None:
    metadata = load_metadata(metadata_path)
    decoys = load_decoys()
    extract_datasets(metadata)
    enriched = augment_datasets(metadata, decoys)
    enriched.to_csv(META_OUT_PATH, index=False)
    print(f"Saved enriched metadata to {META_OUT_PATH}")
    print_sbatch_command(len(enriched))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract raw compound datasets and prepare decoy mappings."
    )
    parser.add_argument(
        "--metadata",
        default=METADATA_PATH,
        help="Path to 01_chembl_datasets_all.csv (default: data/processed/01_chembl_datasets_all.csv).",
    )
    args = parser.parse_args()
    main(args.metadata)
