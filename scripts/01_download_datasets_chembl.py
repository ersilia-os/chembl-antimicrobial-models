"""
Step 01 (ChEMBL) — Download ChEMBL representative datasets.

Downloads the selected binary datasets from the chembl-antimicrobial-tasks repo into
data/raw/chembl/<pathogen>/.

Files downloaded per pathogen:
  - 17_final_datasets.csv
  - 19_final_datasets_metadata.csv
  - 19_final_datasets.zip
  - 20_general_datasets.csv  OR  20_general_no_pubchem_datasets.csv  (no_pubchem preferred)
  - 20_general_datasets_middle.zip  OR  20_general_no_pubchem_datasets_middle.zip  (all pathogens)
  - 20_general_datasets_high.zip    OR  20_general_no_pubchem_datasets_high.zip    (pfalciparum only)

By default, files are copied from a local chembl-antimicrobial-tasks repo assumed to be
in the same parent directory. Pass --eosvc to download from the remote EOS service instead.

Produces data/processed/chembl/<pathogen>/01_chembl_datasets.csv with all datasets merged.
Also produces data/processed/chembl/01_chembl_datasets_all.csv combining all pathogens.

Usage:
    python scripts/01_download_datasets_chembl.py
    python scripts/01_download_datasets_chembl.py --eosvc
"""

import argparse
import io
import os
import shutil
import subprocess
import sys
import zipfile

import pandas as pd
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(ROOT, "..")
sys.path.append(os.path.join(ROOT, "..", "src"))

from default import (
    CHEMBL_CSV_GENERAL,
    CHEMBL_CSV_GENERAL_NO_PUBCHEM,
    CHEMBL_ZIP_GENERAL,
    CHEMBL_ZIP_GENERAL_HIGH,
    CHEMBL_ZIP_GENERAL_NO_PUBCHEM,
    CHEMBL_ZIP_GENERAL_NO_PUBCHEM_HIGH,
    COL_BIN,
    COL_SMILES,
    G_ORG_DR,
    G_ORG_SP,
    PATHOGENS,
    RANDOM_SEED,
)

GENERAL_CSV_TO_ZIP = {
    CHEMBL_CSV_GENERAL: CHEMBL_ZIP_GENERAL,
    CHEMBL_CSV_GENERAL_NO_PUBCHEM: CHEMBL_ZIP_GENERAL_NO_PUBCHEM,
}

GENERAL_CSV_TO_HIGH_ZIP = {
    CHEMBL_CSV_GENERAL: CHEMBL_ZIP_GENERAL_HIGH,
    CHEMBL_CSV_GENERAL_NO_PUBCHEM: CHEMBL_ZIP_GENERAL_NO_PUBCHEM_HIGH,
}

PFALCIPARUM_GENERAL_LEVEL = "high"

REPO_NAME = "chembl-antimicrobial-tasks"
TASKS_REPO_ROOT = os.path.join(REPO_ROOT, "..", REPO_NAME)

# Non-general files — same for all pathogens, no no_pubchem variant
FILES = [
    "17_final_datasets.csv",
    "19_final_datasets_metadata.csv",
    "19_final_datasets.zip",
]


def copy_from_repo(remote_path: str, local_path: str) -> bool:
    source = os.path.join(TASKS_REPO_ROOT, remote_path)
    if not os.path.exists(source):
        print(f"Skipping {remote_path}: not found in local repo.")
        return False
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    shutil.copy2(source, local_path)
    print(f"Copied {remote_path} -> {local_path}")
    return True


def download_file(remote_path: str, local_path: str) -> bool:
    env = os.environ.copy()
    env["EVC_REPO_NAME"] = REPO_NAME
    cmd = ["eosvc", "download", "--path", remote_path]
    print(f"Downloading {remote_path} -> {local_path}")
    result = subprocess.run(cmd, env=env, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        print(f"Skipping {remote_path}: not found in the cloud.")
        return False
    remote_abs = os.path.join(REPO_ROOT, remote_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    shutil.move(remote_abs, local_path)
    return True


def download_pathogen(pathogen: str, use_eosvc: bool) -> None:
    transfer = download_file if use_eosvc else copy_from_repo
    raw_dir = os.path.join(REPO_ROOT, "data", "raw", "chembl", pathogen)

    for filename in FILES:
        transfer(
            remote_path=f"output/{pathogen}/{filename}",
            local_path=os.path.join(raw_dir, filename),
        )

    general_level = PFALCIPARUM_GENERAL_LEVEL if pathogen == "pfalciparum" else "middle"

    # General datasets: prefer no_pubchem when available
    if transfer(
        remote_path=f"output/{pathogen}/{CHEMBL_CSV_GENERAL_NO_PUBCHEM}",
        local_path=os.path.join(raw_dir, CHEMBL_CSV_GENERAL_NO_PUBCHEM),
    ):
        zip_name = CHEMBL_ZIP_GENERAL_NO_PUBCHEM_HIGH if general_level == "high" else CHEMBL_ZIP_GENERAL_NO_PUBCHEM
        transfer(
            remote_path=f"output/{pathogen}/{zip_name}",
            local_path=os.path.join(raw_dir, zip_name),
        )
        general_csv = CHEMBL_CSV_GENERAL_NO_PUBCHEM
    else:
        transfer(
            remote_path=f"output/{pathogen}/{CHEMBL_CSV_GENERAL}",
            local_path=os.path.join(raw_dir, CHEMBL_CSV_GENERAL),
        )
        zip_name = CHEMBL_ZIP_GENERAL_HIGH if general_level == "high" else CHEMBL_ZIP_GENERAL
        transfer(
            remote_path=f"output/{pathogen}/{zip_name}",
            local_path=os.path.join(raw_dir, zip_name),
        )
        general_csv = CHEMBL_CSV_GENERAL

    pathogen_output_dir = os.path.join(REPO_ROOT, "output", pathogen)
    if os.path.isdir(pathogen_output_dir) and not os.listdir(pathogen_output_dir):
        os.rmdir(pathogen_output_dir)

    merge_pathogen(pathogen, general_csv, general_level=general_level)


def build_aggregate_g_datasets(pathogen: str, df_meta: pd.DataFrame, zip_path: str) -> pd.DataFrame:
    """
    Build G_ORG_DR and G_ORG_SP from all general datasets at the chosen cutoff level.
    Deduplicates by InChIKey; active (bin=1) wins over inactive.
    Saves CSVs to data/raw/chembl/{pathogen}/ and returns stat rows for 01_chembl_datasets.
    """
    if not os.path.exists(zip_path):
        return pd.DataFrame()

    stat_rows = []

    for agg_name, activity_types in [("G_ORG_DR", G_ORG_DR), ("G_ORG_SP", G_ORG_SP)]:
        matching = df_meta[df_meta["activity_type"].isin(activity_types)]
        if matching.empty:
            continue

        frames = []
        with zipfile.ZipFile(zip_path) as zf:
            namelist = set(zf.namelist())
            for _, row in matching.iterrows():
                inner = f"ORG_{row['activity_type']}_{row['unit']}_{row['cutoff']}.csv.gz"
                if inner not in namelist:
                    continue
                with zf.open(inner) as f:
                    df = pd.read_csv(io.BytesIO(f.read()), compression="gzip")
                if COL_SMILES not in df.columns or COL_BIN not in df.columns:
                    continue
                frames.append(df[[COL_SMILES, COL_BIN]])

        if not frames:
            continue

        merged = pd.concat(frames, ignore_index=True)

        # Compute canonical SMILES and InChIKey; discard unparseable
        group_keys, canonical = [], []
        for smi in merged[COL_SMILES]:
            mol = Chem.MolFromSmiles(str(smi) if pd.notna(smi) else "")
            if mol is None:
                group_keys.append(None)
                canonical.append(None)
            else:
                can = Chem.MolToSmiles(mol)
                ik = MolToInchiKey(mol)
                group_keys.append(ik if ik else can)
                canonical.append(can)

        merged["_gk"] = group_keys
        merged["_can"] = canonical
        merged = merged.dropna(subset=["_gk"])

        # Activity-conservative dedup: active wins (max of 0/1)
        deduped = (
            merged.groupby("_gk", sort=False)
            .agg({"_can": "first", COL_BIN: "max"})
            .rename(columns={"_can": COL_SMILES})
            .reset_index(drop=True)
        )

        n_actives = int((deduped[COL_BIN] == 1).sum())
        n_compounds = len(deduped)

        if n_actives < 50:
            print(f"  [{pathogen}] {agg_name}: {n_actives} actives < 50, skipping.")
            continue

        raw_dir = os.path.join(REPO_ROOT, "data", "raw", "chembl", pathogen)
        out_path = os.path.join(raw_dir, f"{agg_name}.csv")
        deduped[[COL_SMILES, COL_BIN]].to_csv(out_path, index=False)
        print(f"  [{pathogen}] {agg_name}: {n_compounds} compounds, {n_actives} actives → {out_path}")

        stat_rows.append({
            "name": agg_name,
            "label": "G",
            "assay_type": "general_aggregate",
            "target_type": "ORGANISM",
            "n_assays": len(matching),
            "compounds": n_compounds,
            "positives": n_actives,
        })

    return pd.DataFrame(stat_rows) if stat_rows else pd.DataFrame()


def merge_pathogen(pathogen: str, general_csv: str, general_level: str = "middle") -> None:
    raw_dir = os.path.join(REPO_ROOT, "data", "raw", "chembl", pathogen)
    dfs = []

    metadata_path = os.path.join(raw_dir, "19_final_datasets_metadata.csv")
    if os.path.exists(metadata_path):
        df_final = pd.read_csv(metadata_path)
        df_final = df_final.rename(columns={"original_name": "name", "cpds": "compounds"})
        if "source" in df_final.columns:
            df_final = df_final.rename(columns={"source": "assay_type"})
        path_17 = os.path.join(raw_dir, "17_final_datasets.csv")
        if os.path.exists(path_17):
            df_17 = pd.read_csv(path_17, usecols=["name", "n_assays"])
            n_assays_map = df_17.set_index("name")["n_assays"].to_dict()
            df_final["n_assays"] = df_final["name"].map(n_assays_map)
        dfs.append(df_final)

    general_path = os.path.join(raw_dir, general_csv)
    if os.path.exists(general_path):
        df_general_all = pd.read_csv(general_path)
        df_general_all = df_general_all[df_general_all["level"] == general_level].reset_index(drop=True)
        df_general_all = df_general_all.drop(columns=["level", "n_inactives", "auroc_std"])
        df_general_all = df_general_all.rename(columns={"n_compounds": "compounds", "n_actives": "positives"})

        zip_lookup = GENERAL_CSV_TO_HIGH_ZIP if general_level == "high" else GENERAL_CSV_TO_ZIP
        zip_path = os.path.join(raw_dir, zip_lookup[general_csv])
        agg_df = build_aggregate_g_datasets(pathogen, df_general_all, zip_path)
        if not agg_df.empty:
            dfs.append(agg_df)

        # Individual G datasets: apply quality filter
        df_general = df_general_all[
            (df_general_all["auroc"] >= 0.7) & (df_general_all["positives"] >= 50)
        ].reset_index(drop=True)
        df_general.insert(0, "name", [f"G_ORG{i}_{row.cutoff}" for i, row in enumerate(df_general.itertuples())])
        df_general["target_type"] = "ORGANISM"
        df_general["label"] = "G"
        df_general["assay_type"] = "general"
        dfs.append(df_general)

    if not dfs:
        print(f"Skipping merge for {pathogen}: no datasets found.")
        return

    df = pd.concat(dfs, ignore_index=True)
    df["ratio"] = (df["positives"] / df["compounds"]).round(3)
    df["auroc"] = df["auroc"].round(3)
    df.insert(0, "pathogen", pathogen)
    df["source"] = "chembl"

    first_cols = ["pathogen", "source", "label", "assay_type", "n_assays", "name"]
    rest_cols = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + rest_cols]

    out_path = os.path.join(REPO_ROOT, "data", "processed", "chembl", pathogen, "01_chembl_datasets.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved merged datasets to {out_path}")


def merge_all_pathogens() -> pd.DataFrame | None:
    processed = os.path.join(REPO_ROOT, "data", "processed", "chembl")
    dfs = []
    for pathogen in PATHOGENS:
        path = os.path.join(processed, pathogen, "01_chembl_datasets.csv")
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
    if not dfs:
        print("[ERROR] No per-pathogen datasets found to merge.", file=sys.stderr)
        sys.exit(1)
    df = pd.concat(dfs, ignore_index=True)
    out_path = os.path.join(processed, "01_chembl_datasets_all.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved combined datasets ({len(df)} rows) to {out_path}")
    return df


def print_summary(df: pd.DataFrame) -> None:
    print("\n--- Summary ---")
    print(f"Total datasets      : {len(df)}")
    print(f"Average ratio       : {df['ratio'].mean():.3f} ± {df['ratio'].std():.3f}")
    print(f"\nDatasets per label:")
    for label, count in df['label'].value_counts().items():
        print(f"  {label}: {count}")
    print(f"\nPathogens processed : {df['pathogen'].nunique()} / {len(PATHOGENS)}")
    print(f"\nDatasets per pathogen:")
    counts = df.groupby('pathogen').size()
    cpds = df.groupby('pathogen')['compounds'].sum()
    for pathogen in counts.index:
        print(f"  {pathogen}: {counts[pathogen]} datasets, {cpds[pathogen]:,} compounds")


def main(use_eosvc: bool, pathogens: list[str] | None = None) -> None:
    targets = pathogens if pathogens else PATHOGENS
    for pathogen in targets:
        download_pathogen(pathogen, use_eosvc=use_eosvc)
    df = merge_all_pathogens()
    print_summary(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download ChEMBL binary datasets for all pathogens."
    )
    parser.add_argument(
        "--eosvc",
        action="store_true",
        default=False,
        help="Download from the remote EOS service instead of copying from local repo.",
    )
    parser.add_argument(
        "--pathogen",
        nargs="+",
        choices=PATHOGENS,
        default=None,
        metavar="PATHOGEN",
        help="Run only for the specified pathogen(s). Defaults to all pathogens.",
    )
    args = parser.parse_args()
    main(use_eosvc=args.eosvc, pathogens=args.pathogen)
