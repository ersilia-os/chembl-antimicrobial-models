"""
Step 01 — Download representative datasets.

Downloads the selected binary datasets from the EOS service into data/raw/<pathogen>/.
Files downloaded per pathogen:
  - 17_final_datasets.csv
  - 19_final_datasets_metadata.csv
  - 19_final_datasets.zip
  - 20_general_datasets.csv
  - 20_general_datasets.zip

Produces data/processed/<pathogen>/01_chembl_datasets.csv with all datasets merged.
Also produces data/processed/01_chembl_datasets_all.csv combining all pathogens.

Optional --select_representatives flag runs stratified sampling across compound-count
and activity-ratio bins, writing data/processed/01_chembl_datasets_representatives.csv.

Usage:
    python scripts/01_download_datasets.py --pathogen ecoli
    python scripts/01_download_datasets.py --all
    python scripts/01_download_datasets.py --all --select_representatives [--seed 42]
"""

import argparse
import os
import shutil
import subprocess
import sys
from itertools import product

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(ROOT, "..")
REPO_NAME = "chembl-antimicrobial-tasks"

PATHOGENS = [
    "abaumannii",
    "calbicans",
    "campylobacter",
    "ecoli",
    "efaecium",
    "enterobacter",
    "hpylori",
    "kpneumoniae",
    "mtuberculosis",
    "ngonorrhoeae",
    "paeruginosa",
    "pfalciparum",
    "saureus",
    "smansoni",
    "spneumoniae",
]

FILES = [
    "17_final_datasets.csv",
    "19_final_datasets_metadata.csv",
    "19_final_datasets.zip",
    "20_general_datasets.csv",
    "20_general_datasets.zip",
]

# (min_compounds, max_compounds, n_samples)
COMPOUND_BINS = [
    (100,    1_000,  2),
    (1_000,  10_000, 2),
    (10_000, 50_000, 1),
]

# (min_ratio, max_ratio)
RATIO_BINS = [
    (0.01, 0.3),
    (0.3,  0.5),
]

MANDATORY_DATASET = {"pathogen": "pfalciparum", "name": "CHEMBL4888485_INHIBITION_%_qt_50.0"}


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


def merge_pathogen(pathogen: str) -> None:
    raw_dir = os.path.join(REPO_ROOT, "data", "raw", pathogen)

    metadata_path = os.path.join(raw_dir, "19_final_datasets_metadata.csv")
    general_path = os.path.join(raw_dir, "20_general_datasets.csv")
    dfs = []

    if os.path.exists(metadata_path):
        df_final = pd.read_csv(metadata_path)
        df_final = df_final.rename(columns={"original_name": "name", "cpds": "compounds"})
        path_17 = os.path.join(raw_dir, "17_final_datasets.csv")
        if os.path.exists(path_17):
            df_17 = pd.read_csv(path_17, usecols=["name", "n_assays"])
            n_assays_map = df_17.set_index("name")["n_assays"].to_dict()
            df_final["n_assays"] = df_final["name"].map(n_assays_map)
        dfs.append(df_final)

    if os.path.exists(general_path):
        df_general = pd.read_csv(general_path)
        df_general = df_general.drop(columns=["n_inactives", "auroc_std"])
        df_general = df_general.rename(columns={"n_compounds": "compounds", "n_actives": "positives"})
        df_general = df_general[(df_general["auroc"] >= 0.7) & (df_general["positives"] >= 50)].reset_index(drop=True)
        df_general.insert(0, "name", [f"G_ORG{i}_{row.cutoff}" for i, row in enumerate(df_general.itertuples())])
        df_general["target_type"] = "ORGANISM"
        df_general["label"] = "G"
        df_general["source"] = "general"
        dfs.append(df_general)

    if not dfs:
        print(f"Skipping merge for {pathogen}: no datasets found.")
        return

    df = pd.concat(dfs, ignore_index=True)
    df["ratio"] = (df["positives"] / df["compounds"]).round(3)
    df["auroc"] = df["auroc"].round(3)
    df.insert(0, "pathogen", pathogen)

    first_cols = ["pathogen", "label", "source", "n_assays", "name"]
    rest_cols = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + rest_cols]

    out_path = os.path.join(REPO_ROOT, "data", "processed", pathogen, "01_chembl_datasets.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved merged datasets to {out_path}")


def download_pathogen(pathogen: str) -> None:
    for filename in FILES:
        download_file(
            remote_path=f"output/{pathogen}/{filename}",
            local_path=os.path.join(REPO_ROOT, "data", "raw", pathogen, filename),
        )
    pathogen_output_dir = os.path.join(REPO_ROOT, "output", pathogen)
    if os.path.isdir(pathogen_output_dir) and not os.listdir(pathogen_output_dir):
        os.rmdir(pathogen_output_dir)
    merge_pathogen(pathogen)


def merge_all_pathogens() -> None:
    processed = os.path.join(REPO_ROOT, "data", "processed")
    dfs = []
    for pathogen in PATHOGENS:
        path = os.path.join(processed, pathogen, "01_chembl_datasets.csv")
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
    if not dfs:
        print("No per-pathogen datasets found to merge.", file=sys.stderr)
        return
    df = pd.concat(dfs, ignore_index=True)
    out_path = os.path.join(processed, "01_chembl_datasets_all.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved combined datasets ({len(df)} rows) to {out_path}")
    return df


def print_summary(df: pd.DataFrame, all_pathogens: bool) -> None:
    print("\n--- Summary ---")
    print(f"Total datasets      : {len(df)}")
    print(f"Average ratio       : {df['ratio'].mean():.3f} ± {df['ratio'].std():.3f}")
    print(f"\nDatasets per label:")
    for label, count in df['label'].value_counts().items():
        print(f"  {label}: {count}")
    if all_pathogens:
        print(f"\nPathogens processed : {df['pathogen'].nunique()} / {len(PATHOGENS)}")
        print(f"\nDatasets per pathogen:")
        for pathogen, count in df.groupby('pathogen').size().items():
            n_cpds = df.loc[df['pathogen'] == pathogen, 'compounds'].sum()
            print(f"  {pathogen}: {count} datasets, {n_cpds:,} compounds")


def select_representatives(seed: int) -> None:
    input_path = os.path.join(REPO_ROOT, "data", "processed", "01_chembl_datasets_all.csv")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path).dropna(subset=["compounds", "ratio"])

    selected = []
    for (c_min, c_max, n), (r_min, r_max) in product(COMPOUND_BINS, RATIO_BINS):
        cell = df[
            (df["compounds"] >= c_min) & (df["compounds"] < c_max) &
            (df["ratio"] >= r_min) & (df["ratio"] < r_max)
        ]
        if cell.empty:
            print(f"[WARN] No datasets in compounds=[{c_min},{c_max}) ratio=[{r_min},{r_max})")
            continue
        n_draw = min(n, len(cell))
        selected.append(cell.sample(n=n_draw, random_state=seed))

    mandatory_row = df[
        (df["pathogen"] == MANDATORY_DATASET["pathogen"]) &
        (df["name"] == MANDATORY_DATASET["name"])
    ]
    if mandatory_row.empty:
        print(f"[WARN] Mandatory dataset not found: {MANDATORY_DATASET}")
    else:
        selected.append(mandatory_row)

    result = (
        pd.concat(selected)
        .drop_duplicates(subset=["pathogen", "name"])
        .reset_index(drop=True)
    )
    print(result[["pathogen", "name", "compounds", "ratio"]].to_string())
    out_path = os.path.join(REPO_ROOT, "data", "processed", "01_chembl_datasets_representatives.csv")
    result.to_csv(out_path, index=False)
    print(f"\nSaved {len(result)} datasets to {out_path}")


def main(args: argparse.Namespace) -> None:
    pathogens = PATHOGENS if args.all else [args.pathogen]
    for pathogen in pathogens:
        download_pathogen(pathogen)
    df = merge_all_pathogens()
    if df is not None:
        if not args.all:
            df = df[df['pathogen'] == args.pathogen]
        print_summary(df, all_pathogens=args.all)
    if args.select_representatives:
        select_representatives(args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download ChEMBL binary datasets from eosvc"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pathogen",
        type=str,
        choices=PATHOGENS,
        help="Pathogen code to download (e.g. ecoli, mtuberculosis).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Download datasets for all supported pathogens.",
    )
    parser.add_argument(
        "--select_representatives",
        action="store_true",
        help="After downloading, sample representative datasets into 01_chembl_datasets_representatives.csv.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for representative selection (default: 42).",
    )
    args = parser.parse_args()
    main(args)
