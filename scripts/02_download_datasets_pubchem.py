"""
Step 02 (PubChem) — Download the PubChem annotated bioassay summary.

Downloads output/06_annotate_selected_bioassays/06_summary.csv from the
pubchem-antimicrobial-tasks EOS repository into
data/processed/pubchem/02_pubchem_datasets.csv.

Then derives an organism-only subset (target_type != single_protein, label !=
discarded) with added compounds (actives + inactives) and ratio (actives /
compounds) columns, saved to data/processed/pubchem/02_pubchem_datasets_organism.csv.

Finally downloads the per-assay compound CSVs for those organism assays from
output/06_selected_bioassays/<pathogen_code>/<aid>.csv into
data/raw/pubchem/<pathogen_code>/<aid>.csv.

Usage:
    python scripts/02_download_datasets_pubchem.py
"""

import os
import shutil
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(ROOT, "..")
REPO_NAME = "pubchem-antimicrobial-tasks"

REMOTE_INDEX = "output/06_annotate_selected_bioassays/06_summary.csv"
LOCAL_INDEX = os.path.join(REPO_ROOT, "data", "processed", "pubchem", "02_pubchem_datasets.csv")


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
    src_dir = os.path.dirname(remote_abs)
    repo_root_real = os.path.realpath(REPO_ROOT)
    while os.path.realpath(src_dir) != repo_root_real:
        if os.path.isdir(src_dir) and not os.listdir(src_dir):
            os.rmdir(src_dir)
            src_dir = os.path.dirname(src_dir)
        else:
            break
    return True


def print_summary(df: pd.DataFrame) -> None:
    print("\n--- Summary ---")
    print(f"Total assays        : {len(df)}")
    print(f"Pathogens           : {df['pathogen_code'].nunique()}")
    print(f"\nPer pathogen:")
    for pathogen, group in df.groupby("pathogen_code"):
        print(f"  {pathogen}: {len(group)} assays, {group['cids'].sum():,} compounds")


LOCAL_ORGANISM = os.path.join(REPO_ROOT, "data", "processed", "pubchem", "02_pubchem_datasets_organism.csv")


def main() -> None:
    ok = download_file(REMOTE_INDEX, LOCAL_INDEX)
    if not ok:
        print("Download failed.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(LOCAL_INDEX)
    print(f"Saved summary to {LOCAL_INDEX}")
    print_summary(df)

    df_org = df[(df["target_type"] != "single_protein") & (df["label"] != "discarded")].drop(columns=["protein_id", "chemical_probe"]).copy()
    df_org["compounds"] = df_org["actives"] + df_org["inactives"]
    df_org["ratio"] = (df_org["actives"] / df_org["compounds"]).round(3)
    df_org.to_csv(LOCAL_ORGANISM, index=False)
    print(f"\nSaved organism-only summary to {LOCAL_ORGANISM}")
    print(f"\n--- Organism summary ---")
    print(f"Pathogens           : {df_org['pathogen_code'].nunique()}")
    print(f"Total assays        : {len(df_org)}")
    print(f"Average ratio       : {df_org['ratio'].mean():.3f} ± {df_org['ratio'].std():.3f}")
    print(f"\nPer pathogen:")
    for pathogen, group in df_org.groupby("pathogen_code"):
        print(f"  {pathogen}: {len(group)} assays, {group['compounds'].sum():,} compounds, ratio {group['ratio'].mean():.3f} ± {group['ratio'].std():.3f}")

    for _, row in df_org.iterrows():
        code = row["pathogen_code"]
        aid = int(row["aid"])
        download_file(
            remote_path=f"output/06_selected_bioassays/{code}/{aid}.csv",
            local_path=os.path.join(REPO_ROOT, "data", "raw", "pubchem", code, f"{aid}.csv"),
        )


if __name__ == "__main__":
    main()
