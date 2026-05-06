"""
Step 02 (PubChem) — Download PubChem bioassays selected for modelling.

Downloads data/processed/09_bioassays_to_model/02_bioassays_to_model.csv from the
pubchem-antimicrobial-tasks EOS repository (or uses a local copy via --file),
filters to assays where keep == True, prints a per-pathogen summary, and
downloads the per-assay data CSVs into data/raw/pubchem/<pathogen>/<aid>.csv.

When --file is provided the assay data is fetched from output/results/<pathogen>/,
otherwise from output/05_results/<pathogen>/ (the versioned pipeline output).

The saved index adds a source column (pubchem) and renames code → pathogen for
consistency with the ChEMBL datasets.

Usage:
    python scripts/02_download_datasets_pubchem.py
    python scripts/02_download_datasets_pubchem.py --file /path/to/02_bioassays_to_model.csv
"""

import argparse
import os
import shutil
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(ROOT, "..")
REPO_NAME = "pubchem-antimicrobial-tasks"

REMOTE_INDEX = "data/processed/09_bioassays_to_model/02_bioassays_to_model.csv"
LOCAL_INDEX = os.path.join(REPO_ROOT, "data", "processed", "pubchem", "02_bioassays_to_model.csv")

PUBCHEM_NAMES = {
    "abaumannii":    "acinetobacter_baumannii",
    "calbicans":     "candida_albicans",
    "campylobacter": "campylobacter",
    "ecoli":         "escherichia_coli",
    "efaecium":      "enterococcus_faecium",
    "enterobacter":  "enterobacter",
    "hpylori":       "helicobacter_pylori",
    "kpneumoniae":   "klebsiella_pneumoniae",
    "mtuberculosis": "mycobacterium_tuberculosis",
    "ngonorrhoeae":  "neisseria_gonorrhoeae",
    "paeruginosa":   "pseudomonas_aeruginosa",
    "pfalciparum":   "plasmodium_falciparum",
    "saureus":       "staphylococcus_aureus",
    "smansoni":      "schistosoma_mansoni",
    "spneumoniae":   "streptococcus_pneumoniae",
}


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
    repo_root_real = os.path.realpath(REPO_ROOT)
    src_dir = os.path.dirname(remote_abs)
    while os.path.realpath(src_dir) != repo_root_real:
        if os.path.isdir(src_dir) and not os.listdir(src_dir):
            os.rmdir(src_dir)
            src_dir = os.path.dirname(src_dir)
        else:
            break
    return True


def load_and_filter(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    total = len(df)
    df = df[df["keep"] == True].reset_index(drop=True)
    df = df.rename(columns={"code": "pathogen", "aid": "name", "cids": "compounds", "actives": "positives"})
    df.insert(1, "source", "pubchem")
    print(f"Kept {len(df)}/{total} assays (keep == True)")
    return df


def print_summary(df: pd.DataFrame) -> None:
    print("\n--- Summary ---")
    print(f"Total assays kept   : {len(df)}")
    print(f"Pathogens           : {df['pathogen'].nunique()}")
    print(f"\nPer pathogen:")
    for pathogen, group in df.groupby("pathogen"):
        print(f"  {pathogen}: {len(group)} assays, {group['compounds'].sum():,} compounds")


def download_assay_data(df: pd.DataFrame, results_dir: str) -> None:
    for _, row in df.iterrows():
        pathogen = row["pathogen"]
        name = int(row["name"])
        pubchem_name = PUBCHEM_NAMES[pathogen]
        download_file(
            remote_path=f"output/{results_dir}/{pubchem_name}/{name}.csv",
            local_path=os.path.join(REPO_ROOT, "data", "raw", "pubchem", pathogen, f"{name}.csv"),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download PubChem bioassays selected for modelling."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a local 02_bioassays_to_model.csv to use instead of downloading from eosvc.",
    )
    args = parser.parse_args()

    if args.file:
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        path = args.file
        results_dir = "results"
        if os.path.realpath(path) != os.path.realpath(LOCAL_INDEX):
            os.makedirs(os.path.dirname(LOCAL_INDEX), exist_ok=True)
            shutil.copy2(path, LOCAL_INDEX)
    else:
        ok = download_file(REMOTE_INDEX, LOCAL_INDEX)
        if not ok:
            print("Download failed.", file=sys.stderr)
            sys.exit(1)
        path = LOCAL_INDEX
        results_dir = "05_results"

    df = load_and_filter(path)
    os.makedirs(os.path.dirname(LOCAL_INDEX), exist_ok=True)
    df.to_csv(LOCAL_INDEX, index=False)
    print(f"Saved filtered index to {LOCAL_INDEX}")
    print_summary(df)
    download_assay_data(df, results_dir)


if __name__ == "__main__":
    main()
