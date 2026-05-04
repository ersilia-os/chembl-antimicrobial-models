"""
Step 02 (PubChem) — Download PubChem bioassay datasets.

Downloads curated PubChem bioassay data from the EOS service into data/raw/pubchem/<pathogen>/.
Files downloaded per pathogen:
  - aids.csv           (AID list; one column: aid)
  - summary.csv        (per-AID stats: actives, inactives, etc.)
  - <AID>.csv          (raw compound data for each selected assay)

Produces data/processed/pubchem/<pathogen>/02_pubchem_datasets.csv with per-AID metadata.
Also produces data/processed/pubchem/02_pubchem_datasets_all.csv combining all pathogens.

Usage:
    python scripts/02_download_datasets_pubchem.py --pathogen ecoli
    python scripts/02_download_datasets_pubchem.py --all
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

# Maps short chembl codes → full names used in the pubchem-antimicrobial-tasks repo
PUBCHEM_NAMES = {
    "abaumannii":   "acinetobacter_baumannii",
    "calbicans":    "candida_albicans",
    "campylobacter":"campylobacter",
    "ecoli":        "escherichia_coli",
    "efaecium":     "enterococcus_faecium",
    "enterobacter": "enterobacter",
    "hpylori":      "helicobacter_pylori",
    "kpneumoniae":  "klebsiella_pneumoniae",
    "mtuberculosis":"mycobacterium_tuberculosis",
    "ngonorrhoeae": "neisseria_gonorrhoeae",
    "paeruginosa":  "pseudomonas_aeruginosa",
    "pfalciparum":  "plasmodium_falciparum",
    "saureus":      "staphylococcus_aureus",
    "smansoni":     "schistosoma_mansoni",
    "spneumoniae":  "streptococcus_pneumoniae",
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
    # remove any empty dirs left by eosvc under REPO_ROOT (outside our pubchem/ folders)
    repo_root_real = os.path.realpath(REPO_ROOT)
    src_dir = os.path.dirname(remote_abs)
    while os.path.realpath(src_dir) != repo_root_real:
        if os.path.isdir(src_dir) and not os.listdir(src_dir):
            os.rmdir(src_dir)
            src_dir = os.path.dirname(src_dir)
        else:
            break
    return True


def download_pathogen(pathogen: str) -> bool:
    raw_dir = os.path.join(REPO_ROOT, "data", "raw", "pubchem", pathogen)
    pubchem_name = PUBCHEM_NAMES[pathogen]

    aids_ok = download_file(
        remote_path=f"data/processed/02_bioassays_to_keep/aids_{pubchem_name}.csv",
        local_path=os.path.join(raw_dir, "aids.csv"),
    )
    summary_ok = download_file(
        remote_path=f"data/processed/04_extracted_bioassays/{pubchem_name}/summary.csv",
        local_path=os.path.join(raw_dir, "summary.csv"),
    )

    if not aids_ok or not summary_ok:
        return False

    aids = pd.read_csv(os.path.join(raw_dir, "aids.csv"))["aid"].tolist()

    for aid in aids:
        download_file(
            remote_path=f"output/results/{pubchem_name}/{aid}.csv",
            local_path=os.path.join(raw_dir, f"{aid}.csv"),
        )
    return True


def merge_pathogen(pathogen: str) -> pd.DataFrame | None:
    summary_path = os.path.join(REPO_ROOT, "data", "raw", "pubchem", pathogen, "summary.csv")
    if not os.path.exists(summary_path):
        print(f"Skipping merge for {pathogen}: summary.csv not found.")
        return None

    aids_path = os.path.join(REPO_ROOT, "data", "raw", "pubchem", pathogen, "aids.csv")
    valid_aids = set(pd.read_csv(aids_path)["aid"].astype(str))

    df = pd.read_csv(summary_path)
    df["compounds"] = df["actives"] + df["inactives"]
    df = df[df["compounds"] > 0].copy()
    df = df[df["aid"].astype(str).isin(valid_aids)].copy()
    if df.empty:
        print(f"Skipping merge for {pathogen}: no binary-labelled assays.")
        return None

    df["name"] = df["aid"].astype(str)
    df["positives"] = df["actives"]
    df["ratio"] = (df["positives"] / df["compounds"]).round(3)
    df["source"] = "pubchem"
    df.insert(0, "pathogen", pathogen)

    out_cols = ["pathogen", "source", "name", "compounds", "positives", "ratio"]
    df = df[out_cols]

    out_path = os.path.join(REPO_ROOT, "data", "processed", "pubchem", pathogen, "02_pubchem_datasets.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved merged datasets to {out_path}")
    return df


def merge_all_pathogens() -> pd.DataFrame | None:
    processed = os.path.join(REPO_ROOT, "data", "processed", "pubchem")
    dfs = []
    for pathogen in PATHOGENS:
        path = os.path.join(processed, pathogen, "02_pubchem_datasets.csv")
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
    if not dfs:
        print("No per-pathogen datasets found to merge.", file=sys.stderr)
        return None
    df = pd.concat(dfs, ignore_index=True)
    out_path = os.path.join(processed, "02_pubchem_datasets_all.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved combined datasets ({len(df)} rows) to {out_path}")
    return df


def print_summary(df: pd.DataFrame, all_pathogens: bool) -> None:
    print("\n--- Summary ---")
    print(f"Total datasets      : {len(df)}")
    print(f"Average ratio       : {df['ratio'].mean():.3f} ± {df['ratio'].std():.3f}")
    if all_pathogens:
        present = df["pathogen"].unique()
        skipped = [p for p in PATHOGENS if p not in present]
        print(f"\nPathogens processed : {len(present)} / {len(PATHOGENS)}")
        if skipped:
            print(f"Pathogens with 0 AIDs: {', '.join(skipped)}")
        print(f"\nDatasets per pathogen:")
        for pathogen, grp in df.groupby("pathogen"):
            n_cpds = grp["compounds"].sum()
            n_pos = grp["positives"].sum()
            avg_ratio = grp["ratio"].mean()
            print(f"  {pathogen}: {len(grp)} datasets, {n_cpds:,} compounds, {n_pos:,} positives, avg ratio {avg_ratio:.3f}")


def main(args: argparse.Namespace) -> None:
    pathogens = PATHOGENS if args.all else [args.pathogen]
    for pathogen in pathogens:
        downloaded = download_pathogen(pathogen)
        if downloaded:
            merge_pathogen(pathogen)
    df = merge_all_pathogens()
    if df is not None:
        if not args.all:
            df = df[df["pathogen"] == args.pathogen]
        print_summary(df, all_pathogens=args.all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download PubChem bioassay datasets from eosvc"
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
    args = parser.parse_args()
    main(args)
