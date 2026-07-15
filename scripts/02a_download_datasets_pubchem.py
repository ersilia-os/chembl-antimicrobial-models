"""
Step 02 (PubChem) — Load the ORGANISM pooled datasets from pubchem-antimicrobial-tasks.

The PubChem pipeline now produces its final organism datasets in **step 08**
(`08_transfer_pool_organism`): step 07 folds near-duplicate whole-cell (organism) assays,
then step 08 transfer-pools those datasets (the PubChem analogue of ChEMBL stage4 pooling)
and merges members to one row per InChIKey. We ingest exactly these step-08 pools
(single_protein assays are not used). Each pool file has columns: inchikey, cid, smiles, bin.

Inputs (from the sibling pubchem-antimicrobial-tasks repo, or --eosvc):
    output/08_transfer_pool_organism/08_pool_summary.csv    one row per pool
    output/08_transfer_pool_organism/08_pool_members.csv    pool_id -> member dataset_id + aids
    output/08_transfer_pool_organism/{code}/{pool_id}.csv   the pool data

Produces:
    data/raw/pubchem/{code}/{pool_id}.csv                       the pool data
    data/processed/pubchem/02_pubchem_datasets_organism.csv     metadata (one row per pool)

Usage:
    python scripts/02a_download_datasets_pubchem.py
    python scripts/02a_download_datasets_pubchem.py --eosvc
"""

import argparse
import os
import shutil
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(ROOT, "..")
sys.path.append(os.path.join(ROOT, "..", "src"))

from default import COL_BIN, COL_SMILES, PATHOGENS  # noqa: E402

REPO_NAME = "pubchem-antimicrobial-tasks"
PUBCHEM_TASKS_REPO_ROOT = os.path.join(REPO_ROOT, "..", REPO_NAME)

# Upstream layout (relative to the tasks repo root)
POOL_DIR       = "output/08_transfer_pool_organism"
REMOTE_SUMMARY = f"{POOL_DIR}/08_pool_summary.csv"
REMOTE_MEMBERS = f"{POOL_DIR}/08_pool_members.csv"

LOCAL_SUMMARY = os.path.join(REPO_ROOT, "data", "raw", "pubchem", "08_pool_summary.csv")
LOCAL_MEMBERS = os.path.join(REPO_ROOT, "data", "raw", "pubchem", "08_pool_members.csv")
LOCAL_ORGANISM = os.path.join(
    REPO_ROOT, "data", "processed", "pubchem", "02_pubchem_datasets_organism.csv"
)

TARGET_TYPE = "ORGANISM"

# Output metadata columns (aligned with the ChEMBL 01 contract where they overlap).
META_COLUMNS = [
    "pathogen", "source", "name", "is_merged", "n_members", "member_aids",
    "target_type", "compounds", "positives", "inactives", "ratio",
]


def copy_from_repo(remote_path: str, local_path: str) -> bool:
    source = os.path.join(PUBCHEM_TASKS_REPO_ROOT, remote_path)
    if not os.path.exists(source):
        print(f"  Skipping {remote_path}: not found in local repo.")
        return False
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    shutil.copy2(source, local_path)
    return True


def download_file(remote_path: str, local_path: str) -> bool:
    env = os.environ.copy()
    env["EVC_REPO_NAME"] = REPO_NAME
    cmd = ["eosvc", "download", "--path", remote_path]
    result = subprocess.run(cmd, env=env, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        print(f"  Skipping {remote_path}: not found in the cloud.")
        return False
    remote_abs = os.path.join(REPO_ROOT, remote_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    shutil.move(remote_abs, local_path)
    return True


def _dataset_stats(csv_path: str) -> tuple[int, int] | None:
    """Return (n_compounds, n_positives) from a pool CSV, or None if unreadable."""
    try:
        df = pd.read_csv(csv_path, usecols=[COL_SMILES, COL_BIN])
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] could not read {os.path.basename(csv_path)}: {exc}")
        return None
    return len(df), int((df[COL_BIN] == 1).sum())


def _member_aids_map(members: pd.DataFrame | None) -> dict:
    """(pathogen_code, pool_id) -> sorted unique underlying assay AIDs, from 08_pool_members."""
    out: dict[tuple, list] = {}
    if members is None:
        return out
    for _, mr in members.iterrows():
        key = (mr["pathogen_code"], str(mr["pool_id"]))
        aids = [a for a in str(mr["member_aids"]).split("|") if a]
        out.setdefault(key, []).extend(aids)
    return {k: sorted(set(v)) for k, v in out.items()}


def main(use_eosvc: bool) -> None:
    transfer = download_file if use_eosvc else copy_from_repo

    if not transfer(REMOTE_SUMMARY, LOCAL_SUMMARY):
        print("[ERROR] Failed to obtain the step-08 pool summary.", file=sys.stderr)
        sys.exit(1)
    summary = pd.read_csv(LOCAL_SUMMARY)

    members = pd.read_csv(LOCAL_MEMBERS) if transfer(REMOTE_MEMBERS, LOCAL_MEMBERS) else None
    aids_map = _member_aids_map(members)

    rows = []
    for _, r in summary.iterrows():
        code = r["pathogen_code"]
        if code not in PATHOGENS:
            print(f"  [WARN] unknown pathogen code '{code}', skipping.")
            continue
        pool_id = str(r["pool_id"])
        local = os.path.join(REPO_ROOT, "data", "raw", "pubchem", code, f"{pool_id}.csv")
        if not transfer(f"{POOL_DIR}/{code}/{pool_id}.csv", local):
            continue
        stats = _dataset_stats(local)
        if stats is None:
            continue
        n_compounds, n_positives = stats

        aids = aids_map.get((code, pool_id), [])
        member_aids = "|".join(aids)
        n_members = len(aids)
        rows.append({
            "pathogen": code,
            "source": "pubchem",
            "name": pool_id,
            "is_merged": n_members > 1,           # pool spans >1 underlying assay
            "n_members": n_members,
            "member_aids": member_aids,
            "target_type": TARGET_TYPE,
            "compounds": n_compounds,
            "positives": n_positives,
            "inactives": n_compounds - n_positives,
            "ratio": round(n_positives / n_compounds, 3) if n_compounds else pd.NA,
        })

    if not rows:
        print("[ERROR] No organism pools ingested.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)[META_COLUMNS]
    os.makedirs(os.path.dirname(LOCAL_ORGANISM), exist_ok=True)
    df.to_csv(LOCAL_ORGANISM, index=False)
    print(f"\nSaved organism pool metadata to {LOCAL_ORGANISM}")
    print_summary(df)


def print_summary(df: pd.DataFrame) -> None:
    print("\n--- Summary ---")
    print(f"Total organism pools : {len(df)}")
    print(f"  multi-assay        : {int(df['is_merged'].sum())}")
    print(f"  single-assay       : {int((~df['is_merged']).sum())}")
    print(f"Pathogens with data  : {df['pathogen'].nunique()} / {len(PATHOGENS)}")
    print(f"Average active ratio : {df['ratio'].mean():.3f} ± {df['ratio'].std():.3f}")
    print("\nPools per pathogen:")
    counts = df.groupby("pathogen").size()
    cpds = df.groupby("pathogen")["compounds"].sum()
    for pathogen in counts.index:
        print(f"  {pathogen}: {counts[pathogen]} pools, {cpds[pathogen]:,} compounds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load PubChem organism pooled datasets (step 08).")
    parser.add_argument(
        "--eosvc",
        action="store_true",
        default=False,
        help="Download from the remote EOS service instead of copying from local repo.",
    )
    args = parser.parse_args()
    main(use_eosvc=args.eosvc)
