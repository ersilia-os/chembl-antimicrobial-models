"""
Step 01 (ChEMBL) — Load ChEMBL stage4 signal-based pooled datasets.

Consumes the *signal-based pooled* datasets produced by the rebuilt
chembl-antimicrobial-tasks pipeline (output/stage4/<pathogen>/). Two pool
families are ingested (decided 2026-07-13):

  - 25_pools/{DR,SP}/<pool_id>.csv.gz     grown pools (== the "modelled" pools;
                                          25_pool_summary.csv lists them)
  - 26_pools/{DR,SP}/<category>_catchall.csv.gz   low-data catch-all pools
                                          (26_cv_summary.csv lists them)

Both DR (dose-response) and SP (single-point) categories are kept. All pools
present in 25_pools + 26_pools are retained (no extra quality filter — this is
the "all modelled pools" selection; downstream 10a re-applies an AUROC filter).
First-pass (23_pools) datasets are NOT used, even for pathogens whose growth
step produced no 25 pools (hpylori, ngonorrhoeae) — strict 25+26 per decision.

Each pooled dataset CSV has columns: inchikey, compound_chembl_id, smiles,
value, unit, bin.

Files are copied from a local chembl-antimicrobial-tasks repo assumed to be in
the same parent directory. Pass --eosvc to download from the remote EOS service.

Produces:
  data/raw/chembl/<pathogen>/<pool_id>.csv.gz        (the pool data)
  data/processed/chembl/<pathogen>/01_chembl_datasets.csv
  data/processed/chembl/01_chembl_datasets_all.csv   (all pathogens merged)

Usage:
    python scripts/01_download_datasets_chembl.py
    python scripts/01_download_datasets_chembl.py --eosvc
    python scripts/01_download_datasets_chembl.py --pathogen mtuberculosis
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

REPO_NAME = "chembl-antimicrobial-tasks"
TASKS_REPO_ROOT = os.path.join(REPO_ROOT, "..", REPO_NAME)

# Upstream layout (relative to the tasks repo root)
STAGE4 = "output/stage4"
CATEGORIES = ["DR", "SP"]  # dose-response, single-point

# Per-pathogen summary tables consumed to enumerate pools and pull metadata
SUMMARY_25_POOLS = "25_pool_summary.csv"   # grown pools: pool_id, grown_auroc, ...
SUMMARY_24_CV = "24_cv_summary.csv"        # first-pass CV: pool_id, youden_cutoff, modelled
SUMMARY_25_MEMBERS = "25_pool_members.csv" # post-growth membership: category, pool_id, dataset_id
SUMMARY_26_CV = "26_cv_summary.csv"        # catch-all pools: category, n_datasets, auroc, youden

TARGET_TYPE = "ORGANISM"

# Output metadata columns (kept compatible with the contract read by scripts 03/07a).
META_COLUMNS = [
    "pathogen", "source", "label", "assay_type", "n_assays", "name",
    "activity_type", "unit", "target_type", "cutoff", "auroc",
    "compounds", "positives", "ratio", "pool_step", "member_assay_ids",
]


def copy_from_repo(remote_path: str, local_path: str) -> bool:
    """Copy a file from the sibling chembl-antimicrobial-tasks repo."""
    source = os.path.join(TASKS_REPO_ROOT, remote_path)
    if not os.path.exists(source):
        print(f"  Skipping {remote_path}: not found in local repo.")
        return False
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    shutil.copy2(source, local_path)
    return True


def download_file(remote_path: str, local_path: str) -> bool:
    """Download a file via eosvc from the chembl-antimicrobial-tasks remote."""
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


def _read_summary(transfer, pathogen: str, filename: str) -> pd.DataFrame | None:
    """Transfer and read one stage4 per-pathogen summary CSV. None if absent."""
    remote = f"{STAGE4}/{pathogen}/{filename}"
    local = os.path.join(REPO_ROOT, "data", "raw", "chembl", pathogen, filename)
    if not transfer(remote, local):
        return None
    return pd.read_csv(local)


def _pool_stats(gz_path: str) -> tuple[int, int] | None:
    """Return (n_compounds, n_positives) from a pooled dataset gz, or None."""
    try:
        df = pd.read_csv(gz_path, compression="gzip")
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] could not read {os.path.basename(gz_path)}: {exc}")
        return None
    if COL_SMILES not in df.columns or COL_BIN not in df.columns:
        print(f"  [WARN] {os.path.basename(gz_path)} missing '{COL_SMILES}'/'{COL_BIN}'.")
        return None
    return len(df), int((df[COL_BIN] == 1).sum())


def _ingest_pool(
    transfer, pathogen: str, category: str, pool_id: str, step: str,
    auroc, cutoff, n_assays, raw_dir: str, member_assay_ids=pd.NA,
) -> dict | None:
    """Transfer one pool gz and build its metadata row. None if unavailable."""
    remote = f"{STAGE4}/{pathogen}/{step}_pools/{category}/{pool_id}.csv.gz"
    local = os.path.join(raw_dir, f"{pool_id}.csv.gz")
    if not transfer(remote, local):
        return None
    stats = _pool_stats(local)
    if stats is None:
        return None
    n_compounds, n_positives = stats
    assay_type = "pool" if step == "25" else "catchall"
    return {
        "pathogen": pathogen,
        "source": "chembl",
        "label": category,
        "assay_type": assay_type,
        "n_assays": n_assays,
        "name": pool_id,
        "activity_type": pd.NA,   # pools mix activity types
        "unit": pd.NA,
        "target_type": TARGET_TYPE,
        "cutoff": cutoff,          # youden score cutoff from CV (not an activity cutoff)
        "auroc": round(float(auroc), 3) if pd.notna(auroc) else pd.NA,
        "compounds": n_compounds,
        "positives": n_positives,
        "ratio": round(n_positives / n_compounds, 3) if n_compounds else pd.NA,
        "pool_step": step,
        "member_assay_ids": member_assay_ids,  # pipe-joined ChEMBL assay IDs; NA for catch-alls
    }


def _parse_assay_id(dataset_id: str) -> str:
    """Recover the ChEMBL assay accession from a 25_pool_members.csv dataset_id,
    formatted upstream as '{assay_id}_{activity_type}_{unit_slug}' (unit_slug is
    one of uM/pct/mm, never containing an underscore)."""
    assay_id = dataset_id.rsplit("_", 2)[0]
    if not assay_id.startswith("CHEMBL"):
        raise ValueError(
            f"Unexpected dataset_id format in 25_pool_members.csv: {dataset_id!r} "
            f"(parsed assay_id {assay_id!r} does not start with 'CHEMBL')"
        )
    return assay_id


def _grown_pool_rows(transfer, pathogen: str, raw_dir: str) -> list[dict]:
    """Enumerate 25 (grown) pools from 25_pool_summary and ingest each.

    n_assays / member_assay_ids come from 25_pool_members.csv (post-growth
    membership) rather than 23_pool_summary.csv's n_datasets (pre-growth): step 25
    ("merge leftovers into modelled pools") can substantially grow a pool after
    that count was taken, so the pre-growth figure can understate the true
    constituent-assay count by orders of magnitude for pools that got grown.
    """
    summary = _read_summary(transfer, pathogen, SUMMARY_25_POOLS)
    if summary is None or summary.empty:
        return []
    cv = _read_summary(transfer, pathogen, SUMMARY_24_CV)              # youden_cutoff
    members = _read_summary(transfer, pathogen, SUMMARY_25_MEMBERS)    # dataset_id per pool
    cutoff_map, n_assays_map, assay_ids_map = {}, {}, {}
    if cv is not None:
        cutoff_map = cv.set_index(["category", "pool_id"])["youden_cutoff"].to_dict()
    if members is not None:
        n_assays_map = members.groupby(["category", "pool_id"]).size().to_dict()
        assay_ids_map = {
            key: "|".join(_parse_assay_id(d) for d in group["dataset_id"])
            for key, group in members.groupby(["category", "pool_id"])
        }

    rows = []
    for _, r in summary.iterrows():
        cat, pool_id = r["category"], r["pool_id"]
        key = (cat, pool_id)
        row = _ingest_pool(
            transfer, pathogen, cat, pool_id, step="25",
            auroc=r.get("grown_auroc"),
            cutoff=cutoff_map.get(key, pd.NA),
            n_assays=n_assays_map.get(key, pd.NA),
            member_assay_ids=assay_ids_map.get(key, pd.NA),
            raw_dir=raw_dir,
        )
        if row is not None:
            rows.append(row)
    return rows


def _catchall_pool_rows(transfer, pathogen: str, raw_dir: str) -> list[dict]:
    """Enumerate 26 (catch-all) pools from 26_cv_summary and ingest each."""
    cv = _read_summary(transfer, pathogen, SUMMARY_26_CV)
    if cv is None or cv.empty:
        return []
    rows = []
    for _, r in cv.iterrows():
        cat = r["category"]
        pool_id = f"{cat}_catchall"
        row = _ingest_pool(
            transfer, pathogen, cat, pool_id, step="26",
            auroc=r.get("auroc"),
            cutoff=r.get("youden_cutoff", pd.NA),
            n_assays=r.get("n_datasets", pd.NA),
            raw_dir=raw_dir,
        )
        if row is not None:
            rows.append(row)
    return rows


def download_pathogen(pathogen: str, use_eosvc: bool) -> None:
    transfer = download_file if use_eosvc else copy_from_repo
    raw_dir = os.path.join(REPO_ROOT, "data", "raw", "chembl", pathogen)
    os.makedirs(raw_dir, exist_ok=True)

    print(f"[{pathogen}]")
    rows = _grown_pool_rows(transfer, pathogen, raw_dir)
    rows += _catchall_pool_rows(transfer, pathogen, raw_dir)

    if not rows:
        print(f"  No pooled datasets found for {pathogen}.")
        return

    df = pd.DataFrame(rows)[META_COLUMNS]
    out_path = os.path.join(
        REPO_ROOT, "data", "processed", "chembl", pathogen, "01_chembl_datasets.csv"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    n_grown = int((df["pool_step"] == "25").sum())
    n_catchall = int((df["pool_step"] == "26").sum())
    print(f"  {len(df)} pools ({n_grown} grown, {n_catchall} catch-all) -> {out_path}")


def merge_all_pathogens() -> pd.DataFrame:
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
    print(f"\nSaved combined datasets ({len(df)} rows) to {out_path}")
    return df


def print_summary(df: pd.DataFrame) -> None:
    step = df["pool_step"].astype(str)
    print("\n--- Summary ---")
    print(f"Total pooled datasets : {len(df)}")
    print(f"  grown (25)          : {int((step == '25').sum())}")
    print(f"  catch-all (26)      : {int((step == '26').sum())}")
    print(f"Pathogens with data   : {df['pathogen'].nunique()} / {len(PATHOGENS)}")
    print(f"Average active ratio  : {df['ratio'].mean():.3f} ± {df['ratio'].std():.3f}")
    print("\nDatasets per category (label):")
    for label, count in df["label"].value_counts().items():
        print(f"  {label}: {count}")
    print("\nDatasets per pathogen:")
    counts = df.groupby("pathogen").size()
    cpds = df.groupby("pathogen")["compounds"].sum()
    for pathogen in counts.index:
        print(f"  {pathogen}: {counts[pathogen]} pools, {cpds[pathogen]:,} compounds")


def main(use_eosvc: bool, pathogens: list[str] | None = None) -> None:
    targets = pathogens if pathogens else PATHOGENS
    for pathogen in targets:
        download_pathogen(pathogen, use_eosvc=use_eosvc)
    df = merge_all_pathogens()
    print_summary(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load ChEMBL stage4 pooled datasets for all pathogens."
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
