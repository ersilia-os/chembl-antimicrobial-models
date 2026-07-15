"""
Script 07b — Per-dataset duplication quality report for outputs of script 07.

Audits the post-script-07 datasets in output/07_datasets/{pathogen}/*.csv to verify the
InChIKey-level deduplication and the proven-negative balancing. Each dataset file carries the
upstream `inchikey` column (the key script 07a dedups on), so the audit reads it directly
rather than recomputing InChIKeys. Dataset source (chembl/pubchem) is taken from
07_datasets_metadata.csv.

Per dataset, reports:
  rows                       — total rows (added negatives included)
  no_inchikey                — rows with no InChIKey
  unique_compounds           — distinct InChIKeys
  dup_compounds              — InChIKeys appearing in >1 row (should be 0 after dedup)
  dup_via_diff_smiles        — of dup_compounds, those duplicated via >1 distinct SMILES string
  redundant_rows             — rows - no_inchikey - unique_compounds
  conflict_compounds         — InChIKeys carrying both bin=0 and bin=1 (should be 0)
  added_negatives            — rows flagged added_negative (proven negatives / decoy fallback)
  added_active_collisions    — added-negative rows whose InChIKey is active in any other
                               dataset of the same pathogen (should be 0; non-zero means
                               script 07's pathogen-wide active filter let one slip)

Output:
  output/07_datasets/07_dup_report.csv

Usage:
    python scripts/07b_quality_checks.py
"""

import argparse
import glob
import os
from collections import Counter

import pandas as pd

root = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.abspath(os.path.join(root, ".."))
DATASETS_DIR = os.path.join(REPO_ROOT, "output", "07_datasets")
META_PATH    = os.path.join(DATASETS_DIR, "07_datasets_metadata.csv")
OUT_PATH     = os.path.join(DATASETS_DIR, "07_dup_report.csv")


def _dataset_files(datasets_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(datasets_dir, "*", "*.csv")))


def _source_map(meta_path: str) -> dict:
    """(pathogen, name) -> source, from the step-07 metadata."""
    if not os.path.exists(meta_path):
        return {}
    m = pd.read_csv(meta_path)
    return {(r["pathogen"], str(r["name"])): r["source"] for _, r in m.iterrows()}


def build_pathogen_context(datasets_dir: str) -> tuple[dict, dict]:
    """One pass over all datasets, per pathogen:
      - active InChIKeys (bin=1, not added-negative) — to detect added/active collisions;
      - a Counter of added-negative InChIKeys across the pathogen's datasets — to measure how
        often the same negative is reused across datasets (added_negative sampling shares one
        per-pathogen pool, so the same inactive can land in several datasets).
    """
    actives: dict[str, set[str]] = {}
    added_counts: dict[str, Counter] = {}
    for path in _dataset_files(datasets_dir):
        pathogen = os.path.basename(os.path.dirname(path))
        df = pd.read_csv(path).dropna(subset=["inchikey"])
        has_added = "added_negative" in df.columns
        real = df[~df["added_negative"]] if has_added else df
        actives.setdefault(pathogen, set()).update(real.loc[real["bin"] == 1, "inchikey"])
        if has_added:
            added_counts.setdefault(pathogen, Counter()).update(df.loc[df["added_negative"], "inchikey"])
    return actives, added_counts


def audit_dataset(path: str, source_map: dict, pathogen_active_iks: set, pathogen_added_counts: Counter) -> dict:
    pathogen = os.path.basename(os.path.dirname(path))
    name     = os.path.basename(path)[:-4]

    df = pd.read_csv(path)
    n_rows      = len(df)
    n_no_ik     = int(df["inchikey"].isna().sum())
    valid       = df.dropna(subset=["inchikey"])

    g                       = valid.groupby("inchikey")
    n_compounds             = g.ngroups
    dup_compounds           = int((g.size() > 1).sum())
    dup_compounds_multi_smi = int((g["smiles"].nunique() > 1).sum())
    redundant_rows          = int(n_rows - n_no_ik - n_compounds)
    conflict_compounds      = int((g["bin"].nunique() > 1).sum())

    if "added_negative" in df.columns:
        added_iks = set(valid.loc[valid["added_negative"], "inchikey"])
        added_active_collisions = len(added_iks & pathogen_active_iks)
        n_added = int(df["added_negative"].sum())
        # of this dataset's added negatives, how many were also added to another
        # dataset of the same pathogen (pathogen-wide count > 1)
        added_shared = sum(1 for ik in added_iks if pathogen_added_counts.get(ik, 0) > 1)
    else:
        added_active_collisions = 0
        n_added = 0
        added_shared = 0

    return {
        "pathogen":                pathogen,
        "source":                  source_map.get((pathogen, name), "unknown"),
        "name":                    name,
        "rows":                    n_rows,
        "no_inchikey":             n_no_ik,
        "unique_compounds":        n_compounds,
        "dup_compounds":           dup_compounds,
        "dup_via_diff_smiles":     dup_compounds_multi_smi,
        "redundant_rows":          redundant_rows,
        "conflict_compounds":      conflict_compounds,
        "added_negatives":         n_added,
        "added_neg_shared":        added_shared,
        "added_neg_shared_frac":   round(added_shared / n_added, 3) if n_added else 0.0,
        "added_active_collisions": added_active_collisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_dir", default=DATASETS_DIR)
    parser.add_argument("--output",       default=OUT_PATH)
    args = parser.parse_args()

    files = _dataset_files(args.datasets_dir)
    if not files:
        print(f"No datasets found under {args.datasets_dir}")
        return

    source_map = _source_map(META_PATH)
    print("Building per-pathogen active / added-negative context...")
    pathogen_actives, pathogen_added = build_pathogen_context(args.datasets_dir)
    print(f"  {sum(len(s) for s in pathogen_actives.values()):,} actives across "
          f"{len(pathogen_actives)} pathogens")

    rows = []
    for f in files:
        pathogen = os.path.basename(os.path.dirname(f))
        row = audit_dataset(f, source_map, pathogen_actives.get(pathogen, set()),
                            pathogen_added.get(pathogen, Counter()))
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)

    # Flag anything that should be zero.
    problems = out[(out["dup_compounds"] > 0) | (out["conflict_compounds"] > 0)
                   | (out["added_active_collisions"] > 0)]
    print(f"\n=== SAVED {args.output} ({len(out)} datasets) ===")
    if problems.empty:
        print("All datasets clean: 0 duplicate InChIKeys, 0 label conflicts, 0 added-negative collisions.")
    else:
        print(f"[WARN] {len(problems)} dataset(s) with dup/conflict/collision issues:")
        for _, r in problems.iterrows():
            print(f"  [{r['pathogen']}/{r['name']}] dup={r['dup_compounds']} "
                  f"conflict={r['conflict_compounds']} collisions={r['added_active_collisions']}")

    # Per-pathogen added-negative reuse: how often the shared pool put the same negative
    # into more than one dataset. reuse = total placements / distinct compounds.
    print("\n--- Added-negative reuse across datasets (per pathogen) ---")
    print(f"  {'pathogen':15}{'#ds':>4}{'total':>9}{'unique':>8}{'reuse':>7}{'%reused':>9}")
    reuse_rows = []
    for pathogen, counter in sorted(pathogen_added.items(), key=lambda kv: -(sum(kv[1].values()) - len(kv[1]))):
        total = sum(counter.values())
        unique = len(counter)
        if total == 0:
            continue
        n_ds = int((out["pathogen"] == pathogen).pipe(lambda s: (out.loc[s, "added_negatives"] > 0).sum()))
        pct = round(100 * (total - unique) / total, 1)
        reuse_rows.append((pathogen, n_ds, total, unique, round(total / unique, 3), pct))
        print(f"  {pathogen:15}{n_ds:>4}{total:>9,}{unique:>8,}{total / unique:>7.2f}{pct:>8.1f}%")
    if not reuse_rows:
        print("  (no added negatives)")


if __name__ == "__main__":
    main()
