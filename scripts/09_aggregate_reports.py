"""
Step 09 — Aggregate per-dataset CV reports into a single summarised file.

Iterates over datasets from 06_datasets_metadata.csv, validates that all 5
folds are present, and writes one row per dataset to 09_reports.csv.

Usage:
    python scripts/09_aggregate_reports.py
"""

import json
import os

import numpy as np
import pandas as pd

ROOT      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))

METADATA_PATH = os.path.join(REPO_ROOT, "output", "results", "06_datasets_metadata.csv")
REPORTS_DIR   = os.path.join(REPO_ROOT, "output", "results", "08_reports")
MODELS_DIR    = os.path.join(REPO_ROOT, "output", "results", "08_models")
OUT_PATH      = os.path.join(REPO_ROOT, "output", "results", "09_reports.csv")

N_FOLDS      = 5
DESCRIPTORS  = ["cddd", "chemeleon", "clamp", "morgan", "rdkit"]


def _model_name(mrow: pd.Series, counter: int) -> str:
    label_map = {"A": "individual", "B": "individual", "M": "merged", "G": "general"}
    label_type = label_map[mrow["label"]]
    parts = [
        label_type,
        mrow["activity_type"].lower(),
    ]
    if int(mrow["decoys"]) > 0:
        parts.append("decoys")
    return "_".join(parts)


def _dir_size_mb(path: str) -> float:
    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, files in os.walk(path)
        for f in files
    )
    return round(total / 1e6, 3)


def aggregate(df: pd.DataFrame, pathogen: str, name: str, mrow: pd.Series, counter: int) -> dict:
    row = {"pathogen": pathogen, "name": name, "model_name": _model_name(mrow, counter)}

    row["n_compounds"] = int(df["compounds_test"].sum())
    row["n_positives"] = int(df["positives_test"].sum())

    for col in ("auroc", "auprc", "baseline_auprc"):
        row[f"{col}_mean"] = round(df[col].mean(), 4)
        row[f"{col}_std"]  = round(df[col].std(), 4)

    for desc in DESCRIPTORS:
        col = f"oof_auc_{desc}"
        vals = df[col].dropna()
        if vals.empty:
            row[f"{col}_mean"] = np.nan
            row[f"{col}_std"]  = np.nan
        else:
            row[f"{col}_mean"] = round(vals.mean(), 4)
            row[f"{col}_std"]  = round(vals.std(), 4)

    model_dir = os.path.join(MODELS_DIR, pathogen, name)
    meta_path = os.path.join(model_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            model_meta = json.load(f)
        row["decision_cutoff_rank"] = round(model_meta["decision_cutoff_rank"], 4)
        portfolio = model_meta.get("portfolio", [])
        row["portfolio"] = ";".join(sorted(p.upper() for p in portfolio))
    else:
        row["decision_cutoff_rank"] = np.nan
        row["portfolio"] = np.nan

    for desc in DESCRIPTORS:
        desc_dir = os.path.join(model_dir, desc)
        row[f"model_size_{desc}_mb"] = _dir_size_mb(desc_dir) if os.path.isdir(desc_dir) else np.nan

    row["model_size_total_mb"] = _dir_size_mb(model_dir) if os.path.isdir(model_dir) else np.nan

    row["predict_rank_actives"]   = ";".join(df["predict_rank_actives"].tolist())
    row["predict_rank_inactives"] = ";".join(df["predict_rank_inactives"].tolist())

    return row


def main() -> None:
    meta_df = pd.read_csv(METADATA_PATH)
    n_total = len(meta_df)

    # Build per-pathogen counters (counts ALL rows, including incomplete ones)
    meta_lookup = {}
    pathogen_counters: dict[str, int] = {}
    for _, mrow in meta_df.iterrows():
        p = mrow["pathogen"]
        c = pathogen_counters.get(p, 0)
        meta_lookup[(p, mrow["name"])] = (mrow, c)
        pathogen_counters[p] = c + 1

    records = []

    for i, mrow in enumerate(meta_df.itertuples(), start=1):
        pathogen, name = mrow.pathogen, mrow.name
        prefix = f"[{i}/{n_total}]"

        report_path = os.path.join(REPORTS_DIR, pathogen, f"{name}.csv")
        if not os.path.exists(report_path):
            print(f"{prefix} [WARN] {pathogen}/{name}: report not found, skipping")
            continue

        df = pd.read_csv(report_path)
        if len(df) < N_FOLDS:
            print(f"{prefix} [WARN] {pathogen}/{name}: only {len(df)}/{N_FOLDS} folds complete, skipping")
            continue

        meta_row, counter = meta_lookup[(pathogen, name)]
        records.append(aggregate(df, pathogen, name, meta_row, counter))
        print(f"{prefix} {pathogen}/{name} processed!")

    if not records:
        print("No completed datasets found.")
        return

    # Deduplicate model_name within each pathogen: append _a, _b, ... for clashes
    from collections import Counter as _Counter
    import string as _string
    seen: dict[tuple, int] = {}
    for rec in records:
        key = (rec["pathogen"], rec["model_name"])
        seen[key] = seen.get(key, 0) + 1

    counts: dict[tuple, int] = {}
    suffixes = list(_string.ascii_lowercase)
    for rec in records:
        key = (rec["pathogen"], rec["model_name"])
        if seen[key] > 1:
            n = counts.get(key, 0)
            rec["model_name"] = f"{rec['model_name']}_{suffixes[n]}"
            counts[key] = n + 1

    pd.DataFrame(records).to_csv(OUT_PATH, index=False)
    print(f"\n{len(records)}/{n_total} datasets → {OUT_PATH}")


if __name__ == "__main__":
    main()
