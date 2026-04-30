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


def _dir_size_mb(path: str) -> float:
    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, files in os.walk(path)
        for f in files
    )
    return round(total / 1e6, 3)


def aggregate(df: pd.DataFrame, pathogen: str, name: str) -> dict:
    model_name = df["model_name"].iloc[0]
    row = {"pathogen": pathogen, "name": name, "model_name": model_name}

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

    model_dir = os.path.join(MODELS_DIR, pathogen, model_name)
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

    row["model_size_total_mb"] = _dir_size_mb(model_dir) if os.path.isdir(model_dir) else np.nan

    row["predict_rank_actives"]   = ";".join(df["predict_rank_actives"].tolist())
    row["predict_rank_inactives"] = ";".join(df["predict_rank_inactives"].tolist())

    return row


def main() -> None:
    meta_df = pd.read_csv(METADATA_PATH)
    n_total = len(meta_df)
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

        records.append(aggregate(df, pathogen, name))
        print(f"{prefix} {pathogen}/{name} processed!")

    if not records:
        print("No completed datasets found.")
        return

    pd.DataFrame(records).to_csv(OUT_PATH, index=False)
    print(f"\n{len(records)}/{n_total} datasets → {OUT_PATH}")


if __name__ == "__main__":
    main()
