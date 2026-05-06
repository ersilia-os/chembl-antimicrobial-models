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


def _piecewise_linear(x: float, knots: list[tuple[float, float]]) -> float:
    if x <= knots[0][0]:
        return knots[0][1]
    if x >= knots[-1][0]:
        return knots[-1][1]
    for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return knots[-1][1]


_W1_MAP = {"A": 1.0, "B": 1.0, "M": 0.5, "G": 0.0}


def _w1(label: str) -> float:
    """Dataset type: individual (A/B) → 1, merged (M) → 0.5, general (G) → 0."""
    return _W1_MAP.get(label, 0.0)


def _w2(df: pd.DataFrame, n_decoys: int) -> float:
    """Decoy contamination: 1 if no decoys, decreasing linearly to 0 as decoys fill inactives."""
    n_inactives = int(df["compounds_test"].sum()) - int(df["positives_test"].sum())
    if n_inactives == 0:
        return 1.0
    return round(max(0.0, 1.0 - n_decoys / n_inactives), 4)


def _w3(df: pd.DataFrame) -> float:
    """Mean CV AUROC: 0 at ≤0.7, linear to 1 at 1.0."""
    auroc = df["auroc"].mean()
    if auroc <= 0.7:
        return 0.0
    return round(min((auroc - 0.7) / 0.3, 1.0), 4)


def _enrichment_weight(value: float, baseline: float) -> float:
    """
    Combined weight from two equal sub-scores (each 0–0.5, total 0–1):
      c1: absolute excess — 0 when value=baseline, 0.5 when value=1
      c2: fold enrichment — 0 at ≤1×, 0.5 at ≥10× over baseline
    """
    c1 = 0.5 * float(np.clip((value - baseline) / (1.0 - baseline), 0.0, 1.0))
    c2 = 0.5 * float(np.clip((value / baseline - 1.0) / 9.0, 0.0, 1.0))
    return round(c1 + c2, 4)


def _w4(df: pd.DataFrame) -> float:
    """AUPRC weight: absolute-excess (0–0.5) + fold-enrichment (0–0.5) over prevalence baseline."""
    return _enrichment_weight(df["auprc"].mean(), df["baseline_auprc"].mean())


def _w5(df: pd.DataFrame) -> float:
    """BEDROC weight: absolute-excess (0–0.5) + fold-enrichment (0–0.5) over random baseline."""
    return _enrichment_weight(df["bedroc"].mean(), df["baseline_bedroc"].mean())


_W6_KNOTS = [(100, 0.0), (1_000, 0.25), (10_000, 0.5), (100_000, 1.0)]


def _w6(df: pd.DataFrame) -> float:
    """Total compound count: piecewise linear from <100 → 0 to ≥100k → 1."""
    n = int(df["compounds_test"].sum())
    return round(_piecewise_linear(n, _W6_KNOTS), 4)


_W7_KNOTS = [(50, 0.0), (250, 0.25), (1_000, 0.5), (10_000, 1.0)]


def _w7(df: pd.DataFrame) -> float:
    """Total active compound count: piecewise linear from <50 → 0 to ≥10k → 1."""
    n = int(df["positives_test"].sum())
    return round(_piecewise_linear(n, _W7_KNOTS), 4)


def aggregate(df: pd.DataFrame, pathogen: str, name: str, mrow) -> dict:
    model_name = df["model_name"].iloc[0]

    model_dir = os.path.join(MODELS_DIR, pathogen, model_name)
    meta_path = os.path.join(model_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            model_meta = json.load(f)
        decision_cutoff_rank = round(model_meta["decision_cutoff_rank"], 4)
        portfolio = ";".join(sorted(p.upper() for p in model_meta.get("portfolio", [])))
    else:
        decision_cutoff_rank = np.nan
        portfolio = np.nan
    model_size_total_mb = _dir_size_mb(model_dir) if os.path.isdir(model_dir) else np.nan

    row = {"pathogen": pathogen, "name": name, "model_name": model_name}

    row["n_compounds"] = int(df["compounds_test"].sum())
    row["n_positives"] = int(df["positives_test"].sum())

    for col in ("auroc", "auprc", "baseline_auprc", "bedroc", "baseline_bedroc"):
        row[f"{col}_mean"] = round(df[col].mean(), 4)
        row[f"{col}_std"]  = round(df[col].std(), 4)

    row["w1"] = _w1(mrow.label)
    row["w2"] = _w2(df, int(mrow.decoys))
    row["w3"] = _w3(df)
    row["w4"] = _w4(df)
    row["w5"] = _w5(df)
    row["w6"] = _w6(df)
    row["w7"] = _w7(df)
    row["final_weight"] = round(float(np.mean([row["w1"], row["w2"], row["w3"], row["w4"], row["w5"], row["w6"], row["w7"]])), 4)

    row["decision_cutoff_rank"] = decision_cutoff_rank
    row["portfolio"]            = portfolio
    row["model_size_total_mb"]  = model_size_total_mb

    for desc in DESCRIPTORS:
        col  = f"oof_auc_{desc}"
        vals = df[col].dropna()
        row[f"{col}_mean"] = round(vals.mean(), 4) if not vals.empty else np.nan
        row[f"{col}_std"]  = round(vals.std(),  4) if not vals.empty else np.nan

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

        records.append(aggregate(df, pathogen, name, mrow))
        print(f"{prefix} {pathogen}/{name} processed!")

    if not records:
        print("No completed datasets found.")
        return

    out = pd.DataFrame(records)
    pathogen_totals = out.groupby("pathogen")["final_weight"].transform("sum")
    normalized = (out["final_weight"] / pathogen_totals * 100).round(4)
    out.insert(out.columns.get_loc("final_weight") + 1, "final_normalized_weight", normalized)

    out.to_csv(OUT_PATH, index=False)
    print(f"\n{len(records)}/{n_total} datasets → {OUT_PATH}")


if __name__ == "__main__":
    main()
