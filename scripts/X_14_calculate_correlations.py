"""
Step 14 — Model correlation analysis per pathogen.

Reads raw prob_ranks from output/results/12_drugbank/{pathogen}.csv and the
leave-one-out consensus scores from output/results/13_consensus/{pathogen}.csv.

Produces two files per pathogen in output/results/14_correlations/:
  {pathogen}_models.csv   — one row per unique model pair (i < j):
                            spearman, hit_overlap_100, hit_overlap_1000, compound_overlap
  {pathogen}_consensus.csv — per-model correlation vs excluded and global consensus:
                             spearman + hit_overlap_100/1000 for both

Usage:
    python scripts/14_calculate_correlations.py
    python scripts/14_calculate_correlations.py --pathogen ecoli
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))

DEFAULT_IN_DIR_12 = os.path.join(REPO_ROOT, "output", "results", "12_drugbank")
DEFAULT_IN_DIR_13 = os.path.join(REPO_ROOT, "output", "results", "13_consensus")
DEFAULT_OUT_DIR   = os.path.join(REPO_ROOT, "output", "results", "14_correlations")
REPORTS_PATH      = os.path.join(REPO_ROOT, "output", "results", "10_reports.csv")
POSITIVES_PATH    = os.path.join(REPO_ROOT, "output", "results", "03_selected_positives.csv")
DATASETS_DIR      = os.path.join(REPO_ROOT, "output", "results", "07_datasets")


def hit_overlap_chance(probs1: np.ndarray, probs2: np.ndarray, top: int) -> float:
    """Normalised hit overlap above chance for the top-N predictions."""
    if np.isnan(probs1).any() or np.isnan(probs2).any():
        raise ValueError("hit_overlap_chance: NaN values in input arrays")
    n = len(probs1)
    if n <= top:
        return 1.0
    ind1     = set(np.argsort(probs1)[::-1][:top])
    ind2     = set(np.argsort(probs2)[::-1][:top])
    m        = len(ind1 & ind2)
    expected = top * top / n
    return (m - expected) / (top - expected)


def _build_ik_sets(model_cols: list, model_reports: pd.DataFrame,
                   pathogen: str, smiles_to_ik: dict) -> dict:
    ik_sets = {}
    for model in model_cols:
        name     = model_reports.loc[model, "name"]
        df_train = pd.read_csv(os.path.join(DATASETS_DIR, pathogen, f"{name}.csv"))
        ik_sets[model] = {smiles_to_ik.get(s) for s in df_train["smiles"]} - {None}
    return ik_sets


def run(pathogen: str, in_dir_12: str, in_dir_13: str,
        reports_df: pd.DataFrame, smiles_to_ik: dict, out_dir: str) -> None:
    src12 = os.path.join(in_dir_12, f"{pathogen}.csv")
    src13 = os.path.join(in_dir_13, f"{pathogen}.csv")

    for src in (src12, src13):
        if not os.path.isfile(src):
            print(f"  [SKIP] {pathogen}: {src} not found")
            return

    df12           = pd.read_csv(src12)
    df13           = pd.read_csv(src13)
    model_cols = [c for c in df12.columns if c != "smiles"]
    if len(model_cols) < 2:
        print(f"  [SKIP] {pathogen}: {len(model_cols)} model(s) — correlation requires at least 2")
        return
    model_reports  = reports_df[reports_df["pathogen"] == pathogen].set_index("model_name")
    ik_sets        = _build_ik_sets(model_cols, model_reports, pathogen, smiles_to_ik)

    os.makedirs(out_dir, exist_ok=True)

    # File 1: all-vs-all metrics for each unique model pair
    rows = []
    for i, m1 in enumerate(model_cols):
        for m2 in model_cols[i + 1:]:
            p1, p2 = df12[m1].values, df12[m2].values
            rows.append({
                "model_1":          m1,
                "model_2":          m2,
                "spearman":         spearmanr(p1, p2).statistic,
                "hit_overlap_100":  hit_overlap_chance(p1, p2, 100),
                "hit_overlap_1000": hit_overlap_chance(p1, p2, 1000),
                "compound_overlap": len(ik_sets[m1] & ik_sets[m2]) / min(len(ik_sets[m1]), len(ik_sets[m2])) if min(len(ik_sets[m1]), len(ik_sets[m2])) > 0 else 0.0,
            })
    pd.DataFrame(rows).round(4).to_csv(os.path.join(out_dir, f"{pathogen}_models.csv"), index=False)

    # File 2: each model vs the consensus score that excludes it, and vs the global consensus
    rows = []
    for model in model_cols:
        excluded_col = f"excluded_{model}"
        if excluded_col not in df13.columns:
            continue
        prob = df12[model].values
        excl = df13[excluded_col].values
        glob = df13["consensus_score"].values
        rows.append({
            "model":                     model,
            "spearman_r_excluded":       spearmanr(prob, excl).statistic,
            "spearman_r_global":         spearmanr(prob, glob).statistic,
            "hit_overlap_100_excluded":  hit_overlap_chance(prob, excl, 100),
            "hit_overlap_100_global":    hit_overlap_chance(prob, glob, 100),
            "hit_overlap_1000_excluded": hit_overlap_chance(prob, excl, 1000),
            "hit_overlap_1000_global":   hit_overlap_chance(prob, glob, 1000),
        })
    pd.DataFrame(rows).round(4).to_csv(os.path.join(out_dir, f"{pathogen}_consensus.csv"), index=False)

    print(f"  [{pathogen}] {len(model_cols)} models -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathogen",     default=None)
    parser.add_argument("--input_dir_12", default=DEFAULT_IN_DIR_12)
    parser.add_argument("--input_dir_13", default=DEFAULT_IN_DIR_13)
    parser.add_argument("--output_dir",   default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    reports_df   = pd.read_csv(REPORTS_PATH)
    df03         = pd.read_csv(POSITIVES_PATH, usecols=["canonical_smiles", "inchikey"])
    smiles_to_ik = dict(zip(df03["canonical_smiles"], df03["inchikey"]))
    pathogens    = [args.pathogen] if args.pathogen else list(dict.fromkeys(reports_df["pathogen"]))

    for pathogen in pathogens:
        run(pathogen, args.input_dir_12, args.input_dir_13, reports_df, smiles_to_ik, args.output_dir)


if __name__ == "__main__":
    main()
