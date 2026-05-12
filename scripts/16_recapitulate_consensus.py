"""
Step 16 — Consensus recapitulation per pathogen.

For each model, measures how well the consensus (weighted and unweighted from
step 14) recapitulates that model's individual rankings — both when the model
is excluded from the consensus and when it is included.

Inputs:
  output/results/12_drugbank/{pathogen}.csv         — prob_ranks per model
  output/results/14_consensus/{pathogen}.csv        — weighted consensus
  output/results/14_consensus/{pathogen}_unweighted.csv

Outputs per pathogen in output/results/16_recapitulate_consensus/:
  {pathogen}_exc_weighted.csv    — model vs leave-one-out weighted consensus
  {pathogen}_exc_unweighted.csv  — model vs leave-one-out unweighted consensus
  {pathogen}_weighted.csv        — model vs full weighted consensus
  {pathogen}_unweighted.csv      — model vs full unweighted consensus

Each file: model | spearman | pearson |
           hit_overlap_10 | hit_overlap_100 | hit_overlap_500 |
           auroc_0.1pct | auroc_1pct | auroc_5pct

Usage:
    python scripts/16_recapitulate_consensus.py
    python scripts/16_recapitulate_consensus.py --pathogen ecoli
"""

import argparse
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

ROOT      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))

DEFAULT_IN_DIR_12 = os.path.join(REPO_ROOT, "output", "results", "12_drugbank")
DEFAULT_IN_DIR_14 = os.path.join(REPO_ROOT, "output", "results", "14_consensus")
DEFAULT_OUT_DIR   = os.path.join(REPO_ROOT, "output", "results", "16_recapitulate_consensus")
REPORTS_PATH      = os.path.join(REPO_ROOT, "output", "results", "10_reports.csv")

THRESHOLDS     = [0.001, 0.01, 0.05]
THRESHOLD_SFXS = ["0.1pct", "1pct", "5pct"]


def _hit_overlap(scores_a: np.ndarray, scores_b: np.ndarray, top_n: int) -> float:
    """Raw fraction of compounds shared in the top-N of both score arrays."""
    n = len(scores_a)
    if n <= top_n:
        return 1.0
    top_a = set(np.argsort(scores_a)[::-1][:top_n])
    top_b = set(np.argsort(scores_b)[::-1][:top_n])
    return len(top_a & top_b) / top_n


def _auroc_at_threshold(scores_a: np.ndarray, scores_b: np.ndarray, t: float) -> float:
    """AUROC of A scoring against B binarized at the top-t fraction."""
    n = len(scores_b)
    k = max(1, math.ceil(t * n))
    cutoff = np.sort(scores_b)[::-1][k - 1]
    labels = (scores_b >= cutoff).astype(int)
    if labels.sum() == 0 or labels.sum() == n:
        return float("nan")
    return roc_auc_score(labels, scores_a)


def _compute_rows(model_cols: list, model_scores: dict, consensus_scores: dict) -> list:
    """One row per model: all metrics between model scores and its consensus array."""
    rows = []
    for m in model_cols:
        if m not in consensus_scores:
            continue
        mod  = model_scores[m]
        cons = consensus_scores[m]
        row  = {
            "model":           m,
            "spearman":        spearmanr(mod, cons).statistic,
            "pearson":         pearsonr(mod, cons).statistic,
            "hit_overlap_10":  _hit_overlap(cons, mod, 10),
            "hit_overlap_100": _hit_overlap(cons, mod, 100),
            "hit_overlap_500": _hit_overlap(cons, mod, 500),
        }
        # Binarize the model, evaluate how well the consensus recapitulates it
        for t, sfx in zip(THRESHOLDS, THRESHOLD_SFXS):
            row[f"auroc_{sfx}"] = _auroc_at_threshold(cons, mod, t)
        rows.append(row)
    return rows


def run(pathogen: str, in_dir_12: str, in_dir_14: str, out_dir: str) -> None:
    src12      = os.path.join(in_dir_12, f"{pathogen}.csv")
    src14_w    = os.path.join(in_dir_14, f"{pathogen}.csv")
    src14_uw   = os.path.join(in_dir_14, f"{pathogen}_unweighted.csv")

    for src in (src12, src14_w, src14_uw):
        if not os.path.isfile(src):
            print(f"  [SKIP] {pathogen}: {src} not found")
            return

    df12       = pd.read_csv(src12)
    model_cols = [c for c in df12.columns if c != "smiles"]

    if len(model_cols) < 2:
        print(f"  [SKIP] {pathogen}: {len(model_cols)} model(s) — requires at least 2")
        return

    model_scores = {m: df12[m].fillna(0.0).values for m in model_cols}

    df14_w  = pd.read_csv(src14_w)
    df14_uw = pd.read_csv(src14_uw)

    # Build consensus score dicts for the four combinations
    exc_weighted    = {m: df14_w[f"excluded_{m}"].values  for m in model_cols if f"excluded_{m}" in df14_w.columns}
    exc_unweighted  = {m: df14_uw[f"excluded_{m}"].values for m in model_cols if f"excluded_{m}" in df14_uw.columns}
    inc_weighted    = {m: df14_w["consensus_score"].values  for m in model_cols}
    inc_unweighted  = {m: df14_uw["consensus_score"].values for m in model_cols}

    os.makedirs(out_dir, exist_ok=True)

    outputs = [
        (f"{pathogen}_exc_weighted.csv",   exc_weighted),
        (f"{pathogen}_exc_unweighted.csv", exc_unweighted),
        (f"{pathogen}_weighted.csv",       inc_weighted),
        (f"{pathogen}_unweighted.csv",     inc_unweighted),
    ]

    for filename, consensus_scores in outputs:
        rows     = _compute_rows(model_cols, model_scores, consensus_scores)
        out_path = os.path.join(out_dir, filename)
        pd.DataFrame(rows).round(4).to_csv(out_path, index=False)
        print(f"  [{pathogen}] {len(rows)} models -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathogen",      default=None)
    parser.add_argument("--input_dir_12",  default=DEFAULT_IN_DIR_12)
    parser.add_argument("--input_dir_14",  default=DEFAULT_IN_DIR_14)
    parser.add_argument("--output_dir",    default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    reports_df = pd.read_csv(REPORTS_PATH)
    pathogens  = [args.pathogen] if args.pathogen else list(dict.fromkeys(reports_df["pathogen"]))

    for pathogen in pathogens:
        run(pathogen, args.input_dir_12, args.input_dir_14, args.output_dir)


if __name__ == "__main__":
    main()
