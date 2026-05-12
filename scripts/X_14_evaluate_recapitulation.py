"""
Step 14 — Recapitulation evaluation per pathogen.

For each ordered model pair (scorer, binarized), binarises the binarized model
at 4 thresholds (1%, 5%, 10%, 25%) and computes the AUROC of the scorer's
continuous prob_ranks against those binary labels. Self-pairs are computed.

Also evaluates how well the global consensus and the leave-one-out consensus
recapitulate each individual model.

Outputs per pathogen in output/results/14_recapitulation/:
  {pathogen}_models.csv    — N×N ordered pairs:
                             model_scorer | model_binarized | auroc_1pct … auroc_25pct
  {pathogen}_consensus.csv — N rows, one per model:
                             model | auroc_global_1pct … auroc_global_25pct
                                   | auroc_excluded_1pct … auroc_excluded_25pct

Usage:
    python scripts/14_evaluate_recapitulation.py
    python scripts/14_evaluate_recapitulation.py --pathogen ecoli
"""

import argparse
import math
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))

DEFAULT_IN_DIR_12 = os.path.join(REPO_ROOT, "output", "results", "12_drugbank")
DEFAULT_IN_DIR_13 = os.path.join(REPO_ROOT, "output", "results", "13_consensus")
DEFAULT_OUT_DIR   = os.path.join(REPO_ROOT, "output", "results", "14_recapitulation")
REPORTS_PATH      = os.path.join(REPO_ROOT, "output", "results", "10_reports.csv")

THRESHOLDS     = [0.01, 0.05, 0.10, 0.25]
THRESHOLD_SFXS = ["1pct", "5pct", "10pct", "25pct"]


def _auroc_at_threshold(scores_scorer: np.ndarray,
                        scores_binarized: np.ndarray,
                        t: float) -> float:
    n = len(scores_binarized)
    k = max(1, math.ceil(t * n))
    cutoff = np.sort(scores_binarized)[::-1][k - 1]
    labels = (scores_binarized >= cutoff).astype(int)
    if labels.sum() == 0 or labels.sum() == n:
        return float("nan")
    return roc_auc_score(labels, scores_scorer)


def run(pathogen: str, in_dir_12: str, in_dir_13: str, out_dir: str) -> None:
    src12 = os.path.join(in_dir_12, f"{pathogen}.csv")
    if not os.path.isfile(src12):
        print(f"  [SKIP] {pathogen}: {src12} not found")
        return

    df12       = pd.read_csv(src12)
    model_cols = [c for c in df12.columns if c != "smiles"]
    if len(model_cols) < 2:
        print(f"  [SKIP] {pathogen}: {len(model_cols)} model(s) — recapitulation requires at least 2")
        return

    scores = {m: df12[m].values for m in model_cols}
    os.makedirs(out_dir, exist_ok=True)

    # ── models file: N×N ordered pairs ──
    rows = []
    for m_scorer in model_cols:
        for m_bin in model_cols:
            row = {"model_scorer": m_scorer, "model_binarized": m_bin}
            for t, sfx in zip(THRESHOLDS, THRESHOLD_SFXS):
                row[f"auroc_{sfx}"] = _auroc_at_threshold(scores[m_scorer], scores[m_bin], t)
            rows.append(row)
    pd.DataFrame(rows).round(4).to_csv(os.path.join(out_dir, f"{pathogen}_models.csv"), index=False)

    # ── consensus file: global and excluded consensus as scorer vs each model ──
    src13 = os.path.join(in_dir_13, f"{pathogen}.csv")
    if not os.path.isfile(src13):
        print(f"  [{pathogen}] models written; skipping consensus ({src13} not found)")
        return

    df13         = pd.read_csv(src13)
    glob_scores  = df13["consensus_score"].values

    rows = []
    for model in model_cols:
        excluded_col = f"excluded_{model}"
        if excluded_col not in df13.columns:
            continue
        excl_scores = df13[excluded_col].values
        model_scores = scores[model]
        row = {"model": model}
        for t, sfx in zip(THRESHOLDS, THRESHOLD_SFXS):
            row[f"auroc_global_{sfx}"]   = _auroc_at_threshold(glob_scores,  model_scores, t)
            row[f"auroc_excluded_{sfx}"] = _auroc_at_threshold(excl_scores,  model_scores, t)
        rows.append(row)
    pd.DataFrame(rows).round(4).to_csv(os.path.join(out_dir, f"{pathogen}_consensus.csv"), index=False)

    print(f"  [{pathogen}] {len(model_cols)} models -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathogen",      default=None)
    parser.add_argument("--input_dir_12",  default=DEFAULT_IN_DIR_12)
    parser.add_argument("--input_dir_13",  default=DEFAULT_IN_DIR_13)
    parser.add_argument("--output_dir",    default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    reports_df = pd.read_csv(REPORTS_PATH)
    pathogens  = [args.pathogen] if args.pathogen else list(dict.fromkeys(reports_df["pathogen"]))

    for pathogen in pathogens:
        run(pathogen, args.input_dir_12, args.input_dir_13, args.output_dir)


if __name__ == "__main__":
    main()
