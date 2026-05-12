"""
Step 15 — Pairwise model recapitulation per pathogen.

For every ordered pair (A, B) of models, quantifies how well A's continuous
prob_ranks relate to B's using four metric families:
  - Spearman and Pearson rank correlation
  - Raw hit overlap at top 10, 100, 500
  - AUROC: binarize B at 0.1%, 1%, 5% and score with A

Input:  output/results/12_drugbank/{pathogen}.csv
Output: output/results/15_recapitulate_models/{pathogen}.csv
        model_a | model_b | spearman | pearson |
        hit_overlap_10 | hit_overlap_100 | hit_overlap_500 |
        auroc_0.1pct | auroc_1pct | auroc_5pct

Usage:
    python scripts/15_recapitulate_models.py
    python scripts/15_recapitulate_models.py --pathogen ecoli
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

DEFAULT_IN_DIR  = os.path.join(REPO_ROOT, "output", "results", "12_drugbank")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "output", "results", "15_recapitulate_models")
REPORTS_PATH    = os.path.join(REPO_ROOT, "output", "results", "10_reports.csv")

THRESHOLDS     = [0.001, 0.01, 0.05]
THRESHOLD_SFXS = ["0.1pct", "1pct", "5pct"]


def _hit_overlap(scores_a: np.ndarray, scores_b: np.ndarray, top_n: int) -> float:
    """Raw fraction of compounds shared in the top-N of both models."""
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


def run(pathogen: str, in_dir: str, out_dir: str) -> None:
    src = os.path.join(in_dir, f"{pathogen}.csv")
    if not os.path.isfile(src):
        print(f"  [SKIP] {pathogen}: {src} not found")
        return

    df         = pd.read_csv(src)
    model_cols = [c for c in df.columns if c != "smiles"]

    if len(model_cols) < 2:
        print(f"  [SKIP] {pathogen}: {len(model_cols)} model(s) — pairwise requires at least 2")
        return

    scores = {m: df[m].fillna(0.0).values for m in model_cols}

    rows = []
    for m_a in model_cols:
        for m_b in model_cols:
            if m_a == m_b:
                continue
            a, b = scores[m_a], scores[m_b]
            row = {
                "model_a":         m_a,
                "model_b":         m_b,
                "spearman":        spearmanr(a, b).statistic,
                "pearson":         pearsonr(a, b).statistic,
                "hit_overlap_10":  _hit_overlap(a, b, 10),
                "hit_overlap_100": _hit_overlap(a, b, 100),
                "hit_overlap_500": _hit_overlap(a, b, 500),
            }
            for t, sfx in zip(THRESHOLDS, THRESHOLD_SFXS):
                row[f"auroc_{sfx}"] = _auroc_at_threshold(a, b, t)
            rows.append(row)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{pathogen}.csv")
    pd.DataFrame(rows).round(4).to_csv(out_path, index=False)
    print(f"  [{pathogen}] {len(model_cols)} models -> {len(rows)} pairs -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathogen",   default=None)
    parser.add_argument("--input_dir",  default=DEFAULT_IN_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    reports_df = pd.read_csv(REPORTS_PATH)
    pathogens  = [args.pathogen] if args.pathogen else list(dict.fromkeys(reports_df["pathogen"]))

    for pathogen in pathogens:
        run(pathogen, args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
