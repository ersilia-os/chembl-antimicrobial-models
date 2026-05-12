"""
Step 13 — Consensus scoring of DrugBank compounds per pathogen.

Reads per-pathogen rank matrices from output/results/12_drugbank/{pathogen}.csv
and computes a weighted consensus score per compound using 8 weights:
  W1–W7: model quality weights from 10_reports.csv
  W8:    piecewise linear function of prob_rank vs decision_cutoff_rank (0→0, cutoff→0.5, 1→1)

weight[i,m] = average(W1..W7, W8[i,m], weights=W_WEIGHTS)
score[i]    = sum_m(prob_rank[i,m] * weight[i,m]) / sum_m(weight[i,m])

Output: output/results/13_consensus/{pathogen}.csv
        smiles | excluded_{model} x N | consensus_score
        N+1 score columns: one per model exclusion, then full consensus last.

Usage:
    python scripts/13_consensus_scoring.py
    python scripts/13_consensus_scoring.py --pathogen ecoli
"""

import argparse
import os

import numpy as np
import pandas as pd

ROOT      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))

DEFAULT_IN_DIR  = os.path.join(REPO_ROOT, "output", "results", "12_drugbank")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "output", "results", "13_consensus")
REPORTS_PATH    = os.path.join(REPO_ROOT, "output", "results", "10_reports.csv")
W_COLS    = ["w1", "w2", "w3", "w4", "w5", "w6", "w7"]
W_WEIGHTS = np.ones(len(W_COLS) + 1)  # one per w1..w7 + w8; change here to reweight


def _compute_w8(prob_ranks: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    """Piecewise linear weight: prob_rank=0→0, cutoff→0.5, 1→1."""
    c = cutoffs[np.newaxis, :]
    return np.where(
        prob_ranks <= c,
        0.5 * prob_ranks / c,
        0.5 + 0.5 * (prob_ranks - c) / (1 - c),
    )


def _score(prob_ranks: np.ndarray, w_quality: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    # w_quality: (n_models, 7) — quality weights from 10_reports, one column per W_COLS
    w8 = _compute_w8(prob_ranks, cutoffs)

    n_compounds, n_models = prob_ranks.shape
    w_all = np.empty((n_compounds, n_models, len(W_WEIGHTS)))
    w_all[:, :, :len(W_COLS)] = w_quality   # (n_models, 7) broadcasts over the compounds axis
    w_all[:, :,  len(W_COLS)] = w8

    w = np.average(w_all, axis=-1, weights=W_WEIGHTS)  # (n_compounds, n_models)
    return (prob_ranks * w).sum(axis=1) / w.sum(axis=1)


def run(pathogen: str, in_dir: str, reports_df: pd.DataFrame, out_path: str) -> None:
    src = os.path.join(in_dir, f"{pathogen}.csv")
    if not os.path.isfile(src):
        print(f"  [SKIP] {pathogen}: {src} not found")
        return

    df             = pd.read_csv(src)
    model_reports  = reports_df[reports_df["pathogen"] == pathogen].set_index("model_name")
    model_cols     = [col for col in df.columns if col != "smiles" and col in model_reports.index]

    if len(model_cols) < 2:
        print(f"  [SKIP] {pathogen}: {len(model_cols)} model(s) — consensus requires at least 2")
        return

    prob_ranks = df[model_cols].fillna(0.0).values
    w_quality  = np.array([model_reports.loc[m, W_COLS].values for m in model_cols], dtype=float)
    cutoffs    = np.array([model_reports.loc[m, "decision_cutoff_rank"] for m in model_cols], dtype=float)

    result = {"smiles": df["smiles"]}

    # Leave-one-out: score using all models except model i
    for i, model in enumerate(model_cols):
        other = [j for j in range(len(model_cols)) if j != i]
        result[f"excluded_{model}"] = _score(
            prob_ranks[:, other],
            w_quality[other, :],
            cutoffs[other],
        ).round(4)

    # Full consensus: score using all models
    result["consensus_score"] = _score(prob_ranks, w_quality, cutoffs).round(4)

    out = pd.DataFrame(result)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"  [{pathogen}] {len(model_cols)} models -> {len(out)} compounds -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathogen",   default=None)
    parser.add_argument("--input_dir",  default=DEFAULT_IN_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output",     default=None)
    args = parser.parse_args()

    reports_df = pd.read_csv(REPORTS_PATH)
    pathogens  = [args.pathogen] if args.pathogen else list(dict.fromkeys(reports_df["pathogen"]))

    for pathogen in pathogens:
        out_path = args.output or os.path.join(args.output_dir, f"{pathogen}.csv")
        run(pathogen, args.input_dir, reports_df, out_path)


if __name__ == "__main__":
    main()
