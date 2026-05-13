"""
Step 14 — Consensus scoring of DrugBank compounds per pathogen.

Reads per-pathogen rank matrices from output/results/12_drugbank/{pathogen}.csv
and computes a weighted consensus score per compound using 8 weights:
  W1–W7: model quality weights from 10_reports.csv
  W8:    0 at or below decision_cutoff_rank, linear 0→1 above it

weight[i,m] = average(W1..W7, W8[i,m], weights=W_WEIGHTS)
score[i]    = sum_m(prob_rank[i,m] * weight[i,m]) / sum_m(weight[i,m])

A tanh transformation is then applied to restore the IQR of the consensus scores
toward the average IQR of the individual model prob_ranks. The steepness k depends
only on M (number of models) via a saturating-exponential fit:
  S(M) = 1 + 1.207*(1-exp(-M/6.74)),  k(M) = 2*S(M)
The same k is used for weighted and unweighted files; the center is the empirical
median of each file's global consensus_score. Ranks are preserved exactly (tanh is
strictly monotone). Output values are clipped to [0, 1].

Output: output/results/14_consensus/{pathogen}.csv
        output/results/14_consensus/{pathogen}_unweighted.csv
        output/results/14_consensus/{pathogen}_transformed.csv
        output/results/14_consensus/{pathogen}_unweighted_transformed.csv
        smiles | excluded_{model} x N | consensus_score
        N+1 score columns: one per model exclusion, then full consensus last.

Usage:
    python scripts/14_consensus_scoring.py
    python scripts/14_consensus_scoring.py --pathogen ecoli
"""

import argparse
import os

import numpy as np
import pandas as pd

ROOT      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))

DEFAULT_IN_DIR  = os.path.join(REPO_ROOT, "output", "results", "12_drugbank")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "output", "results", "14_consensus")
REPORTS_PATH    = os.path.join(REPO_ROOT, "output", "results", "10_reports.csv")
W_COLS    = ["w1", "w2", "w3", "w4", "w5", "w6", "w7"]
W_WEIGHTS = np.ones(len(W_COLS) + 1)  # one per w1..w7 + w8; change here to reweight

# Saturating-exponential fit for IQR shrinking factor vs number of models
# S(M) = 1 + _TANH_A*(1-exp(-M/_TANH_TAU)),  k(M) = 2*S(M)
_TANH_A   = 1.207
_TANH_TAU = 6.74


def _compute_w8(prob_ranks: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    """Linear weight above decision cutoff: 0 at or below cutoff, 1 at prob_rank=1."""
    c = np.clip(cutoffs[np.newaxis, :], 0.0, 1.0 - 1e-9)
    return np.where(
        prob_ranks <= c,
        0.0,
        (prob_ranks - c) / (1.0 - c),
    )


def _score(prob_ranks: np.ndarray, w_quality: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    # prob_ranks : (n_compounds, n_models) — normalized rank of each compound under each model
    # w_quality  : (n_models, 7)          — model-level quality weights (w1–w7) from 10_reports
    # cutoffs    : (n_models,)            — decision_cutoff_rank per model, used to compute w8

    # w8 is the only per-compound weight: it rewards compounds ranked above the decision cutoff
    w8 = _compute_w8(prob_ranks, cutoffs)  # (n_compounds, n_models)

    # Stack all 8 weights into a single tensor so we can average them in one call
    n_compounds, n_models = prob_ranks.shape
    w_all = np.empty((n_compounds, n_models, len(W_WEIGHTS)))
    w_all[:, :, :len(W_COLS)] = w_quality  # w1–w7: same for every compound, broadcast over axis 0
    w_all[:, :,  len(W_COLS)] = w8         # w8: varies per compound

    # Collapse the 8 weight dimensions into one scalar per (compound, model)
    w = np.average(w_all, axis=-1, weights=W_WEIGHTS)  # (n_compounds, n_models)

    # Weighted average of prob_ranks across models for each compound
    return (prob_ranks * w).sum(axis=1) / w.sum(axis=1)  # (n_compounds,)


def _score_unweighted(prob_ranks: np.ndarray) -> np.ndarray:
    return prob_ranks.mean(axis=1)


def _k_from_n_models(n: int) -> float:
    """Steepness for the IQR-restoring tanh transform; depends only on number of models."""
    s = 1.0 + _TANH_A * (1.0 - np.exp(-n / _TANH_TAU))
    return 2.0 * s


def _tanh_transform(x: np.ndarray, k: float, center: float) -> np.ndarray:
    return np.clip(0.5 + 0.5 * np.tanh(k * (x - center)), 0.0, 1.0)


def _apply_transform(df: pd.DataFrame, k: float, center: float) -> pd.DataFrame:
    out = df.copy()
    score_cols = [c for c in df.columns if c != "smiles"]
    out[score_cols] = _tanh_transform(df[score_cols].values, k, center).round(4)
    return out


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

    result_unweighted = {"smiles": df["smiles"]}
    for i, model in enumerate(model_cols):
        other = [j for j in range(len(model_cols)) if j != i]
        result_unweighted[f"excluded_{model}"] = _score_unweighted(prob_ranks[:, other]).round(4)
    result_unweighted["consensus_score"] = _score_unweighted(prob_ranks).round(4)

    uw_df = pd.DataFrame(result_unweighted)
    unweighted_path = out_path.replace(".csv", "_unweighted.csv")
    uw_df.to_csv(unweighted_path, index=False)
    print(f"  [{pathogen}] unweighted -> {unweighted_path}")

    # --- tanh IQR-restoring transformation (k depends only on number of models) ---
    k             = _k_from_n_models(len(model_cols))
    avg_model_iqr = float(np.mean([df[m].quantile(0.75) - df[m].quantile(0.25) for m in model_cols]))

    w_cs     = out["consensus_score"]
    w_center = float(w_cs.median())
    w_cons_iqr = float(w_cs.quantile(0.75) - w_cs.quantile(0.25))
    out_t    = _apply_transform(out, k, w_center)
    t_path   = out_path.replace(".csv", "_transformed.csv")
    out_t.to_csv(t_path, index=False)
    w_iqr    = float(out_t["consensus_score"].quantile(0.75) - out_t["consensus_score"].quantile(0.25))
    print(f"  [{pathogen}] weighted transform:   k={k:.3f}  center={w_center:.3f}  "
          f"target_IQR={avg_model_iqr:.4f}  consensus_IQR={w_cons_iqr:.4f}  achieved_IQR={w_iqr:.4f}  -> {t_path}")

    uw_cs      = uw_df["consensus_score"]
    uw_center  = float(uw_cs.median())
    uw_cons_iqr = float(uw_cs.quantile(0.75) - uw_cs.quantile(0.25))
    uw_t       = _apply_transform(uw_df, k, uw_center)
    ut_path    = unweighted_path.replace(".csv", "_transformed.csv")
    uw_t.to_csv(ut_path, index=False)
    uw_iqr     = float(uw_t["consensus_score"].quantile(0.75) - uw_t["consensus_score"].quantile(0.25))
    print(f"  [{pathogen}] unweighted transform: k={k:.3f}  center={uw_center:.3f}  "
          f"target_IQR={avg_model_iqr:.4f}  consensus_IQR={uw_cons_iqr:.4f}  achieved_IQR={uw_iqr:.4f}  -> {ut_path}")


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
