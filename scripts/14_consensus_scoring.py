"""
Step 14 — Consensus scoring of DrugBank compounds per pathogen.

Reads per-pathogen rank matrices from output/12_drugbank/{pathogen}.csv
and computes a weighted consensus score per compound using 8 weights:
  W1–W7: model quality weights from 10_fixed_weights/10_reports.csv
  W8:    0 at or below decision_cutoff_rank, linear 0→1 above it

weight[i,m] = average(W1..W7, W8[i,m], weights=W_WEIGHTS)
score[i]    = sum_m(prob_rank[i,m] * weight[i,m]) / sum_m(weight[i,m])

A tanh transformation is then applied to restore the IQR of the consensus scores
toward the average IQR of the individual model prob_ranks. The steepness k depends
only on M (number of models) via a saturating-exponential fit:
  S(M) = 1 + 1.156*(1-exp(-M/6.47)),  k(M) = 2*S(M)
The center is fixed at 0.5 (neutral point of the prob_rank scale), making the
transformation independent of the compound set being scored. Dividing by tanh(k/2)
normalises the curve through [0,0] and [1,1], guaranteeing scores above 0.5 always
increase and scores below 0.5 always decrease. Ranks are preserved (tanh is strictly
monotone). Output is guaranteed in [0, 1] without clipping.

Output: output/14_consensus/{pathogen}.csv
        output/14_consensus/{pathogen}_unweighted.csv
        output/14_consensus/{pathogen}_transformed.csv
        output/14_consensus/{pathogen}_unweighted_transformed.csv
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

DEFAULT_IN_DIR  = os.path.join(REPO_ROOT, "output", "12_drugbank")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "output", "14_consensus")
REPORTS_PATH    = os.path.join(REPO_ROOT, "output", "10_fixed_weights", "10_reports.csv")
W_COLS    = ["w1", "w2", "w3", "w4", "w5", "w6", "w7"]
W_WEIGHTS = np.ones(len(W_COLS) + 1)  # one per w1..w7 + w8; change here to reweight

# Saturating-exponential fit for IQR shrinking factor vs number of models (fitted on 9 pathogens)
# S(M) = 1 + _TANH_A*(1-exp(-M/_TANH_TAU)),  k(M) = 2*S(M)
_TANH_A   = 1.156
_TANH_TAU = 6.47


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
    # w_quality  : (n_models, 7)          — model-level quality weights (w1–w7) from 10_fixed_weights/10_reports
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


def _tanh_transform(x: np.ndarray, k: float) -> np.ndarray:
    return 0.5 + 0.5 * np.tanh(k * (x - 0.5)) / np.tanh(k / 2)


def _apply_transform(df: pd.DataFrame, k: float) -> pd.DataFrame:
    out = df.copy()
    score_cols = [c for c in df.columns if c != "smiles"]
    out[score_cols] = _tanh_transform(df[score_cols].values, k).round(4)
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

    w_cons_iqr = float(out["consensus_score"].quantile(0.75) - out["consensus_score"].quantile(0.25))
    out_t      = _apply_transform(out, k)
    t_path     = out_path.replace(".csv", "_transformed.csv")
    out_t.to_csv(t_path, index=False)
    w_iqr      = float(out_t["consensus_score"].quantile(0.75) - out_t["consensus_score"].quantile(0.25))
    print(f"  [{pathogen}] weighted transform:   k={k:.3f}  "
          f"target_IQR={avg_model_iqr:.4f}  consensus_IQR={w_cons_iqr:.4f}  achieved_IQR={w_iqr:.4f}  -> {t_path}")

    uw_cons_iqr = float(uw_df["consensus_score"].quantile(0.75) - uw_df["consensus_score"].quantile(0.25))
    uw_t        = _apply_transform(uw_df, k)
    ut_path     = unweighted_path.replace(".csv", "_transformed.csv")
    uw_t.to_csv(ut_path, index=False)
    uw_iqr      = float(uw_t["consensus_score"].quantile(0.75) - uw_t["consensus_score"].quantile(0.25))
    print(f"  [{pathogen}] unweighted transform: k={k:.3f}  "
          f"target_IQR={avg_model_iqr:.4f}  consensus_IQR={uw_cons_iqr:.4f}  achieved_IQR={uw_iqr:.4f}  -> {ut_path}")


def plot_transform_scenarios(reports_df: pd.DataFrame, in_dir: str, out_dir: str) -> None:
    """Generate transform_scenarios.png: left=tanh curves for M=2,5,10,20,100;
    right=M→k fitted curve with real empirical (M, k) points per pathogen."""
    import matplotlib.pyplot as plt  # local import — not required for scoring

    x      = np.linspace(0, 1, 500)
    M_show = [2, 5, 10, 20, 100]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(M_show)))

    # Collect real (M, k_real) per pathogen from output files
    empirical = {}
    for pathogen in dict.fromkeys(reports_df["pathogen"]):
        src  = os.path.join(in_dir, f"{pathogen}.csv")
        cons = os.path.join(out_dir, f"{pathogen}.csv")
        if not os.path.isfile(src) or not os.path.isfile(cons):
            continue
        df_src  = pd.read_csv(src)
        df_cons = pd.read_csv(cons)
        model_names = reports_df[reports_df["pathogen"] == pathogen]["model_name"].values
        model_cols  = [c for c in df_src.columns if c != "smiles" and c in model_names]
        if len(model_cols) < 2:
            continue
        avg_iqr  = float(np.mean([df_src[m].quantile(0.75) - df_src[m].quantile(0.25) for m in model_cols]))
        cons_iqr = float(df_cons["consensus_score"].quantile(0.75) - df_cons["consensus_score"].quantile(0.25))
        if cons_iqr > 0:
            empirical[pathogen] = (len(model_cols), 2.0 * avg_iqr / cons_iqr)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: transformation curves
    ax = axes[0]
    for M, c in zip(M_show, colors):
        k = _k_from_n_models(M)
        ax.plot(x, _tanh_transform(x, k), color=c, lw=2, label=f"M={M}  (k={k:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="identity")
    ax.set_xlabel("Consensus score (input)", fontsize=11)
    ax.set_ylabel("Transformed score (output)", fontsize=11)
    ax.set_title("Tanh IQR-restoring transformation", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    formula = r"$f(x) = 0.5 + \frac{0.5 \cdot \tanh(k(x-0.5))}{\tanh(k/2)}$"
    ax.text(0.97, 0.05, formula, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.grid(True, alpha=0.3)

    # Right: M → k fitted curve + real empirical points
    ax2 = axes[1]
    M_cont = np.linspace(2, 100, 500)
    ax2.plot(M_cont, [_k_from_n_models(m) for m in M_cont], "steelblue", lw=2.5)
    for pname, (M, k_real) in empirical.items():
        ax2.scatter(M, k_real, color="black", s=45, zorder=5)
        ax2.annotate(pname, (M, k_real), textcoords="offset points",
                     xytext=(5, 3), fontsize=7.5, color="black")
    ax2.set_xlabel("Number of models M", fontsize=11)
    ax2.set_ylabel("k (tanh steepness)", fontsize=11)
    ax2.set_title("M → k mapping", fontsize=12)
    formula2 = r"$k(M) = 2\left(1 + 1.156\left(1-e^{-M/6.47}\right)\right)$"
    ax2.text(0.97, 0.07, formula2, transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(2, 100)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "transform_scenarios.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] transform_scenarios -> {out_path}")


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

    if not args.pathogen:
        plot_transform_scenarios(reports_df, args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
