"""
Step 12b - Solve the per-pathogen tanh steepness k* (full + leave-one-out) and
summarize inter-model correlation.

For each pathogen, reconstructs the untransformed weighted consensus from
output/12_drugbank/rank/{pathogen}.csv and the per-model weights in
output/10_reports/10_reports.csv. Before any k* solving, computes the pairwise
Pearson correlation between the pathogen's models (folded in from the former
scripts/20_model_correlation_distributions.py) as a diagnostic for how much
averaging actually reduces variance — high correlation means little benefit
from ensembling. It then numerically solves the steepness k* of the tanh
transform such that the IQR of the transformed consensus equals the average
per-model IQR — once for the full M-model consensus, and once per
leave-one-out exclusion (M-1 models), matching the LOO columns step 14
produces.

There is no meta-curve / global fit across pathogens: each pathogen (and each
of its LOO exclusions) keeps its own directly-solved k*, used as-is downstream
by scripts/14_consensus_scoring.py and scripts/18_update_ersilia_model.py.

Outputs under output/12_drugbank/:
  12b_fit_transformation.csv   — one row per pathogen: M, avg_model_iqr, consensus_iqr,
                                  k_star, k_star_exact, mean/median_pairwise_corr
  12b_k_star_loo.csv           — one row per (pathogen, excluded_model): avg_model_iqr_loo,
                                  consensus_iqr_loo, k_star_loo, k_star_loo_exact
  12b_k_star.json              — {pathogen: {k_star, k_star_exact, M,
                                  k_star_loo: {model: k}, k_star_loo_exact: {model: bool}}}
  12b_fit_transformation.png   — (i) tanh curve at each pathogen's own k*,
                                  (ii) k* vs M, pathogen-labeled (no fitted line)
  12b_pairwise_correlation_distributions.png — KDE of pairwise correlations, one
                                  curve per pathogen (formerly script 20's output)

When the target IQR is unreachable (see `_solve_k_star`'s docstring), the peak-achievable
k is used instead of failing — it is the single closest approximation possible — and the
corresponding `*_exact` flag is set to False so this is never silently mistaken for an
exact IQR match. Every pathogen (and every one of its LOO exclusions) therefore always
gets a usable k; nothing is omitted or left null.
"""

import json
import os

import numpy as np
import pandas as pd
import stylia
from adjustText import adjust_text
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import gaussian_kde
from stylia import ArticleColors, save_figure


root = os.path.dirname(os.path.abspath(__file__))

REPORTS_PATH  = os.path.join(root, "..", "output", "10_reports", "10_reports.csv")
IN_DIR        = os.path.join(root, "..", "output", "12_drugbank")
RANK_DIR      = os.path.join(IN_DIR, "rank")   # per-model rank (0-1) predictions from 12a
OUT_DIR       = os.path.join(root, "..", "output", "12_drugbank")
OUT_PATH      = os.path.join(OUT_DIR, "12b_fit_transformation.csv")
LOO_PATH      = os.path.join(OUT_DIR, "12b_k_star_loo.csv")
FIG_PATH      = os.path.join(OUT_DIR, "12b_fit_transformation.png")
CORR_FIG_PATH = os.path.join(OUT_DIR, "12b_pairwise_correlation_distributions.png")
PARAMS_PATH   = os.path.join(OUT_DIR, "12b_k_star.json")
os.makedirs(OUT_DIR, exist_ok=True)

W_COLS    = ["w1", "w2", "w3", "w4", "w5", "w6"]   # quality weights from 10_reports
W_WEIGHTS = np.ones(len(W_COLS) + 1)               # + w7 (per-compound cutoff ramp)

CORR_GRID = np.linspace(-1.0, 1.0, 400)


def _tanh_transform(x, k):
    return 0.5 + 0.5 * np.tanh(k * (x - 0.5)) / np.tanh(k / 2)


def _compute_w7(prob_ranks, cutoffs):
    """w7 — per-compound cutoff ramp: 0 at/below decision_cutoff_rank, linear to 1 above it."""
    c = np.clip(cutoffs[np.newaxis, :], 0.0, 1.0 - 1e-9)
    return np.where(prob_ranks <= c, 0.0, (prob_ranks - c) / (1.0 - c))


def _score(prob_ranks, w_quality, cutoffs):
    """Untransformed weighted consensus (matches scripts/14_consensus_scoring.py)."""
    w7 = _compute_w7(prob_ranks, cutoffs)
    n_compounds, n_models = prob_ranks.shape
    w_all = np.empty((n_compounds, n_models, len(W_WEIGHTS)))
    w_all[:, :, :len(W_COLS)] = w_quality
    w_all[:, :,  len(W_COLS)] = w7
    w = np.average(w_all, axis=-1, weights=W_WEIGHTS)
    denom = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = (prob_ranks * w).sum(axis=1) / denom
    zero = denom == 0.0          # all weights 0 for a compound -> fall back to plain mean
    if zero.any():
        score[zero] = prob_ranks[zero].mean(axis=1)
    return score


def _iqr(a):
    a = np.asarray(a, dtype=float)
    return float(np.quantile(a, 0.75) - np.quantile(a, 0.25))


def _solve_k_star(consensus, target_iqr, kmax=80.0):
    """k > 0 such that IQR(_tanh_transform(consensus, k)) == target_iqr.

    IQR(k) is hump-shaped when [Q1, Q3] of `consensus` does not bracket 0.5:
    it grows from the untransformed IQR up to a finite peak, then collapses
    to 0 as k -> inf. Locate the peak and search the ascending side. If the
    peak itself can't reach target, return the peak k instead of failing:
    it is provably the single closest achievable point (IQR(k) rises
    monotonically to the peak, then falls back toward 0, so nothing else on
    the curve gets nearer to target_iqr).

    Returns (k, is_exact) — is_exact is False only in that peak-fallback case.
    """
    if target_iqr <= _iqr(consensus):
        return 0.0, True
    iqr_at = lambda k: _iqr(_tanh_transform(consensus, k))
    res = minimize_scalar(lambda k: -iqr_at(k), bounds=(1e-6, kmax), method="bounded")
    k_peak, iqr_peak = float(res.x), -float(res.fun)
    if iqr_peak < target_iqr:
        return k_peak, False
    return float(brentq(lambda k: iqr_at(k) - target_iqr, 1e-6, k_peak, xtol=1e-4)), True


def _pairwise_correlations(df, model_cols):
    """Upper-triangle Pearson correlations between every pair of a pathogen's models."""
    corr = df[model_cols].corr().values
    iu = np.triu_indices_from(corr, k=1)
    return corr[iu]


def _plot_tanh_and_k_star(points):
    stylia.set_format("slide")
    stylia.set_style("article")

    fig, axs = stylia.create_figure(1, 2)
    # Square subplots: total figure ~ 2*h wide by h tall.
    h = 5.0
    fig.set_size_inches(2.5 * h, h)

    # (i) tanh transform at the min-k* and max-k* pathogens only, + identity
    ax = axs.next()
    x = np.linspace(0.0, 1.0, 200)
    nc = ArticleColors()
    min_row = points.loc[points["k_star"].idxmin()]
    max_row = points.loc[points["k_star"].idxmax()]
    for row, c in ((min_row, nc.cobalt), (max_row, nc.crimson)):
        ax.plot(x, _tanh_transform(x, row["k_star"]), color=c, lw=1.5,
                label=f"{row['pathogen']} (M={int(row['M'])}, k*={row['k_star']:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="identity")
    ax.set_xlabel("Untransformed consensus score")
    ax.set_ylabel("Transformed consensus score")
    ax.set_title("Tanh IQR-restoring transformation (min/max k* pathogens)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=6, ncol=2)
    ax.annotate(
        r"$f(x) = 0.5 + \dfrac{0.5 \cdot \tanh(k^*(x-0.5))}{\tanh(k^*/2)}$",
        xy=(0.98, 0.04), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6"),
    )

    # (ii) empirical k* vs M, labelled by pathogen — no fitted curve
    ax = axs.next()
    Ms = points["M"].to_numpy(float)
    ks = points["k_star"].to_numpy(float)
    ax.scatter(Ms, ks, color="black", s=40, zorder=5)
    ax.set_xlabel("Number of models M")
    ax.set_ylabel("k* (tanh steepness)")
    ax.set_title("Empirical k* per pathogen")
    ax.set_xlim(0, 60)
    ax.set_ylim(1, 6)

    texts = [ax.text(row["M"], row["k_star"], row["pathogen"], fontsize=8)
             for _, row in points.iterrows()]
    adjust_text(
        texts, x=Ms, y=ks, ax=ax,
        arrowprops=dict(arrowstyle="-", color="0.5", lw=0.5),
        expand=(1.3, 1.6), force_text=(0.4, 0.6),
    )

    save_figure(FIG_PATH)


def _plot_correlation_distributions(corr_by_pathogen):
    stylia.set_format("slide")
    stylia.set_style("ersilia")

    fig, axs = stylia.create_figure(1, 1)
    ax = axs.next()
    cm = stylia.CyclicColormap("ersilia")
    indices = list(range(len(corr_by_pathogen)))
    cm.fit(indices)
    colors = cm.transform(indices)
    for (pathogen, values), color in zip(corr_by_pathogen.items(), colors):
        kde = gaussian_kde(values)
        ax.plot(CORR_GRID, kde(CORR_GRID), color=color, label=f"{pathogen} (n={len(values)})")
    ax.axvline(0.0, color="0.6", lw=1, linestyle="--")
    ax.set_xlim(-1.0, 1.0)
    stylia.label(
        ax,
        xlabel="Pairwise Pearson correlation (DrugBank rank predictions)",
        ylabel="Density",
        title="Distribution of pairwise model correlations, per pathogen",
    )
    ax.legend(fontsize=7, ncol=2)
    save_figure(CORR_FIG_PATH)


def main():
    reports = pd.read_csv(REPORTS_PATH)
    rows = []
    loo_rows = []
    corr_by_pathogen = {}
    k_star_json = {}

    for pathogen in dict.fromkeys(reports["pathogen"]):
        f12 = os.path.join(RANK_DIR, f"{pathogen}.csv")
        if not os.path.isfile(f12):
            continue
        df12 = pd.read_csv(f12)
        pre = reports[reports["pathogen"] == pathogen].set_index("model_name")
        model_cols = [c for c in df12.columns if c != "smiles" and c in pre.index]
        if len(model_cols) < 2:
            continue
        nan_counts = df12[model_cols].isna().sum()
        if nan_counts.any():
            bad = nan_counts[nan_counts > 0].to_dict()
            raise ValueError(
                f"[{pathogen}] NaN predictions in step-12 output: {bad}. "
                "Decide how to handle these (drop / impute / exclude pairwise) before scoring."
            )
        vmin, vmax = float(df12[model_cols].values.min()), float(df12[model_cols].values.max())
        if vmin < -1e-6 or vmax > 1.0 + 1e-6:
            raise ValueError(
                f"[{pathogen}] predictions are not on the [0,1] rank scale "
                f"(min={vmin:.3f}, max={vmax:.3f}). 12b requires the 'rank' predict type "
                f"({RANK_DIR})."
            )

        # --- correlation diagnostics, computed before any weighting/k* solving ---
        corr_values = _pairwise_correlations(df12, model_cols)
        corr_by_pathogen[pathogen] = corr_values
        mean_pairwise_corr   = float(np.mean(corr_values))
        median_pairwise_corr = float(np.median(corr_values))

        prob_ranks = df12[model_cols].values
        w_quality  = np.array([pre.loc[m, W_COLS].values for m in model_cols], dtype=float)
        cutoffs    = np.array([pre.loc[m, "decision_cutoff_rank"] for m in model_cols], dtype=float)
        M          = len(model_cols)

        # --- full-model k* ---
        consensus            = _score(prob_ranks, w_quality, cutoffs)
        avg_model_iqr         = float(np.mean([_iqr(prob_ranks[:, j]) for j in range(M)]))
        k_star, k_star_exact = _solve_k_star(consensus, avg_model_iqr)
        rows.append({
            "pathogen": pathogen,
            "M": M,
            "avg_model_iqr": round(avg_model_iqr, 4),
            "consensus_iqr": round(_iqr(consensus), 4),
            "k_star": round(k_star, 4),
            "k_star_exact": k_star_exact,
            "mean_pairwise_corr": round(mean_pairwise_corr, 4),
            "median_pairwise_corr": round(median_pairwise_corr, 4),
        })
        if not k_star_exact:
            print(f"[INFO] {pathogen}: target IQR unreachable for the full-model consensus — "
                  f"using k_star={k_star:.4f} (peak of the IQR(k) curve, the closest achievable value).")

        # --- leave-one-out k*, one per excluded model, targeting the IQR of the
        # remaining M-1 models (not the full M-model average) ---
        k_star_loo       = {}
        k_star_loo_exact = {}
        for i, excluded_model in enumerate(model_cols):
            other                         = [j for j in range(M) if j != i]
            loo_consensus                 = _score(prob_ranks[:, other], w_quality[other, :], cutoffs[other])
            loo_avg_iqr                   = float(np.mean([_iqr(prob_ranks[:, j]) for j in other]))
            loo_k_star, loo_k_star_exact  = _solve_k_star(loo_consensus, loo_avg_iqr)
            k_star_loo[excluded_model]       = round(loo_k_star, 4)
            k_star_loo_exact[excluded_model] = loo_k_star_exact
            loo_rows.append({
                "pathogen": pathogen,
                "excluded_model": excluded_model,
                "avg_model_iqr_loo": round(loo_avg_iqr, 4),
                "consensus_iqr_loo": round(_iqr(loo_consensus), 4),
                "k_star_loo": round(loo_k_star, 4),
                "k_star_loo_exact": loo_k_star_exact,
            })

        approx_loo = [m for m, exact in k_star_loo_exact.items() if not exact]
        if approx_loo:
            print(f"[INFO] {pathogen}: target IQR unreachable when excluding {approx_loo} — "
                  "using each exclusion's peak-achievable k_star_loo instead of an exact match.")

        k_star_json[pathogen] = {
            "k_star": round(k_star, 4),
            "k_star_exact": k_star_exact,
            "M": M,
            "k_star_loo": k_star_loo,
            "k_star_loo_exact": k_star_loo_exact,
        }

    points = pd.DataFrame(rows)
    if points.empty:
        print("[WARN] no pathogens with >=2 matching models found — nothing to compute. "
              "Check that 12a wrote rank predictions with matching model_name columns.")
        return

    print(points.to_string(index=False))
    print(f"Total number of models considered: {points['M'].sum()}")

    points.to_csv(OUT_PATH, index=False)
    pd.DataFrame(loo_rows).to_csv(LOO_PATH, index=False)
    with open(PARAMS_PATH, "w") as fh:
        json.dump(k_star_json, fh, indent=2)

    _plot_tanh_and_k_star(points)
    _plot_correlation_distributions(corr_by_pathogen)

    print(f"\nSaved table:       {OUT_PATH}")
    print(f"Saved LOO table:   {LOO_PATH}")
    print(f"Saved k* map:      {PARAMS_PATH}  ({len(k_star_json)}/{len(points)} pathogens)")
    print(f"Saved figure:      {FIG_PATH}")
    print(f"Saved corr figure: {CORR_FIG_PATH}")


if __name__ == "__main__":
    main()
