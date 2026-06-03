"""
Step 12b - Fit the per-pathogen tanh steepness k* and the meta-curve k(M).

For each pathogen, reconstructs the untransformed weighted consensus from
output/12_drugbank/{pathogen}.csv and the per-model weights in
output/10_reports/10_reports.csv, then numerically solves for the steepness
k* of the tanh transform such that the IQR of the transformed consensus
equals the average per-model IQR.

Then fits the saturating-exponential meta-curve
    k(M) = 2 * (1 + a * (1 - exp(-M/tau)))
to all (M, k*) pairs, and plots:
    (i)  the tanh transform at multiple M values
    (ii) the fitted k(M) curve overlaid on the empirical (M, k*) points,
         each labelled with the pathogen code.

Prints a summary table and saves it together with the figure under
output/12_drugbank/ (as 12b_fit_transformation.csv, 12b_fit_transformation.png,
and 12b_tanh_fit.json).
"""

import json
import os

import numpy as np
import pandas as pd
import stylia
from adjustText import adjust_text
from scipy.optimize import brentq, curve_fit, minimize_scalar
from stylia import save_figure


root = os.path.dirname(os.path.abspath(__file__))

REPORTS_PATH = os.path.join(root, "..", "output", "10_reports", "10_reports.csv")
IN_DIR       = os.path.join(root, "..", "output", "12_drugbank")
OUT_DIR      = os.path.join(root, "..", "output", "12_drugbank")
OUT_PATH     = os.path.join(OUT_DIR, "12b_fit_transformation.csv")
FIG_PATH     = os.path.join(OUT_DIR, "12b_fit_transformation.png")
PARAMS_PATH  = os.path.join(OUT_DIR, "12b_tanh_fit.json")
os.makedirs(OUT_DIR, exist_ok=True)

W_COLS    = ["w1", "w2", "w3", "w4", "w5", "w6", "w7"]
W_WEIGHTS = np.ones(len(W_COLS) + 1)


def _tanh_transform(x, k):
    return 0.5 + 0.5 * np.tanh(k * (x - 0.5)) / np.tanh(k / 2)


def _compute_w8(prob_ranks, cutoffs):
    c = np.clip(cutoffs[np.newaxis, :], 0.0, 1.0 - 1e-9)
    return np.where(prob_ranks <= c, 0.0, (prob_ranks - c) / (1.0 - c))


def _score(prob_ranks, w_quality, cutoffs):
    """Untransformed weighted consensus (matches scripts/14_consensus_scoring.py)."""
    w8 = _compute_w8(prob_ranks, cutoffs)
    n_compounds, n_models = prob_ranks.shape
    w_all = np.empty((n_compounds, n_models, len(W_WEIGHTS)))
    w_all[:, :, :len(W_COLS)] = w_quality
    w_all[:, :,  len(W_COLS)] = w8
    w = np.average(w_all, axis=-1, weights=W_WEIGHTS)
    return (prob_ranks * w).sum(axis=1) / w.sum(axis=1)


def _iqr(a):
    a = np.asarray(a, dtype=float)
    return float(np.quantile(a, 0.75) - np.quantile(a, 0.25))


def _solve_k_star(consensus, target_iqr, kmax=80.0):
    """k > 0 such that IQR(_tanh_transform(consensus, k)) == target_iqr.

    IQR(k) is hump-shaped when [Q1, Q3] of `consensus` does not bracket 0.5:
    it grows from the untransformed IQR up to a finite peak, then collapses
    to 0 as k -> inf. Locate the peak and search the ascending side; return
    NaN if the peak itself can't reach target.
    """
    if target_iqr <= _iqr(consensus):
        return 0.0
    iqr_at = lambda k: _iqr(_tanh_transform(consensus, k))
    res = minimize_scalar(lambda k: -iqr_at(k), bounds=(1e-6, kmax), method="bounded")
    k_peak, iqr_peak = float(res.x), -float(res.fun)
    if iqr_peak < target_iqr:
        return float("nan")
    return float(brentq(lambda k: iqr_at(k) - target_iqr, 1e-6, k_peak, xtol=1e-4))


def _k_model(M, a, tau):
    """Saturating-exponential mapping from number of models to tanh steepness."""
    return 2.0 * (1.0 + a * (1.0 - np.exp(-M / tau)))


def _fit_k_model(Ms, ks):
    """Fit (a, tau) of k(M) to the (M, k*) points. Initial guess is data-driven."""
    Ms = np.asarray(Ms, dtype=float)
    ks = np.asarray(ks, dtype=float)
    # Data-driven init: 'a' from the saturation level (k_inf ~ 2*(1+a) ~ ks.max()),
    # 'tau' from the median M (rough scale at which the curve approaches saturation).
    a0   = max(0.1, ks.max() / 2.0 - 1.0)
    tau0 = max(1.0, float(np.median(Ms)))
    (a_hat, tau_hat), _ = curve_fit(
        _k_model, Ms, ks, p0=[a0, tau0],
        bounds=([0.0, 0.1], [10.0, 100.0]), maxfev=20000,
    )
    return float(a_hat), float(tau_hat)


def _plot(points, a_hat, tau_hat):
    stylia.set_format("slide")
    stylia.set_style("article")

    fig, axs = stylia.create_figure(1, 2)
    # Square subplots: total figure ~ 2*h wide by h tall.
    h = 5.0
    fig.set_size_inches(2.5 * h, h)

    # (i) tanh transform at multiple M values
    ax = axs.next()
    x = np.linspace(0.0, 1.0, 200)
    M_show = [2, 5, 10, 20, 100]
    cmap = stylia.ContinuousColorMap()
    cmap.fit(np.array(M_show))
    line_colors = cmap.transform(np.array(M_show))
    for M, c in zip(M_show, line_colors):
        k = _k_model(M, a_hat, tau_hat)
        ax.plot(x, _tanh_transform(x, k), color=c, lw=2, label=f"M={M}  (k={k:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="identity")
    ax.set_xlabel("Untransformed consensus score")
    ax.set_ylabel("Transformed consensus score")
    ax.set_title("Tanh IQR-restoring transformation")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.annotate(
        r"$f(x) = 0.5 + \dfrac{0.5 \cdot \tanh(k(x-0.5))}{\tanh(k/2)}$",
        xy=(0.98, 0.04), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6"),
    )

    # (ii) fitted k(M) curve over empirical k* points, labelled by pathogen
    ax = axs.next()
    Ms = points["M"].to_numpy(float)
    ks = points["k_star"].to_numpy(float)
    M_cont = np.linspace(max(2, Ms.min()), max(100.0, Ms.max()), 400)
    ax.plot(M_cont, _k_model(M_cont, a_hat, tau_hat), lw=2.5, label="k(M) fit")
    ax.scatter(Ms, ks, color="black", s=40, zorder=5, label=r"empirical $k^*$")
    ax.set_xlabel("Number of models M")
    ax.set_ylabel("k (tanh steepness)")
    ax.set_title(r"M $\rightarrow$ k mapping")
    ax.legend(loc="lower right")

    # Formula box, centered on the right edge; placed before label repulsion so
    # adjust_text treats it as an obstacle and keeps pathogen labels clear of it.
    formula = ax.annotate(
        r"$k(M) = 2\,(1 + a\,(1 - e^{-M/\tau}))$" + "\n" +
        rf"$a = {a_hat:.4f},\ \tau = {tau_hat:.4f}$",
        xy=(0.98, 0.5), xycoords="axes fraction", ha="right", va="center",
        fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6"),
    )

    texts = [ax.text(row["M"], row["k_star"], row["pathogen"], fontsize=8)
             for _, row in points.iterrows()]
    adjust_text(
        texts, x=Ms, y=ks, ax=ax, objects=[formula],
        arrowprops=dict(arrowstyle="-", color="0.5", lw=0.5),
        expand=(1.3, 1.6), force_text=(0.4, 0.6),
    )

    save_figure(FIG_PATH)


def main():
    reports = pd.read_csv(REPORTS_PATH)
    rows = []
    for pathogen in dict.fromkeys(reports["pathogen"]):
        f12 = os.path.join(IN_DIR, f"{pathogen}.csv")
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
        prob_ranks    = df12[model_cols].values
        w_quality     = np.array([pre.loc[m, W_COLS].values for m in model_cols], dtype=float)
        cutoffs       = np.array([pre.loc[m, "decision_cutoff_rank"] for m in model_cols], dtype=float)
        consensus     = _score(prob_ranks, w_quality, cutoffs)
        avg_model_iqr = float(np.mean([_iqr(prob_ranks[:, j]) for j in range(len(model_cols))]))
        rows.append({
            "pathogen": pathogen,
            "M": len(model_cols),
            "avg_model_iqr": round(avg_model_iqr, 4),
            "consensus_iqr": round(_iqr(consensus), 4),
            "k_star": round(_solve_k_star(consensus, avg_model_iqr), 4),
        })

    points = pd.DataFrame(rows)

    # Fit the meta-curve k(M) to all valid (M, k*) pairs.
    fit_mask = points["k_star"].notna() & (points["k_star"] > 0)
    a_hat, tau_hat = _fit_k_model(points.loc[fit_mask, "M"], points.loc[fit_mask, "k_star"])
    points["k_fitted"] = np.round(_k_model(points["M"].to_numpy(float), a_hat, tau_hat), 4)
    rmse = float(np.sqrt(np.mean(
        (points.loc[fit_mask, "k_star"] - _k_model(points.loc[fit_mask, "M"].to_numpy(float),
                                                   a_hat, tau_hat)) ** 2
    )))

    print(points.to_string(index=False))
    print(f"Total number of models considered: {points['M'].sum()}")
    print(f"\nFitted k(M) = 2*(1 + a*(1 - exp(-M/tau)))")
    print(f"  a   = {a_hat:.4f}")
    print(f"  tau = {tau_hat:.4f}")
    print(f"  RMSE(k_fitted vs k_star) = {rmse:.4f}  (over {int(fit_mask.sum())} points)")

    points.to_csv(OUT_PATH, index=False)
    with open(PARAMS_PATH, "w") as fh:
        json.dump({"a": a_hat, "tau": tau_hat, "rmse": rmse,
                   "n_points": int(fit_mask.sum())}, fh, indent=2)
    _plot(points, a_hat, tau_hat)
    print(f"\nSaved table:  {OUT_PATH}")
    print(f"Saved params: {PARAMS_PATH}")
    print(f"Saved figure: {FIG_PATH}")


if __name__ == "__main__":
    main()
