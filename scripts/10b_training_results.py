"""
Step 10b — Per-pathogen training-result figures.

For each pathogen, renders a 3-panel figure (one panel stacked per dataset along x):
  (a) AUROC bars with cross-fold std error bars
  (b) Final aggregate weight bars
  (c) Out-of-fold rank-score distributions (boxplot + jittered scatter) for
      actives vs inactives, with the decision_cutoff_rank overlaid as a dashed line.

Inputs:
  - output/10_reports/10_reports.csv
  - output/09_reports/{pathogen}/{model_name}_folds.json  (one per row in 10_reports)

Output:
  - output/10_reports/plots/10_training_{pathogen}.png

Usage:
    python scripts/10b_training_results.py                # all pathogens in 10_reports.csv
    python scripts/10b_training_results.py --pathogen saureus
"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import stylia
from stylia import CategoricalPalette, save_figure


root = os.path.dirname(os.path.abspath(__file__))

REPORT_PATH   = os.path.join(root, "..", "output", "10_reports", "10_reports.csv")
FOLDS_DIR     = os.path.join(root, "..", "output", "09_reports")
PATHOGENS     = os.path.join(root, "..", "config", "pathogens.csv")
OUT_DIR       = os.path.join(root, "..", "output", "10_reports", "plots")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 42


def _load_oof_rank(pathogen: str, model_name: str):
    """Concatenate y_rank across folds; return (rank_actives, rank_inactives)."""
    path = os.path.join(FOLDS_DIR, pathogen, f"{model_name}_folds.json")
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        folds = json.load(f)
    y_true, y_rank = [], []
    for fd in folds.values():
        y_true.extend(fd["y_true"])
        y_rank.extend(fd["y_rank"])
    y_true = np.asarray(y_true)
    y_rank = np.asarray(y_rank, dtype=float)
    actives   = y_rank[y_true == 1]
    inactives = y_rank[y_true == 0]
    return actives, inactives


def plot_pathogen(report_pathogen: pd.DataFrame, pathogen: str, pathogen_name: str,
                  pal: CategoricalPalette, rng: np.random.Generator) -> str:
    report_pathogen = report_pathogen.reset_index(drop=True)
    n = len(report_pathogen)
    x = list(range(n))

    fig, axs = stylia.create_figure(3, 1, width=0.6, height=0.5)
    fig.suptitle(f"{pathogen_name} ({n} datasets)", fontsize=12)

    # (a) AUROC
    ax = axs.next()
    ax.set_xlabel("")
    ax.set_xticks(x)
    ax.set_xticklabels([""] * n)
    ax.set_ylabel("AUROC")
    ax.set_ylim([0.85, 1.01])
    ax.set_xlim([-0.7, n - 0.3])
    aurocs_mean = report_pathogen["auroc_mean"].tolist()
    aurocs_std  = report_pathogen["auroc_std"].tolist()
    has_decoys  = (report_pathogen["n_decoys"] > 0).tolist()
    auroc_fill  = pal.get(2)[1]
    for i in range(n):
        face = "white" if has_decoys[i] else auroc_fill
        ax.bar(i, aurocs_mean[i], color=face, ec="k", lw=0.7)
        ax.plot([i, i],
                [aurocs_mean[i] - aurocs_std[i], aurocs_mean[i] + aurocs_std[i]],
                color="k", lw=0.7)

    # (b) OOF rank scores: actives vs inactives
    ax = axs.next()
    ax.set_xlabel("")
    ax.set_xticks(x)
    ax.set_xticklabels([""] * n)
    ax.set_ylabel("OOF predict\nrank scores")
    ax.set_ylim([0, 1])
    ax.set_xlim([-0.7, n - 0.3])
    w = 0.15
    cutoffs = report_pathogen["decision_cutoff_rank"].tolist()
    for i, row in report_pathogen.iterrows():
        actives, inactives = _load_oof_rank(pathogen, row["model_name"])
        if actives is None or inactives is None or not len(actives) or not len(inactives):
            continue

        x_actives   = i + w + rng.uniform(-w, w, size=len(actives))
        x_inactives = i - w + rng.uniform(-w, w, size=len(inactives))
        ax.scatter(x_actives,   actives,   color=pal.get(8)[3], s=1, alpha=0.4, lw=0)
        ax.scatter(x_inactives, inactives, color=pal.get(8)[7], s=1, alpha=0.4, lw=0)

        cutoff = cutoffs[i]
        if cutoff is not None and not (isinstance(cutoff, float) and np.isnan(cutoff)):
            ax.plot([i - 2 * w, i + 2 * w], [cutoff, cutoff],
                    lw=0.4, c="k", linestyle="dotted")

        def _stats(arr):
            return dict(
                med=np.median(arr),
                q1=np.percentile(arr, 25),
                q3=np.percentile(arr, 75),
                whislo=np.percentile(arr, 5),
                whishi=np.percentile(arr, 95),
                fliers=[],
                min=np.min(arr),
                max=np.max(arr),
            )

        bp = ax.bxp([_stats(inactives), _stats(actives)],
                    positions=[i - w, i + w], widths=w * 2,
                    patch_artist=True, showfliers=False)
        for box in bp["boxes"]:
            box.set_linewidth(0.4)
            box.set_facecolor("none")
        for element in ["whiskers", "caps", "medians"]:
            for line in bp[element]:
                line.set_color("k")
                line.set_linewidth(0.4 if element != "caps" else 0)

    ax.set_xticks(x)
    ax.set_xticklabels([""] * n)

    # (c) Final weights
    ax = axs.next()
    ax.set_xlabel("")
    ax.set_ylabel("Weight\nvalue")
    final_weights = report_pathogen["final_weight"].tolist()
    w_min, w_max = min(final_weights), max(final_weights)
    y_lo = max(0.0, math.floor(w_min * 10) / 10)
    y_hi = min(1.0, math.ceil(w_max * 10) / 10)
    ax.set_ylim([y_lo, y_hi])
    ax.set_xlim([-0.7, n - 0.3])
    weight_fill = pal.get(8)[4]
    for i in range(n):
        face = "white" if has_decoys[i] else weight_fill
        ax.bar(i, final_weights[i], color=face, ec="k", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Model #{i + 1}" for i in x], rotation=90)

    out_path = os.path.join(OUT_DIR, f"10_training_{pathogen}.png")
    save_figure(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--pathogen", default=None,
                        help="Pathogen code to plot (e.g. 'saureus'). "
                             "If omitted, all pathogens in 10_reports.csv are processed.")
    args = parser.parse_args()

    report = pd.read_csv(REPORT_PATH)
    pathogens = pd.read_csv(PATHOGENS)
    code_to_name = dict(zip(pathogens["code"], pathogens["pathogen"]))

    if args.pathogen is not None:
        codes = [args.pathogen]
    else:
        codes = sorted(report["pathogen"].unique().tolist())

    stylia.set_format("slide")
    stylia.set_style("article")
    pal = CategoricalPalette("npg")
    rng = np.random.default_rng(RANDOM_SEED)

    for code in codes:
        sub = report[report["pathogen"] == code]
        if sub.empty:
            print(f"[skip] {code}: no rows in 10_reports.csv")
            continue
        name = code_to_name.get(code, code)
        print(f"Plotting {name} ({code}): {len(sub)} datasets")
        out = plot_pathogen(sub, code, name, pal, rng)
        print(f"  saved: {out}")


if __name__ == "__main__":
    main()
