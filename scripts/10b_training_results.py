"""
Step 10b — Per-pathogen training-result figures.

For each pathogen, renders a 4-panel figure (panels stacked, one bar/group per
dataset along x):
  (a) 5-fold CV AUROC bars with cross-fold std error bars, and a cross per dataset for its
      screening AUC (LazyQSAR >= 3.6: how well the model separates its actives from the
      fixed drug-like reference library, as opposed to from the dataset's own inactives).
  (b) Out-of-fold rank-score distributions (boxplot + jittered scatter) for
      actives vs inactives, with the decision_cutoff_rank overlaid as a dotted line.
  (c) Training-set composition (log scale): grouped bars for actives, original
      inactives, and added negatives (proven negatives + any decoy fallback) per dataset.
  (d) Final aggregate weight bars (mean of w1-w6 and w_screen).

In panels (a) and (d) a lighter bar marks datasets balanced with added negatives.

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
import sys

import numpy as np
import pandas as pd
import stylia
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "..", "src"))
from default import MIN_AUROC, RANDOM_SEED  # noqa: E402

REPORT_PATH   = os.path.join(root, "..", "output", "10_reports", "10_reports.csv")
FOLDS_DIR     = os.path.join(root, "..", "output", "09_reports")
PATHOGENS     = os.path.join(root, "..", "config", "pathogens.csv")
OUT_DIR       = os.path.join(root, "..", "output", "10_reports", "plots")
os.makedirs(OUT_DIR, exist_ok=True)

# Format: slide | Style: article — change with stylia.set_format() / stylia.set_style()
stylia.set_format("slide")
stylia.set_style("article")

HEADROOM = 0.13  # share of each panel's height taken by its one-row legend, right above the data
LIGHTEN = 0.6  # tint of a bar whose dataset was balanced with added negatives (a, d) and of the added-negatives bars (c)


def _legend_band(ax, handles: list, lo: float, data_hi: float, ticks: list):
    """Reserve a band just above the data (*data_hi* is where the data end) and put the legend in
    it, on one row.

    The y axis is extended so the band is HEADROOM of the panel, but *ticks* stop inside the data
    range so the band does not read as part of the scale (an AUROC axis showing 1.1 would
    suggest values that cannot exist).
    """
    ax.set_ylim([lo, data_hi + HEADROOM / (1 - HEADROOM) * (data_hi - lo)])
    ax.set_yticks(ticks)
    ax.legend(handles=handles, loc="upper center", ncol=len(handles))


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


def _box_stats(arr: np.ndarray) -> dict:
    """Median, quartiles and 5th/95th-percentile whiskers, for ax.bxp."""
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


def plot_auroc(ax, report: pd.DataFrame, has_added: list, nc, title: str):
    """(a) 5-fold CV AUROC bars (± std across folds) and a cross per dataset for its screening AUC."""
    n = len(report)
    aurocs_mean = report["auroc_mean"].tolist()
    aurocs_std  = report["auroc_std"].tolist()
    screening   = report["screening_auc"].tolist()
    for i in range(n):
        face = nc.get("turquoise", lighten=LIGHTEN) if has_added[i] else nc.turquoise
        ax.bar(i, aurocs_mean[i], color=face)
        ax.plot([i, i], [aurocs_mean[i] - aurocs_std[i], aurocs_mean[i] + aurocs_std[i]], color="k")
    ax.scatter(range(n), screening, color=nc.tangerine, marker="x", s=stylia.MARKERSIZE_BIG,
               linewidths=stylia.LINEWIDTH_THICK, zorder=3)

    # Every retained model has a mean AUROC >= MIN_AUROC, but a screening AUC can fall below it.
    # Start the axis just under the lowest bar or dot, on a 0.05 grid, so neither is clipped and
    # a bar sitting right at MIN_AUROC does not shrink to nothing.
    y_lo = max(0.0, math.floor((min(min(screening), min(aurocs_mean), MIN_AUROC) - 0.01) * 20) / 20)
    ax.set_xlim([-0.7, n - 0.3])
    ax.set_xticks(range(n))
    ax.set_xticklabels([""] * n)
    handles = [Patch(facecolor=nc.turquoise, label="5-Fold CV AUROC")]
    if any(has_added):
        handles.append(Patch(facecolor=nc.get("turquoise", lighten=LIGHTEN), label="Balanced with added negatives"))
    handles.append(Line2D([], [], marker="x", linestyle="none", markersize=8,
                          markeredgewidth=stylia.LINEWIDTH_THICK, color=nc.tangerine, label="Screening AUC"))
    _legend_band(
        ax,
        handles,
        lo=y_lo, data_hi=max(1.0, max(m + sd for m, sd in zip(aurocs_mean, aurocs_std, strict=True)), max(screening)),
        ticks=list(np.arange(y_lo, 1.0 + 1e-9, 0.05 if y_lo >= 0.5 else 0.1)),
    )
    stylia.label(ax, xlabel="", ylabel="AUROC", title=title)


def plot_oof_ranks(ax, report: pd.DataFrame, pathogen: str, nc, rng: np.random.Generator):
    """(b) Out-of-fold rank scores of actives vs inactives, with each model's decision cutoff."""
    n = len(report)
    w = 0.15
    cutoffs = report["decision_cutoff_rank"].tolist()
    for i, row in report.iterrows():
        actives, inactives = _load_oof_rank(pathogen, row["model_name"])
        if actives is None or not len(actives) or not len(inactives):
            print(f"  [warn] {pathogen}/{row['model_name']}: no out-of-fold ranks in the folds file")
            continue

        # s=4, between a pinpoint and stylia's MARKERSIZE_SMALL (8): there are thousands of points
        # per dataset, which at the default size merge into a solid block over the boxes.
        x_actives   = i + w + rng.uniform(-w, w, size=len(actives))
        x_inactives = i - w + rng.uniform(-w, w, size=len(inactives))
        ax.scatter(x_actives,   actives,   color=nc.crimson, s=4, alpha=0.4, edgecolors="none")
        ax.scatter(x_inactives, inactives, color=nc.silver,  s=4, alpha=0.4, edgecolors="none")
        ax.plot([i - 2 * w, i + 2 * w], [cutoffs[i], cutoffs[i]], color="k", linestyle="dotted")

        bp = ax.bxp([_box_stats(inactives), _box_stats(actives)],
                    positions=[i - w, i + w], widths=w * 2,
                    patch_artist=True, showfliers=False)
        for box in bp["boxes"]:
            box.set_facecolor("none")
        for element in ["whiskers", "caps", "medians"]:
            for line in bp[element]:
                line.set_color("k")
                line.set_visible(element != "caps")

    ax.set_xlim([-0.7, n - 0.3])
    ax.set_xticks(range(n))
    ax.set_xticklabels([""] * n)
    _legend_band(
        ax,
        [
            Patch(facecolor=nc.crimson, label="Actives"),
            Patch(facecolor=nc.silver, label="Inactives (incl. added)"),
            Line2D([], [], color="k", linestyle="dotted", label="Decision cutoff"),
        ],
        lo=0.0, data_hi=1.0, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    stylia.label(ax, xlabel="", ylabel="OOF predict\nrank scores")


def plot_composition(ax, report: pd.DataFrame, n_added_all: pd.Series, nc):
    """(c) Training-set composition on a log scale: actives, inactives, added negatives."""
    n = len(report)
    n_pos   = report["n_positives"].tolist()
    n_added = n_added_all.tolist()
    n_inact = (report["n_compounds"] - report["n_positives"] - n_added_all).tolist()
    bw = 0.27
    for i in range(n):
        for off, val, col in ((-bw, n_pos[i],   nc.crimson),
                              (0.0, n_inact[i], nc.cobalt),
                              (bw,  n_added[i], nc.get("cobalt", lighten=LIGHTEN))):
            if val > 0:
                ax.bar(i + off, val, width=bw, color=col)
    # Log axis, so the band is a share of the *exponent* range: from 10^0 to just above the
    # tallest bar, scaled so that the legend occupies HEADROOM of the panel.
    tallest = math.log10(max(max(n_pos), max(n_inact), max(n_added)))
    ax.set_yscale("log")
    ax.set_ylim([1, 10 ** (tallest / (1 - HEADROOM))])
    ax.set_yticks([10 ** k for k in range(math.floor(tallest) + 1)])
    ax.set_xlim([-0.7, n - 0.3])
    ax.set_xticks(range(n))
    ax.set_xticklabels([""] * n)
    handles = [Patch(facecolor=nc.crimson, label="Actives"), Patch(facecolor=nc.cobalt, label="Inactives")]
    if any(v > 0 for v in n_added):
        handles.append(Patch(facecolor=nc.get("cobalt", lighten=LIGHTEN), label="Added negatives"))
    ax.legend(handles=handles, loc="upper center", ncol=len(handles))
    stylia.label(ax, xlabel="", ylabel="Number of compounds")


def plot_weights(ax, report: pd.DataFrame, has_added: list, nc):
    """(d) Final model weight: the mean of w1-w6 and w_screen."""
    n = len(report)
    final_weights = report["final_weight"].tolist()
    for i in range(n):
        face = nc.get("orchid", lighten=LIGHTEN) if has_added[i] else nc.orchid
        ax.bar(i, final_weights[i], color=face)
    lo = 0.0  # bars start at zero, so their heights are proportional to the weights
    ticks = list(np.arange(lo, max(final_weights) + 1e-9, 0.1))
    if any(has_added):
        _legend_band(
            ax,
            [
                Patch(facecolor=nc.orchid, label="Model weight"),
                Patch(facecolor=nc.get("orchid", lighten=LIGHTEN), label="Balanced with added negatives"),
            ],
            lo=lo, data_hi=max(final_weights), ticks=ticks,
        )
    else:
        ax.set_ylim([lo, max(final_weights) + 0.05 * (max(final_weights) - lo)])
        ax.set_yticks(ticks)
    ax.set_xlim([-0.7, n - 0.3])
    ax.set_xticks(range(n))
    if n > 20:
        ax.set_xticklabels([str(i + 1) if (i + 1) % 5 == 0 else "" for i in range(n)])
    else:
        ax.set_xticklabels([str(i + 1) for i in range(n)])
    stylia.label(ax, xlabel="Model number", ylabel="Average weight (w1-w6, w_screen)")


def plot_pathogen(report_pathogen: pd.DataFrame, pathogen: str, pathogen_name: str,
                  nc, rng: np.random.Generator) -> str:
    report = report_pathogen.reset_index(drop=True)
    n = len(report)
    n_added_all = report["n_added_negatives"] + report["n_added_decoys"]
    has_added   = (n_added_all > 0).tolist()

    # Default width, but an explicit height: the default canvas is a 3:1 strip, which suits
    # panels side by side and squashes four stacked ones until their y labels collide.
    fig, axs = stylia.create_figure(4, 1, height=1.0)
    plot_auroc(axs.next(), report, has_added, nc, title=f"{pathogen_name} ({n} model{'s' if n != 1 else ''})")
    plot_oof_ranks(axs.next(), report, pathogen, nc, rng)
    plot_composition(axs.next(), report, n_added_all, nc)
    plot_weights(axs.next(), report, has_added, nc)

    out_path = os.path.join(OUT_DIR, f"10_training_{pathogen}.png")
    stylia.save_figure(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--pathogen", default=None,
                        help="Pathogen code to plot (e.g. 'saureus'). "
                             "If omitted, all pathogens in 10_reports.csv are processed.")
    args = parser.parse_args()

    report = pd.read_csv(REPORT_PATH)
    if "screening_auc" not in report.columns:
        sys.exit(f"{REPORT_PATH} has no screening_auc column: re-run scripts/10a_aggregate_reports.py")
    pathogens = pd.read_csv(PATHOGENS)
    code_to_name = dict(zip(pathogens["code"], pathogens["pathogen"], strict=True))

    if args.pathogen is not None:
        codes = [args.pathogen]
    else:
        codes = sorted(report["pathogen"].unique().tolist())

    nc = stylia.NamedColors()
    rng = np.random.default_rng(RANDOM_SEED)

    for code in codes:
        sub = report[report["pathogen"] == code]
        if sub.empty:
            print(f"[skip] {code}: no rows in 10_reports.csv")
            continue
        name = code_to_name.get(code, code)
        print(f"Plotting {name} ({code}): {len(sub)} datasets")
        out = plot_pathogen(sub, code, name, nc, rng)
        print(f"  saved: {out}")


if __name__ == "__main__":
    main()
