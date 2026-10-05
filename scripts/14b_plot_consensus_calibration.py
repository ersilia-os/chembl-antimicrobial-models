"""
Step 14b — What the calibration against the reference library does to the consensus score.

For each pathogen, renders a 2-panel figure of the full weighted consensus of step 14:
  (a) Two distributions, each drawn as jittered points with a violin and a boxplot on top: the raw
      consensus, before calibration, and the consensus rank, after calibration against the
      reference library. Each has two groups: the DrugBank compounds (yellow) and the 50,000
      molecules of the reference library (green). Both share the y axis, so the shift is visible. Over the consensus rank, a dashed line marks the decision
      rank 0.65 and the text gives the % of each group at or above it. The reference library is the
      set the calibration is built on, so about 1% of it is at or above 0.65 by design.
  (b) Scatter of the raw consensus (x) against the consensus rank (y), with the Spearman
      correlation of each group in the legend. The calibration is a monotone map, so the order of
      compounds cannot change and Spearman is 1 (up to the ties that the 6-decimal rounding of the
      step-14 files can create).

Only the step-14 files are read; nothing is recomputed. The leave-one-out and unweighted
consensus columns are not shown.

Inputs (per pathogen):
  - output/14_consensus/{pathogen}/{drugbank,reference}_{raw,rank}.csv   (consensus_raw / consensus_rank)
  - output/14_consensus/{pathogen}/anchors.json                          (number of models)
  - output/10_reports/10_reports.csv                                     (which pathogens need a consensus)
  - config/pathogens.csv                                                 (pathogen display names)

Output:
  - output/14b_consensus_calibration/14b_{pathogen}.png

Fails (exit code 1) if a pathogen with two or more retained models has no step-14 output, so a
partial run cannot pass unnoticed.

Usage:
    python scripts/14b_plot_consensus_calibration.py                   # every pathogen with a consensus
    python scripts/14b_plot_consensus_calibration.py --pathogen ecoli
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import stylia
from matplotlib.patches import Patch
from scipy.stats import spearmanr
from stylia import ArticleColors, save_figure

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "..", "src"))
from default import RANDOM_SEED  # noqa: E402

CONSENSUS_DIR  = os.path.join(root, "..", "output", "14_consensus")
REPORTS_PATH   = os.path.join(root, "..", "output", "10_reports", "10_reports.csv")
PATHOGENS_PATH = os.path.join(root, "..", "config", "pathogens.csv")
OUT_DIR        = os.path.join(root, "..", "output", "14b_consensus_calibration")
os.makedirs(OUT_DIR, exist_ok=True)

# Decision rank of LazyQSAR (top 1% of the reference library); the same value as step 14 reads
# from lazyqsar, copied here so that plotting does not need to import it.
DECISION_RANK = 0.65

# Two groups (DrugBank, reference) per boxplot slot: x offsets of their boxes and the half-width
# of a box / of the jitter around it, in x units (one slot = 1 unit).
OFFSETS = (-0.2, 0.2)
HALF_WIDTH = 0.14

# Height at which each group's "% at or above the decision rank" is written.
LABEL_Y = 0.95


def _pct_label(fraction: float) -> str:
    """'84%' from 10% up, one decimal below (so a 2.6% DrugBank hit rate is not rounded to 3%)."""
    pct = 100 * fraction
    return f"{pct:.0f}%" if pct >= 10 else f"{pct:.1f}%"


def _box_stats(arr: np.ndarray) -> dict:
    """Box at the quartiles, whiskers at the 5th and 95th percentiles (as in 10b and 12c)."""
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


def _style_boxes(bp) -> None:
    for box in bp["boxes"]:
        box.set_linewidth(0.8)
        box.set_facecolor("none")
    for element in ["whiskers", "caps", "medians"]:
        for line in bp[element]:
            line.set_color("k")
            line.set_linewidth(0.8 if element != "caps" else 0)


def load_consensus(pathogen: str) -> dict:
    """The full weighted consensus of one pathogen: {"drugbank"|"reference": (raw, rank)} arrays.

    Raises if a file is missing or the raw and rank files are not the same molecules row by row.
    """
    folder = os.path.join(CONSENSUS_DIR, pathogen)
    out = {}
    for dataset in ("drugbank", "reference"):
        raw = pd.read_csv(os.path.join(folder, f"{dataset}_raw.csv"))
        rank = pd.read_csv(os.path.join(folder, f"{dataset}_rank.csv"))
        if len(raw) != len(rank) or not raw["smiles"].equals(rank["smiles"]):
            raise ValueError(f"[{pathogen}] {dataset}_raw.csv and {dataset}_rank.csv are not the same "
                             "molecules in the same order.")
        for frame, column in ((raw, "consensus_raw"), (rank, "consensus_rank")):
            if column not in frame.columns:
                raise ValueError(f"[{pathogen}] column '{column}' is missing from the {dataset} file.")
            if frame[column].isna().any():
                raise ValueError(f"[{pathogen}] NaN in '{column}' of the {dataset} file.")
        out[dataset] = (raw["consensus_raw"].to_numpy(dtype=float), rank["consensus_rank"].to_numpy(dtype=float))
    return out


def plot_distributions(ax, data: dict, colors: tuple, rng, legend_labels: tuple, title: str) -> None:
    """Panel (a): raw consensus and consensus rank, DrugBank and reference side by side in each.

    Each group is the jittered points, a violin over them (kernel density, over the full range of
    the values) and the boxplot (quartiles, whiskers at the 5th and 95th percentiles) on top.
    """
    for slot, column in enumerate((0, 1)):                 # 0 = raw, 1 = rank
        for offset, dataset, color in zip(OFFSETS, ("drugbank", "reference"), colors):
            values = data[dataset][column]
            x = slot + offset + rng.uniform(-HALF_WIDTH, HALF_WIDTH, size=len(values))
            ax.scatter(x, values, color=color, s=1, alpha=0.4, lw=0)
            vp = ax.violinplot([values], positions=[slot + offset], widths=HALF_WIDTH * 2,
                               showextrema=False, showmedians=False)
            for body in vp["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor("k")
                body.set_linewidth(0.6)
                body.set_alpha(0.6)
            bp = ax.bxp([_box_stats(values)], positions=[slot + offset],
                        widths=HALF_WIDTH, patch_artist=True, showfliers=False)
            _style_boxes(bp)
            if column == 1:
                ax.text(slot + offset, LABEL_Y, _pct_label(np.mean(values >= DECISION_RANK)),
                        ha="center", va="center", fontsize=5,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.5))
    ax.plot([1 + OFFSETS[0] - HALF_WIDTH, 1 + OFFSETS[-1] + HALF_WIDTH], [DECISION_RANK] * 2,
            lw=0.4, c="k", linestyle="dashed")
    ax.set_xlim([-0.6, 1.6])
    ax.set_ylim([0, 1])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Raw consensus\n(before calibration)", "Consensus rank\n(after calibration)"])
    ax.legend(handles=[Patch(facecolor=c, edgecolor="none", label=l) for c, l in zip(colors, legend_labels)],
              fontsize=5, loc="upper left", frameon=True, framealpha=0.85, handlelength=1.0,
              handletextpad=0.4, borderpad=0.3)
    stylia.label(ax, xlabel="", ylabel="Consensus score", title=title)


def plot_scatter(ax, data: dict, colors: tuple, legend_labels: tuple) -> list:
    """Panel (b): raw consensus against consensus rank. Returns the Spearman of each group."""
    rhos = []
    for dataset, color in zip(("drugbank", "reference"), colors):                    # reference on top
        raw, rank = data[dataset]
        ax.scatter(raw, rank, color=color, s=1, alpha=0.4, lw=0)
    for dataset in ("drugbank", "reference"):
        raw, rank = data[dataset]
        rhos.append(float(spearmanr(raw, rank).statistic))
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(handles=[Patch(facecolor=c, edgecolor="none", label=f"{l.split(' (')[0]}: Spearman {rho:.4f}")
                       for c, l, rho in zip(colors, legend_labels, rhos)],
              fontsize=5, loc="lower right", frameon=True, framealpha=0.85, handlelength=1.0,
              handletextpad=0.4, borderpad=0.3)
    stylia.label(ax, xlabel="Raw consensus (before calibration)", ylabel="Consensus rank (after calibration)")
    return rhos


def plot_pathogen(pathogen: str, pathogen_name: str, nc, rng) -> tuple:
    """One figure. Returns (path of the PNG, Spearman of DrugBank, Spearman of reference)."""
    data = load_consensus(pathogen)
    with open(os.path.join(CONSENSUS_DIR, pathogen, "anchors.json")) as f:
        n_models = len(json.load(f)["models"])

    colors = (nc.amber, nc.lime)                           # DrugBank, reference (as in 12c)
    labels = (f"DrugBank ({len(data['drugbank'][0]):,} compounds)",
              f"Reference library ({len(data['reference'][0]):,} compounds)")

    fig, axs = stylia.create_figure(1, 2)
    plot_distributions(axs.next(), data, colors, rng, labels,
               title=f"{pathogen_name} ({n_models} models)")
    rhos = plot_scatter(axs.next(), data, colors, labels)

    out_path = os.path.join(OUT_DIR, f"14b_{pathogen}.png")
    save_figure(out_path)
    return out_path, rhos[0], rhos[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--pathogen", default=None,
                        help="Pathogen code to plot (e.g. 'ecoli'). If omitted, every pathogen "
                             "with two or more retained models is plotted.")
    args = parser.parse_args()

    names = pd.read_csv(PATHOGENS_PATH)
    code_to_name = dict(zip(names["code"], names["pathogen"]))
    reports = pd.read_csv(REPORTS_PATH)
    counts = reports.groupby("pathogen").size()
    expected = sorted(counts[counts >= 2].index)

    codes = [args.pathogen] if args.pathogen else expected
    missing = [c for c in codes if not os.path.isfile(os.path.join(CONSENSUS_DIR, c, "anchors.json"))]
    if missing:
        print(f"No step-14 output for: {', '.join(missing)} (run step 14 first)")
        sys.exit(1)

    # Format: slide | Style: article (as in 10b and 12c) — change with stylia.set_format() / stylia.set_style()
    stylia.set_format("slide")
    stylia.set_style("article")
    nc = ArticleColors()
    rng = np.random.default_rng(RANDOM_SEED)

    for code in codes:
        print(f"Plotting {code_to_name.get(code, code)} ({code})")
        path, rho_db, rho_ref = plot_pathogen(code, code_to_name.get(code, code), nc, rng)
        print(f"  Spearman raw vs rank: DrugBank {rho_db:.6f}, reference {rho_ref:.6f}")
        print(f"  saved: {path}")


if __name__ == "__main__":
    main()
