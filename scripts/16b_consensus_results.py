"""
Step 16b — Per-pathogen consensus-recapitulation figures.

For each pathogen, renders a figure with three full-width rows plus a final
row split into two columns:
  [0] DrugBank prob_rank scores per sub-model (+ decision_cutoff_rank line)
  [1] Consensus scores: weighted, tanh-transformed (per excluded-model + global)
  [2] Consensus-without each model (weighted): AUROC (o), top-N overlap as a
      fraction (^) and spearman (s) on one 0-1 axis, colour per depth
  [3] AUROC from per-model recapitulation (off-diagonal pairs): histogram
      (left column) and reversed-cumulative distribution (right column)

Inputs (per pathogen):
  - output/12_drugbank/rank/{pathogen}.csv
  - output/14_consensus/{pathogen}_transformed.csv
  - output/15_recapitulate_models/{pathogen}.csv
  - output/16_recapitulate_consensus/{pathogen}_weighted_transformed.csv (+ _exc_weighted_transformed)
  - output/10_reports/10_reports.csv  (for decision_cutoff_rank)

Output:
  - output/16_recapitulate_consensus/plots/16_consensus_{pathogen}.png

Usage:
    python scripts/16b_consensus_results.py                  # all pathogens
    python scripts/16b_consensus_results.py --pathogen saureus
"""

import argparse
import os
import sys

import matplotlib.patches as mpatches
import numpy as np
from matplotlib.lines import Line2D
import pandas as pd
import stylia
from stylia import ArticleColors, CategoricalPalette, save_figure


root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "..", "src"))

from default import RANDOM_SEED

REPORTS_PATH  = os.path.join(root, "..", "output", "10_reports", "10_reports.csv")
DRUGBANK_DIR  = os.path.join(root, "..", "output", "12_drugbank", "rank")  # per-model rank (0-1) predictions
CONSENSUS_DIR = os.path.join(root, "..", "output", "14_consensus")
RECAP_M_DIR   = os.path.join(root, "..", "output", "15_recapitulate_models")
RECAP_C_DIR   = os.path.join(root, "..", "output", "16_recapitulate_consensus")
PATHOGENS     = os.path.join(root, "..", "config", "pathogens.csv")
OUT_DIR       = os.path.join(root, "..", "output", "16_recapitulate_consensus", "plots")
os.makedirs(OUT_DIR, exist_ok=True)

AUROC_COLS   = ["auroc_0.1pct", "auroc_1pct", "auroc_5pct"]
AUROC_LABELS = ["0.1%", "1%", "5%"]

# Top-N overlap counts, and the denominators that put them on the same 0-1 scale as
# AUROC/spearman. The depths line up with the AUROC thresholds almost exactly (DrugBank
# n=11347: 0.1% = 12 compounds, 1% = 114, 5% = 568), so the two families share one colour
# scale and can be read against each other at matching depth.
HIT_COLS   = ["hit_overlap_10", "hit_overlap_100", "hit_overlap_500"]
HIT_DENOM  = [10, 100, 500]
HIT_LABELS = ["top 10", "top 100", "top 500"]


def _plot_col(ax, values, pos, bw, color, rng):
    jitter = pos + rng.uniform(-bw, bw, size=len(values))
    ax.scatter(jitter, values, color=color, s=1, alpha=0.2, lw=0)
    stats = dict(
        med=np.median(values),
        q1=np.percentile(values, 25),
        q3=np.percentile(values, 75),
        whislo=np.percentile(values, 1),
        whishi=np.percentile(values, 99),
        fliers=[],
    )
    bp = ax.bxp([stats], positions=[pos], widths=bw * 2,
                patch_artist=True, showfliers=False)
    bp["boxes"][0].set_facecolor("none")
    bp["boxes"][0].set_linewidth(0.4)
    for elem in ["whiskers", "caps", "medians"]:
        for line in bp[elem]:
            line.set_color("k")
            line.set_linewidth(0 if elem == "caps" else 0.4)


def _consensus_panel(ax, df, model_cols, nc, rng, ylabel):
    excl_cols = [c for c in df.columns if c.startswith("excluded_")]
    all_cols  = excl_cols + ["consensus_score"]
    xlabels   = list(range(len(excl_cols))) + ["G."]
    NC  = len(all_cols)
    w_c = min(0.35, max(0.15, 1.0 / NC))
    ax.set_ylabel(ylabel)
    ax.set_ylim([0, 1])
    ax.set_xlim([-0.7, NC - 0.3])
    for i, col in enumerate(all_cols):
        color = nc.turquoise if col == "consensus_score" else nc.amber
        _plot_col(ax, df[col].dropna().values, i, w_c, color, rng)
    ax.set_xticks(range(NC))
    ax.set_xticklabels(xlabels, rotation=0, size=9)
    ax.set_xlabel(None)


def _hist_panel(ax, values, pal):
    # Full 0-1 range: a 0.5 lower limit silently drops every pair the consensus
    # ranks backwards, which is 12.5% of all pairs overall and 30% for calbicans -
    # exactly the disagreements this panel exists to show.
    ax.set_xlabel("AUROC")
    ax.set_ylabel("Count")
    ax.set_xlim([0, 1])
    bins = np.arange(0, 1.1, 0.02)
    colors = pal.get(4)
    for col, label, color in zip(AUROC_COLS, AUROC_LABELS, colors):
        ax.hist(values[col].dropna().values, bins=bins,
                alpha=0.6, label=label, color=color)
    ax.axvline(0.5, lw=0.6, ls="--", color="k", alpha=0.4)
    ax.legend(title="Threshold", fontsize=6, ncol=2, loc="upper left")


def _cum_hist_panel(ax, values, pal):
    # Reversed cumulative (count with AUROC >= x): decreasing from top-left to
    # bottom-right.
    ax.set_xlabel("AUROC")
    ax.set_ylabel("Cumulative prop.\n(AUROC ≥ x)")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    bins = np.arange(0, 1.1, 0.02)
    colors = pal.get(4)
    for col, label, color in zip(AUROC_COLS, AUROC_LABELS, colors):
        vals = values[col].dropna().values
        weights = np.ones_like(vals, dtype=float) / len(vals) if len(vals) else None
        ax.hist(vals, bins=bins, cumulative=-1, weights=weights,
                histtype="step", lw=1.2, label=label, color=color)
    ax.axvline(0.5, lw=0.6, ls="--", color="k", alpha=0.4)
    ax.legend(title="Threshold", fontsize=6, ncol=2, loc="upper right")


def _consensus_exc_panel(ax, model_cols, df_exc, cutoff_colors, rng):
    """Per model: how well the leave-one-out consensus recapitulates it.

    Three metric families on one 0-1 axis, shape-coded:
      o  AUROC at 0.1 / 1 / 5%   — the model's own top t% as positives
      ^  top-N overlap / N       — share of the model's top N the consensus also ranks top N
      s  spearman                — whole-ranking agreement, depth-free

    Colour encodes depth, so the circle and triangle at the same colour answer the same
    question at the same cut: AUROC is the lenient reading, overlap the stringent one.
    Two null lines are drawn because the families do not share one: 0.5 is chance for
    AUROC, whereas random top-N overlap is ~N/n (≈0.9% at top-100) and so sits at 0,
    which is also the null for spearman.
    """
    N = len(model_cols)
    ax.set_ylabel("AUROC · overlap · ρ")
    ax.set_xlim([-0.7, N - 0.3])
    ax.axhline(0.5, lw=0.6, ls="--", color="k", alpha=0.4)
    ax.axhline(0.0, lw=0.6, ls=":",  color="k", alpha=0.4)

    # Three groups of marks per model: AUROC left, spearman centre, overlap right.
    # Within the AUROC and overlap groups the offsets run shallow -> deep, left to right.
    auroc_offs = np.linspace(-0.28, -0.16, len(AUROC_COLS))
    hit_offs   = np.linspace(0.16, 0.28, len(HIT_COLS))
    lo = 0.0

    for i, model in enumerate(model_cols):
        row = df_exc[df_exc["model"] == model]
        if row.empty:
            continue
        for col, color, off in zip(AUROC_COLS, cutoff_colors, auroc_offs):
            vals = row[col].dropna().values
            ax.scatter([i + off] * len(vals), vals, color=color, marker="o",
                       s=20, alpha=0.85, lw=0, zorder=3)
        for col, denom, color, off in zip(HIT_COLS, HIT_DENOM, cutoff_colors, hit_offs):
            vals = row[col].dropna().values / denom
            ax.scatter([i + off] * len(vals), vals, color=color, marker="^",
                       s=20, alpha=0.85, lw=0, zorder=3)
        sp = row["spearman"].dropna().values
        ax.scatter([i] * len(sp), sp, color="k", marker="s",
                   s=16, alpha=0.9, lw=0, zorder=4)
        for arr in (row[AUROC_COLS].values, row[HIT_COLS].values / np.array(HIT_DENOM), sp):
            if len(arr) and np.isfinite(arr).any():
                lo = min(lo, float(np.nanmin(arr)))

    ax.set_ylim([min(-0.05, lo - 0.05), 1.05])
    ax.set_xticks(range(N))
    ax.set_xticklabels(range(N), rotation=0, size=9)
    ax.set_xlabel(None)

    # Both legends go above the axes: with 50+ models every corner of the plot area
    # holds data, so an in-axes legend always lands on top of points.
    depth_legend = ax.legend(
        handles=[mpatches.Patch(color=c, label=f"{a} / {h}")
                 for c, a, h in zip(cutoff_colors, AUROC_LABELS, HIT_LABELS)],
        title="Depth (AUROC / overlap)", fontsize=6, title_fontsize=6, ncol=3,
        loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False,
        borderpad=0, columnspacing=1.0, handletextpad=0.5,
    )
    ax.add_artist(depth_legend)
    ax.legend(
        handles=[
            Line2D([], [], ls="", marker="o", color="0.35", ms=4, label="AUROC"),
            Line2D([], [], ls="", marker="^", color="0.35", ms=4, label="overlap / N"),
            Line2D([], [], ls="", marker="s", color="k",    ms=4, label="spearman"),
        ],
        title="Metric", fontsize=6, title_fontsize=6, ncol=3,
        loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False,
        borderpad=0, columnspacing=1.0, handletextpad=0.5,
    )


def plot_pathogen(pathogen, pathogen_name, reports, pal, rng):
    df12        = pd.read_csv(os.path.join(DRUGBANK_DIR,  f"{pathogen}.csv"))
    df14_w_t    = pd.read_csv(os.path.join(CONSENSUS_DIR, f"{pathogen}_transformed.csv"))
    df_recap_m  = pd.read_csv(os.path.join(RECAP_M_DIR,   f"{pathogen}.csv"))
    df_rec_exc  = pd.read_csv(os.path.join(RECAP_C_DIR,   f"{pathogen}_exc_weighted_transformed.csv"))

    model_cols = [c for c in df12.columns if c != "smiles"]
    report_p   = reports[reports["pathogen"] == pathogen].set_index("model_name")
    N          = len(model_cols)
    w_db       = min(0.35, max(0.15, 1.0 / N))

    stylia.set_format("slide")
    stylia.set_style("article")
    pal = CategoricalPalette("npg")
    nc  = ArticleColors()
    # Distinct per-cutoff colors for panel [2]; chosen to not repeat amber/turquoise
    # (panel [1]) or the npg histogram colors (panel [3]).
    cutoff_colors = [nc.cobalt, nc.orchid, nc.lime]

    # 4x2 grid: rows 0-2 are merged into full-width single panels; the last
    # row keeps both columns so panel [3] can be duplicated side by side.
    fig, axs = stylia.create_figure(4, 2, width=0.7, height=0.7)
    fig.suptitle(
        f"{pathogen_name} models ({N}) vs.\nDrugBank compounds ({len(df12)} compounds)",
        fontsize=9, y=0.99,
    )
    cells = [axs.next() for _ in range(8)]
    gs = cells[0].get_subplotspec().get_gridspec()
    merged = []
    for r in range(3):
        cells[2 * r + 1].remove()            # drop the right cell in this row
        cells[2 * r].set_subplotspec(gs[r, :])  # span the left cell across both cols
        merged.append(cells[2 * r])
    ax0, ax1, ax2 = merged
    ax3a, ax3b = cells[6], cells[7]

    # Panels [0]-[2] share an x-axis: model columns at 0..N-1, plus the
    # consensus ("G.") slot at N which only panel [1] fills.
    shared_xlim = [-0.7, N + 0.7]

    # [0] DrugBank prob_rank scores per sub-model
    ax = ax0
    ax.set_ylabel("Rank score")
    ax.set_ylim([0, 1])
    ax.set_xlim(shared_xlim)
    for i, model in enumerate(model_cols):
        _plot_col(ax, df12[model].dropna().values, i, w_db, pal.get(8)[4], rng)
        if model in report_p.index:
            c = report_p.loc[model, "decision_cutoff_rank"]
            ax.plot([i - w_db * 2, i + w_db * 2], [c, c],
                    lw=0.5, c="k", linestyle="dotted")
    ax.set_xticks(range(N))
    ax.set_xticklabels(range(N), rotation=0, size=9)
    ax.set_xlabel(None)

    # [1] Consensus scores: weighted, tanh-transformed
    _consensus_panel(ax1, df14_w_t, model_cols, nc, rng, "Consensus score\ntransf.")
    ax1.set_xlim(shared_xlim)

    # [2] AUROC consensus-without each model (weighted), colored per cutoff
    _consensus_exc_panel(ax2, model_cols, df_rec_exc, cutoff_colors, rng)
    ax2.set_xlim(shared_xlim)

    # [3] AUROC recapitulation per-model (off-diagonal): histogram (left) and
    # reversed-cumulative distribution (right)
    df_recap_off = df_recap_m[df_recap_m["model_scorer"] != df_recap_m["model_binarized"]]
    _hist_panel(ax3a, df_recap_off, pal)
    _cum_hist_panel(ax3b, df_recap_off, pal)

    out_path = os.path.join(OUT_DIR, f"16_consensus_{pathogen}.png")
    save_figure(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--pathogen", default=None,
                        help="Pathogen code (e.g. 'saureus'). "
                             "If omitted, all pathogens in config/pathogens.csv are processed.")
    args = parser.parse_args()

    reports   = pd.read_csv(REPORTS_PATH)
    pathogens = pd.read_csv(PATHOGENS)
    code_to_name = dict(zip(pathogens["code"], pathogens["pathogen"]))

    if args.pathogen is not None:
        codes = [args.pathogen]
    else:
        codes = pathogens["code"].tolist()

    stylia.set_format("slide")
    stylia.set_style("ersilia")
    pal = CategoricalPalette("ersilia")
    rng = np.random.default_rng(RANDOM_SEED)

    for code in codes:
        name = code_to_name.get(code, code)
        transformed_path = os.path.join(CONSENSUS_DIR, f"{code}_transformed.csv")
        if not os.path.isfile(transformed_path):
            print(f"  [SKIP] {code}: no consensus output (fewer than 2 retained models)")
            continue
        print(f"Plotting {name} ({code})")
        out = plot_pathogen(code, name, reports, pal, rng)
        print(f"  saved: {out}")


if __name__ == "__main__":
    main()
