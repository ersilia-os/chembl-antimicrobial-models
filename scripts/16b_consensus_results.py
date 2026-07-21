"""
Step 16b — Per-pathogen consensus-recapitulation figures.

For each pathogen, renders a figure with three full-width rows plus a final
row split into two columns:
  [0] DrugBank prob_rank scores per sub-model (+ decision_cutoff_rank line)
  [1] Consensus scores: weighted, tanh-transformed (per excluded-model + global)
  [2] AUROC: consensus-without each model (weighted), one color per cutoff
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
    ax.set_xlabel("AUROC")
    ax.set_ylabel("Count")
    ax.set_xlim([0.5, 1])
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
    ax.set_xlim([0.5, 1])
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
    N = len(model_cols)
    ax.set_ylabel("AUROC")
    ax.set_ylim([0.3, 1.05])
    ax.set_xlim([-0.7, N - 0.3])
    ax.axhline(0.5, lw=0.6, ls="--", color="k", alpha=0.4)
    # Fixed x offset per cutoff, ordered low -> high cutoff (0.1% left, 5% right).
    offs = np.linspace(-0.1, 0.1, len(AUROC_COLS))
    for i, model in enumerate(model_cols):
        row = df_exc[df_exc["model"] == model]
        if row.empty:
            continue
        for col, color, off in zip(AUROC_COLS, cutoff_colors, offs):
            vals = row[col].dropna().values
            ax.scatter([i + off] * len(vals), vals, color=color,
                       s=20, alpha=0.85, lw=0, zorder=3)
    ax.set_xticks(range(N))
    ax.set_xticklabels(range(N), rotation=0, size=9)
    ax.set_xlabel(None)
    ax.legend(handles=[mpatches.Patch(color=c, label=l)
                       for c, l in zip(cutoff_colors, AUROC_LABELS)],
              title="Threshold", fontsize=6, ncol=2, loc="lower right")


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
