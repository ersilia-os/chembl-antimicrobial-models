"""
Step 16b — Per-pathogen consensus-recapitulation figures, on DrugBank and on the reference library.

For each pathogen and each dataset (drugbank, reference), renders a figure with three full-width
rows plus a final row split into two columns:
  [0] Rank scores per sub-model (+ decision_cutoff_rank line)
  [1] Consensus rank (weighted; per excluded-model + global)
  [2] Consensus-without each model (weighted): AUROC of the consensus recapitulating
      the model's top 0.1 / 1 / 5%, colour per depth
  [3] AUROC from per-model recapitulation (off-diagonal pairs): histogram
      (left column) and reversed-cumulative distribution (right column)

On the reference library panels [0] and [1] are a calibration check, not a measurement: every
rank there is a position against that same library, so each distribution is fixed by construction
(and 1% of the consensus sits at rank >= 0.65). Panels [2] and [3] are real agreement measurements.

Inputs (per pathogen and dataset):
  - output/12_{drugbank,reference}/rank/{pathogen}.csv
  - output/14_consensus/{pathogen}/{dataset}_rank.csv
  - output/15_recapitulate_models/{pathogen}/{dataset}.csv
  - output/16_recapitulate_consensus/{pathogen}/{dataset}_exc_weighted.csv
  - output/10_reports/10_reports.csv  (for decision_cutoff_rank)

Output:
  - output/16_recapitulate_consensus/plots/16_consensus_{pathogen}_{dataset}.png

The script raises if the step-12 file, the step-14 columns, the step-15 and step-16a tables and
10_reports.csv do not all list the same models, and exits with code 1 if a pathogen with two or
more retained models has no step-14 output, so a stale or partial run cannot pass unnoticed.

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

from default import RANDOM_SEED, THRESHOLDS, THRESHOLD_SFXS

REPORTS_PATH  = os.path.join(root, "..", "output", "10_reports", "10_reports.csv")
# dataset -> (step-12 rank folder with the per-model predictions, name used in the figure title)
DATASETS = {
    "drugbank":  (os.path.join(root, "..", "output", "12_drugbank", "rank"),  "DrugBank compounds"),
    "reference": (os.path.join(root, "..", "output", "12_reference", "rank"), "reference-library compounds"),
}
CONSENSUS_DIR = os.path.join(root, "..", "output", "14_consensus")
RECAP_M_DIR   = os.path.join(root, "..", "output", "15_recapitulate_models")
RECAP_C_DIR   = os.path.join(root, "..", "output", "16_recapitulate_consensus")
PATHOGENS     = os.path.join(root, "..", "config", "pathogens.csv")
OUT_DIR       = os.path.join(root, "..", "output", "16_recapitulate_consensus", "plots")
os.makedirs(OUT_DIR, exist_ok=True)

# AUROC is read at three depths: the top 0.1% / 1% / 5% of the dataset.
AUROC_COLS   = [f"auroc_{s}" for s in THRESHOLD_SFXS]
AUROC_LABELS = [f"{t * 100:g}%" for t in THRESHOLDS]


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
    # One leave-one-out column per model, in the model order of panel [0]
    excl_cols = [f"consensus_rank_without_{m}" for m in model_cols]
    all_cols  = excl_cols + ["consensus_rank"]
    xlabels   = list(range(len(excl_cols))) + ["G."]
    NC  = len(all_cols)
    w_c = min(0.35, max(0.15, 1.0 / NC))
    ax.set_ylabel(ylabel)
    ax.set_ylim([0, 1])
    ax.set_xlim([-0.7, NC - 0.3])
    for i, col in enumerate(all_cols):
        color = nc.turquoise if col == "consensus_rank" else nc.amber
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
    """Per model: how well the leave-one-out consensus recapitulates it, as AUROC.

    One circle per depth: the model's own top 0.1 / 1 / 5% as positives, the consensus built
    without that model as the score. Colour encodes depth. The dashed line is chance (0.5).
    """
    N = len(model_cols)
    ax.set_ylabel("AUROC")
    ax.set_xlim([-0.7, N - 0.3])
    ax.axhline(0.5, lw=0.6, ls="--", color="k", alpha=0.4)

    # The three depths sit side by side around each model's tick, shallow -> deep, left to right.
    offs = np.linspace(-0.12, 0.12, len(AUROC_COLS))
    lo = 0.5

    for i, model in enumerate(model_cols):
        row = df_exc[df_exc["model"] == model]
        if row.empty:
            continue
        for col, color, off in zip(AUROC_COLS, cutoff_colors, offs):
            vals = row[col].dropna().values
            ax.scatter([i + off] * len(vals), vals, color=color, marker="o",
                       s=20, alpha=0.85, lw=0, zorder=3)
            if len(vals):
                lo = min(lo, float(np.nanmin(vals)))

    ax.set_ylim([min(0.45, lo - 0.05), 1.05])
    ax.set_xticks(range(N))
    ax.set_xticklabels(range(N), rotation=0, size=9)
    ax.set_xlabel(None)

    # The legend goes above the axes: with 50+ models every corner of the plot area holds data,
    # so an in-axes legend always lands on top of points.
    ax.legend(
        handles=[mpatches.Patch(color=c, label=f"top {a}")
                 for c, a in zip(cutoff_colors, AUROC_LABELS)],
        title="Depth", fontsize=6, title_fontsize=6, ncol=3,
        loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False,
        borderpad=0, columnspacing=1.0, handletextpad=0.5,
    )


def _check_models(pathogen, dataset, model_cols, report_models, df14_w, df_recap_m, df_rec_exc):
    """Raise unless every table behind the figure covers exactly the models of the step-12 file."""
    prefix = "consensus_rank_without_"
    expected = set(model_cols)
    sources = {
        "10_reports.csv": set(report_models),
        "the step-14 leave-one-out columns": {c[len(prefix):] for c in df14_w.columns if c.startswith(prefix)},
        "the step-15 table (scorer)": set(df_recap_m["model_scorer"].astype(str)),
        "the step-15 table (binarized)": set(df_recap_m["model_binarized"].astype(str)),
        "the step-16a table": set(df_rec_exc["model"].astype(str)),
    }
    for name, models in sources.items():
        if models != expected:
            raise ValueError(
                f"[{pathogen}] {dataset}: {name} and the step-12 rank file list different models. "
                f"Only in {name}: {sorted(models - expected)}; only in the step-12 file: "
                f"{sorted(expected - models)}. Re-run the steps after the one that changed.")


def plot_pathogen(pathogen, pathogen_name, dataset, reports, pal, rng):
    in_dir_12, dataset_label = DATASETS[dataset]
    df12        = pd.read_csv(os.path.join(in_dir_12,     f"{pathogen}.csv"))
    df14_w      = pd.read_csv(os.path.join(CONSENSUS_DIR, pathogen, f"{dataset}_rank.csv"))
    df_recap_m  = pd.read_csv(os.path.join(RECAP_M_DIR,   pathogen, f"{dataset}.csv"))
    df_rec_exc  = pd.read_csv(os.path.join(RECAP_C_DIR,   pathogen, f"{dataset}_exc_weighted.csv"))

    model_cols = [c for c in df12.columns if c != "smiles"]
    report_p   = reports[reports["pathogen"] == pathogen].set_index("model_name")
    N          = len(model_cols)
    w_db       = min(0.35, max(0.15, 1.0 / N))
    _check_models(pathogen, dataset, model_cols, list(report_p.index), df14_w, df_recap_m, df_rec_exc)

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
        f"{pathogen_name} models ({N}) vs.\n{dataset_label} ({len(df12)} compounds)",
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

    # [0] Rank scores per sub-model
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

    # [1] Consensus rank: weighted
    _consensus_panel(ax1, df14_w, model_cols, nc, rng, "Consensus rank")
    ax1.set_xlim(shared_xlim)

    # [2] AUROC consensus-without each model (weighted), colored per cutoff
    _consensus_exc_panel(ax2, model_cols, df_rec_exc, cutoff_colors, rng)
    ax2.set_xlim(shared_xlim)

    # [3] AUROC recapitulation per-model (off-diagonal): histogram (left) and
    # reversed-cumulative distribution (right)
    df_recap_off = df_recap_m[df_recap_m["model_scorer"] != df_recap_m["model_binarized"]]
    _hist_panel(ax3a, df_recap_off, pal)
    _cum_hist_panel(ax3b, df_recap_off, pal)

    out_path = os.path.join(OUT_DIR, f"16_consensus_{pathogen}_{dataset}.png")
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
    n_models = reports.groupby("pathogen").size()

    if args.pathogen is not None:
        if args.pathogen not in n_models.index:
            parser.error(f"unknown pathogen '{args.pathogen}'; 10_reports.csv has: {', '.join(sorted(n_models.index))}")
        codes = [args.pathogen]
    else:
        codes = pathogens["code"].tolist()

    stylia.set_format("slide")
    stylia.set_style("ersilia")
    pal = CategoricalPalette("ersilia")
    rng = np.random.default_rng(RANDOM_SEED)

    missing = []
    for code in codes:
        name = code_to_name.get(code, code)
        if n_models.get(code, 0) < 2:
            print(f"  [SKIP] {code}: {n_models.get(code, 0)} retained model(s), no consensus")
            continue
        if not os.path.isdir(os.path.join(CONSENSUS_DIR, code)):
            print(f"  [MISSING] {code}: no step-14 output in {CONSENSUS_DIR}")
            missing.append(code)
            continue
        for dataset in DATASETS:
            print(f"Plotting {name} ({code}) on {dataset}")
            out = plot_pathogen(code, name, dataset, reports, pal, rng)
            print(f"  saved: {out}")

    if missing:
        print("\nNo step-14 output for: " + ", ".join(missing) + "\n(run steps 14, 15 and 16a for them first)")
        sys.exit(1)


if __name__ == "__main__":
    main()
