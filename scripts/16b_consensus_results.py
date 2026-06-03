"""
Step 16b — Per-pathogen consensus-recapitulation figures.

For each pathogen, renders a 4x1 panel figure summarising:
  [0] DrugBank prob_rank scores per sub-model (+ decision_cutoff_rank line)
  [1] Consensus scores: weighted, tanh-transformed (per excluded-model + global)
  [2] AUROC: consensus-with vs consensus-without each model (weighted)
  [3] AUROC histogram from per-model recapitulation (off-diagonal pairs)

Inputs (per pathogen):
  - output/12_drugbank/{pathogen}.csv
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
from stylia import CategoricalPalette, save_figure


root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "..", "src"))

from default import RANDOM_SEED

REPORTS_PATH  = os.path.join(root, "..", "output", "10_reports", "10_reports.csv")
DRUGBANK_DIR  = os.path.join(root, "..", "output", "12_drugbank")
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


def _consensus_panel(ax, df, model_cols, pal, rng, ylabel):
    excl_cols = [c for c in df.columns if c.startswith("excluded_")]
    all_cols  = excl_cols + ["consensus_score"]
    xlabels   = list(range(len(excl_cols))) + ["G."]
    NC  = len(all_cols)
    w_c = min(0.35, max(0.15, 1.0 / NC))
    ax.set_ylabel(ylabel)
    ax.set_ylim([0, 1])
    ax.set_xlim([-0.7, NC - 0.3])
    for i, col in enumerate(all_cols):
        color = pal.get(2)[1] if col == "consensus_score" else pal.get(8)[4]
        _plot_col(ax, df[col].dropna().values, i, w_c, color, rng)
    ax.set_xticks(range(NC))
    ax.set_xticklabels(xlabels, rotation=0, size=6)
    ax.set_xlabel(None)


def _hist_panel(ax, values, pal):
    ax.set_xlabel("AUROC")
    ax.set_ylabel("Count")
    ax.set_xlim([0.2, 1])
    bins = np.arange(0, 1.1, 0.05)
    colors = pal.get(4)
    for col, label, color in zip(AUROC_COLS, AUROC_LABELS, colors):
        ax.hist(values[col].dropna().values, bins=bins,
                alpha=0.6, label=label, color=color)
    ax.axvline(0.5, lw=0.6, ls="--", color="k", alpha=0.4)
    ax.legend(title="Threshold", fontsize=6, ncol=2, loc="upper left")


def _consensus_vs_excl_panel(ax, model_cols, df_inc, df_exc, pal, rng, legend_labels):
    colors = pal.get(2)
    offset = 0.15
    N = len(model_cols)
    ax.set_ylabel("AUROC")
    ax.set_ylim([0.3, 1.05])
    ax.set_xlim([-0.7, N - 0.3])
    ax.axhline(0.5, lw=0.6, ls="--", color="k", alpha=0.4)
    for i, model in enumerate(model_cols):
        for df, sign, color in [(df_exc, -1, colors[0]),
                                (df_inc, +1, colors[1])]:
            row = df[df["model"] == model]
            if not row.empty:
                vals = row[AUROC_COLS].values.flatten()
                xs   = i + sign * offset + rng.uniform(-0.04, 0.04, size=len(vals))
                ax.scatter(xs, vals, color=color, s=6, alpha=0.8, lw=0, zorder=3)
    ax.set_xticks(range(N))
    ax.set_xticklabels(range(N), rotation=0, size=6)
    ax.set_xlabel(None)
    ax.legend(handles=[mpatches.Patch(color=c, label=l)
                       for c, l in zip(colors, legend_labels)],
              fontsize=6, loc="lower right")


def plot_pathogen(pathogen, pathogen_name, reports, pal, rng):
    df12        = pd.read_csv(os.path.join(DRUGBANK_DIR,  f"{pathogen}.csv"))
    df14_w_t    = pd.read_csv(os.path.join(CONSENSUS_DIR, f"{pathogen}_transformed.csv"))
    df_recap_m  = pd.read_csv(os.path.join(RECAP_M_DIR,   f"{pathogen}.csv"))
    df_rec_inc  = pd.read_csv(os.path.join(RECAP_C_DIR,   f"{pathogen}_weighted_transformed.csv"))
    df_rec_exc  = pd.read_csv(os.path.join(RECAP_C_DIR,   f"{pathogen}_exc_weighted_transformed.csv"))

    model_cols = [c for c in df12.columns if c != "smiles"]
    report_p   = reports[reports["pathogen"] == pathogen].set_index("model_name")
    N          = len(model_cols)
    w_db       = min(0.35, max(0.15, 1.0 / N))

    stylia.set_format("slide")
    stylia.set_style("article")
    pal = CategoricalPalette("npg")

    fig, axs = stylia.create_figure(4, 1, width=1.3, height=1)
    fig.suptitle(
        f"{pathogen_name} models ({N}) vs.\nDrugBank compounds ({len(df12)} compounds)",
        fontsize=9, y=0.99,
    )

    # [0] DrugBank prob_rank scores per sub-model
    ax = axs.next()
    ax.set_ylabel("prob rank")
    ax.set_ylim([0, 1])
    ax.set_xlim([-0.7, N - 0.3])
    for i, model in enumerate(model_cols):
        _plot_col(ax, df12[model].dropna().values, i, w_db, pal.get(8)[4], rng)
        if model in report_p.index:
            c = report_p.loc[model, "decision_cutoff_rank"]
            ax.plot([i - w_db * 2, i + w_db * 2], [c, c],
                    lw=0.5, c="k", linestyle="dotted")
    ax.set_xticks(range(N))
    ax.set_xticklabels(range(N), rotation=0, size=6)
    ax.set_xlabel(None)

    # [1] Consensus scores: weighted, tanh-transformed
    _consensus_panel(axs.next(), df14_w_t, model_cols, pal, rng, "consensus score\ntransformed")

    # [2] AUROC consensus-incl vs consensus-excl (weighted)
    _consensus_vs_excl_panel(axs.next(), model_cols, df_rec_inc, df_rec_exc, pal, rng,
                             legend_labels=["model excl.", "model incl."])

    # [3] AUROC recapitulation per-model (off-diagonal)
    ax = axs.next()
    df_recap_off = df_recap_m[df_recap_m["model_scorer"] != df_recap_m["model_binarized"]]
    _hist_panel(ax, df_recap_off, pal)

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
        print(f"Plotting {name} ({code})")
        out = plot_pathogen(code, name, reports, pal, rng)
        print(f"  saved: {out}")


if __name__ == "__main__":
    main()
