"""
Step 12c — Per-organism summary: score distributions of the training folds vs DrugBank.

For each pathogen, renders a 2-panel figure (panels stacked, one slot per dataset along x):
  (a) Predicted probabilities, three boxplots (+ jittered points) per dataset: out-of-fold
      actives (red), out-of-fold inactives (blue) and the DrugBank compounds (yellow), with
      the model's decision_cutoff_proba as a dotted line.
  (b) The same three groups for the rank scores, with decision_cutoff_rank as a dotted line.
Above each group (y = 0.95) the text gives the % of its compounds at or above the cutoff
(>=, the comparison lazyqsar's `binary` output uses). In (b) that is the rank 0.65 every model
was cut at, for all three groups. In (a) only the DrugBank group has a cutoff line and a label:
its probabilities come from the final model, so that model's decision_cutoff_proba is exact for
them. The out-of-fold probabilities come from five fold models, and the same rank 0.65 sits at a
different probability in each of them (they differ by a median of 0.05, up to 0.28), so no single
probability cutoff applies to the out-of-fold groups and panel (a) shows none for them; their
percentages are in panel (b).

All five folds are used, so every compound of a dataset is scored out-of-fold exactly once. The
DrugBank scores are those of the dataset's final model, trained on all data. The x order is the
column order of the DrugBank rank file. The raw model score has no panel: 09 never saved
out-of-fold scores (only probabilities and ranks).

Inputs (per pathogen):
  - output/09_reports/{pathogen}/{name}_folds.json    (per-fold y_true / y_hat / y_rank)
  - output/09_models/{pathogen}/{name}/metadata.json  (decision_cutoff_proba / _rank)
  - output/12_drugbank/{rank,proba}/{pathogen}.csv    (DrugBank scores, one column per model)
  - config/pathogens.csv                              (pathogen display names)

Output:
  - output/12c_organism_summary/12c_{pathogen}.png

Usage:
    python scripts/12c_plot_organism_summary.py                  # every pathogen in 12_drugbank/rank
    python scripts/12c_plot_organism_summary.py --pathogen kpneumoniae
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import stylia
from matplotlib.patches import Patch
from stylia import ArticleColors, save_figure

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "..", "src"))
from default import RANDOM_SEED  # noqa: E402

REPORTS_DIR    = os.path.join(root, "..", "output", "09_reports")
MODELS_DIR     = os.path.join(root, "..", "output", "09_models")
DRUGBANK_DIR   = os.path.join(root, "..", "output", "12_drugbank")
PATHOGENS_PATH = os.path.join(root, "..", "config", "pathogens.csv")
OUT_DIR        = os.path.join(root, "..", "output", "12c_organism_summary")
os.makedirs(OUT_DIR, exist_ok=True)

# Three groups per dataset slot (actives, inactives, DrugBank): x offsets of their boxes and
# the half-width of a box / of the jitter around it, in x units (one dataset = 1 unit).
OFFSETS = (-0.31, 0.0, 0.31)
HALF_WIDTH = 0.11

# Height at which each group's "% of compounds at or above the cutoff" is written.
LABEL_Y = 0.95


def _pct_label(fraction: float) -> str:
    """'84%' from 10% up, one decimal below (so a 2.6% DrugBank hit rate is not rounded to 3%)."""
    pct = 100 * fraction
    return f"{pct:.0f}%" if pct >= 10 else f"{pct:.1f}%"


def _load_oof(pathogen: str, name: str, key: str):
    """Concatenate the *key* scores ('y_hat' = proba, 'y_rank') across folds.

    Returns (scores_of_actives, scores_of_inactives).
    """
    with open(os.path.join(REPORTS_DIR, pathogen, f"{name}_folds.json")) as f:
        folds = json.load(f)
    y_true, scores = [], []
    for fd in folds.values():
        y_true.extend(fd["y_true"])
        scores.extend(fd[key])
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    return scores[y_true == 1], scores[y_true == 0]


def _cutoff(pathogen: str, name: str, key: str) -> float:
    with open(os.path.join(MODELS_DIR, pathogen, name, "metadata.json")) as f:
        return json.load(f)[key]


def _box_stats(arr: np.ndarray) -> dict:
    """Box at the quartiles, whiskers at the 5th and 95th percentiles (as in 10b)."""
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


def _set_x(ax, names: list, show_labels: bool) -> None:
    n = len(names)
    ax.set_xlim([-0.7, n - 0.3])
    ax.set_xticks(list(range(n)))
    if show_labels:
        ax.set_xticklabels(names, rotation=45, ha="right")
    else:
        ax.set_xticklabels([""] * n)


def plot_distributions(ax, names: list, groups: list, cutoffs: list, colors: tuple, rng,
                       ylabel: str, xlabel: str = "", title: str = "",
                       legend_labels: tuple = None, oof_cutoff: bool = True) -> None:
    """Three boxplots (+ jittered points) per dataset slot.

    *groups[i]* holds the three score arrays of dataset i (actives, inactives, DrugBank), in the
    order of *colors*; the dotted line marks that dataset's decision cutoff. With
    *oof_cutoff* False only the last group, DrugBank, gets its line and its % label.
    """
    last = len(OFFSETS) - 1
    for i, arrays in enumerate(groups):
        for g, (offset, values, color) in enumerate(zip(OFFSETS, arrays, colors)):
            x = i + offset + rng.uniform(-HALF_WIDTH, HALF_WIDTH, size=len(values))
            ax.scatter(x, values, color=color, s=1, alpha=0.4, lw=0)
            bp = ax.bxp([_box_stats(values)], positions=[i + offset],
                        widths=HALF_WIDTH * 2, patch_artist=True, showfliers=False)
            _style_boxes(bp)
            if oof_cutoff or g == last:
                ax.text(i + offset, LABEL_Y, _pct_label(np.mean(values >= cutoffs[i])),
                        ha="center", va="center", fontsize=5,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.5))
        first = 0 if oof_cutoff else last
        ax.plot([i + OFFSETS[first] - HALF_WIDTH, i + OFFSETS[last] + HALF_WIDTH],
                [cutoffs[i], cutoffs[i]], lw=0.4, c="k", linestyle="dotted")
    ax.set_ylim([0, 1])
    _set_x(ax, names, show_labels=bool(xlabel))
    if legend_labels:
        ax.legend(
            handles=[Patch(facecolor=c, edgecolor="none", label=l)
                     for c, l in zip(colors, legend_labels)],
            fontsize=5, loc="upper left", bbox_to_anchor=(1.005, 1.0), frameon=True,
            framealpha=0.85, handlelength=1.0, handletextpad=0.4, borderpad=0.3,
        )
    stylia.label(ax, xlabel=xlabel, ylabel=ylabel, title=title)


def plot_pathogen(pathogen: str, pathogen_name: str, nc, rng) -> str:
    drugbank_rank = pd.read_csv(os.path.join(DRUGBANK_DIR, "rank", f"{pathogen}.csv"))
    drugbank_proba = pd.read_csv(os.path.join(DRUGBANK_DIR, "proba", f"{pathogen}.csv"))
    names = [c for c in drugbank_rank.columns if c != "smiles"]
    n = len(names)

    def groups(key: str, drugbank: pd.DataFrame) -> list:
        out = []
        for name in names:
            actives, inactives = _load_oof(pathogen, name, key)
            out.append((actives, inactives, drugbank[name].to_numpy(dtype=float)))
        return out

    proba_groups = groups("y_hat", drugbank_proba)
    rank_groups = groups("y_rank", drugbank_rank)
    proba_cutoffs = [_cutoff(pathogen, name, "decision_cutoff_proba") for name in names]
    rank_cutoffs = [_cutoff(pathogen, name, "decision_cutoff_rank") for name in names]

    colors = (nc.crimson, nc.cobalt, nc.amber)  # actives, inactives, DrugBank
    labels = ("Actives (OOF)", "Inactives (OOF, incl. added)", "DrugBank")

    # Explicit size (like 10b): the default grid is a very wide, short strip in which the
    # y labels and 10 datasets x 3 boxplots per panel do not fit.
    fig, axs = stylia.create_figure(2, 1, width=0.8, height=0.4)
    plot_distributions(axs.next(), names, proba_groups, proba_cutoffs, colors, rng,
                       ylabel="Predict\nprobabilities", legend_labels=labels,
                       title=f"{pathogen_name} ({n} dataset{'s' if n != 1 else ''})",
                       oof_cutoff=False)
    plot_distributions(axs.next(), names, rank_groups, rank_cutoffs, colors, rng,
                       ylabel="Predict\nrank scores", xlabel="Dataset")

    out_path = os.path.join(OUT_DIR, f"12c_{pathogen}.png")
    save_figure(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--pathogen", default=None,
                        help="Pathogen code to plot (e.g. 'kpneumoniae'). "
                             "If omitted, every pathogen in output/12_drugbank/rank is plotted.")
    args = parser.parse_args()

    names = pd.read_csv(PATHOGENS_PATH)
    code_to_name = dict(zip(names["code"], names["pathogen"]))

    if args.pathogen is not None:
        codes = [args.pathogen]
    else:
        codes = sorted(f[:-4] for f in os.listdir(os.path.join(DRUGBANK_DIR, "rank"))
                       if f.endswith(".csv"))

    # Format: slide | Style: article (as in 10b) — change with stylia.set_format() / stylia.set_style()
    stylia.set_format("slide")
    stylia.set_style("article")
    nc = ArticleColors()
    rng = np.random.default_rng(RANDOM_SEED)

    for code in codes:
        print(f"Plotting {code_to_name.get(code, code)} ({code})")
        print(f"  saved: {plot_pathogen(code, code_to_name.get(code, code), nc, rng)}")


if __name__ == "__main__":
    main()
