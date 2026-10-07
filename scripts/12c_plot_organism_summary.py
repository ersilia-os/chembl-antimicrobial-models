"""
Step 12c — Per-organism summary: score distributions of the training folds vs DrugBank and the
reference library.

For each pathogen, renders a 2-panel figure (panels stacked, one slot per dataset along x):
  (a) Predicted probabilities, four boxplots (+ jittered points) per dataset: out-of-fold
      actives (red), out-of-fold inactives (blue), the DrugBank compounds (yellow) and the
      50,000 molecules of the LazyQSAR reference library (green), with the model's
      decision_cutoff_proba as a dotted line over the last two.
  (b) The same four groups for the rank scores, with decision_cutoff_rank as a dotted line.
Every boxplot: box from the 25th to the 75th percentile with a line at the median, whiskers at the 5th
and 95th percentiles, and circles at the 90th (orchid, purple) and 99th (fuchsia) percentiles; the
legend of panel (a) says so.
Above each group (y = 0.95) the text gives the % of its compounds at or above the cutoff
(>=, the comparison lazyqsar's `binary` output uses). In (b) that is the rank 0.65 every model
was cut at, for all four groups. In (a) only the DrugBank and reference groups have a cutoff
line and a label: their probabilities come from the final model, so that model's
decision_cutoff_proba is exact for them. The out-of-fold probabilities come from five fold
models, and the same rank 0.65 sits at a different probability in each of them (they differ by a
median of 0.05, up to 0.28), so no single probability cutoff applies to the out-of-fold groups
and panel (a) shows none for them; their percentages are in panel (b).

The reference library is the set every model's rank is a position against, so about 1% of it
reaches rank 0.65 by design (0.98% to 1.21% per model, see the 12a entry in scripts/README.md):
the green group is a calibration reference for the others, not an independent test.

All five folds are used, so every compound of a dataset is scored out-of-fold exactly once. The
DrugBank and reference scores are those of the dataset's final model, trained on all data. The x
order is the column order of the DrugBank rank file. The raw model score has no panel: 09 never
saved out-of-fold scores (only probabilities and ranks).

Inputs (per pathogen):
  - output/09_reports/{pathogen}/{name}_folds.json    (per-fold y_true / y_hat / y_rank)
  - output/09_models/{pathogen}/{name}/metadata.json  (decision_cutoff_proba / _rank)
  - output/12_drugbank/{rank,proba}/{pathogen}.csv    (DrugBank scores, one column per model)
  - output/12_reference/{rank,proba}/{pathogen}.csv   (reference-library scores, same columns)
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from stylia import ArticleColors, save_figure

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "..", "src"))
from default import RANDOM_SEED  # noqa: E402

REPORTS_DIR    = os.path.join(root, "..", "output", "09_reports")
MODELS_DIR     = os.path.join(root, "..", "output", "09_models")
DRUGBANK_DIR   = os.path.join(root, "..", "output", "12_drugbank")
REFERENCE_DIR  = os.path.join(root, "..", "output", "12_reference")
PATHOGENS_PATH = os.path.join(root, "..", "config", "pathogens.csv")
OUT_DIR        = os.path.join(root, "..", "output", "12c_organism_summary")
os.makedirs(OUT_DIR, exist_ok=True)

# Four groups per dataset slot (actives, inactives, DrugBank, reference): x offsets of their boxes
# and the half-width of a box / of the jitter around it, in x units (one dataset = 1 unit). The
# first N_OOF groups are out-of-fold; the rest are scored by the final model.
OFFSETS = (-0.375, -0.125, 0.125, 0.375)
HALF_WIDTH = 0.09
N_OOF = 2

# Height at which each group's "% of compounds at or above the cutoff" is written.
LABEL_Y = 0.95

# Extra percentiles marked on every boxplot, beyond the box (25-75) and whiskers (5-95), as circles:
# (percentile, fill), the fill being "white" or an ArticleColors name not used by the four groups
# (orchid is the palette's purple). Black outline, thinner than stylia's default of 1.0.
# On the rank scale p90 and p99 of the reference group sit at the 0.50 and 0.65 anchors by construction.
MARKED_PERCENTILES = ((90, "orchid"), (99, "fuchsia"))
MARKER_EDGE_WIDTH = 0.5


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


def _marker_fills(nc) -> list:
    """(percentile, fill colour) of the marked percentiles, the colour names resolved through *nc*."""
    return [(q, fill if fill == "white" else getattr(nc, fill)) for q, fill in MARKED_PERCENTILES]


def plot_distributions(ax, names: list, groups: list, cutoffs: list, colors: tuple, rng,
                       marker_fills: list, ylabel: str, xlabel: str = "", title: str = "",
                       legend_labels: tuple = None, oof_cutoff: bool = True) -> None:
    """Three boxplots (+ jittered points) per dataset slot.

    *groups[i]* holds the four score arrays of dataset i (actives, inactives, DrugBank,
    reference), in the order of *colors*; the dotted line marks that dataset's decision cutoff.
    With *oof_cutoff* False only the groups scored by the final model (DrugBank and reference,
    the last ones) get their line and their % label.
    """
    last = len(OFFSETS) - 1
    for i, arrays in enumerate(groups):
        for g, (offset, values, color) in enumerate(zip(OFFSETS, arrays, colors)):
            x = i + offset + rng.uniform(-HALF_WIDTH, HALF_WIDTH, size=len(values))
            ax.scatter(x, values, color=color, s=1, alpha=0.4, lw=0)
            bp = ax.bxp([_box_stats(values)], positions=[i + offset],
                        widths=HALF_WIDTH * 2, patch_artist=True, showfliers=False)
            _style_boxes(bp)
            for q, fill in marker_fills:
                ax.plot([i + offset], [np.percentile(values, q)], marker="o", linestyle="none",
                        markerfacecolor=fill, markeredgecolor="k", markeredgewidth=MARKER_EDGE_WIDTH)
            if oof_cutoff or g >= N_OOF:
                ax.text(i + offset, LABEL_Y, _pct_label(np.mean(values >= cutoffs[i])),
                        ha="center", va="center", fontsize=5,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.5))
        first = 0 if oof_cutoff else N_OOF
        ax.plot([i + OFFSETS[first] - HALF_WIDTH, i + OFFSETS[last] + HALF_WIDTH],
                [cutoffs[i], cutoffs[i]], lw=0.4, c="k", linestyle="dotted")
    ax.set_ylim([0, 1])
    _set_x(ax, names, show_labels=bool(xlabel))
    if legend_labels:
        handles = [Patch(facecolor=c, edgecolor="none", label=l) for c, l in zip(colors, legend_labels)]
        handles += [
            Patch(facecolor="none", edgecolor="k", linewidth=0.8, label="Box: p25 to p75, line at the median"),
            Line2D([], [], color="k", label="Whiskers: p5 to p95"),
        ] + [
            Line2D([], [], marker="o", linestyle="none", markeredgecolor="k", markerfacecolor=fill,
                   markeredgewidth=MARKER_EDGE_WIDTH, label=f"p{q}")
            for q, fill in marker_fills
        ]
        ax.legend(
            handles=handles,
            fontsize=5, loc="upper left", bbox_to_anchor=(1.005, 1.0), frameon=True,
            framealpha=0.85, handlelength=1.0, handletextpad=0.4, borderpad=0.3,
        )
    stylia.label(ax, xlabel=xlabel, ylabel=ylabel, title=title)


def plot_pathogen(pathogen: str, pathogen_name: str, nc, rng) -> str:
    drugbank_rank = pd.read_csv(os.path.join(DRUGBANK_DIR, "rank", f"{pathogen}.csv"))
    drugbank_proba = pd.read_csv(os.path.join(DRUGBANK_DIR, "proba", f"{pathogen}.csv"))
    reference_rank = pd.read_csv(os.path.join(REFERENCE_DIR, "rank", f"{pathogen}.csv"))
    reference_proba = pd.read_csv(os.path.join(REFERENCE_DIR, "proba", f"{pathogen}.csv"))
    names = [c for c in drugbank_rank.columns if c != "smiles"]
    n = len(names)
    for label, frame in (("DrugBank proba", drugbank_proba), ("reference rank", reference_rank),
                         ("reference proba", reference_proba)):
        if [c for c in frame.columns if c != "smiles"] != names:
            raise ValueError(f"[{pathogen}] the {label} file does not have the same models, in the "
                             "same order, as the DrugBank rank file.")

    def groups(key: str, drugbank: pd.DataFrame, reference: pd.DataFrame) -> list:
        out = []
        for name in names:
            actives, inactives = _load_oof(pathogen, name, key)
            out.append((actives, inactives, drugbank[name].to_numpy(dtype=float),
                        reference[name].to_numpy(dtype=float)))
        return out

    proba_groups = groups("y_hat", drugbank_proba, reference_proba)
    rank_groups = groups("y_rank", drugbank_rank, reference_rank)
    proba_cutoffs = [_cutoff(pathogen, name, "decision_cutoff_proba") for name in names]
    rank_cutoffs = [_cutoff(pathogen, name, "decision_cutoff_rank") for name in names]

    colors = (nc.crimson, nc.cobalt, nc.amber, nc.lime)  # actives, inactives, DrugBank, reference
    labels = ("Actives (OOF)", "Inactives (OOF, incl. added)",
              f"DrugBank ({len(drugbank_rank):,} compounds)",
              f"Reference library ({len(reference_rank):,} compounds)")

    # Explicit size (like 10b): the default grid is a very wide, short strip in which the
    # y labels and 10 datasets x 4 boxplots per panel do not fit.
    fig, axs = stylia.create_figure(2, 1, width=0.8, height=0.4)
    marker_fills = _marker_fills(nc)
    plot_distributions(axs.next(), names, proba_groups, proba_cutoffs, colors, rng, marker_fills,
                       ylabel="Predict\nprobabilities", legend_labels=labels,
                       title=f"{pathogen_name} ({n} model{'s' if n != 1 else ''})",
                       oof_cutoff=False)
    plot_distributions(axs.next(), names, rank_groups, rank_cutoffs, colors, rng, marker_fills,
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
