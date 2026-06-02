"""
Step 07c — Plot post-augmentation dataset stats per pathogen.

Mirrors notebook 06: three horizontal panels sharing the pathogen axis.
  (a) datasets per pathogen, stacked by decoy augmentation
      (solid = without decoys; white fill / colored edge = with decoys)
  (b) three pies per pathogen at final_ratio categories (<0.05, 0.05-0.25,
      >0.25); pie area ∝ fraction of that pathogen's datasets in the range.
      The middle range is split by decoy augmentation (no decoys = filled,
      with decoys = white).
  (c) total decoys added per pathogen (log-scaled x)

Input:  output/07_datasets/07_datasets_metadata.csv
Output: output/07_datasets/07c_datasets.png
"""

import os

import numpy as np
import pandas as pd
import stylia
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from stylia import CategoricalPalette, save_figure


root = os.path.dirname(os.path.abspath(__file__))

METADATA_PATH = os.path.join(root, "..", "output", "07_datasets",
                             "07_datasets_metadata.csv")
PATHOGENS     = os.path.join(root, "..", "config", "pathogens.csv")
OUT_DIR       = os.path.join(root, "..", "output", "07_datasets")
FIG_PATH      = os.path.join(OUT_DIR, "07_datasets.png")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED     = 42
JITTER_STRENGTH = 0.18


def main():
    metadata  = pd.read_csv(METADATA_PATH)
    pathogens = pd.read_csv(PATHOGENS)
    code_to_name = dict(zip(pathogens["code"], pathogens["pathogen"]))

    counts = (
        metadata.groupby("pathogen")
        .size()
        .reset_index(name="n_datasets")
        .sort_values("n_datasets", ascending=True)
    )
    counts["name"] = counts["pathogen"].map(code_to_name)

    pathogen_order = counts["pathogen"].tolist()
    metadata = metadata.copy()
    metadata["name"] = metadata["pathogen"].map(code_to_name)
    metadata["pathogen_rank"] = metadata["pathogen"].map(
        {p: i for i, p in enumerate(pathogen_order)}
    )

    rng = np.random.default_rng(RANDOM_SEED)
    metadata["pathogen_y_jitter"] = (
        metadata["pathogen_rank"]
        + rng.uniform(-JITTER_STRENGTH, JITTER_STRENGTH, size=len(metadata))
    )

    stylia.set_format("slide")
    stylia.set_style("article")

    pal = CategoricalPalette("npg")
    color = pal.get(8)[0]

    name_order = counts["name"].tolist()
    pathogen_to_y = {name: i for i, name in enumerate(name_order)}
    counts = counts.copy()
    counts["pathogen_y"] = counts["name"].map(pathogen_to_y)

    # Split each pathogen's datasets by whether decoys were added.
    n_no_decoy = metadata.loc[metadata["decoys"] == 0].groupby("pathogen").size()
    n_with_decoy = metadata.loc[metadata["decoys"] > 0].groupby("pathogen").size()
    counts["n_no_decoy"] = counts["pathogen"].map(n_no_decoy).fillna(0).astype(int)
    counts["n_with_decoy"] = counts["pathogen"].map(n_with_decoy).fillna(0).astype(int)

    decoy_counts = (
        metadata.groupby("name")["decoys"]
        .sum()
        .reset_index(name="n_decoys")
    )
    decoy_counts["pathogen_y"] = decoy_counts["name"].map(pathogen_to_y)
    decoy_counts = decoy_counts.sort_values("pathogen_y")

    fig, axs = stylia.create_figure(1, 3, width=0.6, height=0.3)

    ax_a = axs.next()
    ax_a.barh(counts["pathogen_y"], counts["n_no_decoy"], facecolor=color,
              edgecolor=color, linewidth=0.8, label="without decoys")
    ax_a.barh(counts["pathogen_y"], counts["n_with_decoy"],
              left=counts["n_no_decoy"], facecolor="white", edgecolor=color,
              linewidth=0.8, label="with decoys")
    ax_a.set_yticks(counts["pathogen_y"])
    ax_a.set_yticklabels(counts["name"])
    ax_a.set_xlabel("Number of datasets")
    ax_a.set_ylabel("Pathogen")
    ax_a.legend(loc="lower right")
    stylia.label(ax_a, title="Datasets per pathogen")

    # Per pathogen, three pies at the active-ratio (final_ratio) categories.
    # Each pie's AREA is proportional to the fraction of that pathogen's
    # datasets in the range; PIE_FULL_AREA is the area (in^2) of a pie that
    # would represent 100% of a pathogen's datasets. The middle range
    # (0.05-0.25) is split into a 2-slice pie by decoy augmentation:
    # without decoys filled with color, with decoys white (both colored edge).
    RATIO_TICKS = [0, 1, 2]
    RATIO_LABELS = ["<0.05", "0.05-0.25", ">0.25"]
    PIE_FULL_AREA = 0.045  # in^2; pie area at proportion == 1

    metadata["ratio_bin"] = np.select(
        [metadata["final_ratio"] < 0.05,
         metadata["final_ratio"] <= 0.25],
        [0, 1], default=2,
    )
    prop = (
        pd.crosstab(metadata["pathogen"], metadata["ratio_bin"], normalize="index")
        .reindex(columns=RATIO_TICKS, fill_value=0.0)
    )
    # Decoy composition of the middle (0.05-0.25) range, per pathogen.
    mid = metadata.loc[metadata["ratio_bin"] == 1]
    mid_decoy = (
        mid.assign(_has=mid["decoys"] > 0)
        .groupby(["pathogen", "_has"]).size().unstack(fill_value=0)
        .reindex(columns=[False, True], fill_value=0)
    )

    ax_b = axs.next()
    ax_b.sharey(ax_a)
    ax_b.set_xlim([-0.5, 2.5])
    for _, row in counts.iterrows():
        pcode = row["pathogen"]
        for xi in RATIO_TICKS:
            p = prop.loc[pcode, xi] if pcode in prop.index else 0.0
            if p <= 0:
                continue
            diameter = 2 * np.sqrt(PIE_FULL_AREA * p / np.pi)
            axins = inset_axes(ax_b, width=diameter, height=diameter, loc="center",
                               bbox_to_anchor=(xi, row["pathogen_y"]),
                               bbox_transform=ax_b.transData, borderpad=0)
            if xi == 1 and pcode in mid_decoy.index:
                n_no = mid_decoy.loc[pcode, False]
                n_with = mid_decoy.loc[pcode, True]
                axins.pie([n_no, n_with], colors=[color, "white"],
                          wedgeprops={"edgecolor": color, "linewidth": 0.65})
            else:
                axins.pie([1.0], colors=[color],
                          wedgeprops={"edgecolor": color, "linewidth": 0.65})

    ax_b.set_xticks(RATIO_TICKS)
    ax_b.set_xticklabels(RATIO_LABELS)
    ax_b.set_xlabel("Active ratio")
    ax_b.set_ylabel("")
    ax_b.tick_params(axis="y", left=False, labelleft=False)
    stylia.label(ax_b, title="Active ratio per dataset")

    ax_c = axs.next()
    ax_c.set_xscale("log")
    ax_c.sharey(ax_a)
    ax_c.barh(decoy_counts["pathogen_y"], decoy_counts["n_decoys"],
              facecolor="white", edgecolor=color, linewidth=0.8)
    ax_c.set_xlabel("Number of decoys")
    ax_c.set_ylabel("")
    ax_c.tick_params(axis="y", left=False, labelleft=False)
    stylia.label(ax_c, title="Decoys added per pathogen")

    save_figure(FIG_PATH)
    print(f"Saved figure: {FIG_PATH}")


if __name__ == "__main__":
    main()
