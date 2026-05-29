"""
Step 07c — Plot post-augmentation dataset stats per pathogen.

Mirrors notebook 06: three horizontal panels sharing the pathogen axis.
  (a) number of datasets per pathogen
  (b) per-dataset active ratio (final_ratio) with jittered y
  (c) total decoys added per pathogen (log-scaled x)

Input:  output/07_datasets/07_datasets_metadata.csv
Output: output/07_datasets/07c_datasets.png
"""

import os

import numpy as np
import pandas as pd
import stylia
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

    decoy_counts = (
        metadata.groupby("name")["decoys"]
        .sum()
        .reset_index(name="n_decoys")
    )
    decoy_counts["pathogen_y"] = decoy_counts["name"].map(pathogen_to_y)
    decoy_counts = decoy_counts.sort_values("pathogen_y")

    fig, axs = stylia.create_figure(1, 3, width=0.6, height=0.3)

    ax_a = axs.next()
    ax_a.barh(counts["pathogen_y"], counts["n_datasets"], color=color)
    ax_a.set_yticks(counts["pathogen_y"])
    ax_a.set_yticklabels(counts["name"])
    ax_a.set_xlabel("Number of datasets")
    ax_a.set_ylabel("Pathogen")
    stylia.label(ax_a, title="Datasets per pathogen")

    ax_b = axs.next()
    ax_b.sharey(ax_a)
    augmented = metadata["decoys"] > 0
    native = metadata.loc[~augmented]
    decoyed = metadata.loc[augmented]
    ax_b.scatter(native["final_ratio"], native["pathogen_y_jitter"],
                 color=color, s=stylia.MARKERSIZE,
                 label="no decoys")
    ax_b.scatter(decoyed["final_ratio"], decoyed["pathogen_y_jitter"],
                 facecolors="none", edgecolors=color, linewidths=0.5, alpha=0.5,
                 s=stylia.MARKERSIZE-2,
                 label="with decoys")
    ax_b.set_xlabel("Active ratio")
    ax_b.set_ylabel("")
    ax_b.set_xlim([-0.05, 1.05])
    ax_b.tick_params(axis="y", left=False, labelleft=False)
    stylia.label(ax_b, title="Active ratio per dataset")

    ax_c = axs.next()
    ax_c.set_xscale("log")
    ax_c.sharey(ax_a)
    ax_c.barh(decoy_counts["pathogen_y"], decoy_counts["n_decoys"], color=color)
    ax_c.set_xlabel("Number of decoys")
    ax_c.set_ylabel("")
    ax_c.tick_params(axis="y", left=False, labelleft=False)
    stylia.label(ax_c, title="Decoys added per pathogen")

    save_figure(FIG_PATH)
    print(f"Saved figure: {FIG_PATH}")


if __name__ == "__main__":
    main()
