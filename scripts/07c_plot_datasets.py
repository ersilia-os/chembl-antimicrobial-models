"""
Step 07c — Plot post-balancing dataset stats per pathogen.

Three horizontal panels sharing the pathogen axis:
  (a) datasets per pathogen, split by whether they were balanced with added negatives
      (solid = as-is; white fill / colored edge = balanced)
  (b) final active ratio per dataset (jittered scatter); balanced datasets are hollow and
      sit at the 0.5 balance target (marked with a reference line)
  (c) total negatives added per pathogen (log-scaled x)

Input:  output/07_datasets/07_datasets_metadata.csv
Output: output/07_datasets/07_datasets.png
"""

import glob
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import stylia
from stylia import CategoricalPalette, NamedColors, save_figure

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "..", "src"))
from default import RANDOM_SEED  # noqa: E402

METADATA_PATH = os.path.join(root, "..", "output", "07_datasets", "07_datasets_metadata.csv")
PATHOGENS     = os.path.join(root, "..", "config", "pathogens.csv")
OUT_DIR       = os.path.join(root, "..", "output", "07_datasets")
FIG_PATH      = os.path.join(OUT_DIR, "07_datasets.png")
os.makedirs(OUT_DIR, exist_ok=True)

BALANCE_RATIO   = 0.5
JITTER_STRENGTH = 0.18


def pathogen_reuse(datasets_dir: str) -> pd.DataFrame:
    """Per pathogen, the % of added negatives that were reused across datasets.

    Added negatives are sampled from one shared per-pathogen pool, so the same inactive can
    land in several datasets. reuse% = 100 * (total placements - distinct compounds) / total.
    """
    per: dict[str, Counter] = {}
    for f in glob.glob(os.path.join(datasets_dir, "*", "*.csv")):
        p = os.path.basename(os.path.dirname(f))
        df = pd.read_csv(f)
        if "added_negative" not in df.columns:
            continue
        iks = df.loc[df["added_negative"] & df["inchikey"].notna(), "inchikey"]
        per.setdefault(p, Counter()).update(iks)
    rows = []
    for p, c in per.items():
        total = sum(c.values())
        if total:
            rows.append({"pathogen": p, "pct_reused": 100 * (total - len(c)) / total})
    return pd.DataFrame(rows)


def main():
    metadata  = pd.read_csv(METADATA_PATH)
    pathogens = pd.read_csv(PATHOGENS)
    code_to_name = dict(zip(pathogens["code"], pathogens["pathogen"]))

    # Total negatives added per dataset (proven negatives + any decoy fallback).
    metadata = metadata.copy()
    metadata["added"] = metadata["added_negatives"] + metadata["added_decoys"]

    counts = (
        metadata.groupby("pathogen").size().reset_index(name="n_datasets")
        .sort_values("n_datasets", ascending=True)
    )
    counts["name"] = counts["pathogen"].map(code_to_name)
    pathogen_order = counts["pathogen"].tolist()
    pathogen_to_y = {p: i for i, p in enumerate(pathogen_order)}
    counts["pathogen_y"] = counts["pathogen"].map(pathogen_to_y)

    metadata["pathogen_rank"] = metadata["pathogen"].map(pathogen_to_y)
    rng = np.random.default_rng(RANDOM_SEED)
    metadata["y_jitter"] = metadata["pathogen_rank"] + rng.uniform(
        -JITTER_STRENGTH, JITTER_STRENGTH, size=len(metadata)
    )

    n_asis = metadata.loc[metadata["added"] == 0].groupby("pathogen").size()
    n_bal  = metadata.loc[metadata["added"] > 0].groupby("pathogen").size()
    counts["n_asis"] = counts["pathogen"].map(n_asis).fillna(0).astype(int)
    counts["n_bal"]  = counts["pathogen"].map(n_bal).fillna(0).astype(int)

    added_per_pathogen = metadata.groupby("pathogen")["added"].sum()
    counts["n_added"] = counts["pathogen"].map(added_per_pathogen).fillna(0).astype(int)

    reuse = pathogen_reuse(os.path.join(root, "..", "output", "07_datasets"))
    reuse_map = dict(zip(reuse["pathogen"], reuse["pct_reused"])) if len(reuse) else {}
    counts["pct_reused"] = counts["pathogen"].map(reuse_map).fillna(0.0)

    stylia.set_format("slide")
    stylia.set_style("article")
    pal = CategoricalPalette("npg")
    color = pal.get(8)[0]
    nc = NamedColors()

    fig, axs = stylia.create_figure(1, 4)

    # (a) datasets per pathogen, split by balancing
    ax_a = axs.next()
    ax_a.barh(counts["pathogen_y"], counts["n_asis"], facecolor=color,
              edgecolor=color, linewidth=0.8, label="as-is")
    ax_a.barh(counts["pathogen_y"], counts["n_bal"], left=counts["n_asis"],
              facecolor="white", edgecolor=color, linewidth=0.8, label="balanced")
    ax_a.set_yticks(counts["pathogen_y"])
    ax_a.set_yticklabels(counts["name"])
    ax_a.legend(loc="lower right")
    stylia.label(ax_a, xlabel="Number of datasets", ylabel="Pathogen", title="Datasets per pathogen")

    # (b) final active ratio per dataset
    ax_b = axs.next()
    ax_b.sharey(ax_a)
    asis = metadata[metadata["added"] == 0]
    bal  = metadata[metadata["added"] > 0]
    ax_b.axvline(BALANCE_RATIO, color=nc.silver, linewidth=0.8, zorder=0)
    ax_b.scatter(asis["final_ratio"], asis["y_jitter"], color=color, alpha=0.6, label="as-is")
    # Balanced datasets all sit at exactly 0.5; a distinct filled colour makes that
    # column visible instead of vanishing into the 0.5 reference line.
    ax_b.scatter(bal["final_ratio"], bal["y_jitter"], color=nc.cobalt, alpha=0.85,
                 zorder=3, label="balanced (→0.5)")
    ax_b.set_xlim([-0.05, 1.05])
    ax_b.tick_params(axis="y", left=False, labelleft=False)
    ax_b.legend(loc="lower right")
    stylia.label(ax_b, xlabel="Final active ratio", ylabel="", title="Active ratio per dataset")

    # (c) negatives added per pathogen
    ax_c = axs.next()
    ax_c.set_xscale("log")
    ax_c.sharey(ax_a)
    ax_c.barh(counts["pathogen_y"], counts["n_added"].clip(lower=0),
              facecolor="white", edgecolor=color, linewidth=0.8)
    ax_c.tick_params(axis="y", left=False, labelleft=False)
    stylia.label(ax_c, xlabel="Negatives added", ylabel="", title="Negatives added per pathogen")

    # (d) how much the shared negative pool was reused across a pathogen's datasets
    ax_d = axs.next()
    ax_d.sharey(ax_a)
    ax_d.barh(counts["pathogen_y"], counts["pct_reused"], facecolor=color, edgecolor=color, linewidth=0.8)
    ax_d.tick_params(axis="y", left=False, labelleft=False)
    stylia.label(ax_d, xlabel="% of added negatives reused", ylabel="", title="Negative reuse across datasets")

    save_figure(FIG_PATH)
    print(f"Saved figure: {FIG_PATH}")


if __name__ == "__main__":
    main()
