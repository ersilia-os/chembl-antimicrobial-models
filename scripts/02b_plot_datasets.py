"""
Step 02b - Plot dataset counts and active ratios per pathogen.

Stacked horizontal bar chart (ChEMBL + PubChem) and a scatter of active ratios
per dataset. PubChem is restricted to organism-level assays
(data/processed/pubchem/02_pubchem_datasets_organism.csv) and rows labelled
'discarded' are excluded.

Output: output/02b_plot_datasets/02b_datasets.png
"""

import os

import numpy as np
import pandas as pd
import stylia
from stylia import CategoricalPalette, save_figure


root = os.path.dirname(os.path.abspath(__file__))

CHEMBL_PATH  = os.path.join(root, "..", "data", "processed", "chembl",
                            "01_chembl_datasets_all.csv")
PUBCHEM_PATH = os.path.join(root, "..", "data", "processed", "pubchem",
                            "02_pubchem_datasets_organism.csv")
PATHOGENS    = os.path.join(root, "..", "config", "pathogens.csv")
OUT_DIR      = os.path.join(root, "..", "output", "02_datasets")
FIG_PATH     = os.path.join(OUT_DIR, "02_datasets.png")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED     = 42
JITTER_STRENGTH = 0.18


def _print_summary(chembl, pubchem):
    def block(name, df):
        n_above = int((df["ratio"] > 0.5).sum())
        mean, std = df["ratio"].mean(), df["ratio"].std()
        lab = df["label"].value_counts()
        parts = [f"{int(lab.get(L, 0))} {L}s" for L in ["A", "B", "M", "G"] if L in lab.index]
        print(f"— {name}")
        print(f"— {df['pathogen'].nunique()} pathogens")
        print(f"— {len(df)} binarized datasets obtained from {name} antimicrobial tasks")
        print(f"— {n_above} datasets have ratio>0.5 ({mean:.2f}±{std:.2f})")
        print(f"— {', '.join(parts[:-1])} and {parts[-1]}" if len(parts) > 1 else f"— {parts[0]}")

    block("ChEMBL", chembl)
    print()
    block("PubChem", pubchem)

    ids = set(pubchem["chembl_id"].dropna())
    if ids:
        pat = "|".join(map(str, ids))
        n_super = int(chembl["name"].str.contains(pat, na=False, regex=True).sum())
        print(f"— {n_super} ChEMBL datasets are superseded by PubChem assays")


def main():
    chembl = pd.read_csv(CHEMBL_PATH)
    pubchem = pd.read_csv(PUBCHEM_PATH)
    pubchem = pubchem[pubchem["label"] != "discarded"].copy()
    pubchem = pubchem.rename(columns={"pathogen_code": "pathogen"})

    _print_summary(chembl, pubchem)
    print()

    pathogens = pd.read_csv(PATHOGENS)
    code_to_name = dict(zip(pathogens["code"], pathogens["pathogen"]))

    n_chembl  = chembl.groupby("pathogen").size().rename("n_chembl")
    n_pubchem = pubchem.groupby("pathogen").size().rename("n_pubchem")
    counts = (
        pd.concat([n_chembl, n_pubchem], axis=1)
        .fillna(0).astype(int).reset_index()
    )
    counts["n_total"] = counts["n_chembl"] + counts["n_pubchem"]
    counts = counts.sort_values("n_total", ascending=True).reset_index(drop=True)
    counts["name"] = counts["pathogen"].map(code_to_name)

    pathogen_order = counts["pathogen"].tolist()
    pathogen_to_y  = {p: i for i, p in enumerate(pathogen_order)}

    rng = np.random.default_rng(RANDOM_SEED)
    def _add_jitter(df):
        df = df.copy()
        df["name"] = df["pathogen"].map(code_to_name)
        df["pathogen_rank"] = df["pathogen"].map(pathogen_to_y)
        df = df.dropna(subset=["pathogen_rank"])
        df["pathogen_y_jitter"] = (
            df["pathogen_rank"]
            + rng.uniform(-JITTER_STRENGTH, JITTER_STRENGTH, size=len(df))
        )
        return df

    chembl_pts  = _add_jitter(chembl)
    pubchem_pts = _add_jitter(pubchem)

    stylia.set_format("slide")
    stylia.set_style("article")

    pal = CategoricalPalette("npg")
    chembl_color  = pal.get(8)[0]
    pubchem_color = "#1f77b4"  # blue

    fig, axs = stylia.create_figure(1, 4, width=1.0, height=0.3)

    ax_a = axs.next()
    y = counts["pathogen"].map(pathogen_to_y).to_numpy()
    ax_a.barh(y, counts["n_chembl"], color=chembl_color, label="ChEMBL")
    ax_a.barh(y, counts["n_pubchem"], left=counts["n_chembl"],
              color=pubchem_color, label="PubChem")
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(counts["name"])
    ax_a.set_xlabel("Number of datasets")
    ax_a.set_ylabel("Pathogen")
    ax_a.legend(loc="lower right")
    stylia.label(ax_a, title="Datasets per pathogen")

    ax_b = axs.next()
    ax_b.sharey(ax_a)
    ax_b.scatter(chembl_pts["ratio"], chembl_pts["pathogen_y_jitter"],
                 color=chembl_color, s=stylia.MARKERSIZE_SMALL, alpha=0.75,
                 label="ChEMBL")
    ax_b.scatter(pubchem_pts["ratio"], pubchem_pts["pathogen_y_jitter"],
                 color=pubchem_color, s=stylia.MARKERSIZE_SMALL, alpha=0.75,
                 label="PubChem")
    ax_b.set_xlabel("Active ratio")
    ax_b.set_ylabel("")
    ax_b.set_xlim([-0.05, 1.05])
    ax_b.tick_params(axis="y", left=False, labelleft=False)
    stylia.label(ax_b, title="Active ratio per dataset")

    ax_c = axs.next()
    ax_c.sharey(ax_a)
    ax_c.scatter(chembl_pts["compounds"], chembl_pts["pathogen_y_jitter"],
                 color=chembl_color, s=stylia.MARKERSIZE_SMALL, alpha=0.75,
                 label="ChEMBL")
    ax_c.scatter(pubchem_pts["compounds"], pubchem_pts["pathogen_y_jitter"],
                 color=pubchem_color, s=stylia.MARKERSIZE_SMALL, alpha=0.75,
                 label="PubChem")
    ax_c.set_xscale("log")
    ax_c.set_xlabel("Number of compounds")
    ax_c.set_ylabel("")
    ax_c.tick_params(axis="y", left=False, labelleft=False)
    stylia.label(ax_c, title="Compounds per dataset")

    ax_d = axs.next()
    ax_d.sharey(ax_a)
    all_pts = pd.concat([chembl_pts, pubchem_pts], ignore_index=True)
    label_order  = ["A", "B", "M", "G"]
    label_colors = dict(zip(label_order, pal.get(len(label_order))))
    label_counts = (
        all_pts.groupby(["pathogen", "label"]).size().unstack(fill_value=0)
        .reindex(columns=label_order, fill_value=0)
        .reindex(pathogen_order, fill_value=0)
    )
    y = np.array([pathogen_to_y[p] for p in label_counts.index])
    left = np.zeros(len(y))
    for L in label_order:
        vals = label_counts[L].to_numpy()
        ax_d.barh(y, vals, left=left, color=label_colors[L], label=L)
        left = left + vals
    ax_d.set_xlabel("Number of datasets")
    ax_d.set_ylabel("")
    ax_d.tick_params(axis="y", left=False, labelleft=False)
    ax_d.legend(loc="lower right")
    stylia.label(ax_d, title="Dataset types per pathogen")

    save_figure(FIG_PATH)
    print(f"Saved figure: {FIG_PATH}")


if __name__ == "__main__":
    main()
