"""
Step 09c — AUROC delta between random (09) and scaffold (09b) CV splits.

For every dataset with a completed report in *both* output/09_reports/ (random
5-fold split) and output/09b_reports/ (scaffold-grouped split), computes the
per-model mean CV AUROC under each split and their delta:

    delta_auroc = auroc_random_mean - auroc_scaffold_mean

A positive delta means the random split overestimated generalization relative
to the harder scaffold split. Datasets with a report on only one side (most
commonly: still pending in the 09b SLURM array job) are excluded from the
comparison and counted in the console summary, not guessed.

Outputs to output/09c_scaffold_vs_random/:
  - 09c_delta_auroc.csv  — one row per included model, with its final (post-balancing)
    training set size (n_compounds, from 07_datasets_metadata.csv)
  - 09c_scaffold_vs_random.png — jittered scatter, delta AUROC per pathogen; dot color
    is pathogen (same npg palette/order as ersilia-model-hub-paper's
    xx_chembl_models_drugbank.py) and dot size is proportional to sqrt(training set size)

Usage:
    python scripts/09c_plot_scaffold_vs_random.py
"""

import os
import sys

import numpy as np
import pandas as pd
import stylia
from stylia import CategoricalPalette, save_figure

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "..", "src"))
from default import PATHOGENS, RANDOM_SEED  # noqa: E402

METADATA_PATH  = os.path.join(root, "..", "output", "07_datasets", "07_datasets_metadata.csv")
REPORTS_09_DIR  = os.path.join(root, "..", "output", "09_reports")
REPORTS_09B_DIR = os.path.join(root, "..", "output", "09b_reports")
PATHOGENS_CSV   = os.path.join(root, "..", "config", "pathogens.csv")
OUT_DIR         = os.path.join(root, "..", "output", "09c_scaffold_vs_random")
os.makedirs(OUT_DIR, exist_ok=True)

JITTER = 0.15
MARKER_SIZE_RANGE = (10.0, 200.0)  # matplotlib scatter `s` (points^2), rendering-only
LEGEND_SIZES = [100, 1_000, 10_000]  # reference training-set sizes shown in the size legend


def _mean_auroc(reports_dir: str, pathogen: str, name: str):
    path = os.path.join(reports_dir, pathogen, f"{name}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    mean = df["auroc"].mean()
    return None if pd.isna(mean) else round(float(mean), 4)


def build_delta_table() -> pd.DataFrame:
    meta = pd.read_csv(METADATA_PATH)

    records = []
    excluded = {p: {"missing_09": 0, "missing_09b": 0} for p in PATHOGENS}
    for row in meta.itertuples():
        pathogen, name = row.pathogen, str(row.name)
        auroc_random   = _mean_auroc(REPORTS_09_DIR, pathogen, name)
        auroc_scaffold = _mean_auroc(REPORTS_09B_DIR, pathogen, name)

        if auroc_random is None and auroc_scaffold is None:
            continue  # untrainable in both splits (same underlying class-balance rule)
        if auroc_random is None:
            excluded[pathogen]["missing_09"] += 1
            continue
        if auroc_scaffold is None:
            excluded[pathogen]["missing_09b"] += 1
            continue

        records.append({
            "pathogen": pathogen,
            "name": name,
            "n_compounds": int(row.final_compounds),
            "auroc_random_mean": auroc_random,
            "auroc_scaffold_mean": auroc_scaffold,
            "delta_auroc": round(auroc_random - auroc_scaffold, 4),
        })

    print("Per-pathogen coverage (included / missing in 09b / missing in 09):")
    included_counts = pd.Series([r["pathogen"] for r in records]).value_counts()
    for pathogen in PATHOGENS:
        n_incl = int(included_counts.get(pathogen, 0))
        n_miss_09b = excluded[pathogen]["missing_09b"]
        n_miss_09  = excluded[pathogen]["missing_09"]
        print(f"  {pathogen:15s} included={n_incl:3d}  missing_09b={n_miss_09b:2d}  missing_09={n_miss_09:2d}")

    return pd.DataFrame(records)


def _sqrt_range(n_compounds: pd.Series) -> tuple:
    """sqrt(n_compounds) min/max over the plotted data — the fixed scale that both the
    real points and the (fixed-value) size-legend markers are normalized against."""
    sqrt_n = np.sqrt(n_compounds.to_numpy(dtype=float))
    return sqrt_n.min(), sqrt_n.max()


def _marker_sizes(n_compounds, sqrt_lo: float, sqrt_hi: float) -> np.ndarray:
    """Marker area scaled by sqrt(n_compounds) (area ~ value, the standard bubble-chart
    convention), min-max normalized to MARKER_SIZE_RANGE using the given sqrt(n) range."""
    sqrt_n = np.sqrt(np.asarray(n_compounds, dtype=float))
    s_lo, s_hi = MARKER_SIZE_RANGE
    if sqrt_hi == sqrt_lo:
        return np.full(len(sqrt_n), (s_lo + s_hi) / 2)
    return s_lo + (sqrt_n - sqrt_lo) / (sqrt_hi - sqrt_lo) * (s_hi - s_lo)


def plot_delta(df: pd.DataFrame) -> str:
    # Format: slide | Style: article — matches every other plotting script in this repo.
    stylia.set_format("slide")
    stylia.set_style("article")
    rng = np.random.default_rng(RANDOM_SEED)

    # Same palette + pathogen order as ersilia-model-hub-paper's
    # xx_chembl_models_drugbank.py, so a pathogen's color is consistent across repos.
    pal = CategoricalPalette("npg")
    pathogen_color = dict(zip(PATHOGENS, pal.get(len(PATHOGENS))))

    sqrt_lo, sqrt_hi = _sqrt_range(df["n_compounds"])
    sizes = pd.Series(_marker_sizes(df["n_compounds"], sqrt_lo, sqrt_hi), index=df.index)

    # Single wide panel (categorical x, 15 pathogens) — omit width/height so stylia
    # uses its default wide sizing, not the square sizing meant for ROC/heatmap-style
    # same-scale x/y panels.
    fig, axs = stylia.create_figure(1, 1)
    ax = axs.next()

    for i, pathogen in enumerate(PATHOGENS):
        sub = df[df["pathogen"] == pathogen]
        if sub.empty:
            continue
        x = np.full(len(sub), i) + rng.uniform(-JITTER, JITTER, size=len(sub))
        ax.scatter(x, sub["delta_auroc"], s=sizes.loc[sub.index],
                   color=pathogen_color[pathogen], alpha=0.7, lw=0)

    ax.axhline(0.0, color="k", lw=0.7, linestyle="dashed")
    ax.set_xticks(range(len(PATHOGENS)))
    ax.set_xticklabels(PATHOGENS, rotation=90, ha="center")
    ax.set_xlim(-0.5, len(PATHOGENS) - 0.5)

    stylia.label(ax, xlabel="Pathogen", ylabel="Delta AUROC (random - scaffold)")

    # Marker-size legend: fixed reference sizes, scaled with the same sqrt-range as the
    # real points so the legend markers are directly comparable to the plotted ones.
    legend_n = np.array(LEGEND_SIZES)
    legend_s = _marker_sizes(legend_n, sqrt_lo, sqrt_hi)
    size_handles = [
        ax.scatter([], [], s=s, color="gray", alpha=0.7, lw=0, label=f"n={n:,}")
        for n, s in zip(legend_n, legend_s)
    ]
    ax.legend(handles=size_handles, title="Training set size",
              fontsize=5, title_fontsize=5, loc="upper right",
              frameon=True, framealpha=0.85, handletextpad=0.6, borderpad=0.4)

    out_path = os.path.join(OUT_DIR, "09c_scaffold_vs_random.png")
    save_figure(out_path)
    return out_path


def main() -> None:
    df = build_delta_table()
    if df.empty:
        print("No datasets with reports in both 09 and 09b — nothing to plot.")
        return

    csv_path = os.path.join(OUT_DIR, "09c_delta_auroc.csv")
    df.to_csv(csv_path, index=False)
    print(f"{len(df)} models included -> {csv_path}")

    png_path = plot_delta(df)
    print(f"Plot saved: {png_path}")


if __name__ == "__main__":
    main()
