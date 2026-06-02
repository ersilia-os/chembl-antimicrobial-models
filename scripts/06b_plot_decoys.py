"""
Step 06b — Build decoy-quality comparison groups and scaffold the figure.

Loads the aggregated eos3e6s decoy table (output/06_decoys/06_eos3e6s_v1.csv),
where each row is one reference compound (`input`) plus 100 generated decoys
(`smi_00`..`smi_99`). It then assembles three groups for a later
physicochemical-matching comparison:

  - references       : N reference compounds (random seeded sample of rows)
  - assigned decoys  : 10 random decoys per reference (drawn from that row's
                       own smi_* columns)            -> N * 10 SMILES
  - random compounds : N * 10 reference compounds (`input`) sampled at random
                       across the ENTIRE table       -> N * 10 SMILES

The big table is released from memory before plotting. The 2x3 stylia figure
shows, for each reference, how decoys compare to random reference compounds:
  - panel 1 : Tanimoto similarity (Morgan/ECFP4) to the reference
  - panels 2-6 : absolute per-pair difference in MW, LogP, HBA, HBD, and
                 rotatable bonds
Each panel overlays the ref-decoy vs ref-random distributions.

Reproducibility: row selection, decoy assignment, and random sampling all use
RANDOM_SEED from src/default.py.

Usage:
    python scripts/06b_plot_decoys.py
    python scripts/06b_plot_decoys.py --compounds 200
"""

import argparse
import gc
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "..", "src"))

from default import DECOY_MODEL, RANDOM_SEED  # noqa: E402

import stylia  # noqa: E402
from stylia import CategoricalPalette, save_figure  # noqa: E402


INPUT_PATH = os.path.join(root, "..", "output", "06_decoys", f"06_{DECOY_MODEL}_v1.csv")
OUTPUT_DIR = os.path.join(root, "..", "output", "06_decoys")
FIG_PATH = os.path.join(OUTPUT_DIR, "06b_decoys.png")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_COMPOUNDS = 100
DECOYS_PER_COMPOUND = 10
REFERENCE_COL = "input"
DECOY_COLS = [f"smi_{i:02d}" for i in range(100)]

# Fingerprint for Tanimoto similarity: Morgan radius 2, 2048 bits (ECFP4).
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
_MFPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_NBITS)


def _fingerprint(smiles):
    """Morgan fingerprint, or None if the SMILES does not parse."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _MFPGEN.GetFingerprint(mol)


# Physicochemical descriptors: (key, axis label, RDKit function, x-upper, binsize).
# x-upper and binsize set the |Δ| histogram range/resolution for that panel.
PHYSCHEM = [
    ("mw", "Molecular weight", Descriptors.MolWt, 1000.0, 50.0),
    ("logp", "LogP", Descriptors.MolLogP, 7.5, 0.1),
    ("hba", "H-bond acceptors", Descriptors.NumHAcceptors, 20.0, 1.0),
    ("hbd", "H-bond donors", Descriptors.NumHDonors, 20.0, 1.0),
    ("rb", "Rotatable bonds", Descriptors.NumRotatableBonds, 20.0, 1.0),
]

# Tanimoto similarity panel: x-upper and binsize.
TANIMOTO_XHI = 0.4
TANIMOTO_BINSIZE = 0.01


def abs_diff_distributions(props, key, n_ref):
    """Absolute per-pair difference in descriptor `key` between each reference
    and its partners. Returns (ref-decoy, ref-random) flat (N*K,) arrays.

    NaN propagates from any unparsable molecule and is filtered at plot time.
    """
    ref = props["references"][key][:, None]            # (N, 1)
    dec = props["decoys"][key].reshape(n_ref, -1)       # (N, K)
    rnd = props["random"][key].reshape(n_ref, -1)       # (N, K)
    d_decoy = np.abs(ref - dec).reshape(-1)
    d_random = np.abs(ref - rnd).reshape(-1)
    return d_decoy, d_random


def compute_physchem(smiles_array, desc="physchem"):
    """Compute the PHYSCHEM descriptors for an array of SMILES.

    Returns a dict {key: float array} aligned with smiles_array; unparsable
    SMILES yield NaN across all descriptors (recorded, not dropped).
    """
    n = len(smiles_array)
    out = {key: np.full(n, np.nan) for key, *_ in PHYSCHEM}
    for i, smi in enumerate(tqdm(smiles_array, desc=desc, unit="mol")):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for key, _, fn, *_ in PHYSCHEM:
            out[key][i] = fn(mol)
    return out


def tanimoto_to_partners(references, partners):
    """Tanimoto similarity of each reference to each of its K partner SMILES.

    references: (N,) array of SMILES; partners: (N, K) array of SMILES.
    Returns a flat (N*K,) float array; entries where either molecule fails to
    parse are NaN (recorded, not dropped).
    """
    sims = np.full(partners.shape, np.nan)
    ref_fps = [_fingerprint(s) for s in references]
    for i, rfp in enumerate(tqdm(ref_fps, desc="Tanimoto", unit="ref")):
        if rfp is None:
            continue
        for j in range(partners.shape[1]):
            pfp = _fingerprint(partners[i, j])
            if pfp is None:
                continue
            sims[i, j] = DataStructs.TanimotoSimilarity(rfp, pfp)
    return sims.reshape(-1)


def build_groups(df: pd.DataFrame, n_compounds: int, rng: np.random.Generator):
    """Return (references, assigned_decoys, random_compounds) as 1-D arrays.

    assigned_decoys holds n_compounds * DECOYS_PER_COMPOUND SMILES; each
    reference contributes DECOYS_PER_COMPOUND decoys drawn from its own row.
    random_compounds holds the same number of `input` SMILES sampled across the
    full table.
    """
    n_total = len(df)
    if n_compounds > n_total:
        print(f"[WARN] Requested {n_compounds} compounds but only {n_total} rows "
              f"available; clamping to {n_total}.")
        n_compounds = n_total

    selected = df.sample(n=n_compounds, random_state=RANDOM_SEED)
    references = selected[REFERENCE_COL].to_numpy()

    # 10 random decoys per reference, drawn from that row's own decoy columns.
    decoy_matrix = selected[DECOY_COLS].to_numpy()
    assigned = np.empty((n_compounds, DECOYS_PER_COMPOUND), dtype=object)
    for i in range(n_compounds):
        idx = rng.choice(len(DECOY_COLS), size=DECOYS_PER_COMPOUND, replace=False)
        assigned[i] = decoy_matrix[i, idx]
    assigned_decoys = assigned.reshape(-1)

    # Random comparison group: reference compounds sampled across the whole table.
    n_random = n_compounds * DECOYS_PER_COMPOUND
    all_inputs = df[REFERENCE_COL].to_numpy()
    rand_idx = rng.choice(len(all_inputs), size=n_random, replace=False)
    random_compounds = all_inputs[rand_idx]

    return references, assigned_decoys, random_compounds


def main(n_compounds: int) -> None:
    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns.")

    rng = np.random.default_rng(RANDOM_SEED)
    references, assigned_decoys, random_compounds = build_groups(df, n_compounds, rng)

    n_pool = len(df)
    # Release the large table before plotting.
    del df
    gc.collect()

    n_ref = len(references)
    decoys_2d = assigned_decoys.reshape(n_ref, DECOYS_PER_COMPOUND)
    random_2d = random_compounds.reshape(n_ref, DECOYS_PER_COMPOUND)

    # Tanimoto similarity of each reference to its decoys and to its randoms.
    ref_decoy_sims = tanimoto_to_partners(references, decoys_2d)
    ref_random_sims = tanimoto_to_partners(references, random_2d)

    # Physicochemical descriptors for each group.
    props = {
        "references": compute_physchem(references),
        "decoys": compute_physchem(assigned_decoys),
        "random": compute_physchem(random_compounds),
    }

    print("\n=== Summary ===")
    print(f"Model:                {DECOY_MODEL}")
    print(f"Random seed:          {RANDOM_SEED}")
    print(f"Rows in full table:   {n_pool:,}")
    print(f"References:           {n_ref:,}")
    print(f"Decoys per reference: {DECOYS_PER_COMPOUND}")
    print(f"Assigned decoys:      {len(assigned_decoys):,}")
    print(f"Random compounds:     {len(random_compounds):,} (sampled from `{REFERENCE_COL}`)")
    print(f"Fingerprint:          Morgan r={MORGAN_RADIUS}, {MORGAN_NBITS} bits (ECFP4)")
    print(f"Tanimoto ref-decoy:   mean {np.nanmean(ref_decoy_sims):.3f}  "
          f"({len(ref_decoy_sims):,} values, {np.isnan(ref_decoy_sims).sum()} NaN)")
    print(f"Tanimoto ref-random:  mean {np.nanmean(ref_random_sims):.3f}  "
          f"({len(ref_random_sims):,} values, {np.isnan(ref_random_sims).sum()} NaN)")

    print("\nMean |Δ| of diff distributions (ref-decoy / ref-random):")
    for key, label, *_ in PHYSCHEM:
        d_decoy, d_random = abs_diff_distributions(props, key, n_ref)
        md = np.nanmean(d_decoy)
        mr = np.nanmean(d_random)
        print(f"  {label:<18} {md:8.2f} / {mr:8.2f}")

    # 2x3 figure: panel 1 = Tanimoto similarity; panels 2-6 = per-property
    # absolute differences, each overlaying ref-decoy vs ref-random.
    stylia.set_format("slide")
    stylia.set_style("article")
    fig, axs = stylia.create_figure(2, 3, width=1.0, height=0.6)

    pal = CategoricalPalette("npg").get(8)
    c_decoy, c_random = pal[0], pal[1]

    ax = axs.next()
    sim_bins = np.arange(0, TANIMOTO_XHI + TANIMOTO_BINSIZE, TANIMOTO_BINSIZE)
    ax.hist(ref_decoy_sims[~np.isnan(ref_decoy_sims)], bins=sim_bins, color=c_decoy,
            alpha=0.6, density=True, label="ref-decoy")
    ax.hist(ref_random_sims[~np.isnan(ref_random_sims)], bins=sim_bins, color=c_random,
            alpha=0.6, density=True, label="ref-random")
    ax.set_xlabel("Tanimoto similarity to reference")
    ax.set_ylabel("Density")
    ax.set_xlim([0, TANIMOTO_XHI])
    ax.legend()
    stylia.label(ax, title="Reference similarity")

    for key, label, _, xhi, binsize in PHYSCHEM:
        ax = axs.next()
        d_decoy, d_random = abs_diff_distributions(props, key, n_ref)
        d_decoy = d_decoy[~np.isnan(d_decoy)]
        d_random = d_random[~np.isnan(d_random)]
        bins = np.arange(0, xhi + binsize, binsize)
        ax.hist(d_decoy, bins=bins, color=c_decoy, alpha=0.6, density=True, label="ref-decoy")
        ax.hist(d_random, bins=bins, color=c_random, alpha=0.6, density=True, label="ref-random")
        ax.set_xlabel(f"|Δ {label}|")
        ax.set_ylabel("Density")
        ax.set_xlim([0, xhi])
        if float(binsize).is_integer():
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        stylia.label(ax, title=label)

    save_figure(FIG_PATH)
    print(f"\nSaved figure: {FIG_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build decoy-quality comparison groups and scaffold the 2x3 figure."
    )
    parser.add_argument(
        "--compounds",
        type=int,
        default=DEFAULT_COMPOUNDS,
        help=f"Number of reference compounds (rows) to select (default {DEFAULT_COMPOUNDS}).",
    )
    args = parser.parse_args()
    main(n_compounds=args.compounds)
