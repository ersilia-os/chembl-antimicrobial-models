"""
Consensus scoring on the LazyQSAR reference scale.

Pure functions: no file I/O and no printing. Step 14 (scripts/14_consensus_scoring.py) reads and
writes the files; step 18b imports the same functions to apply the anchors it ships and to check
that they still match the weights it ships, so the pipeline and the shipped model cannot drift apart.

The idea. A sub-model's `rank` is a position against a fixed library of 50,000 drug-like
molecules: rank 0.65 means "beats 99% of that library". The consensus is a weighted MEAN of several
such positions, which is not itself a position against anything. So we place it on the same scale
the same way: compute the consensus of the 50,000 reference molecules, read where it sits at the
reference's p50 / p90 / p99 / p99.9, and map those four values to 0.25 / 0.50 / 0.65 / 0.75 with
the piecewise-linear rule LazyQSAR uses for a single model (the anchor table).

The anchors and the decision rank are read from LazyQSAR, not copied, so the consensus scale is
the sub-models' scale by construction. Above the last reference anchor the table runs in a straight
line to (raw 1.0 -> rank 1.0): LazyQSAR's own fallback for a model with no out-of-fold-actives
anchor, which a consensus cannot have.
"""

import hashlib
import json

import numpy as np
from lazyqsar.utils.ranking import (
    DECISION_RANK,
    TAIL_ANCHORS,
    prepare_knots,
    reference_anchor_table,
)

ANCHOR_PERCENTILES = [float(q) for q, _ in TAIL_ANCHORS]   # [50, 90, 99, 99.9]
ANCHOR_RANKS       = [float(r) for _, r in TAIL_ANCHORS]   # [0.25, 0.50, 0.65, 0.75]

# Step 14, the README and the Hub model descriptions all say "rank 0.65 = beats 99% of drug-like
# chemistry". If a later LazyQSAR release moves the scale, stop here instead of silently publishing
# a different meaning under the same words.
if DECISION_RANK != 0.65 or ANCHOR_PERCENTILES != [50.0, 90.0, 99.0, 99.9] \
        or ANCHOR_RANKS != [0.25, 0.50, 0.65, 0.75]:
    raise ImportError(
        f"LazyQSAR's reference scale changed (TAIL_ANCHORS={TAIL_ANCHORS}, DECISION_RANK={DECISION_RANK}). "
        "The consensus text and thresholds in this repo assume 50/90/99/99.9 -> 0.25/0.50/0.65/0.75; "
        "review them before running step 14."
    )


# ---------------------------------------------------------------------------------------------
# The consensus itself (the recipe of the previous step 14, unchanged)
# ---------------------------------------------------------------------------------------------

def compute_w7(ranks: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    """Per-molecule weight: 0 at or below the model's decision cutoff, rising linearly to 1 at rank 1."""
    c = np.clip(cutoffs[np.newaxis, :], 0.0, 1.0 - 1e-9)
    return np.where(ranks <= c, 0.0, (ranks - c) / (1.0 - c))


def weighted_consensus(ranks: np.ndarray, w_quality: np.ndarray, cutoffs: np.ndarray,
                       w_weights: np.ndarray) -> np.ndarray:
    """Quality-weighted mean of the sub-models' ranks, one value per molecule.

    ranks     : (n_molecules, n_models)    each model's rank for each molecule
    w_quality : (n_models, n_quality)      model-level weights (QUALITY_WEIGHT_COLS in default.py)
    cutoffs   : (n_models,)                each model's decision_cutoff_rank
    w_weights : (n_quality + 1,)           how the quality weights and w7 combine (all ones = equal)

    weight[i, m] = average(quality weights of m, w7[i, m], weights=w_weights)
    consensus[i] = sum_m(ranks[i, m] * weight[i, m]) / sum_m(weight[i, m])

    A molecule whose weights are all zero (every model at or below its cutoff, and every quality
    weight zero) falls back to the plain mean instead of 0/0.
    """
    n, m = ranks.shape
    w_all = np.empty((n, m, len(w_weights)))
    w_all[:, :, :-1] = w_quality[np.newaxis, :, :]       # same for every molecule, broadcast
    w_all[:, :, -1] = compute_w7(ranks, cutoffs)          # depends on the molecule
    w = np.average(w_all, axis=-1, weights=w_weights)     # (n, m)
    denom = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = (ranks * w).sum(axis=1) / denom
    zero = denom == 0.0
    if zero.any():
        score[zero] = ranks[zero].mean(axis=1)
    return score


def plain_consensus(ranks: np.ndarray) -> np.ndarray:
    """Unweighted consensus: the plain mean of the sub-models' ranks (no quality weights, no w7)."""
    return ranks.mean(axis=1)


# ---------------------------------------------------------------------------------------------
# Anchoring on the reference library
# ---------------------------------------------------------------------------------------------

def anchor_values(raw_reference: np.ndarray) -> np.ndarray:
    """The raw consensus at the reference's p50 / p90 / p99 / p99.9, read the way LazyQSAR reads
    them (inverse interpolation of the tie-aware ECDF)."""
    vals, midranks = prepare_knots(raw_reference)
    return np.array([np.interp(q / 100.0, midranks, vals) for q in ANCHOR_PERCENTILES])


def build_anchor_table(raw_reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The piecewise-linear map (x = raw consensus, y = rank) for one set of models.

    Built by LazyQSAR's own `reference_anchor_table`: (0, 0), the four reference anchors, (1, 1).
    With no actives anchor, the table runs straight from the last reference anchor to (1, 1).
    Anchors that fall on the same value are collapsed (keeping the higher rank) so x is strictly
    increasing and the map stays monotone.
    """
    prepared = prepare_knots(raw_reference)
    x, y, _ = reference_anchor_table(prepared=prepared, anchor_high=None)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def apply_anchor_table(raw: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Place raw consensus values on the rank scale. Monotone, so the order of molecules is unchanged."""
    return np.clip(np.interp(raw, x, y), 0.0, 1.0)


def bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    """Resampling indices for `anchor_standard_errors`: n_boot draws of n molecules with replacement."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n), dtype=np.int32)


def anchor_standard_errors(raw_reference: np.ndarray, boot_idx: np.ndarray, chunk: int = 50) -> np.ndarray:
    """How much each of the four anchors would wobble with a different draw of reference molecules.

    The standard deviation of the anchors over the resamples in `boot_idx`. Diagnostic only: it
    never changes a score. The resamples use plain percentiles, which differ from LazyQSAR's
    tie-aware reading only in the last decimals; the spread is what is wanted here.
    """
    out = []
    for start in range(0, len(boot_idx), chunk):
        sample = raw_reference[boot_idx[start:start + chunk]]
        out.append(np.percentile(sample, ANCHOR_PERCENTILES, axis=1).T)
    return np.concatenate(out).std(axis=0, ddof=1)


# ---------------------------------------------------------------------------------------------
# Fingerprints: detect anchors that no longer match the weights they were built with
# ---------------------------------------------------------------------------------------------

def weights_fingerprint(models, weight_cols, w_quality, cutoffs, w_weights) -> str:
    """SHA-256 of everything the consensus weights depend on.

    Anchors describe the consensus of ONE set of models under ONE set of weights. If a model is
    retrained, or 10a is re-run with different weights, the anchors are stale. Step 14 stores this
    in the anchors JSON; step 18b recomputes it from the reports.csv it is about to ship and
    refuses to ship on a mismatch.
    """
    payload = {
        "models": list(models),
        "weight_cols": list(weight_cols),
        "w_quality": [[float(v) for v in row] for row in np.asarray(w_quality, dtype=float)],
        "cutoffs": [float(v) for v in np.asarray(cutoffs, dtype=float)],
        "w_weights": [float(v) for v in np.asarray(w_weights, dtype=float)],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def file_sha256(path: str, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()
