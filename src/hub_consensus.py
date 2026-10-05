"""Quality-weighted consensus across the LazyQSAR sub-models of this model, on the rank scale.

Shipped to every Ersilia Hub model as model/framework/code/consensus.py. Stand-alone on purpose
(numpy and pandas only: it runs inside the Hub model, outside the repository that makes it).
Step 18b of chembl-antimicrobial-models copies this file and bakes in the anchor table of the
pathogen; the same formulas live in that repository's src/consensus.py (step 14), and the two are
tested against each other.

Each sub-model's rank (predict_type="rank") is the position of a molecule against a fixed reference
library of 50,000 drug-like molecules: 0.65 means "scores higher than 99% of that library". A weighted
mean of such positions is not a position against anything, so it is placed on the same scale here:
step 14 computed the same weighted mean on the reference library, read its p50 / p90 / p99 / p99.9, and
mapped them to 0.25 / 0.50 / 0.65 / 0.75 (and 0 -> 0, 1 -> 1). That table is _ANCHOR_X -> _ANCHOR_Y
below, one per pathogen. Linear interpolation through it keeps the order of molecules and makes
consensus_score read like a sub-model's rank: 0.65 is the recommended threshold, and about 1% of
generic drug-like chemistry reaches it.

The weights:
- W1..W6 and W_screen are per-sub-model quality weights, from reports.csv.
- W7 is a per-compound weight that ramps from 0 to 1 above each sub-model's decision_cutoff_rank.
- The eight are averaged with equal weight into an effective weight per compound and sub-model; the
  raw consensus is the weighted mean of the ranks.

NaN policy: if any sub-model returns NaN for a compound, that compound's consensus_score is NaN (no
weighting, no averaging). The per-sub-model columns keep the real ranks where prediction succeeded
and NaN where it did not.
"""

import os
import numpy as np
import pandas as pd

_W_COLS = ["w1", "w2", "w3", "w4", "w5", "w6", "w_screen"]
_ANCHOR_X = None  # baked by step 18b: raw consensus at 0, p50, p90, p99, p99.9 of the reference, and 1
_ANCHOR_Y = None  # baked by step 18b: the ranks they map to: 0, 0.25, 0.50, 0.65, 0.75, 1


def weighted_mean(ranks, w_quality, cutoffs):
    """Raw consensus of a (n, M) matrix of ranks without NaN: the quality-weighted mean per row.

    w_quality: (M, len(_W_COLS)) model-level weights; cutoffs: (M,) decision_cutoff_rank of each model.
    """
    c = np.clip(cutoffs[np.newaxis, :], 0.0, 1.0 - 1e-9)
    w7 = np.where(ranks <= c, 0.0, (ranks - c) / (1.0 - c))

    n, M = ranks.shape
    n_w = len(_W_COLS) + 1
    w_all = np.empty((n, M, n_w))
    w_all[:, :, :len(_W_COLS)] = w_quality
    w_all[:, :, len(_W_COLS)] = w7
    w_eff = np.average(w_all, axis=-1, weights=np.ones(n_w))

    denom = w_eff.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = (ranks * w_eff).sum(axis=1) / denom
    zero = denom == 0.0          # all weights 0 for a compound -> fall back to the plain mean
    if zero.any():
        raw[zero] = ranks[zero].mean(axis=1)
    return raw


def to_rank_scale(raw, x, y):
    """Place raw consensus values on the rank scale through the anchor table (monotone)."""
    return np.clip(np.interp(raw, x, y), 0.0, 1.0)


def compute_consensus(R, cols_ordered, model_names, checkpoints_dir):
    """Build the model's output matrix.

    Args:
        R:               (n, K) rank matrix returned by lqsar_predict.
        cols_ordered:    list of K column names matching R's columns (also from lqsar_predict).
        model_names:     canonical sub-model order for this pathogen (length M, M <= K).
        checkpoints_dir: path to model/checkpoints/ (must contain reports.csv).

    Returns:
        results: (n, 1+M) float array, rounded to 4 decimals.
                 results[:, 0] is the consensus score on the rank scale.
                 results[:, 1:] is the per-sub-model rank reordered to match model_names.
        header:  ["consensus_score", *model_names].
    """
    name_to_idx = {c: i for i, c in enumerate(cols_ordered)}
    ranks = R[:, [name_to_idx[m] for m in model_names]].astype(float)

    # Single sub-model: the consensus would only be a copy of the sole rank. Skip it and emit only
    # the sub-model column.
    if len(model_names) == 1:
        return np.round(ranks, 4), list(model_names)

    if _ANCHOR_X is None or _ANCHOR_Y is None:
        raise RuntimeError("consensus.py has no anchor table baked in (step 18b fills it in).")

    reports = pd.read_csv(os.path.join(checkpoints_dir, "reports.csv")).set_index("model_name")
    w_quality = np.array([reports.loc[m, _W_COLS].values for m in model_names], dtype=float)
    cutoffs = np.array([reports.loc[m, "decision_cutoff_rank"] for m in model_names], dtype=float)

    # Compounds with any NaN rank get consensus = NaN; the rest go through the weighting and the
    # anchor table. Splitting like this keeps the NaN policy explicit and avoids invalid-value warnings.
    nan_rows = np.isnan(ranks).any(axis=1)
    consensus = np.full(ranks.shape[0], np.nan)
    if (~nan_rows).any():
        raw = weighted_mean(ranks[~nan_rows], w_quality, cutoffs)
        consensus[~nan_rows] = to_rank_scale(raw, _ANCHOR_X, _ANCHOR_Y)

    results = np.round(np.column_stack([consensus, ranks]), 4)
    header = ["consensus_score", *model_names]
    return results, header
