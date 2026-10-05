"""
Recapitulation metrics shared by step 15 (model vs model) and step 16a (consensus vs model).

Each score vector is turned once into a `Profile` (its ranks, its top-k sets), and a pair of
profiles is then compared in a few vector operations. The numbers are the ones the earlier
scripts computed with `scipy.stats.spearmanr`, `pearsonr` and `sklearn.metrics.roc_auc_score`:

  spearman   Pearson correlation of the average ranks
  pearson    Pearson correlation of the scores
  hit_overlap_{t}   molecules shared by the top-k of both vectors, k = max(1, ceil(t * n))
  auroc_{t}  AUROC of A scoring the molecules that B puts in its top t (B >= its k-th score,
             so ties at the cut are all positives), computed as the Mann-Whitney rank sum

Computing the ranks once per vector instead of once per pair makes a pair cost milliseconds
instead of about a second on the 50,000-molecule reference library.
"""

import math

import numpy as np
from scipy.stats import rankdata

from default import THRESHOLD_SFXS, THRESHOLDS


def top_k(n: int, t: float) -> int:
    """Number of molecules in the top fraction t of n."""
    return max(1, math.ceil(t * n))


def metric_columns() -> list:
    """Column names of a metrics row, in order (after the identifying columns)."""
    return (["spearman", "pearson"]
            + [f"top_k_{s}" for s in THRESHOLD_SFXS]
            + [f"hit_overlap_{s}" for s in THRESHOLD_SFXS]
            + [f"auroc_{s}" for s in THRESHOLD_SFXS])


def _standardize(v: np.ndarray):
    """Centred, unit-norm copy of v, or None for a constant vector (its correlation is undefined)."""
    c = v - v.mean()
    norm = np.sqrt(c @ c)
    return None if norm == 0.0 else c / norm


class Profile:
    """Everything about one score vector that the pairwise metrics reuse."""

    def __init__(self, scores):
        v = np.asarray(scores, dtype=float)
        if np.isnan(v).any():
            raise ValueError("NaN scores: decide how to handle them before computing metrics.")
        self.n      = len(v)
        self.ranks  = rankdata(v)                  # average ranks: ties share their mean rank
        self.z_val  = _standardize(v)
        self.z_rank = _standardize(self.ranks)

        order = np.argsort(v)[::-1]                # the same sort the previous scripts used
        self.k, self.top, self.labels = [], [], []
        for t in THRESHOLDS:
            k = top_k(self.n, t)
            top = np.zeros(self.n, dtype=bool)
            top[order[:k]] = True                  # exactly k molecules, ties broken by the sort
            cutoff = v[order[k - 1]]
            self.k.append(k)
            self.top.append(top)
            self.labels.append(v >= cutoff)        # all molecules at or above the k-th score


def pair_metrics(scorer: Profile, target: Profile) -> dict:
    """How well `scorer` recapitulates `target`: the row of metrics, without identifying columns.

    Spearman, Pearson and hit overlap are symmetric; the AUROC is not (the target is binarized).
    """
    if scorer.n != target.n:
        raise ValueError(f"score vectors differ in length: {scorer.n} vs {target.n}")
    n = scorer.n
    nan = float("nan")
    row = {
        "spearman": nan if scorer.z_rank is None or target.z_rank is None
                    else float(scorer.z_rank @ target.z_rank),
        "pearson":  nan if scorer.z_val is None or target.z_val is None
                    else float(scorer.z_val @ target.z_val),
    }
    for s, k in zip(THRESHOLD_SFXS, target.k):
        row[f"top_k_{s}"] = k
    for s, top_a, top_b in zip(THRESHOLD_SFXS, scorer.top, target.top):
        row[f"hit_overlap_{s}"] = int((top_a & top_b).sum())
    for s, labels in zip(THRESHOLD_SFXS, target.labels):
        n_pos = int(labels.sum())
        n_neg = n - n_pos
        if n_pos == 0 or n_neg == 0:
            row[f"auroc_{s}"] = nan
        else:
            rank_sum = scorer.ranks[labels].sum()
            row[f"auroc_{s}"] = float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    return row
