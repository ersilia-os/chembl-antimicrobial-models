"""
Step 14 — Consensus scoring of DrugBank compounds per pathogen.

Reads per-pathogen rank matrices from output/12_drugbank/rank/{pathogen}.csv
and computes a weighted consensus score per compound using 7 weights:
  W1–W6: model quality weights from 10_reports.csv
  W7:    0 at or below decision_cutoff_rank, linear 0→1 above it (per-compound)

weight[i,m] = average(W1..W6, W7[i,m], weights=W_WEIGHTS)
score[i]    = sum_m(prob_rank[i,m] * weight[i,m]) / sum_m(weight[i,m])

A tanh transformation is then applied to restore the IQR of the consensus scores
toward the average IQR of the individual model prob_ranks. The steepness k* is
solved directly per pathogen (no shared meta-curve across pathogens): the full
consensus column uses the pathogen's own k_star; each leave-one-out excluded_*
column uses the k_star_loo solved specifically for that exclusion (M-1 models).
Both are loaded from output/12_drugbank/12b_k_star.json, produced by
scripts/12b_fit_transformation.py. If the target IQR was unreachable for a given
column (see 12b), its k is the closest achievable approximation rather than an
exact match (flagged via k_star_exact/k_star_loo_exact and printed here) — every
column still gets transformed. Only if the whole pathogen is missing from
12b_k_star.json (12b hasn't been run for it) are its outputs left untransformed.
The center is fixed at 0.5 (neutral point of the prob_rank scale), making the
transformation independent of the compound set being scored. Dividing by tanh(k/2)
normalises the curve through [0,0] and [1,1], guaranteeing scores above 0.5 always
increase and scores below 0.5 always decrease. Ranks are preserved (tanh is strictly
monotone). Output is guaranteed in [0, 1] without clipping.

Output: output/14_consensus/{pathogen}.csv
        output/14_consensus/{pathogen}_unweighted.csv
        output/14_consensus/{pathogen}_transformed.csv
        output/14_consensus/{pathogen}_unweighted_transformed.csv
        smiles | excluded_{model} x N | consensus_score
        N+1 score columns: one per model exclusion, then full consensus last.

Usage:
    python scripts/14_consensus_scoring.py
    python scripts/14_consensus_scoring.py --pathogen ecoli
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

ROOT      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))

DEFAULT_IN_DIR  = os.path.join(REPO_ROOT, "output", "12_drugbank")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "output", "14_consensus")
REPORTS_PATH    = os.path.join(REPO_ROOT, "output", "10_reports", "10_reports.csv")
W_COLS    = ["w1", "w2", "w3", "w4", "w5", "w6"]  # quality weights from 10_reports
W_WEIGHTS = np.ones(len(W_COLS) + 1)  # + w7 (per-compound cutoff ramp); change here to reweight

# Per-pathogen tanh steepness (full-model k_star + per-exclusion k_star_loo),
# produced by scripts/12b_fit_transformation.py. No shared meta-curve across
# pathogens: {pathogen: {"k_star": float, "M": int, "k_star_loo": {model: float|null}}}
_K_STAR_PATH = os.path.join(REPO_ROOT, "output", "12_drugbank", "12b_k_star.json")
with open(_K_STAR_PATH) as _fh:
    _K_STAR_BY_PATHOGEN = json.load(_fh)


def _compute_w7(prob_ranks: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    """Linear weight above decision cutoff: 0 at or below cutoff, 1 at prob_rank=1."""
    c = np.clip(cutoffs[np.newaxis, :], 0.0, 1.0 - 1e-9)
    return np.where(
        prob_ranks <= c,
        0.0,
        (prob_ranks - c) / (1.0 - c),
    )


def _score(prob_ranks: np.ndarray, w_quality: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    # prob_ranks : (n_compounds, n_models) — normalized rank of each compound under each model
    # w_quality  : (n_models, 6)          — model-level quality weights (w1–w6) from 10_reports
    # cutoffs    : (n_models,)            — decision_cutoff_rank per model, used to compute w7

    # w7 is the only per-compound weight: it rewards compounds ranked above the decision cutoff
    w7 = _compute_w7(prob_ranks, cutoffs)  # (n_compounds, n_models)

    # Stack all 7 weights into a single tensor so we can average them in one call
    n_compounds, n_models = prob_ranks.shape
    w_all = np.empty((n_compounds, n_models, len(W_WEIGHTS)))
    w_all[:, :, :len(W_COLS)] = w_quality  # w1–w6: same for every compound, broadcast over axis 0
    w_all[:, :,  len(W_COLS)] = w7         # w7: varies per compound

    # Collapse the 7 weight dimensions into one scalar per (compound, model)
    w = np.average(w_all, axis=-1, weights=W_WEIGHTS)  # (n_compounds, n_models)

    # Weighted average of prob_ranks across models for each compound.
    # Guard compounds where every weight is 0 (all models rank at/below their cutoff and
    # all quality weights are 0): fall back to the plain mean rather than 0/0 = NaN.
    denom = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = (prob_ranks * w).sum(axis=1) / denom
    zero = denom == 0.0
    if zero.any():
        score[zero] = prob_ranks[zero].mean(axis=1)
    return score  # (n_compounds,)


def _score_unweighted(prob_ranks: np.ndarray) -> np.ndarray:
    return prob_ranks.mean(axis=1)


def _tanh_transform(x: np.ndarray, k: float) -> np.ndarray:
    return 0.5 + 0.5 * np.tanh(k * (x - 0.5)) / np.tanh(k / 2)


def _apply_transform(df: pd.DataFrame, k_by_column: dict, pathogen: str) -> pd.DataFrame:
    # Each column gets its own k (consensus_score -> k_star; excluded_{model} ->
    # that exclusion's k_star_loo), so the IQR-restoring strength matches the
    # exact set of models actually averaged into that column. 12b always
    # provides a usable k (falling back to the peak-achievable value rather
    # than NaN); a column missing from k_by_column entirely (stale/partial
    # 12b_k_star.json) is left untransformed instead of failing the pathogen.
    out = df.copy()
    for c in df.columns:
        if c == "smiles":
            continue
        k = k_by_column.get(c)
        if k is None:
            print(f"  [WARN] {pathogen}: no k_star for column '{c}' — left untransformed")
            out[c] = df[c].values.round(4)
        else:
            out[c] = _tanh_transform(df[c].values, k).round(4)
    return out


def run(pathogen: str, in_dir: str, reports_df: pd.DataFrame, out_path: str) -> None:
    src = os.path.join(in_dir, "rank", f"{pathogen}.csv")
    if not os.path.isfile(src):
        print(f"  [SKIP] {pathogen}: {src} not found")
        return

    df             = pd.read_csv(src)
    model_reports  = reports_df[reports_df["pathogen"] == pathogen].set_index("model_name")
    model_cols     = [col for col in df.columns if col != "smiles" and col in model_reports.index]

    if len(model_cols) < 2:
        print(f"  [SKIP] {pathogen}: {len(model_cols)} model(s) — consensus requires at least 2")
        return

    nan_counts = df[model_cols].isna().sum()
    if nan_counts.any():
        bad = nan_counts[nan_counts > 0].to_dict()
        raise ValueError(
            f"[{pathogen}] NaN predictions in step-12 output: {bad}. "
            "Decide how to handle these (drop / impute / exclude pairwise) before scoring."
        )
    prob_ranks = df[model_cols].values
    if prob_ranks.min() < -1e-6 or prob_ranks.max() > 1.0 + 1e-6:
        raise ValueError(
            f"[{pathogen}] predictions are not on the [0,1] rank scale "
            f"(min={prob_ranks.min():.3f}, max={prob_ranks.max():.3f}). "
            f"14 requires the 'rank' predict type ({src})."
        )
    w_quality  = np.array([model_reports.loc[m, W_COLS].values for m in model_cols], dtype=float)
    cutoffs    = np.array([model_reports.loc[m, "decision_cutoff_rank"] for m in model_cols], dtype=float)

    result = {"smiles": df["smiles"]}

    # Leave-one-out: score using all models except model i
    for i, model in enumerate(model_cols):
        other = [j for j in range(len(model_cols)) if j != i]
        result[f"excluded_{model}"] = _score(
            prob_ranks[:, other],
            w_quality[other, :],
            cutoffs[other],
        ).round(4)

    # Full consensus: score using all models
    result["consensus_score"] = _score(prob_ranks, w_quality, cutoffs).round(4)

    out = pd.DataFrame(result)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"  [{pathogen}] {len(model_cols)} models -> {len(out)} compounds -> {out_path}")

    result_unweighted = {"smiles": df["smiles"]}
    for i, model in enumerate(model_cols):
        other = [j for j in range(len(model_cols)) if j != i]
        result_unweighted[f"excluded_{model}"] = _score_unweighted(prob_ranks[:, other]).round(4)
    result_unweighted["consensus_score"] = _score_unweighted(prob_ranks).round(4)

    uw_df = pd.DataFrame(result_unweighted)
    unweighted_path = out_path.replace(".csv", "_unweighted.csv")
    uw_df.to_csv(unweighted_path, index=False)
    print(f"  [{pathogen}] unweighted -> {unweighted_path}")

    # --- tanh IQR-restoring transformation (per-pathogen k_star / k_star_loo, no meta-curve) ---
    entry = _K_STAR_BY_PATHOGEN.get(pathogen)
    if entry is None:
        print(f"  [SKIP transform] {pathogen}: no entry in 12b_k_star.json "
              "(run scripts/12b_fit_transformation.py first) — untransformed outputs only")
        return

    k_by_column = {"consensus_score": entry["k_star"]}
    for model in model_cols:
        k_by_column[f"excluded_{model}"] = entry["k_star_loo"].get(model)

    approx = [] if entry.get("k_star_exact", True) else ["consensus_score"]
    approx += [f"excluded_{m}" for m in model_cols if not entry.get("k_star_loo_exact", {}).get(m, True)]
    if approx:
        print(f"  [{pathogen}] using peak-achievable (non-exact) k for: {approx}")

    avg_model_iqr = float(np.mean([df[m].quantile(0.75) - df[m].quantile(0.25) for m in model_cols]))
    loo_ks        = [k_by_column[f"excluded_{m}"] for m in model_cols if k_by_column[f"excluded_{m}"] is not None]
    loo_range     = f"{min(loo_ks):.3f}-{max(loo_ks):.3f}" if loo_ks else "n/a"

    w_cons_iqr = float(out["consensus_score"].quantile(0.75) - out["consensus_score"].quantile(0.25))
    out_t      = _apply_transform(out, k_by_column, pathogen)
    t_path     = out_path.replace(".csv", "_transformed.csv")
    out_t.to_csv(t_path, index=False)
    w_iqr      = float(out_t["consensus_score"].quantile(0.75) - out_t["consensus_score"].quantile(0.25))
    print(f"  [{pathogen}] weighted transform:   k_star={entry['k_star']:.3f} "
          f"(k_star_loo range={loo_range})  "
          f"target_IQR={avg_model_iqr:.4f}  consensus_IQR={w_cons_iqr:.4f}  achieved_IQR={w_iqr:.4f}  -> {t_path}")

    uw_cons_iqr = float(uw_df["consensus_score"].quantile(0.75) - uw_df["consensus_score"].quantile(0.25))
    uw_t        = _apply_transform(uw_df, k_by_column, pathogen)
    ut_path     = unweighted_path.replace(".csv", "_transformed.csv")
    uw_t.to_csv(ut_path, index=False)
    uw_iqr      = float(uw_t["consensus_score"].quantile(0.75) - uw_t["consensus_score"].quantile(0.25))
    print(f"  [{pathogen}] unweighted transform: k_star={entry['k_star']:.3f} "
          f"(k_star_loo range={loo_range})  "
          f"target_IQR={avg_model_iqr:.4f}  consensus_IQR={uw_cons_iqr:.4f}  achieved_IQR={uw_iqr:.4f}  -> {ut_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathogen",   default=None)
    parser.add_argument("--input_dir",  default=DEFAULT_IN_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output",     default=None)
    args = parser.parse_args()

    reports_df = pd.read_csv(REPORTS_PATH)
    pathogens  = [args.pathogen] if args.pathogen else list(dict.fromkeys(reports_df["pathogen"]))

    for pathogen in pathogens:
        out_path = args.output or os.path.join(args.output_dir, f"{pathogen}.csv")
        run(pathogen, args.input_dir, reports_df, out_path)


if __name__ == "__main__":
    main()
