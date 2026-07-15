"""
Step 10a — Aggregate per-dataset CV reports into a single summarised file.

Iterates over datasets from 07_datasets/07_datasets_metadata.csv, validates that all
N_FOLDS folds are present, and applies a mean AUROC ≥ MIN_AUROC filter. Writes to
output/10_reports/:
  - 10_reports.csv          — one row per retained dataset with aggregated metrics + weights
  - 10_discarded_models.csv — datasets dropped for failing the AUROC threshold

Model files are keyed by the dataset `name` (unique per pathogen), so reports/models are
found directly with no positional recomputation.

Quality weight = mean of six 0–1 components (all guarded against NaN/inf):
  w1 real negatives    — 1 − (added_negatives + added_decoys) / n_negatives
                          (penalises negatives borrowed from other assays / decoy fallback)
  w2 mean CV AUROC     — 0 at ≤0.7, linear to 1 at 1.0
  w3 AUPRC enrichment  — absolute excess + fold enrichment over prevalence
  w4 BEDROC enrichment — absolute excess + fold enrichment over random
  w5 total compounds   — piecewise linear
  w6 total actives      — piecewise linear
(The old flat dataset-type weight was removed and the rest renumbered.) A per-compound
seventh weight w7 — the decision-cutoff ramp — is added only in the consensus (steps 12b/14).
final_normalized_weight rescales final_weight within each pathogen to sum to 100.

Usage:
    python scripts/10a_aggregate_reports.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
sys.path.append(os.path.join(ROOT, "..", "src"))

from default import DESCRIPTORS, MIN_AUROC, N_FOLDS  # noqa: E402

METADATA_PATH  = os.path.join(REPO_ROOT, "output", "07_datasets", "07_datasets_metadata.csv")
REPORTS_DIR    = os.path.join(REPO_ROOT, "output", "09_reports")
MODELS_DIR     = os.path.join(REPO_ROOT, "output", "09_models")
OUT_DIR        = os.path.join(REPO_ROOT, "output", "10_reports")
OUT_PATH       = os.path.join(OUT_DIR, "10_reports.csv")
DISCARDED_PATH = os.path.join(OUT_DIR, "10_discarded_models.csv")
os.makedirs(OUT_DIR, exist_ok=True)


def _dir_size_mb(path: str) -> float:
    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, files in os.walk(path)
        for f in files
    )
    return round(total / 1e6, 3)


def _piecewise_linear(x: float, knots: list[tuple[float, float]]) -> float:
    if x <= knots[0][0]:
        return knots[0][1]
    if x >= knots[-1][0]:
        return knots[-1][1]
    for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return knots[-1][1]


# Six quality weights (w1..w6), each 0–1; final_weight = their mean. The old flat
# dataset-type weight was removed and the rest renumbered (w1 = real-negative fraction, …).
def _w_real_neg(df: pd.DataFrame, n_added: int) -> float:
    """w1 — fraction of the negative class that is original data (not added from other
    assays or decoy fallback): 1 − n_added / n_negatives. 1.0 when nothing was added."""
    n_negatives = int(df["compounds_test"].sum()) - int(df["positives_test"].sum())
    if n_negatives <= 0:
        return 0.0
    return round(float(np.clip(1.0 - n_added / n_negatives, 0.0, 1.0)), 4)


def _w_auroc(df: pd.DataFrame) -> float:
    """w2 — mean CV AUROC: 0 at ≤0.7, linear to 1 at 1.0."""
    auroc = df["auroc"].mean()
    if auroc <= 0.7:
        return 0.0
    return round(min((auroc - 0.7) / 0.3, 1.0), 4)


def _enrichment_weight(value: float, baseline: float) -> float:
    """Two equal sub-scores (each 0–0.5, total 0–1):
      c1: absolute excess — 0 when value=baseline, 0.5 when value=1
      c2: fold enrichment — 0 at ≤1×, 0.5 at ≥10× over baseline
    Baseline is clamped to (0, 1) so degenerate prevalence can't produce inf/NaN."""
    baseline = float(min(max(baseline, 1e-9), 1.0 - 1e-9))
    c1 = 0.5 * float(np.clip((value - baseline) / (1.0 - baseline), 0.0, 1.0))
    c2 = 0.5 * float(np.clip((value / baseline - 1.0) / 9.0, 0.0, 1.0))
    return round(c1 + c2, 4)


def _w_auprc(df: pd.DataFrame) -> float:
    """w3 — AUPRC: absolute-excess (0–0.5) + fold-enrichment (0–0.5) over prevalence baseline."""
    return _enrichment_weight(df["auprc"].mean(), df["baseline_auprc"].mean())


def _w_bedroc(df: pd.DataFrame) -> float:
    """w4 — BEDROC: absolute-excess (0–0.5) + fold-enrichment (0–0.5) over random baseline."""
    return _enrichment_weight(df["bedroc"].mean(), df["baseline_bedroc"].mean())


_W_COMPOUNDS_KNOTS = [(100, 0.0), (1_000, 0.25), (10_000, 0.5), (100_000, 1.0)]


def _w_compounds(df: pd.DataFrame) -> float:
    """w5 — total compound count: piecewise linear from <100 → 0 to ≥100k → 1."""
    n = int(df["compounds_test"].sum())
    return round(_piecewise_linear(n, _W_COMPOUNDS_KNOTS), 4)


_W_ACTIVES_KNOTS = [(50, 0.0), (250, 0.25), (1_000, 0.5), (10_000, 1.0)]


def _w_actives(df: pd.DataFrame) -> float:
    """w6 — total active count: piecewise linear from <50 → 0 to ≥10k → 1."""
    n = int(df["positives_test"].sum())
    return round(_piecewise_linear(n, _W_ACTIVES_KNOTS), 4)


def aggregate(df: pd.DataFrame, pathogen: str, name: str, mrow) -> dict:
    model_name = name  # model files are keyed by the dataset name

    model_dir = os.path.join(MODELS_DIR, pathogen, model_name)
    meta_path = os.path.join(model_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            model_meta = json.load(f)
        decision_cutoff_rank = round(model_meta["decision_cutoff_rank"], 4)
        portfolio = ";".join(sorted(p.upper() for p in model_meta.get("portfolio", [])))
    else:
        decision_cutoff_rank = np.nan
        portfolio = np.nan
    model_size_total_mb = _dir_size_mb(model_dir) if os.path.isdir(model_dir) else np.nan

    _an = getattr(mrow, "added_negatives", 0)
    _ad = getattr(mrow, "added_decoys", 0)
    added_negatives = 0 if pd.isna(_an) else int(_an)   # pd.isna handles NaN/None (nan is truthy)
    added_decoys    = 0 if pd.isna(_ad) else int(_ad)
    n_added = added_negatives + added_decoys

    row = {"pathogen": pathogen, "name": name, "model_name": model_name}
    row["n_compounds"]      = int(df["compounds_test"].sum())
    row["n_positives"]      = int(df["positives_test"].sum())
    row["n_added_negatives"] = added_negatives
    row["n_added_decoys"]    = added_decoys

    for col in ("auroc", "auprc", "baseline_auprc", "bedroc", "baseline_bedroc"):
        row[f"{col}_mean"] = round(df[col].mean(), 4)
        row[f"{col}_std"]  = round(df[col].std(), 4)

    row["w1"] = _w_real_neg(df, n_added)
    row["w2"] = _w_auroc(df)
    row["w3"] = _w_auprc(df)
    row["w4"] = _w_bedroc(df)
    row["w5"] = _w_compounds(df)
    row["w6"] = _w_actives(df)
    row["final_weight"] = round(float(np.mean([row[f"w{i}"] for i in range(1, 7)])), 4)

    row["decision_cutoff_rank"] = decision_cutoff_rank
    row["portfolio"]            = portfolio
    row["model_size_total_mb"]  = model_size_total_mb

    for desc in DESCRIPTORS:
        col  = f"oof_auc_{desc}"
        vals = df[col].dropna()
        row[f"{col}_mean"] = round(vals.mean(), 4) if not vals.empty else np.nan
        row[f"{col}_std"]  = round(vals.std(),  4) if not vals.empty else np.nan

    return row


def _type_rank(mrow) -> int:
    """Display order within a pathogen: ChEMBL DR pool, SP pool, DR catch-all, SP catch-all,
    then PubChem merged, PubChem single."""
    if mrow.source == "pubchem":
        return 4 if bool(getattr(mrow, "is_merged", False)) else 5
    cat = 0 if mrow.label == "DR" else 1
    tier = 0 if getattr(mrow, "assay_type", "") == "pool" else 2
    return tier + cat


def main() -> None:
    meta_df = pd.read_csv(METADATA_PATH)
    n_total = len(meta_df)
    records = []
    discarded = []

    for i, mrow in enumerate(meta_df.itertuples(), start=1):
        pathogen, name = mrow.pathogen, str(mrow.name)
        prefix = f"[{i}/{n_total}]"

        report_path = os.path.join(REPORTS_DIR, pathogen, f"{name}.csv")
        if not os.path.exists(report_path):
            # No report: distinguish "untrainable" (step 09's CV guard skipped it because the
            # minority class < N_FOLDS — e.g. all-inactive pools kept only to enlarge the
            # negative pool) from a genuinely missing/failed training run.
            n_pos = int(mrow.positives)
            n_neg = int(mrow.final_compounds) - n_pos
            if min(n_pos, n_neg) < N_FOLDS:
                reason = f"untrainable: min class {min(n_pos, n_neg)} < {N_FOLDS} folds"
            else:
                reason = "no report found (training missing/failed)"
            print(f"{prefix} [WARN] {pathogen}/{name}: {reason}")
            discarded.append({"pathogen": pathogen, "name": name, "mean_auroc": np.nan, "reason": reason})
            continue

        df = pd.read_csv(report_path)
        if len(df) < N_FOLDS:
            reason = f"incomplete CV: {len(df)}/{N_FOLDS} folds"
            print(f"{prefix} [WARN] {pathogen}/{name}: {reason}, skipping")
            discarded.append({"pathogen": pathogen, "name": name, "mean_auroc": np.nan, "reason": reason})
            continue

        mean_auroc = round(df["auroc"].mean(), 4)
        if mean_auroc < MIN_AUROC:
            reason = f"mean AUROC {mean_auroc:.3f} < {MIN_AUROC}"
            print(f"{prefix} [SKIP] {pathogen}/{name}: {reason}, discarding")
            discarded.append({"pathogen": pathogen, "name": name, "mean_auroc": mean_auroc, "reason": reason})
            continue

        rec = aggregate(df, pathogen, name, mrow)
        rec["_type_rank"] = _type_rank(mrow)
        rec["_orig_compounds"] = int(mrow.compounds)
        records.append(rec)
        print(f"{prefix} {pathogen}/{name} processed!")

    disc_df = pd.DataFrame(discarded, columns=["pathogen", "name", "mean_auroc", "reason"])
    disc_df.to_csv(DISCARDED_PATH, index=False)
    print(f"{len(disc_df)} discarded models → {DISCARDED_PATH}")
    if len(disc_df):
        for kind, n in disc_df["reason"].str.split(":").str[0].str.split(" <").str[0].value_counts().items():
            print(f"    {n}× {kind}")

    if not records:
        print("No completed datasets found.")
        return

    out = pd.DataFrame(records)
    out = out.sort_values(
        ["pathogen", "_type_rank", "_orig_compounds"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    pathogen_totals = out.groupby("pathogen")["final_weight"].transform("sum")
    normalized = np.where(pathogen_totals > 0, out["final_weight"] / pathogen_totals * 100, 0.0)
    out.insert(out.columns.get_loc("final_weight") + 1, "final_normalized_weight", np.round(normalized, 4))

    out = out.drop(columns=["_type_rank", "_orig_compounds"])
    out.to_csv(OUT_PATH, index=False)
    print(f"{len(records)}/{n_total} datasets → {OUT_PATH}")

    n_nan_cutoff = int(out["decision_cutoff_rank"].isna().sum())
    print(f"decision_cutoff_rank NaN: {n_nan_cutoff}/{len(out)}")
    if n_nan_cutoff:
        missing = out.loc[out["decision_cutoff_rank"].isna(), ["pathogen", "name", "model_name"]]
        for _, r in missing.iterrows():
            print(f"  [no metadata.json] {r['pathogen']}/{r['model_name']} ({r['name']})")


if __name__ == "__main__":
    main()
