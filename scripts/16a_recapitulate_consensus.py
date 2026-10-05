"""
Step 16a — Consensus recapitulation per pathogen, on DrugBank and on the reference library.

For each model, measures how well the consensus (weighted and unweighted, from step 14)
recapitulates that model's own ranking — both when the model is left out of the consensus and when
it is part of it. Metrics and depths are those of step 15 (src/recapitulation.py): Spearman,
Pearson, hit overlap at the top 0.1% / 1% / 5%, and the AUROC of the consensus scoring the
molecules the model puts in its top 0.1% / 1% / 5%.

The consensus used is the one step 14 puts on the LazyQSAR rank scale (`consensus_rank`). The
mapping from raw consensus to rank is monotone, so Spearman, hit overlap and AUROC are identical on
the raw consensus; only Pearson differs. Pass --raw to use `consensus_raw` instead (files then get
a `_raw` suffix so the two variants do not overwrite each other).

On the reference library the consensus is calibrated by construction (1% of it at rank >= 0.65),
but these metrics only read the order of the molecules, so the calibration does not enter them.

Inputs, per pathogen and dataset (drugbank, reference):
  output/12_{drugbank,reference}/rank/{pathogen}.csv         — rank per model (step 12)
  output/14_consensus/{pathogen}/{dataset}_rank.csv          — weighted consensus (default)
  output/14_consensus/{pathogen}/{dataset}_unweighted_rank.csv
  output/14_consensus/{pathogen}/{dataset}_raw.csv           (--raw)
  output/14_consensus/{pathogen}/{dataset}_unweighted_raw.csv

Outputs in output/16_recapitulate_consensus/{pathogen}/ (`_raw` appended to the name with --raw):
  {dataset}_exc_weighted.csv    — model vs the weighted consensus WITHOUT that model
  {dataset}_exc_unweighted.csv  — model vs the unweighted consensus WITHOUT that model
  {dataset}_weighted.csv        — model vs the full weighted consensus
  {dataset}_unweighted.csv      — model vs the full unweighted consensus

Each file: model | spearman | pearson | top_k_0.1pct | top_k_1pct | top_k_5pct |
           hit_overlap_0.1pct | hit_overlap_1pct | hit_overlap_5pct |
           auroc_0.1pct | auroc_1pct | auroc_5pct

The top-k sets hold exactly k molecules, so when several molecules tie at the k-th score (the
consensus is written with 6 decimals) which of them are in the set is arbitrary. That can move a
hit overlap by one molecule; AUROC counts every tied molecule as a positive and is not affected.

Safeguards, so that a partial or stale run cannot pass unnoticed. The script exits with code 1 if a
pathogen with two or more retained models has no step-14 folder. It raises if the step-12 file and
10_reports.csv list different models, or if the step-14 files were not built from the step-12 files
it reads: different columns or models, a reference rank file whose SHA-256 is not the one recorded
in anchors.json, or an unweighted consensus that is not the mean of the step-12 ranks.

Usage:
    python scripts/16a_recapitulate_consensus.py
    python scripts/16a_recapitulate_consensus.py --pathogen ecoli
    python scripts/16a_recapitulate_consensus.py --raw
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "..", "src"))

import recapitulation as rc  # noqa: E402  (src/recapitulation.py)

# dataset name -> folder with the per-model rank predictions of step 12
DEFAULT_IN_DIRS_12 = {
    "drugbank":  os.path.join(root, "..", "output", "12_drugbank", "rank"),
    "reference": os.path.join(root, "..", "output", "12_reference", "rank"),
}
DEFAULT_IN_DIR_14 = os.path.join(root, "..", "output", "14_consensus")
DEFAULT_OUT_DIR   = os.path.join(root, "..", "output", "16_recapitulate_consensus")
REPORTS_PATH      = os.path.join(root, "..", "output", "10_reports", "10_reports.csv")
os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)


def _compute_rows(model_cols: list, profiles: dict, consensus_profiles: dict) -> list:
    """One row per model: the metrics of its consensus array (scorer) against the model (target)."""
    return [{"model": m, **rc.pair_metrics(consensus_profiles[m], profiles[m])} for m in model_cols]


def _sha256(path: str, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_consensus(path: str, smiles: pd.Series, what: str) -> pd.DataFrame:
    """A step-14 table, checked to be row-for-row the same molecules as the step-12 table."""
    df = pd.read_csv(path)
    if len(df) != len(smiles) or df["smiles"].tolist() != smiles.tolist():
        raise ValueError(f"{what}: the SMILES of {path} do not match the step-12 predictions row by row.")
    return df


def _check_step14(what: str, dir14: str, reference_file: str, df12: pd.DataFrame, model_cols: list,
                  col: str, tables: dict, unweighted_raw: pd.DataFrame) -> None:
    """Raise unless step 14 was built from exactly the step-12 models and files read here.

    The SMILES check cannot see a consensus that averages a different set of models, or one that was
    computed before step 12 was regenerated: the molecules are the same, the values are not.
    """
    expected = {"smiles", col} | {f"{col}_without_{m}" for m in model_cols}
    for name, df in tables.items():
        if set(df.columns) != expected:
            raise ValueError(
                f"{what}: the step-14 {name} table has different columns than the step-12 models imply. "
                f"Only in step 14: {sorted(set(df.columns) - expected)}; only expected: "
                f"{sorted(expected - set(df.columns))}. Re-run step 14.")

    with open(os.path.join(dir14, "anchors.json")) as f:
        anchors = json.load(f)
    if sorted(anchors["models"]) != sorted(model_cols):
        raise ValueError(f"{what}: anchors.json was built from other models than the step-12 file "
                         f"(only in anchors.json: {sorted(set(anchors['models']) - set(model_cols))}; "
                         f"only in step 12: {sorted(set(model_cols) - set(anchors['models']))}). Re-run step 14.")
    if anchors["reference"]["rank_file_sha256"] != _sha256(reference_file):
        raise ValueError(f"{what}: step 14 was built from a different reference rank file than {reference_file} "
                         "(it changed after step 14 ran). Re-run step 14.")

    # The plain mean of the ranks is the unweighted raw consensus, up to the 6-decimal rounding.
    deviation = np.abs(unweighted_raw["consensus_raw"].to_numpy() - df12[model_cols].mean(axis=1).to_numpy()).max()
    if deviation > 1e-5:
        raise ValueError(f"{what}: the unweighted consensus of step 14 is not the mean of the step-12 ranks "
                         f"(largest difference {deviation:.1e}); step 12 changed after step 14 ran. Re-run step 14.")


def run(pathogen: str, dataset: str, in_dir_12: str, reference_dir: str, in_dir_14: str, out_dir: str,
        report_models: list, scale: str = "rank") -> str:
    """Consensus recapitulation of one pathogen on one dataset.

    `report_models` are the retained models of the pathogen in 10_reports.csv. Returns 'ok', 'skip'
    (fewer than 2 retained models) or 'missing' (step 14 has no folder for the pathogen). Files that
    exist but do not belong together raise.

    scale is "rank" (consensus on the LazyQSAR scale) or "raw" (the weighted mean before anchoring).
    """
    sfx        = "" if scale == "rank" else f"_{scale}"
    col        = f"consensus_{scale}"
    src12      = os.path.join(in_dir_12, f"{pathogen}.csv")
    dir14      = os.path.join(in_dir_14, pathogen)
    src14_w    = os.path.join(dir14, f"{dataset}_{scale}.csv")
    src14_uw   = os.path.join(dir14, f"{dataset}_unweighted_{scale}.csv")
    src14_uwr  = os.path.join(dir14, f"{dataset}_unweighted_raw.csv")

    if len(report_models) < 2:
        print(f"  [SKIP] {pathogen} {dataset}: {len(report_models)} retained model(s) — requires at least 2")
        return "skip"
    if not os.path.isdir(dir14):
        print(f"  [MISSING] {pathogen} {dataset}: no step-14 folder {dir14}")
        return "missing"
    for src in (src12, src14_w, src14_uw, src14_uwr, os.path.join(dir14, "anchors.json")):
        if not os.path.isfile(src):
            raise FileNotFoundError(f"[{pathogen}] {dataset}: {src} not found, but step 14 produced a folder for it.")

    df12       = pd.read_csv(src12)
    model_cols = [c for c in df12.columns if c != "smiles"]
    if sorted(model_cols) != sorted(report_models):
        raise ValueError(
            f"[{pathogen}] the {dataset} rank file and 10_reports.csv list different models. Only in the "
            f"file: {sorted(set(model_cols) - set(report_models))}; only in the reports: "
            f"{sorted(set(report_models) - set(model_cols))}. Re-run 10a, then steps 12a and 12b.")

    nan_counts = df12[model_cols].isna().sum()
    if nan_counts.any():
        bad = nan_counts[nan_counts > 0].to_dict()
        raise ValueError(
            f"[{pathogen}] NaN predictions in step-12 {dataset} output: {bad}. "
            "Decide how to handle these (drop / impute / exclude pairwise) before scoring."
        )

    what    = f"[{pathogen}] {dataset}"
    df14_w  = _read_consensus(src14_w,  df12["smiles"], what)
    df14_uw = _read_consensus(src14_uw, df12["smiles"], what)

    profiles = {m: rc.Profile(df12[m].values) for m in model_cols}

    unweighted_raw = df14_uw if scale == "raw" else _read_consensus(src14_uwr, df12["smiles"], what)
    _check_step14(what, dir14, os.path.join(reference_dir, f"{pathogen}.csv"), df12, model_cols, col,
                  {"weighted": df14_w, "unweighted": df14_uw}, unweighted_raw)

    # The four consensus arrays per model: left out of the consensus, and included in it.
    # _check_step14 has made sure that step 14 wrote one leave-one-out column for every model.
    full_w  = rc.Profile(df14_w[col].values)
    full_uw = rc.Profile(df14_uw[col].values)
    outputs = [
        (f"{dataset}_exc_weighted{sfx}.csv",   {m: rc.Profile(df14_w[f"{col}_without_{m}"].values)  for m in model_cols}),
        (f"{dataset}_exc_unweighted{sfx}.csv", {m: rc.Profile(df14_uw[f"{col}_without_{m}"].values) for m in model_cols}),
        (f"{dataset}_weighted{sfx}.csv",       {m: full_w  for m in model_cols}),
        (f"{dataset}_unweighted{sfx}.csv",     {m: full_uw for m in model_cols}),
    ]

    pathogen_dir = os.path.join(out_dir, pathogen)
    os.makedirs(pathogen_dir, exist_ok=True)
    for filename, consensus_profiles in outputs:
        rows     = _compute_rows(model_cols, profiles, consensus_profiles)
        out_path = os.path.join(pathogen_dir, filename)
        pd.DataFrame(rows).round(4).to_csv(out_path, index=False)
        print(f"  [{pathogen}] {dataset}: {len(rows)} models, {len(df12)} molecules -> {out_path}")
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathogen",      default=None)
    parser.add_argument("--drugbank_dir",  default=DEFAULT_IN_DIRS_12["drugbank"],
                        help="Step-12 DrugBank rank folder.")
    parser.add_argument("--reference_dir", default=DEFAULT_IN_DIRS_12["reference"],
                        help="Step-12 reference-library rank folder.")
    parser.add_argument("--input_dir_14",  default=DEFAULT_IN_DIR_14)
    parser.add_argument("--output_dir",    default=DEFAULT_OUT_DIR)
    parser.add_argument("--raw", action="store_true",
                        help="Use the raw step-14 consensus (before the rank anchoring) instead of the rank.")
    args = parser.parse_args()

    in_dirs_12    = {"drugbank": args.drugbank_dir, "reference": args.reference_dir}
    reports_df    = pd.read_csv(REPORTS_PATH)
    report_models = reports_df.groupby("pathogen")["model_name"].apply(list).to_dict()
    if args.pathogen and args.pathogen not in report_models:
        parser.error(f"unknown pathogen '{args.pathogen}'; 10_reports.csv has: {', '.join(sorted(report_models))}")
    pathogens = [args.pathogen] if args.pathogen else list(dict.fromkeys(reports_df["pathogen"]))

    missing = []
    for pathogen in pathogens:
        for dataset, in_dir_12 in in_dirs_12.items():
            status = run(pathogen, dataset, in_dir_12, args.reference_dir, args.input_dir_14, args.output_dir,
                         report_models[pathogen], scale="raw" if args.raw else "rank")
            if status == "missing":
                missing.append(f"{pathogen} ({dataset})")

    if missing:
        print("\nNo step-14 output for: " + ", ".join(missing) + "\n(run step 14 for them first)")
        sys.exit(1)


if __name__ == "__main__":
    main()
