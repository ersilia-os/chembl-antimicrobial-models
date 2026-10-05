"""
Step 15 — Pairwise model recapitulation per pathogen, on DrugBank and on the reference library.

For every ordered pair (A, B) of models, quantifies how well A's rank predictions relate to B's:
  - Spearman and Pearson correlation
  - Hit overlap: molecules shared by the top-k of A and of B
  - AUROC: B binarized at its top 0.1%, 1%, 5%, scored with A
The depth is a fraction of the dataset, k = max(1, ceil(t * n)), so the two datasets are read at
the same depth: DrugBank (n = 11,347) 12 / 114 / 568 molecules, reference library (n = 50,000)
50 / 500 / 2,500. The k used is stored in the top_k_* columns.

The two datasets answer different questions. DrugBank holds known drugs, enriched in real
bioactives: do the models agree on molecules that may truly be active? The reference library is
generic drug-like chemistry, almost all inactive: do the models agree across ordinary chemical
space? Every metric is rank-based, so none of them depends on the anchoring of step 14.

Input:  output/12_drugbank/rank/{pathogen}.csv
        output/12_reference/rank/{pathogen}.csv
Output: output/15_recapitulate_models/{pathogen}/drugbank.csv
        output/15_recapitulate_models/{pathogen}/reference.csv
        model_scorer | model_binarized | spearman | pearson |
        top_k_0.1pct | top_k_1pct | top_k_5pct |
        hit_overlap_0.1pct | hit_overlap_1pct | hit_overlap_5pct |
        auroc_0.1pct | auroc_1pct | auroc_5pct

Fails (exit code 1) if a pathogen with two or more retained models in 10_reports.csv has no rank
file, and raises if a rank file does not hold exactly the models 10_reports.csv lists for that
pathogen, so a partial run cannot pass unnoticed. Pathogens with a single retained model are skipped.

Usage:
    python scripts/15_recapitulate_models.py
    python scripts/15_recapitulate_models.py --pathogen ecoli
"""

import argparse
import os
import sys

import pandas as pd

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "..", "src"))

import recapitulation as rc  # noqa: E402  (src/recapitulation.py)

# dataset name -> folder with the per-model rank predictions of step 12
DEFAULT_IN_DIRS = {
    "drugbank":  os.path.join(root, "..", "output", "12_drugbank", "rank"),
    "reference": os.path.join(root, "..", "output", "12_reference", "rank"),
}
DEFAULT_OUT_DIR = os.path.join(root, "..", "output", "15_recapitulate_models")
REPORTS_PATH    = os.path.join(root, "..", "output", "10_reports", "10_reports.csv")
os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)


def run(pathogen: str, dataset: str, in_dir: str, out_dir: str, report_models: list) -> str:
    """Pairwise metrics for one pathogen on one dataset.

    `report_models` are the retained models of the pathogen in 10_reports.csv. Returns 'ok', 'skip'
    (fewer than 2 retained models) or 'missing' (the rank file is not there). A rank file that
    exists but does not hold exactly those models raises.
    """
    if len(report_models) < 2:
        print(f"  [SKIP] {pathogen} {dataset}: {len(report_models)} retained model(s) — pairwise requires at least 2")
        return "skip"

    src = os.path.join(in_dir, f"{pathogen}.csv")
    if not os.path.isfile(src):
        print(f"  [MISSING] {pathogen} {dataset}: {src} not found")
        return "missing"

    df         = pd.read_csv(src)
    model_cols = [c for c in df.columns if c != "smiles"]
    if sorted(model_cols) != sorted(report_models):
        raise ValueError(
            f"[{pathogen}] the {dataset} rank file and 10_reports.csv list different models. Only in the "
            f"file: {sorted(set(model_cols) - set(report_models))}; only in the reports: "
            f"{sorted(set(report_models) - set(model_cols))}. Re-run 10a, then steps 12a and 12b."
        )

    nan_counts = df[model_cols].isna().sum()
    if nan_counts.any():
        bad = nan_counts[nan_counts > 0].to_dict()
        raise ValueError(
            f"[{pathogen}] NaN predictions in step-12 {dataset} output: {bad}. "
            "Decide how to handle these (drop / impute / exclude pairwise) before scoring."
        )

    profiles = {m: rc.Profile(df[m].values) for m in model_cols}

    rows = []
    for m_a in model_cols:
        for m_b in model_cols:
            if m_a == m_b:
                continue
            rows.append({"model_scorer": m_a, "model_binarized": m_b,
                         **rc.pair_metrics(profiles[m_a], profiles[m_b])})

    pathogen_dir = os.path.join(out_dir, pathogen)
    os.makedirs(pathogen_dir, exist_ok=True)
    out_path = os.path.join(pathogen_dir, f"{dataset}.csv")
    pd.DataFrame(rows).round(4).to_csv(out_path, index=False)
    print(f"  [{pathogen}] {dataset}: {len(model_cols)} models, {len(df)} molecules -> {len(rows)} pairs -> {out_path}")
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathogen",         default=None)
    parser.add_argument("--drugbank_dir",     default=DEFAULT_IN_DIRS["drugbank"])
    parser.add_argument("--reference_dir",    default=DEFAULT_IN_DIRS["reference"])
    parser.add_argument("--output_dir",       default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    in_dirs       = {"drugbank": args.drugbank_dir, "reference": args.reference_dir}
    reports_df    = pd.read_csv(REPORTS_PATH)
    report_models = reports_df.groupby("pathogen")["model_name"].apply(list).to_dict()
    if args.pathogen and args.pathogen not in report_models:
        parser.error(f"unknown pathogen '{args.pathogen}'; 10_reports.csv has: {', '.join(sorted(report_models))}")
    pathogens = [args.pathogen] if args.pathogen else list(dict.fromkeys(reports_df["pathogen"]))

    missing = []
    for pathogen in pathogens:
        for dataset, in_dir in in_dirs.items():
            if run(pathogen, dataset, in_dir, args.output_dir, report_models[pathogen]) == "missing":
                missing.append(f"{pathogen} ({dataset})")

    if missing:
        print("\nMissing rank files for: " + ", ".join(missing) + "\n(steps 12a and 12b must finish for them first)")
        sys.exit(1)


if __name__ == "__main__":
    main()
