"""
Step 18 — Package per-pathogen EMH files for the Ersilia Model Hub.

For each pathogen, reads:
  - output/10_reports/10_reports.csv      — model quality metrics and weights
  - output/07_datasets/07_datasets_metadata.csv — assay details for column descriptions

And writes two files to output/18_emh_files/{pathogen}/:
  reports.csv     — model report, stripped of pipeline-internal columns; to be
                    committed as model/checkpoints/reports.csv in the Ersilia repo
  run_columns.csv — output column descriptions and recommended thresholds; to be
                    committed as model/framework/columns/run_columns.csv

Usage:
    python scripts/18_emh_files.py
    python scripts/18_emh_files.py --pathogen abaumannii
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "..", "src"))
from default import TANH_A, TANH_TAU

REPO_ROOT     = os.path.abspath(os.path.join(root, ".."))
REPORTS_PATH  = os.path.join(REPO_ROOT, "output", "10_reports", "10_reports.csv")
METADATA_PATH = os.path.join(REPO_ROOT, "output", "07_datasets", "07_datasets_metadata.csv")
OUTPUT_DIR    = os.path.join(REPO_ROOT, "output", "18_emh_files")

_DROP_COLS = ["predict_rank_actives", "predict_rank_inactives"]


# ---------------------------------------------------------------------------
# Consensus threshold helpers
# ---------------------------------------------------------------------------

def _k(n: int) -> float:
    """Tanh steepness for n models (saturating-exponential fit, same as script 14)."""
    return 2.0 * (1.0 + TANH_A * (1.0 - np.exp(-n / TANH_TAU)))


def _tanh_transform(x: float, k: float) -> float:
    return 0.5 + 0.5 * float(np.tanh(k * (x - 0.5))) / float(np.tanh(k / 2))


# ---------------------------------------------------------------------------
# run_columns description builders
# ---------------------------------------------------------------------------

def _cutoff_str(cutoff: float, unit: str) -> str:
    val = int(cutoff) if float(cutoff) % 1 == 0 else float(cutoff)
    if unit == "%":
        return f"{val}%"
    if unit == "umol.L-1":
        return f"{val} uM"
    return f"{val} {unit}"


def _measurement_phrase(activity_type: str, unit: str) -> str:
    if pd.isna(activity_type):
        return ""
    if unit == "%":
        if activity_type == "INHIBITION":
            return "inhibition %"
        if activity_type == "ACTIVITY":
            return "single-point % activity"
        if activity_type == "GI":
            return "growth inhibition %"
        if activity_type == "PERCENTEFFECT":
            return "percent effect"
        return activity_type.lower()
    return f"{activity_type} measurements"


def _sort_priority(meta_row: pd.Series, dataset_name: str) -> int:
    """Row order in run_columns.csv: SP aggregate, DR aggregate, other general,
    individual ChEMBL, individual PubChem, merged."""
    assay_type = meta_row["assay_type"]
    source     = meta_row.get("source", "chembl")
    if assay_type == "general_aggregate":
        return 0 if "SP" in dataset_name else 1
    if assay_type == "general":
        return 2
    if assay_type == "individual":
        return 3 if source == "chembl" else 4
    if assay_type == "merged":
        return 5
    return 6


def _build_description(meta_row: pd.Series, dataset_name: str, dcr: float) -> str:
    assay_type    = meta_row["assay_type"]
    activity_type = meta_row.get("activity_type", None)
    unit          = meta_row.get("unit", None)
    cutoff        = meta_row.get("cutoff", None)
    n_assays      = int(meta_row["n_assays"]) if not pd.isna(meta_row["n_assays"]) else None
    n_compounds   = int(meta_row["final_compounds"])
    decoys        = int(meta_row["decoys"]) if not pd.isna(meta_row.get("decoys", np.nan)) else 0

    decoys_str    = " incl. decoys" if decoys > 0 else ""
    threshold_str = f"Recommended threshold: {round(dcr, 3)}."

    if assay_type == "individual":
        assay_id = dataset_name.split("_")[0]
        phrase   = _measurement_phrase(activity_type, unit)
        cutoff_s = _cutoff_str(cutoff, unit)
        body = f"ChEMBL assay {assay_id} ({phrase}; cutoff {cutoff_s}; n={n_compounds})"
        return f"Probability from sub-model trained on {body}. {threshold_str}"

    if assay_type == "merged":
        phrase   = _measurement_phrase(activity_type, unit)
        cutoff_s = _cutoff_str(cutoff, unit)
        body = (
            f"{phrase} merged across {n_assays} ChEMBL assays "
            f"(cutoff {cutoff_s}; n={n_compounds}{decoys_str})"
        )
        return f"Probability from sub-model trained on {body}. {threshold_str}"

    if assay_type == "general":
        phrase   = _measurement_phrase(activity_type, unit)
        cutoff_s = _cutoff_str(cutoff, unit)
        body = (
            f"{phrase} aggregated across {n_assays} ChEMBL assays "
            f"(cutoff {cutoff_s}; n={n_compounds}{decoys_str})"
        )
        return f"Probability from sub-model trained on {body}. {threshold_str}"

    if assay_type == "general_aggregate":
        phrase = "dose-response measurements" if "DR" in dataset_name else "single-point activity measurements"
        body   = f"{phrase} aggregated across {n_assays} ChEMBL assays (n={n_compounds})"
        return f"Probability from sub-model trained on {body}. {threshold_str}"

    return f"Probability from sub-model trained on dataset {dataset_name} (n={n_compounds}). {threshold_str}"


# ---------------------------------------------------------------------------
# Per-pathogen run
# ---------------------------------------------------------------------------

def run(pathogen: str, reports_df: pd.DataFrame, meta_df: pd.DataFrame) -> None:
    df = reports_df[reports_df["pathogen"] == pathogen].copy()
    if df.empty:
        print(f"  [SKIP] {pathogen}: no rows in reports")
        return

    out_dir = os.path.join(OUTPUT_DIR, pathogen)
    os.makedirs(out_dir, exist_ok=True)

    # --- reports.csv ---
    cols_to_drop = [c for c in _DROP_COLS if c in df.columns]
    df.drop(columns=cols_to_drop).to_csv(os.path.join(out_dir, "reports.csv"), index=False)

    # --- run_columns.csv ---
    n            = len(df)
    k            = _k(n)
    wavg_cutoff  = float(np.average(df["decision_cutoff_rank"], weights=df["final_normalized_weight"]))
    cons_threshold = round(_tanh_transform(wavg_cutoff, k), 3)

    rows = [{
        "name":      "consensus_score",
        "type":      "float",
        "direction": "high",
        "description": (
            f"Tanh-transformed quality-weighted consensus probability across the {n} sub-models. "
            f"Recommended threshold: {cons_threshold}."
        ),
    }]

    path_meta    = meta_df[meta_df["pathogen"] == pathogen].set_index("name")
    submodel_rows = []
    for _, model_row in df.iterrows():
        dataset_name = model_row["name"]
        if dataset_name not in path_meta.index:
            print(f"  [WARN] {pathogen}: dataset '{dataset_name}' not in metadata — skipping")
            continue
        meta_row = path_meta.loc[dataset_name]
        submodel_rows.append((
            _sort_priority(meta_row, dataset_name),
            {
                "name":      model_row["model_name"],
                "type":      "float",
                "direction": "high",
                "description": _build_description(meta_row, dataset_name, model_row["decision_cutoff_rank"]),
            },
        ))

    rows.extend(row for _, row in sorted(submodel_rows, key=lambda x: x[0]))

    pd.DataFrame(rows, columns=["name", "type", "direction", "description"]).to_csv(
        os.path.join(out_dir, "run_columns.csv"), index=False
    )

    print(f"  [{pathogen}] {n} models -> {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathogen", default=None)
    args = parser.parse_args()

    reports_df = pd.read_csv(REPORTS_PATH)
    meta_df    = pd.read_csv(METADATA_PATH)
    pathogens  = [args.pathogen] if args.pathogen else list(dict.fromkeys(reports_df["pathogen"]))

    for pathogen in pathogens:
        run(pathogen, reports_df, meta_df)


if __name__ == "__main__":
    main()
