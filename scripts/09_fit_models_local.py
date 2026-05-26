"""
Step 09 (local) — Train LazyQSAR models for all datasets on a local machine.

Equivalent to 09_run_models.py / 09_run_models.sh but runs all datasets
sequentially without SLURM. Uses LazyQSAR's default weight paths.

For each dataset in 07_datasets/07_datasets_metadata.csv:
  1. 5-fold stratified CV with full metrics
       → output/09_reports/{pathogen}/{name}.csv         (step-10 compatible)
       → output/09_reports/{pathogen}/{name}_folds.json  (raw fold arrays)
  2. Final model fit
       → output/09_models/{pathogen}/{model_name}/

Existing outputs are skipped so the script is safe to re-run after interruption.

Usage:
    python scripts/09_fit_models_local.py
    python scripts/09_fit_models_local.py --pathogens abaumannii saureus
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from lazyqsar.qsar import LazyClassifierQSAR
from lazyqsar.utils.metrics import bedroc_random_baseline, bedroc_score
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

root      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(root, ".."))
sys.path.append(os.path.join(root, "..", "src"))

from default import DESCRIPTORS, N_FOLDS, RANDOM_SEED
from model_name import compute_model_name

METADATA_PATH = os.path.join(REPO_ROOT, "output", "07_datasets", "07_datasets_metadata.csv")
DATASETS_DIR  = os.path.join(REPO_ROOT, "output", "07_datasets")
REPORTS_DIR   = os.path.join(REPO_ROOT, "output", "09_reports")
MODELS_DIR    = os.path.join(REPO_ROOT, "output", "09_models")

MODE = "slow"


# ---------------------------------------------------------------------------
# CV + reports
# ---------------------------------------------------------------------------

def run_cv(smiles: list, y: list, pathogen: str, name: str, model_name: str) -> None:
    """5-fold stratified CV. Saves CSV (metrics) and JSON (raw fold arrays)."""
    records = []
    fold_data = {}
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    for fold, (train_idx, test_idx) in enumerate(kf.split(smiles, y)):
        smiles_train = [smiles[i] for i in train_idx]
        y_train      = [y[i]      for i in train_idx]
        smiles_test  = [smiles[i] for i in test_idx]
        y_test       = [y[i]      for i in test_idx]

        model = LazyClassifierQSAR(mode=MODE)
        model.fit(smiles_list=smiles_train, y=y_train)
        scores_proba = model.predict_proba(smiles_list=smiles_test)[:, 1]
        scores_rank  = model.predict_rank(smiles_list=smiles_test)[:, 1]

        auroc           = roc_auc_score(y_test, scores_rank)
        auprc           = average_precision_score(y_test, scores_rank)
        baseline_auroc  = 0.5
        baseline_auprc  = sum(y_test) / len(y_test)
        bedroc          = bedroc_score(np.array(y_test), scores_rank)
        baseline_bedroc = bedroc_random_baseline(np.array(y_test))

        oof_auc_map = dict(zip(model.descriptor_types, model.oof_aucs_))
        oof_per_descriptor = {
            f"oof_auc_{desc}": round(oof_auc_map[desc], 4) if desc in oof_auc_map else np.nan
            for desc in DESCRIPTORS
        }
        num_batches = len(model.models[0]._model.models) if model.models else np.nan

        y_arr = np.array(y_test)

        def fmt(arr, mask):
            return ";".join(str(round(float(v), 3)) for v in arr[mask])

        records.append({
            "pathogen":                pathogen,
            "name":                    name,
            "model_name":              model_name,
            "fold":                    fold,
            "compounds_train":         len(y_train),
            "compounds_test":          len(y_test),
            "positives_train":         sum(y_train),
            "positives_test":          sum(y_test),
            "auroc":                   round(auroc, 4),
            "auprc":                   round(auprc, 4),
            "baseline_auroc":          baseline_auroc,
            "baseline_auprc":          round(baseline_auprc, 4),
            "bedroc":                  round(bedroc, 4),
            "baseline_bedroc":         round(baseline_bedroc, 4),
            "num_batches":             num_batches,
            **oof_per_descriptor,
            "predict_proba_actives":   fmt(scores_proba, y_arr == 1),
            "predict_proba_inactives": fmt(scores_proba, y_arr == 0),
            "predict_rank_actives":    fmt(scores_rank,  y_arr == 1),
            "predict_rank_inactives":  fmt(scores_rank,  y_arr == 0),
        })

        fold_data[str(fold)] = {
            "y_true":  y_test,
            "y_hat":   scores_proba.tolist(),
            "y_rank":  scores_rank.tolist(),
            "roc_auc": round(auroc, 4),
        }

        print(f"  fold {fold}: auroc={auroc:.3f}  auprc={auprc:.3f}  bedroc={bedroc:.3f}")

    report_dir = os.path.join(REPORTS_DIR, pathogen)
    os.makedirs(report_dir, exist_ok=True)

    pd.DataFrame(records).to_csv(os.path.join(report_dir, f"{model_name}.csv"), index=False)
    with open(os.path.join(report_dir, f"{model_name}_folds.json"), "w") as f:
        json.dump(fold_data, f)
    print(f"  Report saved: {os.path.join(report_dir, model_name)}.csv / _folds.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(pathogens: list | None) -> None:
    for path in [REPORTS_DIR, MODELS_DIR]:
        os.makedirs(path, exist_ok=True)

    meta = pd.read_csv(METADATA_PATH)
    if pathogens:
        meta = meta[meta["pathogen"].isin(pathogens)].reset_index(drop=True)

    # Precompute model names for all rows (needed in both phases)
    model_name_map = {task_id: compute_model_name(meta, task_id) for task_id in meta.index}

    # -----------------------------------------------------------------------
    # Phase 1: CV + final models
    # -----------------------------------------------------------------------
    for task_id, row in meta.iterrows():
        pathogen, name = row["pathogen"], row["name"]
        model_name = model_name_map[task_id]

        print(f"\n{'='*60}")
        print(f"  [{task_id}] {pathogen}/{name} ({model_name})")
        print(f"{'='*60}")

        report_csv = os.path.join(REPORTS_DIR, pathogen, f"{model_name}.csv")
        model_path = os.path.join(MODELS_DIR,  pathogen, model_name)
        report_done = os.path.exists(report_csv)
        model_done  = os.path.exists(os.path.join(model_path, "metadata.json"))

        if report_done and model_done:
            print("  Report and model exist — skipping")
            continue

        try:
            df     = pd.read_csv(os.path.join(DATASETS_DIR, pathogen, f"{name}.csv"))
            smiles = df["smiles"].tolist()
            y      = df["bin"].tolist()

            if not report_done:
                run_cv(smiles, y, pathogen, name, model_name)
            else:
                print("  Report exists — skipping CV")

            if not model_done:
                print("  Training final model")
                model = LazyClassifierQSAR(mode=MODE)
                model.fit(smiles_list=smiles, y=y)
                os.makedirs(model_path, exist_ok=True)
                model.save(model_path)
                print(f"  Model saved: {model_path}")
            else:
                print("  Final model exists — skipping")
        except Exception as e:
            print(f"  [ERROR] {pathogen}/{name}: {e} — skipping")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LazyQSAR models locally (sequential, no SLURM)."
    )
    parser.add_argument(
        "--pathogens",
        nargs="+",
        default=None,
        help="Restrict to specific pathogens (e.g. --pathogens abaumannii saureus)",
    )
    args = parser.parse_args()
    main(args.pathogens)
