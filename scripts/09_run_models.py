"""
Step 09 — Train LazyQSAR models for each dataset.

Intended to run as a SLURM array job via 09_run_models.sh, with one task
per row in 07_datasets/07_datasets_metadata.csv.

For each dataset (identified by SLURM_ARRAY_TASK_ID):
  1. Loads the prepared CSV from output/07_datasets/{pathogen}/{name}.csv
  2. Runs 5-fold stratified cross-validation and records per-fold metrics
     (AUROC, AUPRC, BEDROC and their baselines, OOF AUCs, raw score arrays)
     in output/09_reports/{pathogen}/{name}.csv
  3. Trains a final model on all data and saves it to
     output/09_models/{pathogen}/{model_name}/

Usage:
    python scripts/09_run_models.py <task_id>
    # task_id: 0-based index into 07_datasets/07_datasets_metadata.csv
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from lazyqsar.qsar import LazyClassifierQSAR
from lazyqsar.utils.metrics import bedroc_random_baseline, bedroc_score
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
sys.path.append(os.path.join(ROOT, "..", "src"))

from default import DESCRIPTORS, N_FOLDS, RANDOM_SEED

METADATA_PATH = os.path.join(REPO_ROOT, "output", "07_datasets", "07_datasets_metadata.csv")
DATASETS_DIR  = os.path.join(REPO_ROOT, "output", "07_datasets")
REPORTS_DIR   = os.path.join(REPO_ROOT, "output", "09_reports")
MODELS_DIR    = os.path.join(REPO_ROOT, "output", "09_models")

MODE = "slow"

def run(task_id: int) -> None:
    meta = pd.read_csv(METADATA_PATH)
    row  = meta.iloc[task_id]
    pathogen, name = row["pathogen"], str(row["name"])
    model_name = name  # model files are keyed by the dataset name (unique per pathogen)

    print(f"[{task_id}] {pathogen}/{name} ({model_name})")

    dataset_path = os.path.join(DATASETS_DIR, pathogen, f"{name}.csv")
    df = pd.read_csv(dataset_path)
    smiles = df["smiles"].tolist()
    y      = df["bin"].tolist()

    # Skip datasets that cannot support stratified N-fold CV (too few of a class, or
    # degenerate all-active/all-inactive). These produce no report and no model.
    n_pos = int(sum(y))
    n_neg = len(y) - n_pos
    if min(n_pos, n_neg) < N_FOLDS:
        print(f"[SKIP] {pathogen}/{name}: min class size {min(n_pos, n_neg)} < {N_FOLDS} folds "
              f"({n_pos} active, {n_neg} inactive) — not trainable")
        return

    # 5-fold CV
    records = []
    fold_data = {}
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for fold, (train_idx, test_idx) in enumerate(kf.split(smiles, y)):
        smiles_train = [smiles[i] for i in train_idx]
        y_train      = [y[i]      for i in train_idx]
        smiles_test   = [smiles[i] for i in test_idx]
        y_test        = [y[i]      for i in test_idx]

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

        fold_data[str(fold)] = {
            "y_true":  y_test,
            "y_hat":   scores_proba.tolist(),
            "y_rank":  scores_rank.tolist(),
            "roc_auc": round(auroc, 4),
        }

        records.append({
            "pathogen":               pathogen,
            "name":                   name,
            "model_name":             model_name,
            "fold":                   fold,
            "compounds_train":        len(y_train),
            "compounds_test":         len(y_test),
            "positives_train":        sum(y_train),
            "positives_test":         sum(y_test),
            "auroc":                  round(auroc, 4),
            "auprc":                  round(auprc, 4),
            "baseline_auroc":         baseline_auroc,
            "baseline_auprc":         round(baseline_auprc, 4),
            "bedroc":                 round(bedroc, 4),
            "baseline_bedroc":        round(baseline_bedroc, 4),
            "num_batches":            num_batches,
            **oof_per_descriptor,
        })
        print(f"  fold {fold}: auroc={auroc:.3f}  auprc={auprc:.3f}  bedroc={bedroc:.3f}  (baseline auprc={baseline_auprc:.3f}  baseline bedroc={baseline_bedroc:.3f})")

        

    report_dir = os.path.join(REPORTS_DIR, pathogen)
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"{model_name}.csv")
    pd.DataFrame(records).to_csv(report_path, index=False)
    print(f"  Report saved: {report_path}")
    folds_path = os.path.join(report_dir, f"{model_name}_folds.json")
    with open(folds_path, "w") as f:
        json.dump(fold_data, f)
    print(f"  Folds saved:  {folds_path}")

    # Full fit
    model = LazyClassifierQSAR(mode=MODE)
    model.fit(smiles_list=smiles, y=y)

    model_path = os.path.join(MODELS_DIR, pathogen, model_name)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"  Model saved:  {model_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 09_run_models.py <task_id>", file=sys.stderr)
        sys.exit(1)
    run(int(sys.argv[1]))
