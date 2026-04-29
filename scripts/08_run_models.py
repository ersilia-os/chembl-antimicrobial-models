"""
Step 08 — Train LazyQSAR models for each dataset.

Intended to run as a SLURM array job via 08_run_models.sh, with one task
per row in 06_datasets_metadata.csv.

For each dataset (identified by SLURM_ARRAY_TASK_ID):
  1. Loads the prepared CSV from output/results/06_datasets/{pathogen}/{name}.csv
  2. Runs 5-fold stratified cross-validation and records AUROC, AUPRC, and their
     prevalence/random baselines in output/results/08_reports/{pathogen}/{name}.csv
  3. Trains a final model on all data and saves it to
     output/results/08_models/{pathogen}/{name}/

Usage:
    python scripts/08_run_models.py <task_id>
    # task_id: 0-based index into 06_datasets_metadata.csv
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from lazyqsar.qsar import LazyClassifierQSAR

ROOT      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))

METADATA_PATH = os.path.join(REPO_ROOT, "output", "results", "06_datasets_metadata.csv")
DATASETS_DIR  = os.path.join(REPO_ROOT, "output", "results", "06_datasets")
REPORTS_DIR   = os.path.join(REPO_ROOT, "output", "results", "08_reports")
MODELS_DIR    = os.path.join(REPO_ROOT, "output", "results", "08_models")

N_FOLDS = 5
MODE    = "slow"

DESCRIPTOR_TYPES = {
    "slow": ["cddd", "chemeleon", "clamp", "morgan", "rdkit"],
    "fast": ["morgan"],
}


def run(task_id: int) -> None:
    meta = pd.read_csv(METADATA_PATH)
    row  = meta.iloc[task_id]
    pathogen, name = row["pathogen"], row["name"]

    print(f"[{task_id}] {pathogen}/{name}")

    dataset_path = os.path.join(DATASETS_DIR, pathogen, f"{name}.csv")
    df = pd.read_csv(dataset_path)
    smiles = df["smiles"].tolist()
    y      = df["bin"].tolist()

    # 5-fold CV
    records = []
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(kf.split(smiles, y)):
        smiles_train = [smiles[i] for i in train_idx]
        y_train      = [y[i]      for i in train_idx]
        smiles_test   = [smiles[i] for i in test_idx]
        y_test        = [y[i]      for i in test_idx]

        model = LazyClassifierQSAR(mode=MODE)
        model.fit(smiles_list=smiles_train, y=y_train)
        scores_proba = model.predict_proba(smiles_list=smiles_test)[:, 1]
        scores_rank  = model.predict_rank(smiles_list=smiles_test)[:, 1]

        auroc          = roc_auc_score(y_test, scores_proba)
        auprc          = average_precision_score(y_test, scores_proba)
        baseline_auroc = 0.5
        baseline_auprc = sum(y_test) / len(y_test)

        oof_auc_map = dict(zip(model.descriptor_types, model.oof_aucs_))
        oof_per_descriptor = {
            f"oof_auc_{desc}": round(oof_auc_map[desc], 4) if desc in oof_auc_map else np.nan
            for desc in DESCRIPTOR_TYPES[MODE]
        }

        y_arr = np.array(y_test)

        def fmt(arr, mask):
            return ";".join(str(round(float(v), 3)) for v in arr[mask])

        records.append({
            "pathogen":               pathogen,
            "name":                   name,
            "fold":                   fold,
            "compounds_train":        len(y_train),
            "compounds_test":         len(y_test),
            "positives_train":        sum(y_train),
            "positives_test":         sum(y_test),
            "auroc":                  round(auroc, 4),
            "auprc":                  round(auprc, 4),
            "baseline_auroc":         baseline_auroc,
            "baseline_auprc":         round(baseline_auprc, 4),
            **oof_per_descriptor,
            "predict_proba_actives":  fmt(scores_proba, y_arr == 1),
            "predict_proba_inactives":fmt(scores_proba, y_arr == 0),
            "predict_rank_actives":   fmt(scores_rank,  y_arr == 1),
            "predict_rank_inactives": fmt(scores_rank,  y_arr == 0),
        })
        print(f"  fold {fold}: auroc={auroc:.3f}  auprc={auprc:.3f}  (baseline auprc={baseline_auprc:.3f})")

        

    report_path = os.path.join(REPORTS_DIR, pathogen, f"{name}.csv")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    pd.DataFrame(records).to_csv(report_path, index=False)
    print(f"  Report saved: {report_path}")

    # Full fit
    model = LazyClassifierQSAR(mode=MODE)
    model.fit(smiles_list=smiles, y=y)

    model_path = os.path.join(MODELS_DIR, pathogen, name)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"  Model saved:  {model_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 08_run_models.py <task_id>", file=sys.stderr)
        sys.exit(1)
    run(int(sys.argv[1]))
