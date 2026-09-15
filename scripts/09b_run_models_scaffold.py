"""
Step 09b — Evaluate LazyQSAR models with a scaffold split (companion to step 09).

Same 5-fold CV evaluation as 09_run_models.py, but folds are assigned by
Bemis-Murcko scaffold instead of at random, so no scaffold is split across
train/test within a fold. This is a diagnostic report only: it does not
train or save a final full-dataset model (that model already exists from
step 09 and would be identical here), and it is not wired into the
downstream pipeline (10a onward).

Intended to run as a SLURM array job via 09b_run_models_scaffold.sh, with
one task per row in 07_datasets/07_datasets_metadata.csv — the same
task_id -> dataset mapping as step 09, so the same set of datasets is
attempted.

For each dataset (identified by SLURM_ARRAY_TASK_ID):
  1. Loads the prepared CSV from output/07_datasets/{pathogen}/{name}.csv
  2. Computes a Bemis-Murcko scaffold per compound (RDKit
     MurckoScaffoldSmiles, standard/non-generic, includeChirality=False;
     acyclic compounds all share the empty-string scaffold).
  3. Runs 5-fold StratifiedGroupKFold CV (scaffold groups never split
     across folds; class balance per fold is best-effort) and records
     per-fold metrics (AUROC, AUPRC, BEDROC and baselines, OOF AUCs per
     descriptor, raw score arrays, and the number of distinct scaffold
     groups in train/test) in output/09b_reports/{pathogen}/{name}.csv.

     A scaffold-constrained fold can end up with a single-class test set
     even when the dataset passes the min-class-size trainability check
     below. When that happens, that fold alone is skipped (no model is
     fit for it, since its AUROC/AUPRC/BEDROC would be undefined): the
     report row keeps only the always-defined fields (compound/positive
     counts, baseline_auroc=0.5) and leaves the rest NaN, and the fold is
     omitted from the _folds.json file.

Usage:
    python scripts/09b_run_models_scaffold.py <task_id>
    # task_id: 0-based index into 07_datasets/07_datasets_metadata.csv
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from lazyqsar.qsar import LazyClassifierQSAR
from lazyqsar.utils.metrics import bedroc_random_baseline, bedroc_score
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
sys.path.append(os.path.join(ROOT, "..", "src"))

from default import DESCRIPTORS, N_FOLDS, RANDOM_SEED

METADATA_PATH = os.path.join(REPO_ROOT, "output", "07_datasets", "07_datasets_metadata.csv")
DATASETS_DIR  = os.path.join(REPO_ROOT, "output", "07_datasets")
REPORTS_DIR   = os.path.join(REPO_ROOT, "output", "09b_reports")

MODE = "slow"


def compute_scaffolds(smiles_list: list) -> list:
    """Standard (non-generic) Bemis-Murcko scaffold per SMILES. Acyclic
    compounds all map to the empty-string scaffold and are grouped together."""
    return [
        MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=False)
        for smi in smiles_list
    ]


def run(task_id: int) -> None:
    meta = pd.read_csv(METADATA_PATH)
    row  = meta.iloc[task_id]
    pathogen, name = row["pathogen"], str(row["name"])
    model_name = name  # report files are keyed by the dataset name (unique per pathogen)

    print(f"[{task_id}] {pathogen}/{name} ({model_name}) — scaffold split")

    dataset_path = os.path.join(DATASETS_DIR, pathogen, f"{name}.csv")
    df = pd.read_csv(dataset_path)
    smiles = df["smiles"].tolist()
    y      = df["bin"].tolist()

    # Skip datasets that cannot support stratified N-fold CV (too few of a class, or
    # degenerate all-active/all-inactive). Same rule as step 09, so both steps attempt
    # the same set of datasets.
    n_pos = int(sum(y))
    n_neg = len(y) - n_pos
    if min(n_pos, n_neg) < N_FOLDS:
        print(f"[SKIP] {pathogen}/{name}: min class size {min(n_pos, n_neg)} < {N_FOLDS} folds "
              f"({n_pos} active, {n_neg} inactive) — not trainable")
        return

    scaffolds = compute_scaffolds(smiles)

    # 5-fold scaffold-grouped CV
    records = []
    fold_data = {}
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(smiles, y, groups=scaffolds)):
        smiles_train = [smiles[i] for i in train_idx]
        y_train      = [y[i]      for i in train_idx]
        smiles_test   = [smiles[i] for i in test_idx]
        y_test        = [y[i]      for i in test_idx]

        n_groups_train = len(set(scaffolds[i] for i in train_idx))
        n_groups_test  = len(set(scaffolds[i] for i in test_idx))

        n_pos_test = sum(y_test)
        n_neg_test = len(y_test) - n_pos_test
        if n_pos_test == 0 or n_neg_test == 0:
            print(f"  [SKIP FOLD] fold {fold}: single-class test set from scaffold grouping "
                  f"({n_pos_test} active, {n_neg_test} inactive) — metrics undefined, no model fit")
            records.append({
                "pathogen":               pathogen,
                "name":                   name,
                "model_name":             model_name,
                "fold":                   fold,
                "compounds_train":        len(y_train),
                "compounds_test":         len(y_test),
                "scaffolds_train":        n_groups_train,
                "scaffolds_test":         n_groups_test,
                "positives_train":        sum(y_train),
                "positives_test":         n_pos_test,
                "auroc":                  np.nan,
                "auprc":                  np.nan,
                "baseline_auroc":         0.5,
                "baseline_auprc":         np.nan,
                "bedroc":                 np.nan,
                "baseline_bedroc":        np.nan,
                "num_batches":            np.nan,
                **{f"oof_auc_{desc}": np.nan for desc in DESCRIPTORS},
            })
            continue

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
            "scaffolds_train":        n_groups_train,
            "scaffolds_test":         n_groups_test,
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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 09b_run_models_scaffold.py <task_id>", file=sys.stderr)
        sys.exit(1)
    run(int(sys.argv[1]))
