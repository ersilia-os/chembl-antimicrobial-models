"""
Step 03 — Train LazyQSAR models.

Reads data/processed/<pathogen>/01_chembl_datasets.csv to determine which
datasets to train. For each dataset, loads SMILES from the appropriate zip
file in data/raw/<pathogen>/ and trains a LazyClassifierQSAR model saved
under output/models/<pathogen>/<name>/.

  individual/merged datasets → 19_final_datasets.zip  (<name>.csv)
  general datasets           → 20_general_datasets.zip (ORG_<activity>_<unit>_<cutoff>.csv.gz)

A stratified 80/20 split is used to benchmark vanilla RF and LR baselines
(Morgan fingerprints) against LazyClassifierQSAR on the same held-out set.
The saved model is the LazyClassifierQSAR trained on the 80% train split.

Usage:
    python scripts/03_train_models.py --pathogen abaumannii --dataset <name>
    python scripts/03_train_models.py --pathogen abaumannii --dataset <name> --mode fast
"""

import argparse
import io
import os
import time
import zipfile

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

import lazyqsar
from lazyqsar.descriptors.morgan import MorganFingerprint
from lazyqsar.qsar import LazyClassifierQSAR

lazyqsar.set_verbosity(True)

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(ROOT, "..")

PATHOGENS = [
    "abaumannii", "calbicans", "campylobacter", "ecoli", "efaecium",
    "enterobacter", "hpylori", "kpneumoniae", "mtuberculosis", "ngonorrhoeae",
    "paeruginosa", "pfalciparum", "saureus", "smansoni", "spneumoniae",
]


def load_dataset(row: pd.Series, raw_dir: str) -> pd.DataFrame:
    source = row["source"]
    if source in ("individual", "merged"):
        zip_path = os.path.join(raw_dir, "19_final_datasets.zip")
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(f"{row['name']}.csv") as f:
                return pd.read_csv(f)[["smiles", "bin"]]
    else:
        filename = f"ORG_{row['activity_type']}_{row['unit']}_{row['cutoff']}.csv.gz"
        zip_path = os.path.join(raw_dir, "20_general_datasets.zip")
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(filename) as f:
                return pd.read_csv(io.BytesIO(f.read()), compression="gzip")[["smiles", "bin"]]


def _print_summary(results: list[tuple[str, float, float]]) -> None:
    print("\n" + "═" * 62)
    print(f"{'Method':<36}  {'AUC':>8}  {'Time (s)':>10}")
    print("─" * 62)
    for name, auc, elapsed in results:
        print(f"{name:<36}  {auc:>8.4f}  {elapsed:>10.1f}")
    print("═" * 62 + "\n")


def train_dataset(pathogen: str, dataset: str, mode: str) -> None:
    metadata_path = os.path.join(REPO_ROOT, "data", "processed", pathogen, "01_chembl_datasets.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No metadata found for {pathogen}")

    metadata = pd.read_csv(metadata_path)
    duplicates = metadata[metadata["name"].duplicated()]
    if not duplicates.empty:
        raise ValueError(f"Duplicate dataset names found: {duplicates['name'].tolist()}")

    matches = metadata[metadata["name"] == dataset]
    if matches.empty:
        raise ValueError(f"Dataset '{dataset}' not found in {pathogen} metadata.")

    row = matches.iloc[0]
    raw_dir = os.path.join(REPO_ROOT, "data", "raw", pathogen)
    df = load_dataset(row, raw_dir)

    smiles_list = df["smiles"].tolist()
    y = np.array(df["bin"].tolist(), dtype=int)

    print(f"Training {pathogen}/{dataset} ({len(smiles_list)} compounds, mode={mode})")

    smiles_train, smiles_test, y_train, y_test = train_test_split(
        smiles_list, y, test_size=0.2, stratify=y, random_state=42
    )

    # --- Morgan fingerprints for baselines ---
    morgan = MorganFingerprint()
    X_train = morgan.transform(smiles_train)
    X_test = morgan.transform(smiles_test)

    results = []

    # --- RF baseline ---
    t0 = time.perf_counter()
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    results.append(("RandomForest (Morgan, n=100)", rf_auc, time.perf_counter() - t0))

    # --- LR baseline ---
    t0 = time.perf_counter()
    scaler = MaxAbsScaler()
    lr = LogisticRegression(C=0.1, solver="saga", penalty="l1", class_weight="balanced",
                            max_iter=10_000, random_state=42)
    lr.fit(scaler.fit_transform(X_train), y_train)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(scaler.transform(X_test))[:, 1])
    results.append(("LogisticRegression (Morgan, L1)", lr_auc, time.perf_counter() - t0))

    # --- LazyClassifierQSAR ---
    t0 = time.perf_counter()
    model = LazyClassifierQSAR(mode=mode)
    model.fit(smiles_train, y_train)
    lazy_auc = roc_auc_score(y_test, model.predict_proba(smiles_test)[:, 1])
    results.append((f"LazyClassifierQSAR (mode={mode})", lazy_auc, time.perf_counter() - t0))

    _print_summary(results)

    model_dir = os.path.join(REPO_ROOT, "output", "models", pathogen, dataset)
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_dir)
    print(f"Saved to {model_dir}")


def main(args: argparse.Namespace) -> None:
    train_dataset(args.pathogen, args.dataset, args.mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LazyQSAR binary classifiers for antimicrobial datasets."
    )
    parser.add_argument(
        "--pathogen",
        type=str,
        required=True,
        choices=PATHOGENS,
        help="Pathogen code to train models for.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name to train (e.g. CHEMBL4296188_INHIBITION_%%_qt_25.0).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="default",
        choices=["fast", "default", "slow"],
        help="LazyQSAR descriptor mode (default: default).",
    )
    args = parser.parse_args()
    main(args)
