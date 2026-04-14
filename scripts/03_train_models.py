"""
Step 03 — Train LazyQSAR models.

For each binary dataset in data/processed/<pathogen>/ (SMILES + bin columns),
trains a LazyClassifierQSAR model and saves the result under
output/models/<pathogen>/<dataset_name>/. Datasets without a processed version
fall back to data/raw/<pathogen>/.

Usage:
    python scripts/03_train_models.py --pathogen ecoli
    python scripts/03_train_models.py --pathogen ecoli --mode fast
    python scripts/03_train_models.py --pathogen ecoli --output_dir output/models
"""

import argparse
import os

import pandas as pd
from lazyqsar import LazyClassifierQSAR


PATHOGENS = [
    "abaumannii", "calbicans", "campylobacter", "ecoli", "efaecium",
    "enterobacter", "hpylori", "kpneumoniae", "mtuberculosis", "ngonorrhoeae",
    "paeruginosa", "pfalciparum", "saureus", "smansoni", "spneumoniae",
]


def train_pathogen(pathogen: str, mode: str, output_dir: str) -> None:
    processed_dir = os.path.join("data", "processed", pathogen)
    raw_dir = os.path.join("data", "raw", pathogen)
    data_dir = processed_dir if os.path.isdir(processed_dir) else raw_dir

    if not os.path.isdir(data_dir):
        print(f"[SKIP] No data found for {pathogen} in {data_dir}")
        return

    dataset_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not dataset_files:
        print(f"[SKIP] No CSV datasets found for {pathogen}")
        return

    for dataset_file in sorted(dataset_files):
        dataset_name = dataset_file.replace(".csv", "")
        model_dir = os.path.join(output_dir, pathogen, dataset_name)

        df = pd.read_csv(os.path.join(data_dir, dataset_file))
        smiles_list = df["smiles"].tolist()
        y = df["bin"].tolist()

        print(f"Training {pathogen}/{dataset_name} ({len(smiles_list)} compounds, mode={mode})")
        model = LazyClassifierQSAR(mode=mode)
        model.fit(smiles_list, y)
        model.save(model_dir)
        print(f"  Saved to {model_dir}")


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    train_pathogen(args.pathogen, args.mode, args.output_dir)


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
        "--mode",
        type=str,
        default="default",
        choices=["fast", "default", "slow"],
        help="LazyQSAR descriptor mode (default: default).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("output", "models"),
        help="Root directory where trained models are saved (default: output/models).",
    )
    args = parser.parse_args()
    main(args)
