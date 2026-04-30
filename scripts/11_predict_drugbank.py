"""
Step 11 — Predict DrugBank ranks for all models of a given pathogen.

Loads every trained LazyQSAR model for the specified pathogen and runs
predict_rank on all DrugBank compounds, writing one column per model.

Usage:
    python scripts/11_predict_drugbank.py --pathogen ecoli
    python scripts/11_predict_drugbank.py --pathogen ecoli --drugbank path/to/smiles.csv
"""

import argparse
import os

import pandas as pd

from lazyqsar.qsar import LazyClassifierQSAR

ROOT      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))

DEFAULT_DRUGBANK   = os.path.join(REPO_ROOT, "data", "processed", "10_drugbank_smiles.csv")
DEFAULT_MODELS_DIR = os.path.join(REPO_ROOT, "output", "results", "08_models")
DEFAULT_OUT_DIR    = os.path.join(REPO_ROOT, "output", "results", "11_drugbank")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict DrugBank ranks for a pathogen.")
    parser.add_argument("--pathogen",   required=True,              help="Pathogen code (e.g. ecoli)")
    parser.add_argument("--drugbank",   default=DEFAULT_DRUGBANK,   help="DrugBank SMILES CSV")
    parser.add_argument("--models_dir", default=DEFAULT_MODELS_DIR, help="Base directory for trained models")
    parser.add_argument("--output",     default=None,               help="Output CSV path")
    args = parser.parse_args()

    out_path = args.output or os.path.join(DEFAULT_OUT_DIR, f"{args.pathogen}.csv")

    smiles = pd.read_csv(args.drugbank)["smiles"].tolist()
    print(f"DrugBank: {len(smiles)} compounds")

    pathogen_dir = os.path.join(args.models_dir, args.pathogen)
    if not os.path.isdir(pathogen_dir):
        raise FileNotFoundError(f"No models found for pathogen '{args.pathogen}' at {pathogen_dir}")

    model_names = sorted(
        d for d in os.listdir(pathogen_dir)
        if os.path.isdir(os.path.join(pathogen_dir, d))
    )
    if not model_names:
        raise FileNotFoundError(f"No model subdirectories found in {pathogen_dir}")

    print(f"Models for {args.pathogen}: {model_names}")

    results = {"smiles": smiles}
    for model_name in model_names:
        model_dir = os.path.join(pathogen_dir, model_name)
        print(f"  Loading {model_name} ...", end=" ", flush=True)
        model = LazyClassifierQSAR.load(model_dir)
        scores = model.predict_rank(smiles_list=smiles)[:, 1]
        results[model_name] = [round(float(s), 4) for s in scores]
        print("done")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved {len(smiles)} rows × {len(model_names)} models → {out_path}")


if __name__ == "__main__":
    main()
