"""
Step 12 — Predict DrugBank ranks for all models of a given pathogen.

Uses the lazy-qsar multi-model predict() API so descriptors are computed
once per featurizer type and shared across all models, rather than
recomputing them for each model independently.

Model order follows 10_reports.csv (which mirrors 07_datasets_metadata.csv).

Usage:
    python scripts/12_predict_drugbank.py --pathogen ecoli
    python scripts/12_predict_drugbank.py --all_pathogens
    python scripts/12_predict_drugbank.py --pathogen ecoli --drugbank path/to/smiles.csv
"""

import argparse
import os

import pandas as pd

from lazyqsar.api.classifier_predict import predict as lqsar_predict

ROOT      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))

# Point lazyqsar to the project weights directory (mirrors what 09_run_models.sh does via $HOME).
os.environ["HOME"] = os.path.join(REPO_ROOT, "output", "results", "08_weights")

DEFAULT_DRUGBANK   = os.path.join(REPO_ROOT, "data", "processed", "10_drugbank_smiles.csv")
DEFAULT_MODELS_DIR = os.path.join(REPO_ROOT, "output", "results", "09_models")
DEFAULT_OUT_DIR    = os.path.join(REPO_ROOT, "output", "results", "12_drugbank")
REPORTS_PATH       = os.path.join(REPO_ROOT, "output", "results", "10_reports.csv")


def _ordered_model_names(pathogen: str, models_dir: str) -> list[str]:
    """Return model names for a pathogen in 10_reports.csv order, keeping only those on disk."""
    reports = pd.read_csv(REPORTS_PATH)
    rows = reports[reports["pathogen"] == pathogen]
    pathogen_dir = os.path.join(models_dir, pathogen)
    return [
        name for name in rows["model_name"].tolist()
        if os.path.isdir(os.path.join(pathogen_dir, name))
    ]


def _ordered_pathogens() -> list[str]:
    """Return pathogens in the order they first appear in 10_reports.csv."""
    reports = pd.read_csv(REPORTS_PATH)
    return list(dict.fromkeys(reports["pathogen"].tolist()))


def run_pathogen(pathogen: str, drugbank_csv: str, models_dir: str, out_path: str) -> None:
    pathogen_dir = os.path.join(models_dir, pathogen)
    if not os.path.isdir(pathogen_dir):
        print(f"  [SKIP] {pathogen}: no model directory at {pathogen_dir}")
        return

    model_names = _ordered_model_names(pathogen, models_dir)
    if not model_names:
        print(f"  [SKIP] {pathogen}: no models found in reports or on disk")
        return

    print(f"\n[{pathogen}] {len(model_names)} models: {model_names}")

    model_dir_dict = {
        os.path.join(pathogen_dir, name): name for name in model_names
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    lqsar_predict(
        model_dir=model_dir_dict,
        input_csv=drugbank_csv,
        output_csv=out_path,
        predict_type="rank",
    )

    smiles = pd.read_csv(drugbank_csv)["smiles"].tolist()
    df = pd.read_csv(out_path)
    df.insert(0, "smiles", smiles)
    df.to_csv(out_path, index=False)

    print(f"  Saved {len(smiles)} rows x {len(model_names)} models -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict DrugBank ranks for a pathogen.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pathogen",      help="Pathogen code (e.g. ecoli)")
    group.add_argument("--all_pathogens", action="store_true",
                       help="Run all pathogens in metadata order")
    parser.add_argument("--drugbank",   default=DEFAULT_DRUGBANK,   help="DrugBank SMILES CSV")
    parser.add_argument("--models_dir", default=DEFAULT_MODELS_DIR, help="Base directory for trained models")
    parser.add_argument("--output",     default=None,
                        help="Output CSV path (single pathogen only)")
    args = parser.parse_args()

    n_compounds = len(pd.read_csv(args.drugbank))
    print(f"DrugBank: {n_compounds} compounds")

    if args.all_pathogens:
        for pathogen in _ordered_pathogens():
            out_path = os.path.join(DEFAULT_OUT_DIR, f"{pathogen}.csv")
            run_pathogen(pathogen, args.drugbank, args.models_dir, out_path)
    else:
        out_path = args.output or os.path.join(DEFAULT_OUT_DIR, f"{args.pathogen}.csv")
        run_pathogen(args.pathogen, args.drugbank, args.models_dir, out_path)


if __name__ == "__main__":
    main()
