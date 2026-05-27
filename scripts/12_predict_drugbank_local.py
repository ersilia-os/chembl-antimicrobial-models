"""
Step 12 (local) — Predict DrugBank ranks for trained models, using the lazyqsar
default installation weights instead of the project's output/08_weights directory.

Identical to 12_predict_drugbank.py except the HOME override is omitted.

Usage:
    python scripts/12_predict_drugbank_local.py --pathogen ecoli
    python scripts/12_predict_drugbank_local.py --all_pathogens
"""

import argparse
import os
import time

import pandas as pd

from lazyqsar.api.classifier_predict import predict as lqsar_predict

ROOT      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))

DRUGBANK_PATH = os.path.join(REPO_ROOT, "data", "processed", "11_drugbank_smiles.csv")
MODELS_DIR    = os.path.join(REPO_ROOT, "output", "09_models")
OUT_DIR       = os.path.join(REPO_ROOT, "output", "12_drugbank")
REPORTS_PATH  = os.path.join(REPO_ROOT, "output", "10_reports", "10_reports.csv")
os.makedirs(OUT_DIR, exist_ok=True)


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


def run_all_pathogens(drugbank_csv: str, models_dir: str, out_dir: str) -> int:
    pathogens = _ordered_pathogens()
    model_dir_dict = {}
    for pathogen in pathogens:
        pathogen_dir = os.path.join(models_dir, pathogen)
        if not os.path.isdir(pathogen_dir):
            print(f"  [SKIP] {pathogen}: no model directory")
            continue
        for name in _ordered_model_names(pathogen, models_dir):
            model_path = os.path.join(pathogen_dir, name)
            col_name = f"{pathogen}/{name}"
            model_dir_dict[col_name] = model_path

    if not model_dir_dict:
        print("No models found across any pathogen.")
        return 0

    n_models = len(model_dir_dict)
    print(f"\n[all_pathogens] {n_models} models total")

    os.makedirs(out_dir, exist_ok=True)
    tmp_path = os.path.join(out_dir, "_tmp_all_pathogens.csv")
    lqsar_predict(
        model_dir=model_dir_dict,
        input_csv=drugbank_csv,
        output_csv=tmp_path,
        predict_type="rank",
    )

    smiles = pd.read_csv(drugbank_csv)["smiles"].tolist()
    df = pd.read_csv(tmp_path)
    df.insert(0, "smiles", smiles)

    for pathogen in pathogens:
        prefix = f"{pathogen}/"
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            continue
        per_pathogen_df = df[["smiles"] + cols].rename(
            columns={c: c[len(prefix):] for c in cols}
        )
        out_path = os.path.join(out_dir, f"{pathogen}.csv")
        per_pathogen_df.to_csv(out_path, index=False)
        print(f"  [{pathogen}] {len(smiles)} rows x {len(cols)} models -> {out_path}")

    os.remove(tmp_path)
    return n_models


def run_pathogen(pathogen: str, drugbank_csv: str, models_dir: str, out_path: str) -> int:
    pathogen_dir = os.path.join(models_dir, pathogen)
    if not os.path.isdir(pathogen_dir):
        print(f"  [SKIP] {pathogen}: no model directory at {pathogen_dir}")
        return 0

    model_names = _ordered_model_names(pathogen, models_dir)
    if not model_names:
        print(f"  [SKIP] {pathogen}: no models found in reports or on disk")
        return 0

    print(f"\n[{pathogen}] {len(model_names)} models: {model_names}")

    model_dir_dict = {
        name: os.path.join(pathogen_dir, name) for name in model_names
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
    return len(model_names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict DrugBank ranks (local lazyqsar weights).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pathogen",      help="Pathogen code (e.g. ecoli)")
    group.add_argument("--all_pathogens", action="store_true",
                       help="Run all pathogens in metadata order")
    parser.add_argument("--drugbank",   default=DRUGBANK_PATH, help="DrugBank SMILES CSV")
    parser.add_argument("--models_dir", default=MODELS_DIR,    help="Base directory for trained models")
    parser.add_argument("--output",     default=None,
                        help="Output CSV path (--pathogen mode only)")
    args = parser.parse_args()

    n_compounds = len(pd.read_csv(args.drugbank))
    print(f"DrugBank: {n_compounds} compounds")

    t0 = time.time()
    if args.all_pathogens:
        n_models = run_all_pathogens(args.drugbank, args.models_dir, OUT_DIR)
    else:
        out_path = args.output or os.path.join(OUT_DIR, f"{args.pathogen}.csv")
        n_models = run_pathogen(args.pathogen, args.drugbank, args.models_dir, out_path)
    elapsed_min = (time.time() - t0) / 60

    expected_min = round(elapsed_min * 10_000 / n_compounds, 1) if n_compounds else float("nan")
    print(
        f"\n--- Inference summary ---"
        f"\n  Compounds : {n_compounds}"
        f"\n  Models    : {n_models}"
        f"\n  Total time: {elapsed_min:.1f} min"
        f"\n  Est. time for 10,000 compounds: {expected_min} min"
    )


if __name__ == "__main__":
    main()
