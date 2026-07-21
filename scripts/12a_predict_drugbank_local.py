"""
Step 12a (local) — Predict DrugBank scores for trained models, across predict types.

Uses the lazy-qsar multi-model predict() API so that, within a single call,
descriptors are computed once per featurizer type and shared across all models,
rather than recomputing them for each model independently.

Runs every predict type in PREDICT_TYPES in series. Each type writes to its own
subdirectory: output/12_drugbank/{type}/{pathogen}.csv.

NOTE: the multi-model predict() API rebuilds (and deletes) a scratch descriptor
matrix on every call, so descriptors ARE recomputed once per predict type. The
within-object _ensemble_cache from lazy-qsar issue #26 is a different (single-
model) code path that this script does not use.

--pathogen:      predict for one pathogen; output is one CSV per pathogen.
--all_pathogens: build descriptors once (per type) and score ALL models across
                 ALL pathogens in a single pass, then split into one CSV per
                 pathogen. The combined intermediate file is removed after
                 splitting.

Outputs are CSVs with columns: smiles, model_name_1, model_name_2, ...
Model order follows output/10_reports/10_reports.csv.

Usage:
    python scripts/12a_predict_drugbank_local.py --pathogen ecoli
    python scripts/12a_predict_drugbank_local.py --all_pathogens
"""

import argparse
import os
import time

import pandas as pd

from lazyqsar.api.classifier_predict import predict as lqsar_predict

# Predict types to produce, in run order. Each writes to its own subdirectory
# under OUT_DIR: {OUT_DIR}/{type}/{pathogen}.csv.
PREDICT_TYPES = ["rank", "proba", "score", "logit", "lift", "binary"]

ROOT      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))

# Point lazyqsar to the project weights directory (mirrors what 09_run_models.sh does via $HOME).
os.environ["HOME"] = os.path.join(REPO_ROOT, "output", "08_weights")

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


def run_all_pathogens(drugbank_csv: str, models_dir: str, out_dir: str,
                      predict_type: str) -> int:
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
    print(f"\n[all_pathogens | {predict_type}] {n_models} models total")

    os.makedirs(out_dir, exist_ok=True)
    tmp_path = os.path.join(out_dir, "_tmp_all_pathogens.csv")
    lqsar_predict(
        model_dir=model_dir_dict,
        input_csv=drugbank_csv,
        output_csv=tmp_path,
        predict_type=predict_type,
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


def run_pathogen(pathogen: str, drugbank_csv: str, models_dir: str, out_path: str,
                 predict_type: str) -> int:
    pathogen_dir = os.path.join(models_dir, pathogen)
    if not os.path.isdir(pathogen_dir):
        print(f"  [SKIP] {pathogen}: no model directory at {pathogen_dir}")
        return 0

    model_names = _ordered_model_names(pathogen, models_dir)
    if not model_names:
        print(f"  [SKIP] {pathogen}: no models found in reports or on disk")
        return 0

    print(f"\n[{pathogen} | {predict_type}] {len(model_names)} models: {model_names}")

    model_dir_dict = {
        name: os.path.join(pathogen_dir, name) for name in model_names
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    lqsar_predict(
        model_dir=model_dir_dict,
        input_csv=drugbank_csv,
        output_csv=out_path,
        predict_type=predict_type,
    )

    smiles = pd.read_csv(drugbank_csv)["smiles"].tolist()
    df = pd.read_csv(out_path)
    df.insert(0, "smiles", smiles)
    df.to_csv(out_path, index=False)

    print(f"  Saved {len(smiles)} rows x {len(model_names)} models -> {out_path}")
    return len(model_names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict DrugBank ranks for a pathogen.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pathogen",      help="Pathogen code (e.g. ecoli)")
    group.add_argument("--all_pathogens", action="store_true",
                       help="Run all pathogens in metadata order")
    parser.add_argument("--drugbank",   default=DRUGBANK_PATH, help="DrugBank SMILES CSV")
    parser.add_argument("--models_dir", default=MODELS_DIR,    help="Base directory for trained models")
    parser.add_argument("--output_dir", default=None,
                        help="Base output directory (default output/12_drugbank); "
                             "each type writes to {output_dir}/{type}/{pathogen}.csv")
    args = parser.parse_args()

    n_compounds = len(pd.read_csv(args.drugbank))
    print(f"DrugBank: {n_compounds} compounds")
    print(f"Predict types: {PREDICT_TYPES}  (descriptors recomputed per type)")

    base_dir = args.output_dir or OUT_DIR

    t0 = time.time()
    n_models = 0
    for predict_type in PREDICT_TYPES:
        out_dir = os.path.join(base_dir, predict_type)
        os.makedirs(out_dir, exist_ok=True)
        if args.all_pathogens:
            n_models = run_all_pathogens(args.drugbank, args.models_dir, out_dir, predict_type)
        else:
            out_path = os.path.join(out_dir, f"{args.pathogen}.csv")
            n_models = run_pathogen(args.pathogen, args.drugbank, args.models_dir,
                                    out_path, predict_type)
    elapsed_min = (time.time() - t0) / 60

    per_type_min = elapsed_min / len(PREDICT_TYPES) if PREDICT_TYPES else float("nan")
    expected_min = round(per_type_min * 10_000 / n_compounds, 1) if n_compounds else float("nan")
    print(
        f"\n--- Inference summary ---"
        f"\n  Compounds   : {n_compounds}"
        f"\n  Models      : {n_models}"
        f"\n  Predict types: {len(PREDICT_TYPES)} ({', '.join(PREDICT_TYPES)})"
        f"\n  Total time  : {elapsed_min:.1f} min (all types)"
        f"\n  Est. time for 10,000 compounds (per type): {expected_min} min"
    )


if __name__ == "__main__":
    main()
