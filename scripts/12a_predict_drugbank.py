"""
Step 12a (array) — Predict DrugBank scores for one (pathogen, predict type) per SLURM task.

Intended to run as a SLURM array job via 12a_run_array.sh, with the task_id mapping to one
(pathogen, predict_type) combination:
    pathogen_idx, type_idx = divmod(task_id, len(PREDICT_TYPES))
over the 15 pathogens (src/default.py PATHOGENS, fixed order) x 6 predict types below.

Same underlying computation as 12a_predict_drugbank_local.py --pathogen <p> for a single predict
type, but split so each (pathogen, type) pair runs as its own cluster job instead of looping
over all 6 types in series in one process. Descriptors ARE recomputed independently per task
(not shared across pathogens or types) — that's the accepted tradeoff for wall-clock
parallelism on the cluster.

Skip-if-exists: if the target output/12_drugbank/{type}/{pathogen}.csv already exists (e.g.
produced by a concurrent `12a_predict_drugbank_local.py --all_pathogens` run), the task exits
immediately without recomputing — this lets the array run safely alongside that local job,
picking up only the (pathogen, type) combinations it hasn't produced yet.

Usage:
    python scripts/12a_predict_drugbank.py <task_id>
    # task_id: 0-based index, 0 to (n_pathogens * n_predict_types - 1)
"""

import os
import sys

import pandas as pd

from lazyqsar.api.classifier_predict import predict as lqsar_predict

ROOT      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
sys.path.append(os.path.join(ROOT, "..", "src"))

from default import PATHOGENS  # noqa: E402

# Point lazyqsar to the project weights directory (mirrors 12a_predict_drugbank_local.py / 09_run_models.sh).
os.environ["HOME"] = os.path.join(REPO_ROOT, "output", "08_weights")

# Same list, same order, as 12a_predict_drugbank_local.py — kept in sync manually since that script's
# filename starts with a digit and can't be imported as a normal Python module.
PREDICT_TYPES = ["rank", "proba", "score", "logit", "lift", "binary"]

DRUGBANK_PATH = os.path.join(REPO_ROOT, "data", "processed", "11_drugbank_smiles.csv")
MODELS_DIR    = os.path.join(REPO_ROOT, "output", "09_models")
OUT_DIR       = os.path.join(REPO_ROOT, "output", "12_drugbank")
REPORTS_PATH  = os.path.join(REPO_ROOT, "output", "10_reports", "10_reports.csv")


def _ordered_model_names(pathogen: str) -> list[str]:
    """Model names for a pathogen in 10_reports.csv order, keeping only those on disk."""
    reports = pd.read_csv(REPORTS_PATH)
    rows = reports[reports["pathogen"] == pathogen]
    pathogen_dir = os.path.join(MODELS_DIR, pathogen)
    return [
        name for name in rows["model_name"].tolist()
        if os.path.isdir(os.path.join(pathogen_dir, name))
    ]


def run(task_id: int) -> None:
    n_types = len(PREDICT_TYPES)
    n_total = len(PATHOGENS) * n_types
    if not 0 <= task_id < n_total:
        raise ValueError(
            f"task_id {task_id} out of range: {len(PATHOGENS)} pathogens x {n_types} "
            f"predict types = {n_total} tasks (0-{n_total - 1})"
        )

    pathogen_idx, type_idx = divmod(task_id, n_types)
    pathogen = PATHOGENS[pathogen_idx]
    predict_type = PREDICT_TYPES[type_idx]

    print(f"[{task_id}] {pathogen} | {predict_type}")

    out_dir = os.path.join(OUT_DIR, predict_type)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{pathogen}.csv")

    if os.path.exists(out_path):
        print(f"  [SKIP] already exists (produced by another run): {out_path}")
        return

    pathogen_dir = os.path.join(MODELS_DIR, pathogen)
    if not os.path.isdir(pathogen_dir):
        print(f"  [SKIP] no model directory at {pathogen_dir}")
        return

    model_names = _ordered_model_names(pathogen)
    if not model_names:
        print(f"  [SKIP] no models found in reports or on disk for {pathogen}")
        return

    print(f"  {len(model_names)} models: {model_names}")
    model_dir_dict = {name: os.path.join(pathogen_dir, name) for name in model_names}

    lqsar_predict(
        model_dir=model_dir_dict,
        input_csv=DRUGBANK_PATH,
        output_csv=out_path,
        predict_type=predict_type,
    )

    smiles = pd.read_csv(DRUGBANK_PATH)["smiles"].tolist()
    df = pd.read_csv(out_path)
    df.insert(0, "smiles", smiles)
    df.to_csv(out_path, index=False)

    print(f"  Saved {len(smiles)} rows x {len(model_names)} models -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 12a_predict_drugbank.py <task_id>", file=sys.stderr)
        sys.exit(1)
    run(int(sys.argv[1]))
