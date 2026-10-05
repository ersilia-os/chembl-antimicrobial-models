"""
Step 12a (array) — Predict reference-library scores for one (pathogen, predict type) per SLURM task.

The twin of 12b_predict_drugbank.py: same models, same predict types, same task mapping,
same output layout — only the molecules differ. Instead of DrugBank it scores the 50,000
drug-like molecules of the LazyQSAR >= 3.6 reference library, the same library every
model's `rank` is anchored on:
    pathogen_idx, type_idx = divmod(task_id, len(PREDICT_TYPES))
over the 15 pathogens (src/default.py PATHOGENS, fixed order) x 6 predict types below.

Why: a single model's rank is a position against that library, so rank 0.65 means "beats 99%
of drug-like chemistry" and admits about 1% of it (0.98% to 1.21% per model once the library is
re-scored at inference, not exactly 1%). The consensus is a weighted mean
of several models' ranks, which is not itself a position against anything — with models that
agree only weakly, far less than 1% of generic chemistry reaches a mean of 0.65. Scoring the
library itself gives the consensus the same reference distribution the sub-models have, so a
consensus cutoff can be defined by the hit rate it admits rather than by transforming 0.65.

The molecule list is read from the cached bundle through lazyqsar's own `reference.identity`,
never from a path spelled out here, so it follows the bundle that the checkpoints were fitted
against (`manifest_sha256` in every metadata.json). Step 08 fetches it; this script never
downloads, so a compute node with no network fails loudly instead of silently diverging.

Skip-if-exists: if the target output/12_reference/{type}/{pathogen}.csv already exists, the
task exits immediately without recomputing, so a partially finished array can be resubmitted.
The output is written in place, so a task killed mid-write leaves a file that a rerun would accept:
after a kill, delete that task's file first.

Reproducibility: each (pathogen, type) is a separate task that recomputes everything on whichever
node it lands on, and the six types of a pathogen are not bit-consistent with each other (CPU-
dependent floating point amplified by the tree heads). See scripts/README.md for the size.

Usage:
    python scripts/12a_predict_reference.py <task_id>
    # task_id: 0-based index, 0 to (n_pathogens * n_predict_types - 1)
"""

import os
import sys

import pandas as pd

ROOT      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
sys.path.append(os.path.join(ROOT, "..", "src"))

# Set before importing lazyqsar, not after as in 12b: this script also asks lazyqsar where the
# reference bundle is cached, and parts of lazyqsar read these at import time. HOME alone would
# work through the ~/.lazyqsar default, but LAZYQSAR_HOME is the variable it actually consults.
WEIGHTS_ROOT = os.path.join(REPO_ROOT, "output", "08_weights")
os.environ["HOME"] = WEIGHTS_ROOT
os.environ["LAZYQSAR_HOME"] = os.path.join(WEIGHTS_ROOT, ".lazyqsar")

from lazyqsar.api.classifier_predict import predict as lqsar_predict  # noqa: E402
from lazyqsar.reference import identity  # noqa: E402

from default import PATHOGENS  # noqa: E402

# Same list, same order, as 12b_predict_drugbank.py — the two scripts are twins and the task
# mapping only lines up while these agree.
PREDICT_TYPES = ["rank", "proba", "score", "logit", "lift", "binary"]

MODELS_DIR   = os.path.join(REPO_ROOT, "output", "09_models")
OUT_DIR      = os.path.join(REPO_ROOT, "output", "12_reference")
REPORTS_PATH = os.path.join(REPO_ROOT, "output", "10_reports", "10_reports.csv")
os.makedirs(OUT_DIR, exist_ok=True)


def reference_smiles_path() -> str:
    """The cached reference molecule list, located through lazyqsar rather than hardcoded."""
    path = identity.reference_dir() / identity.smiles_filename()
    if not path.is_file():
        sys.exit(
            f"Reference library not cached at {path}.\n"
            f"Run scripts/08_download_weights_and_reference.py first — this script never "
            f"downloads, so that every task reads the bundle the models were fitted against."
        )
    return str(path)


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

    reference_path = reference_smiles_path()
    print(f"  reference library: {identity.REFERENCE_ID} ({identity.default_n():,} molecules)")
    print(f"  {len(model_names)} models: {model_names}")
    model_dir_dict = {name: os.path.join(pathogen_dir, name) for name in model_names}

    lqsar_predict(
        model_dir=model_dir_dict,
        input_csv=reference_path,
        output_csv=out_path,
        predict_type=predict_type,
    )

    smiles = pd.read_csv(reference_path)["smiles"].tolist()
    df = pd.read_csv(out_path)
    df.insert(0, "smiles", smiles)
    df.to_csv(out_path, index=False)

    print(f"  Saved {len(smiles)} rows x {len(model_names)} models -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 12a_predict_reference.py <task_id>", file=sys.stderr)
        sys.exit(1)
    run(int(sys.argv[1]))
