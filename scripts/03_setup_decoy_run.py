"""
Step 03 — Set up decoy run environment.

(0) Checks that ersilia_apptainer is available in the project conda env
    (envs/camm/bin/ersilia_apptainer). The env must be created beforehand:
        conda env create -f environment.yml --prefix ./envs/camm
(1) Splits output/results/02_selected_positives.csv into per-split CSVs
    (single 'smiles' column) → output/results/03_positives_splits/split_XXX.csv
(2) Builds the eos3e6s Singularity/Apptainer SIF image via ersilia_apptainer create
    → output/results/03_eos3e6s_v1.sif
(3) Writes scripts/04_run_decoys.sh — submit with: sbatch scripts/04_run_decoys.sh

All steps overwrite existing outputs.

Usage:
    python scripts/03_setup_decoy_run.py
    python scripts/03_setup_decoy_run.py --positives path/to/other.csv --version v1.0.0
"""

import argparse
import os
import stat
import subprocess

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
MODEL = "eos3e6s"
CAMM_BIN = os.path.join(REPO_ROOT, "envs", "camm", "bin", "ersilia_apptainer")
RESULTS_DIR = os.path.join(REPO_ROOT, "output", "results")
DIRS = {
    "splits": os.path.join(RESULTS_DIR, "03_positives_splits"),
    "decoys": os.path.join(RESULTS_DIR, "04_decoys"),
    "logs":   os.path.join(RESULTS_DIR, "04_logs"),
}


# ---------------------------------------------------------------------------
# Step 0 — sanity check + directories
# ---------------------------------------------------------------------------

def check_camm_env() -> None:
    if not os.path.isfile(CAMM_BIN):
        raise RuntimeError(
            f"ersilia_apptainer binary not found at {CAMM_BIN}.\n"
            "Create the project conda env first:\n"
            "    conda env create -f environment.yml --prefix ./envs/camm"
        )


def make_dirs() -> None:
    for path in DIRS.values():
        os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1 — split positives
# ---------------------------------------------------------------------------

def split_positives(positives_path: str) -> int:
    df = pd.read_csv(positives_path)
    for split_idx, group in df.groupby("split"):
        out_path = os.path.join(DIRS["splits"], f"split_{int(split_idx):03d}.csv")
        group[["smiles"]].to_csv(out_path, index=False)
    n_splits = df["split"].nunique()
    print(f"Split positives into {n_splits} files in {DIRS['splits']}")
    return n_splits


# ---------------------------------------------------------------------------
# Step 2 — build SIF image
# ---------------------------------------------------------------------------

def build_sif(version: str) -> str:
    major = version.split(".")[0]  # e.g. "v1.0.0" → "v1"
    raw_path = os.path.join(RESULTS_DIR, f"{MODEL}_{major}.sif")
    sif_path = os.path.join(RESULTS_DIR, f"03_{MODEL}_{major}.sif")

    if os.path.exists(sif_path):
        os.remove(sif_path)

    subprocess.run(
        [CAMM_BIN, "create", "--model", MODEL, "--version", version,
         "--output-dir", RESULTS_DIR, "--verbose"],
        check=True,
    )
    os.rename(raw_path, sif_path)
    print(f"Built SIF image: {sif_path}")
    return sif_path


# ---------------------------------------------------------------------------
# Step 3 — write 04_run_decoys.sh
# ---------------------------------------------------------------------------

def write_run_script(n_splits: int, sif_path: str) -> None:
    script_path = os.path.join(ROOT, "04_run_decoys.sh")
    max_idx = n_splits - 1

    log_dir = DIRS["logs"]
    inp_dir = DIRS["splits"]
    res_dir = DIRS["decoys"]

    # All static paths are resolved by Python f-strings at write time.
    # Only genuine SLURM runtime variables remain as shell variables.
    # Submit with: sbatch scripts/04_run_decoys.sh
    content = f"""\
#!/bin/bash
#SBATCH --job-name=camm-{MODEL}
#SBATCH --chdir={REPO_ROOT}
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=0-{max_idx}%40
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output={log_dir}/%x_%a.out
#SBATCH --partition=spot_cpu
#SBATCH --nodelist=irbccn16,irbccn41,irbccn42
#SBATCH --requeue

export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export SINGULARITY_BINDPATH="/home/sbnb:/aloy/home,/data/sbnb/data:/aloy/data,/data/sbnb/scratch:/aloy/scratch"
export LD_LIBRARY_PATH=/apps/manual/software/CUDA/11.6.1/lib64:/apps/manual/software/CUDA/11.6.1/targets/x86_64-linux/lib:/apps/manual/software/CUDA/11.6.1/extras/CUPTI/lib64/:/apps/manual/software/CUDA/11.6.1/nvvm/lib64/:$LD_LIBRARY_PATH
export PYTHONDONTWRITEBYTECODE=1

alpha_padded="$(printf "%03d" "$SLURM_ARRAY_TASK_ID")"

{CAMM_BIN} run \\
  --sif "{sif_path}" \\
  --input "{inp_dir}/split_${{alpha_padded}}.csv" \\
  --output "{res_dir}/{MODEL}_${{alpha_padded}}.csv" \\
  --verbose
"""

    with open(script_path, "w") as f:
        f.write(content)

    os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Wrote run script: {script_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(positives_path: str, version: str) -> None:
    check_camm_env()
    make_dirs()
    n_splits = split_positives(positives_path)
    sif_path = build_sif(version)
    write_run_script(n_splits, sif_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Set up the decoy run environment (splits, SIF, SLURM script)."
    )
    parser.add_argument(
        "--positives",
        default=os.path.join(REPO_ROOT, "output", "results", "02_selected_positives.csv"),
        help="Path to 02_selected_positives.csv (default: output/results/02_selected_positives.csv).",
    )
    parser.add_argument(
        "--version",
        default="v1.0.0",
        help="DockerHub version tag for the model image (default: v1.0.0).",
    )
    args = parser.parse_args()
    main(args.positives, args.version)
