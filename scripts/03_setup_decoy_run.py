"""
Step 03 — Set up decoy run environment.

(0) Creates output directories.
(1) Splits output/results/02_selected_positives.csv into per-split CSVs
    (single 'smiles' column) → output/results/03_positives_splits/split_XXX.csv
(2) Clones git@github.com:ersilia-os/ersilia-apptainer.git into
    output/results/03_ersilia_apptainer and installs the latest version
    into output/results/03_conda_camm/, exposed as ersilia_apptainer_camm.
(3) Builds the eos3e6s Singularity/Apptainer SIF image
    → output/results/03_eos3e6s/eos3e6s.sif
(4) Writes scripts/04_run_decoys.sh — submit with: sbatch scripts/04_run_decoys.sh

All steps overwrite existing outputs.

Usage:
    python scripts/03_setup_decoy_run.py
    python scripts/03_setup_decoy_run.py --positives path/to/other.csv
"""

import argparse
import os
import shutil
import stat
import subprocess

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
MODEL = "eos3e6s"
DIRS = {
    "splits":    os.path.join(REPO_ROOT, "output", "results", "03_positives_splits"),
    "apptainer": os.path.join(REPO_ROOT, "output", "results", "03_ersilia_apptainer"),
    "conda_env": os.path.join(REPO_ROOT, "output", "results", "03_conda_camm"),
    "sif":       os.path.join(REPO_ROOT, "output", "results", f"03_{MODEL}"),
    "decoys":    os.path.join(REPO_ROOT, "output", "results", "04_decoys"),
    "logs":      os.path.join(REPO_ROOT, "output", "results", "04_logs"),
}


def _find_conda() -> str:
    conda = shutil.which("conda") or shutil.which("mamba")
    if conda:
        return conda
    raise RuntimeError(
        "conda/mamba not found in PATH. Run this script from an active conda environment."
    )


# ---------------------------------------------------------------------------
# Step 0 — directories
# ---------------------------------------------------------------------------

def make_dirs() -> None:
    for key, path in DIRS.items():
        if key == "conda_env":
            continue  # conda create handles this directory itself
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
# Step 2 — clone ersilia-apptainer and install into project conda env
# ---------------------------------------------------------------------------

def setup_ersilia_apptainer() -> str:
    apptainer_dir = DIRS["apptainer"]
    conda_env_dir = DIRS["conda_env"]
    camm_bin = os.path.join(conda_env_dir, "bin", "ersilia_apptainer_camm")

    if os.path.exists(apptainer_dir):
        shutil.rmtree(apptainer_dir)
    subprocess.run(
        ["git", "clone", "git@github.com:ersilia-os/ersilia-apptainer.git", apptainer_dir],
        check=True,
    )
    subprocess.run(
        ["git", "-C", apptainer_dir, "checkout", "a7d8be09df0efc72114ed554f111d25f1f6587cb"],
        check=True,
    )
    print(f"Cloned ersilia-apptainer to {apptainer_dir}")

    conda = _find_conda()
    if os.path.exists(conda_env_dir):
        shutil.rmtree(conda_env_dir)
    subprocess.run(
        [conda, "create", "--prefix", conda_env_dir, "python=3.11", "--yes", "--quiet"],
        check=True,
    )
    conda_pip = os.path.join(conda_env_dir, "bin", "pip")
    subprocess.run([conda_pip, "install", apptainer_dir], check=True)
    # Expose as ersilia_apptainer_camm (relative symlink, same bin/ dir)
    os.symlink("ersilia_apptainer", camm_bin)
    print(f"Installed ersilia_apptainer_camm in {conda_env_dir}")

    return camm_bin


# ---------------------------------------------------------------------------
# Step 3 — build SIF image
# ---------------------------------------------------------------------------

def build_sif() -> None:
    sif_dir = DIRS["sif"]
    def_path = os.path.join(sif_dir, f"{MODEL}.def")
    sif_path = os.path.join(sif_dir, f"{MODEL}.sif")

    if os.path.exists(sif_path):
        os.remove(sif_path)

    with open(def_path, "w") as f:
        f.write(f"""Bootstrap: docker
From: ersiliaos/{MODEL}:latest

%post
    mkdir -p /opt/ersilia
    mv /root/bundles /opt/ersilia/bundles
    mv /root/model /opt/ersilia/model
    chmod -R 755 /opt/ersilia
    export ERSILIA_PATH=/opt/ersilia

%environment
    export ERSILIA_PATH=/opt/ersilia
""")

    subprocess.run(["singularity", "build", sif_path, def_path], check=True)
    subprocess.run(["apptainer", "cache", "clean", "-f"])
    print(f"Built SIF image: {sif_path}")


# ---------------------------------------------------------------------------
# Step 4 — write 04_run_decoys.sh
# ---------------------------------------------------------------------------

def write_run_script(n_splits: int, camm_bin: str) -> None:
    script_path = os.path.join(ROOT, "04_run_decoys.sh")
    max_idx = n_splits - 1

    log_dir = DIRS["logs"]
    inp_dir = DIRS["splits"]
    res_dir = DIRS["decoys"]
    app_dir = DIRS["sif"]

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
#SBATCH --array=0-{max_idx}%10
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

{camm_bin} run \\
  --sif "{app_dir}/{MODEL}.sif" \\
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

def main(positives_path: str) -> None:
    make_dirs()

    n_splits = split_positives(positives_path)
    camm_bin = setup_ersilia_apptainer()
    build_sif()
    write_run_script(n_splits, camm_bin)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Set up the decoy run environment (splits, SIF, SLURM script)."
    )
    parser.add_argument(
        "--positives",
        default=os.path.join(REPO_ROOT, "output", "results", "02_selected_positives.csv"),
        help="Path to 02_selected_positives.csv (default: output/results/02_selected_positives.csv).",
    )
    args = parser.parse_args()
    main(args.positives)
