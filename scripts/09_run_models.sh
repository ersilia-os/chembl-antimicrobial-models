#!/bin/bash
# Step 09 — Train LazyQSAR models on the HPC cluster.
#
# Runs in the camm environment (LazyQSAR 3.6.0); run script 08 first, which fetches both the
# descriptor weights and the reference library.
#
# Submit with --array set to the row indices (0-based) of 07_datasets_metadata.csv to train;
# script 08 prints a command covering every dataset:
#     sbatch --chdir=<repo_root> --array=<indices> scripts/09_run_models.sh
# All paths are relative to --chdir (the repository root).

#SBATCH --job-name=camm-lq
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=output/09_logs/%x_%a.out
#SBATCH --error=output/09_logs/_%x_%a.err
#SBATCH --partition=spot_cpu
#SBATCH --nodelist=irbccn16,irbccn41,irbccn42
#SBATCH --requeue

export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export SINGULARITY_BINDPATH="/home/sbnb:/aloy/home,/data/sbnb/data:/aloy/data,/data/sbnb/scratch:/aloy/scratch"
export LD_LIBRARY_PATH=/apps/manual/software/CUDA/11.6.1/lib64:/apps/manual/software/CUDA/11.6.1/targets/x86_64-linux/lib:/apps/manual/software/CUDA/11.6.1/extras/CUPTI/lib64/:/apps/manual/software/CUDA/11.6.1/nvvm/lib64/:$LD_LIBRARY_PATH
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HOME="$(pwd)/output/08_weights"

# LazyQSAR >= 3.6 needs the reference library at fit time. Script 08 fetches it once into
# $HOME/.lazyqsar/reference/, so tasks must never try to download it (they would all write
# to the same cache at once, and compute nodes may have no network).
export LAZYQSAR_REFERENCE_OFFLINE=1

envs/camm/bin/python -u scripts/09_run_models.py "$SLURM_ARRAY_TASK_ID"
