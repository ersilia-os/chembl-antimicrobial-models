#!/bin/bash
# Step 09b — Evaluate LazyQSAR models with a scaffold split, on the HPC cluster.
#
# Companion to 09_run_models.sh: same array indexing over
# 07_datasets/07_datasets_metadata.csv, but scripts/09b_run_models_scaffold.py
# splits folds by Bemis-Murcko scaffold instead of at random, and only writes
# a report (no final model fit/save).
#
# output/09b_logs/ must exist before submission (sbatch does not create it):
#     mkdir -p output/09b_logs
# Submit with the same array range used for script 09:
#     sbatch --chdir=<repo_root> --array=0-<N>%20 scripts/09b_run_models_scaffold.sh
# All paths are relative to --chdir (the repository root).

#SBATCH --job-name=camm-lq-scaffold
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=output/09b_logs/%x_%a.out
#SBATCH --error=output/09b_logs/_%x_%a.err
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

envs/camm/bin/python -u scripts/09b_run_models_scaffold.py "$SLURM_ARRAY_TASK_ID"
