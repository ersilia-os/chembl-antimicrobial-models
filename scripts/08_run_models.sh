#!/bin/bash
# Step 08 — Train LazyQSAR models on the HPC cluster.
#
# Submit via the command printed by script 07:
#     sbatch --chdir=<repo_root> --array=0-<N>%20 scripts/08_run_models.sh
# All paths are relative to --chdir (the repository root).

#SBATCH --job-name=camm-lq
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=output/results/08_logs/%x_%a.out
#SBATCH --error=output/results/08_logs/_%x_%a.err
#SBATCH --partition=spot_cpu
#SBATCH --nodelist=irbccn16,irbccn41,irbccn42
#SBATCH --requeue

export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export SINGULARITY_BINDPATH="/home/sbnb:/aloy/home,/data/sbnb/data:/aloy/data,/data/sbnb/scratch:/aloy/scratch"
export LD_LIBRARY_PATH=/apps/manual/software/CUDA/11.6.1/lib64:/apps/manual/software/CUDA/11.6.1/targets/x86_64-linux/lib:/apps/manual/software/CUDA/11.6.1/extras/CUPTI/lib64/:/apps/manual/software/CUDA/11.6.1/nvvm/lib64/:$LD_LIBRARY_PATH
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HOME="$(pwd)/output/results/07_weights"

envs/camm/bin/python -u scripts/08_run_models.py "$SLURM_ARRAY_TASK_ID"
