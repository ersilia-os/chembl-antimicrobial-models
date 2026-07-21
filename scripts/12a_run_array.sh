#!/bin/bash
# Step 12a (array) — Predict DrugBank scores on the HPC cluster, one (pathogen, predict type)
# per array task.
#
# Submit via (90 tasks = 15 pathogens x 6 predict types):
#     sbatch --chdir=<repo_root> --array=0-89%20 scripts/12a_run_array.sh
# All paths are relative to --chdir (the repository root).
#
# Safe to run alongside a concurrent `12a_predict_drugbank_local.py --all_pathogens` local run —
# each task skips immediately if its output/12_drugbank/{type}/{pathogen}.csv already exists.

#SBATCH --job-name=camm-lq-drugbank
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=output/12_logs/%x_%a.out
#SBATCH --error=output/12_logs/_%x_%a.err
#SBATCH --partition=spot_cpu
#SBATCH --nodelist=irbccn16,irbccn41,irbccn42
#SBATCH --requeue

export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export SINGULARITY_BINDPATH="/home/sbnb:/aloy/home,/data/sbnb/data:/aloy/data,/data/sbnb/scratch:/aloy/scratch"
export LD_LIBRARY_PATH=/apps/manual/software/CUDA/11.6.1/lib64:/apps/manual/software/CUDA/11.6.1/targets/x86_64-linux/lib:/apps/manual/software/CUDA/11.6.1/extras/CUPTI/lib64/:/apps/manual/software/CUDA/11.6.1/nvvm/lib64/:$LD_LIBRARY_PATH
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

envs/camm/bin/python -u scripts/12a_predict_drugbank.py "$SLURM_ARRAY_TASK_ID"
