#!/bin/bash
# Step 12a (array) — Score the LazyQSAR reference library on the HPC cluster, one
# (pathogen, predict type) per array task. The twin of 12b_run_array.sh.
#
# Submit via (90 tasks = 15 pathogens x 6 predict types):
#     sbatch --chdir=<repo_root> --array=0-89%20 scripts/12a_run_array.sh
# Only the consensus re-anchoring needs `rank`, which is predict type 0, so in practice the
# rank tasks alone are submitted: --array=0,6,12,...  (pathogen index x 6).
# All paths are relative to --chdir (the repository root).
#
# Needs the reference bundle cached by script 08. Each task skips immediately if its
# output/12_reference/{type}/{pathogen}.csv already exists.

#SBATCH --job-name=reference
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=output/12a_logs/%x_%a.out
#SBATCH --error=output/12a_logs/_%x_%a.err
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

# The bundle is read, never fetched: script 08 cached it once, and a compute node reaching for
# the network here would either hang or race the other tasks into the same cache.
export LAZYQSAR_REFERENCE_OFFLINE=1

envs/camm/bin/python -u scripts/12a_predict_reference.py "$SLURM_ARRAY_TASK_ID"
