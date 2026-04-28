#!/bin/bash
#SBATCH --job-name=camm-lq
#SBATCH --chdir=/aloy/home/acomajuncosa/Ersilia/chembl-antimicrobial-models
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=0-351%20
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/aloy/home/acomajuncosa/Ersilia/chembl-antimicrobial-models/output/results/07_logs/%x_%a.out
#SBATCH --partition=spot_cpu
#SBATCH --nodelist=irbccn16,irbccn41,irbccn42
#SBATCH --requeue

export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export SINGULARITY_BINDPATH="/home/sbnb:/aloy/home,/data/sbnb/data:/aloy/data,/data/sbnb/scratch:/aloy/scratch"
export LD_LIBRARY_PATH=/apps/manual/software/CUDA/11.6.1/lib64:/apps/manual/software/CUDA/11.6.1/targets/x86_64-linux/lib:/apps/manual/software/CUDA/11.6.1/extras/CUPTI/lib64/:/apps/manual/software/CUDA/11.6.1/nvvm/lib64/:$LD_LIBRARY_PATH
export PYTHONDONTWRITEBYTECODE=1

/aloy/home/acomajuncosa/Ersilia/chembl-antimicrobial-models/envs/camm/bin/python \
  /aloy/home/acomajuncosa/Ersilia/chembl-antimicrobial-models/scripts/07_run_models.py \
  "$SLURM_ARRAY_TASK_ID"
