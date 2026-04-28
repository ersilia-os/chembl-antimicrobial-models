#!/bin/bash
#SBATCH --job-name=camm-eos3e6s
#SBATCH --chdir=/aloy/home/acomajuncosa/Ersilia/chembl-antimicrobial-models
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=0-464%40
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/aloy/home/acomajuncosa/Ersilia/chembl-antimicrobial-models/output/results/04_logs/%x_%a.out
#SBATCH --partition=spot_cpu
#SBATCH --nodelist=irbccn16,irbccn41,irbccn42
#SBATCH --requeue

export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export SINGULARITY_BINDPATH="/home/sbnb:/aloy/home,/data/sbnb/data:/aloy/data,/data/sbnb/scratch:/aloy/scratch"
export LD_LIBRARY_PATH=/apps/manual/software/CUDA/11.6.1/lib64:/apps/manual/software/CUDA/11.6.1/targets/x86_64-linux/lib:/apps/manual/software/CUDA/11.6.1/extras/CUPTI/lib64/:/apps/manual/software/CUDA/11.6.1/nvvm/lib64/:$LD_LIBRARY_PATH
export PYTHONDONTWRITEBYTECODE=1

alpha_padded="$(printf "%03d" "$SLURM_ARRAY_TASK_ID")"

/aloy/home/acomajuncosa/Ersilia/chembl-antimicrobial-models/envs/camm/bin/ersilia_apptainer run \
  --sif "/aloy/home/acomajuncosa/Ersilia/chembl-antimicrobial-models/output/results/03_eos3e6s_v1.sif" \
  --input "/aloy/home/acomajuncosa/Ersilia/chembl-antimicrobial-models/output/results/03_positives_splits/split_${alpha_padded}.csv" \
  --output "/aloy/home/acomajuncosa/Ersilia/chembl-antimicrobial-models/output/results/04_decoys/eos3e6s_${alpha_padded}.csv" \
  --verbose
