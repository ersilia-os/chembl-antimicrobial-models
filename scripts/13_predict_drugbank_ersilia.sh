#!/bin/bash
# Step 13 — Predict DrugBank compounds using an Ersilia Hub model.
#
# NOTE: run this script with a conda environment that has ersilia installed,
#       NOT camm (ersilia conflicts with lazyqsar's numpy requirement).
#       Example setup: conda create -n ersilia python=3.10 && pip install ersilia==0.1.58
#
# Usage:
#     conda activate ersilia
#     bash scripts/13_predict_drugbank_ersilia.sh <model_id> [batch_size]
#
# Examples:
#     bash scripts/13_predict_drugbank_ersilia.sh eos4rw4
#     bash scripts/13_predict_drugbank_ersilia.sh eos18ie 10   # use small batch for heavy models

set -e

model="$1"
batch_size="${2:-100}"
if [ -z "$model" ]; then
    echo "Usage: $0 <model_id> [batch_size]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

path_to_csv="$REPO_ROOT/data/processed/11_drugbank_smiles.csv"
path_to_output="$REPO_ROOT/output/results/13_drugbank_ersilia/${model}.csv"

mkdir -p "$(dirname "$path_to_output")"

ersilia fetch "$model"
ersilia serve "$model"
ersilia run -i "$path_to_csv" -o "$path_to_output" -b "$batch_size"
ersilia delete "$model"
