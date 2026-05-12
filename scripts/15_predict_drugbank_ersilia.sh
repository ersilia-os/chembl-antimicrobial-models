#!/bin/bash
# Step 15 — Predict DrugBank compounds using an Ersilia Hub model.
#
# NOTE: run this script with a conda environment that has ersilia installed,
#       NOT camm (ersilia conflicts with lazyqsar's numpy requirement).
#       Example setup: conda create -n ersilia python=3.10 && pip install ersilia==0.1.58
#
# Usage:
#     conda activate ersilia
#     bash scripts/15_predict_drugbank_ersilia.sh <model_id>
#
# Example:
#     bash scripts/15_predict_drugbank_ersilia.sh eos4rw4

set -e

model="$1"
if [ -z "$model" ]; then
    echo "Usage: $0 <model_id>" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

path_to_csv="$REPO_ROOT/data/processed/11_drugbank_smiles.csv"
path_to_output="$REPO_ROOT/output/results/15_drugbank_ersilia/${model}.csv"

mkdir -p "$(dirname "$path_to_output")"

ersilia fetch "$model"
ersilia serve "$model"
ersilia run -i "$path_to_csv" -o "$path_to_output"
ersilia delete "$model"
