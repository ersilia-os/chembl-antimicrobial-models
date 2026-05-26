#!/usr/bin/env bash
set -euo pipefail
mkdir -p output/05_decoys
ersilia serve eos3e6s
ersilia -v run \
    -i output/03_select_positives/selected_positive_smiles.csv \
    -o output/05_decoys/eos3e6s_all.csv
ersilia close
