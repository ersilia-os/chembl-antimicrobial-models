#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

EUOPENSCREEN_REPO="eu-openscreen-antimicrobial-tasks"
REMOTE_PATH="data/processed/02_only_smiles.csv"
LOCAL_INPUT="$REPO_ROOT/data/raw/euopenscreen/02_only_smiles.csv"

# --- Parse pathogen argument ---
PATHOGEN="${1:-}"
if [ -z "$PATHOGEN" ]; then
    echo "Usage: bash scripts/19_euopenscreen_benchmark.sh <pathogen>"
    echo "Available: abaumannii calbicans campylobacter ecoli efaecium enterobacter"
    echo "           hpylori kpneumoniae mtuberculosis ngonorrhoeae paeruginosa"
    echo "           pfalciparum saureus smansoni spneumoniae"
    exit 1
fi

# --- Look up model ID from src/default.py ---
MODEL_ID=$(python3 - <<PYEOF
import sys, os
sys.path.insert(0, "$REPO_ROOT/src")
from default import ERSILIA_MODEL_IDS
pathogen = "$PATHOGEN"
if pathogen not in ERSILIA_MODEL_IDS:
    print(f"Unknown pathogen: {pathogen}", file=sys.stderr)
    sys.exit(1)
print(ERSILIA_MODEL_IDS[pathogen])
PYEOF
)

echo "Pathogen : $PATHOGEN"
echo "Model ID : $MODEL_ID"

OUTPUT_DIR="$REPO_ROOT/output/19_euopenscreen_benchmark"
mkdir -p "$(dirname "$LOCAL_INPUT")"
mkdir -p "$OUTPUT_DIR"

# --- Download input via eosvc (skipped if already present) ---
if [ ! -f "$LOCAL_INPUT" ]; then
    echo "Downloading $REMOTE_PATH from $EUOPENSCREEN_REPO ..."
    EVC_REPO_NAME="$EUOPENSCREEN_REPO" eosvc download --path "$REMOTE_PATH"
    # eosvc places the file at <cwd>/<REMOTE_PATH>; move it to the canonical location
    mv "$REPO_ROOT/$REMOTE_PATH" "$LOCAL_INPUT"
    echo "Saved to $LOCAL_INPUT"
else
    echo "Input already present: $LOCAL_INPUT"
fi

# --- Run Ersilia model ---
ersilia serve "$MODEL_ID"
ersilia -v run \
    -i "$LOCAL_INPUT" \
    -o "$OUTPUT_DIR/${PATHOGEN}_euopenscreen_preds.csv" \
    --batch_size 1000
ersilia close
