#!/bin/bash
# Pre-stage model weights for offline OSC compute nodes (run on a LOGIN/transfer node
# WITH internet, after building the .sif). Data parquets travel with the repo (bind-mounted),
# so only the HF weights need pre-downloading. Idempotent.
#
# Usage:
#   bash cloud/osc/stage.sh Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-32B-Instruct
set -euo pipefail

PROJECT="${PWD}"
SIF="$PROJECT/cloud/osc/ecvol-llm.sif"
# OSC defines no $SCRATCH; default to the project scratch (override by exporting SCRATCH).
: "${SCRATCH:=/fs/scratch/PAS3453/$USER}"
mkdir -p "$SCRATCH"
export HF_HOME="${HF_HOME:-$SCRATCH/ecvol_hf}"
mkdir -p "$HF_HOME"
module load apptainer 2>/dev/null || true

if [ ! -f "$SIF" ]; then
    echo "ERROR: $SIF not found — build it first (see cloud/osc/README.md)" >&2
    exit 1
fi

for MODEL_ID in "$@"; do
    echo ">>> pre-downloading $MODEL_ID into $HF_HOME"
    # HF_HUB_OFFLINE=1 is baked into the image for the offline compute nodes; unset it here
    # (login node has internet) so the download can reach the Hub. Model id is passed via env
    # to avoid nested-quote hell. The `unset` runs after %environment, so order doesn't matter.
    APPTAINERENV_HF_HOME="$HF_HOME" APPTAINERENV_ECVOL_MODEL="$MODEL_ID" \
        apptainer exec --nv -B "$SCRATCH:$SCRATCH" "$SIF" \
        bash -c 'unset HF_HUB_OFFLINE; python3 -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ[\"ECVOL_MODEL\"])"'
done

echo "staged. confirm data parquets are present under data/{fincall,maec}/ and data/splits/."
echo "verify offline load works on a compute node before launching the full panel."
