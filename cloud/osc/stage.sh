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
export HF_HOME="${HF_HOME:-$SCRATCH/ecvol_hf}"
mkdir -p "$HF_HOME"
module load apptainer 2>/dev/null || true

if [ ! -f "$SIF" ]; then
    echo "ERROR: $SIF not found — build it first (see cloud/osc/README.md)" >&2
    exit 1
fi

for MODEL_ID in "$@"; do
    echo ">>> pre-downloading $MODEL_ID into $HF_HOME"
    APPTAINERENV_HF_HOME="$HF_HOME" apptainer exec --nv -B "$SCRATCH:$SCRATCH" "$SIF" \
        python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_ID')"
done

echo "staged. confirm data parquets are present under data/{fincall,maec}/ and data/splits/."
echo "verify offline load works on a compute node before launching the full panel."
