#!/bin/bash
# T6.2 cloud-burst LLM extraction on OSC (one model per submission; resumable).
#
# CONFIG — edit the #SBATCH account/partition and the model panel to match your
# OSC allocation (see cloud/osc/README.md). Resumable: the per-model parquet is the
# checkpoint, so a short walltime that requeues is fine — just resubmit.
#
# Submit (per model):
#   smoke test (3 calls, real ETA/validate YaRN before the full run):
#     sbatch --export=ALL,MODEL_ID=Qwen/Qwen2.5-7B-Instruct,DATASET=fincall,LIMIT=3 cloud/osc/slurm/extract.sbatch
#   full corpus:
#     sbatch --export=ALL,MODEL_ID=Qwen/Qwen2.5-7B-Instruct,DATASET=fincall cloud/osc/slurm/extract.sbatch
#     sbatch --export=ALL,MODEL_ID=Qwen/Qwen2.5-32B-Instruct,DATASET=maec   cloud/osc/slurm/extract.sbatch

#SBATCH --job-name=ecvol-llm
#SBATCH --account=PAS0541           # OSC project code
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1           # Ascend=A100-80GB, Cardinal=H100; one GPU is plenty for <=32B
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00            # resumable → short walltime + resubmit is safe
#SBATCH --output=cloud/osc/logs/%x-%j.out

set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
REVISION="${REVISION:-}"            # pin a HF commit for reproducibility (DESIGN §12)
DATASET="${DATASET:-fincall}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"   # context policy: extend to 64k (covers the 61k tail)
YARN="${YARN:-1}"                          # 1 = YaRN rope-scaling on (the chosen >32k policy)
LIMIT="${LIMIT:-0}"                         # 0 = full corpus; set e.g. 3 for a smoke-test job
PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
SIF="$PROJECT/cloud/osc/ecvol-llm.sif"

export HF_HOME="${HF_HOME:-$SCRATCH/ecvol_hf}"   # weights pre-staged here by stage.sh
export APPTAINERENV_HF_HOME="$HF_HOME"
export APPTAINERENV_HF_HUB_OFFLINE=1             # OSC compute nodes have no/limited internet

module load apptainer 2>/dev/null || true
mkdir -p "$PROJECT/cloud/osc/logs"

echo "model=$MODEL_ID rev=${REVISION:-<latest>} dataset=$DATASET sif=$SIF host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

apptainer exec --nv \
    -B "$PROJECT:/workspace" \
    -B "$SCRATCH:$SCRATCH" \
    "$SIF" \
    ecvol featurize llm \
        --dataset "$DATASET" \
        --model-id "$MODEL_ID" \
        ${REVISION:+--revision "$REVISION"} \
        --engine vllm \
        --max-model-len "$MAX_MODEL_LEN" \
        $([ "$YARN" = "1" ] && echo --yarn) \
        $([ "$LIMIT" != "0" ] && echo --limit "$LIMIT") \
        --root /workspace/data

echo "done; features at data/$DATASET/llm_features__$(echo "$MODEL_ID" | tr '/' '_')*.parquet"
