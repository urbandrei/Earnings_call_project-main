#!/bin/bash
# Idempotent Colab session setup for T6.2 LLM extraction. Run once per Colab session, AFTER
# mounting Drive and `cd`-ing into the repo (which lives on Drive so code + outputs persist
# across the VM's disconnects). Unlike OSC, the Colab VM has internet + a GPU + torch/CUDA
# preinstalled, so there is no container build and no offline weight-staging: weights download
# from the HF Hub on first model use.
#
# Usage (in a Colab cell):
#   !bash cloud/colab/setup.sh
set -euo pipefail

REPO="${1:-$PWD}"
cd "$REPO"

# The package pins requires-python>=3.12; recent Colab runtimes satisfy this. Fail loudly rather
# than let a --no-deps install silently target the wrong interpreter.
python -c 'import sys; assert sys.version_info[:2] >= (3, 12), f"need python>=3.12, got {sys.version.split()[0]} — see RUNBOOK gotcha index"'

# vLLM manages its own torch pin — install it first so it wins any version resolution. Then the
# ecvol CLI (console script `ecvol`) with --no-deps + only the runtime deps `featurize llm`
# imports (mirrors cloud/osc/apptainer/ecvol-llm.def; skips arch/lightgbm/yfinance/etc.).
pip install -q "vllm>=0.6" "outlines>=0.1"
pip install -q --no-deps .
pip install -q "pandas>=3.0.3" "pyarrow>=24" "pydantic>=2.13" "typer>=0.15" scipy scikit-learn pyyaml

echo "== versions =="
python -c "import vllm, outlines, pandas, pydantic; print('vllm', vllm.__version__, '| outlines', outlines.__version__, '| pandas', pandas.__version__, '| pydantic', pydantic.VERSION)"
echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "== ecvol =="
ecvol --help | head -3
echo "== inputs (must exist under data/) =="
ls -la data/fincall/chunks.parquet data/maec/chunks.parquet 2>/dev/null || \
  echo "MISSING chunks.parquet — stage the two gitignored payloads to Drive (see README §Data)"
