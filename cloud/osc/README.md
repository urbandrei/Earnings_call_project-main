# OSC cloud burst — T6.2 LLM feature extraction

Run constrained LLM extraction (`ecvol featurize llm --engine vllm`) at scale on the Ohio
Supercomputer Center when the local RTX 5060 Ti is too slow (the **>20h rule** — see
DECISIONS 2026-06-24). The corpus is **FinCall + MAEC**; the run is a **multi-model panel**
(exploration: does a stronger model produce more signal?), one Slurm submission per model.

**Reproducibility rule (do not break):** a model's κ-audit must score the *exact*
weights+quant that produced its corpus features. So the 50 audited calls must be extracted by
the same container+model that does the corpus. Don't audit a 7B locally and ship a 32B corpus
from here without re-auditing the 32B.

## Files

| File | Role |
|---|---|
| `apptainer/ecvol-llm.def` | Apptainer image: vLLM base + Outlines + the `ecvol` CLI. |
| `slurm/extract.sbatch` | One extraction job; parameterized by `MODEL_ID`, `DATASET`, `REVISION`. |
| `stage.sh` | Pre-download model weights to `$SCRATCH` (compute nodes are offline). |

## Config block — fill these in

| Where | Set |
|---|---|
| `slurm/extract.sbatch` `#SBATCH --account=` | your OSC project code (e.g. `PAS1234`). |
| `slurm/extract.sbatch` `#SBATCH --gpus-per-node=` | keep `1` (A100-80GB / H100 handle ≤32B). |
| `slurm/extract.sbatch` `#SBATCH --time=` | walltime; resumable, so 2–4h + resubmit is safe. |
| `HF_HOME` (env) | defaults to `$SCRATCH/ecvol_hf`; weights live here. |
| `REVISION` | pin the HF commit per model for reproducibility (DESIGN §12). |

## Cluster choice (you said "pick for me")

- **Ascend (A100-80GB)** — recommended default: 80 GB runs 7B/32B unquantized, even 72B at
  4-bit, comfortably on one GPU.
- **Cardinal (H100)** — fastest; use if you have a Cardinal allocation.
- **Pitzer (V100-16/32GB)** — works for 7B; 32B needs 4-bit/AWQ and is slow. Last resort.

The package is cluster-agnostic; only `--account` and `--gpus-per-node` may need tweaking.

## Context policy — >32k sections (already wired)

FinCall's longest sections reach ~61k tokens, past Qwen2.5's 32k native context. Policy
(DECISIONS 2026-06-29): **extend, don't truncate** — `extract.sbatch` runs with
`MAX_MODEL_LEN=65536` + `YARN=1` (vLLM YaRN rope-scaling) by default, so every section is
processed whole. To override per submission, pass `--export=...,MAX_MODEL_LEN=...,YARN=0`. The
policy must be identical for a model's κ-audit and its corpus run — since the 50 audit calls are
a FinCall subset extracted by the same job, this holds automatically.

## Workflow

```bash
# 0. clone the repo on OSC, then transfer the gitignored payloads `featurize llm` READS.
#    Extraction reads ONLY chunks.parquet (calls.parquet is gitignored too but NOT needed;
#    data/splits/*.csv are git-tracked and arrive with the clone). So only two files move:
#    rsync -avPR data/fincall/chunks.parquet data/maec/chunks.parquet \
#                <osc_user>@ascend.osc.edu:~/Earnings_call_project-main/
#    (scp alternative — Windows-native, no rsync: see the Data transfer section below)

# 1. build the image (login node with internet; ~10 min)
module load apptainer
apptainer build cloud/osc/ecvol-llm.sif cloud/osc/apptainer/ecvol-llm.def

# 2. pre-stage weights for every model in the panel (login node; offline compute nodes)
bash cloud/osc/stage.sh Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-32B-Instruct

# 3. submit extraction — one job per (model × dataset)
for M in Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-32B-Instruct; do
  for D in fincall maec; do
    sbatch --export=ALL,MODEL_ID=$M,DATASET=$D cloud/osc/slurm/extract.sbatch
  done
done

# 4. outputs land in the bind-mounted repo: data/{dataset}/llm_features__{model}.parquet
#    rsync them back to your workstation.

# 5. κ-audit each model LOCALLY against the filled human sheet (the content gate).
#    The corpus run already extracted the 50 audit calls (they're a FinCall subset), so the
#    same container+model covers them — no separate audit job (reproducibility rule above).
#    Gate is on the confirmatory core; weak/exploratory fields are reported only.
#    ecvol llm-kappa --sheet data/coverage/fincall_llm_labels_rater1.csv \
#                    --features data/fincall/llm_features__Qwen__Qwen2.5-32B-Instruct.parquet
```

## Data transfer (from Windows)

Only two gitignored files move: `data/fincall/chunks.parquet` (138 MB) +
`data/maec/chunks.parquet` (46 MB). `--account` is the **project** code (`PAS0541`); the
ssh/scp login uses your **personal OSC username**, not the project code.

```powershell
# run from the project root. Clone the repo on OSC FIRST (the data/ dirs need to exist).
# OSC will prompt for your password + Duo MFA.

# scp — built into Windows 10/11, no rsync needed:
ssh <osc_user>@ascend.osc.edu "mkdir -p ~/Earnings_call_project-main/data/fincall ~/Earnings_call_project-main/data/maec"
scp data/fincall/chunks.parquet <osc_user>@ascend.osc.edu:~/Earnings_call_project-main/data/fincall/
scp data/maec/chunks.parquet    <osc_user>@ascend.osc.edu:~/Earnings_call_project-main/data/maec/
```

- **Resumable alternative (rsync):** needs WSL or Git-Bash-with-rsync (Windows has no native
  rsync). From a WSL/Git-Bash shell in the project root:
  `rsync -avPR data/fincall/chunks.parquet data/maec/chunks.parquet <osc_user>@ascend.osc.edu:~/Earnings_call_project-main/`
  (`-R` recreates the `data/fincall/…` path + dirs; `-P` resumes a partial transfer.)
- **No-CLI alternative:** OSC OnDemand (`ondemand.osc.edu`) → Files → navigate to
  `~/Earnings_call_project-main/data/fincall/` → drag-drop `chunks.parquet`; repeat for `maec`.
- After transfer, sanity-check on OSC: `ls -la ~/Earnings_call_project-main/data/{fincall,maec}/chunks.parquet`.

## Model panel

Core ladder (clean capability comparison, same family): **Qwen2.5-7B-Instruct →
Qwen2.5-32B-Instruct**. Optional within budget: **Qwen2.5-72B-Instruct** (4-bit) and a
**cross-family** check (e.g. `meta-llama/Llama-3.1-8B-Instruct`). Add models by extending the
loop in step 3 + `stage.sh`. The best-κ model that clears **0.6** is the confirmatory feature
set (T6.3); the rest are exploratory comparisons.

## Cost / ETA

- Budget: **$1000** (DECISIONS 2026-06-24). vLLM on one A100/H100 does the FinCall+MAEC corpus
  (~2.6k FinCall + MAEC calls × 2 sections, long-context prefill-bound) in a handful of
  GPU-hours per model — the full panel should land **well under budget**; cost is not the
  binding constraint, the local >20h convenience rule is.
- Get a real per-model number with a short sample job first: `--export=...,MODEL_ID=...` plus a
  temporary `--limit` (add to the sbatch) before committing the full panel.
- **No paid job runs until the DECISIONS spend entry is approved** (it is — see the 2026-06-24
  entry — within the $1000 envelope; log any overage as a new entry).
