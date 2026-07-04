# Colab Pro+ — T6.2 LLM feature extraction

Run constrained LLM extraction (`ecvol featurize llm --engine vllm`) on a Google Colab Pro+ GPU.
This is the **rerouted** compute path after the OSC allocation lapsed (DECISIONS 2026-07-04,
superseding the 2026-06-24 OSC plan). Same engine as OSC — vLLM + Outlines + the YaRN >32k
policy — just on Colab's GPU instead of a Slurm A100. The corpus is **FinCall + MAEC**; the run
is a **multi-model panel** (exploration: does a stronger model produce more signal?), one model
at a time.

**Reproducibility rule (do not break):** a model's κ-audit must score the *exact* weights+quant
that produced its corpus features. So the 50 audited calls (`--audit-sample`) are extracted by
the same Colab session + model that does the corpus. Don't audit a 7B and ship a 32B corpus
without re-auditing the 32B.

## Why the pipeline barely changes

`ecvol featurize llm --engine vllm` already exists and is engine-agnostic; the only new piece is
`--audit-sample`, which extracts exactly the 50 train-only κ-gate calls through the vLLM engine
so the gate is scored on the same weights as the corpus. Everything else — resume store, YaRN,
the confirmatory-core κ-gate (`ecvol llm-kappa`) — is unchanged from the OSC package.

## Files

| File | Role |
|---|---|
| `run.ipynb` | The Colab launcher notebook: mount Drive → `setup.sh` → smoke → audit+κ → corpus. |
| `setup.sh`  | Idempotent per-session install (vLLM + Outlines + the `ecvol` CLI). |
| `RUNBOOK.md`| From-scratch ordered procedure + the Colab gotcha index. |

## The two things that make Colab different from OSC

1. **Persistence = Google Drive.** The Colab VM's local disk is wiped on every disconnect, so the
   repo (code + the `data/` outputs) lives on **Drive**, mounted at `/content/drive`. Because the
   per-model parquet is the resume store and it sits on Drive, a session timeout + reconnect just
   resumes — already-done `(call_id, section)` rows are skipped. There is **no deadline** (flat
   Pro+ subscription; DECISIONS 2026-07-04), so a multi-session corpus run is fine.
2. **GPU class is not guaranteed.** Pro+ usually offers an **A100-40GB** (ideal), sometimes an
   **L4-24GB** (ok), occasionally a **T4-16GB** (insufficient for 7B fp16 + long-context KV).
   Check `nvidia-smi` at the top of `setup.sh`; if you land a T4, factory-reset the runtime to
   re-roll, or fall back to an AWQ checkpoint (see the panel + gotcha notes).

## Config / staging — do these once

| Where | Set |
|---|---|
| Google Drive | Put the **repo** at `MyDrive/ecvol/Earnings_call_project-main` (git-clone with a PAT, or upload a zip). |
| Drive `data/fincall/chunks.parquet`, `data/maec/chunks.parquet` | The **two gitignored payloads** (138 MB + 46 MB) `featurize llm` reads. Everything else it needs — `data/splits/*.csv`, `data/coverage/fincall_llm_labels_rater1.csv` — is git-tracked and arrives with the repo. |
| `run.ipynb` `REPO=` | Path to the repo on Drive (default `/content/drive/MyDrive/ecvol/Earnings_call_project-main`). |
| `REVISION` (optional) | Pin the HF commit per model for reproducibility (DESIGN §12). |

## Context policy — >32k sections (already wired)

FinCall's longest sections reach ~61k tokens, past Qwen2.5's 32k native context. Policy
(DECISIONS 2026-06-29): **extend, don't truncate** — run with `--max-model-len 65536 --yarn`, so
every section is processed whole. The policy must be identical for a model's κ-audit and its
corpus run — since `--audit-sample` uses the same command, this holds automatically. On a smaller
GPU (L4/T4) the 65k KV cache is the first thing to OOM; that is the GPU-class limitation above,
not a policy change — don't quietly lower `--max-model-len` for one run and not the audit.

## Workflow (the notebook automates this)

```bash
# 0. (once) repo + the two chunks.parquet on Drive; Pro+ runtime with a GPU attached.

# 1. mount Drive + install (run.ipynb cells 1-2, or:)
bash cloud/colab/setup.sh

# 2. smoke test — 3 calls, validates the vLLM/YaRN path cheaply
ecvol featurize llm --dataset fincall --model-id Qwen/Qwen2.5-7B-Instruct \
    --engine vllm --max-model-len 65536 --yarn --limit 3 --root data

# 3. κ-GATE — extract the 50 audit calls (same engine) then score; BLOCKS the corpus on κ>0.6
ecvol featurize llm --dataset fincall --model-id Qwen/Qwen2.5-7B-Instruct \
    --engine vllm --max-model-len 65536 --yarn --audit-sample --root data
ecvol llm-kappa --sheet data/coverage/fincall_llm_labels_rater1.csv \
    --features data/fincall/llm_features__Qwen__Qwen2.5-7B-Instruct.parquet
#   PASS (κ>0.6 confirmatory core) → step 4.  FAIL → stop; a stronger model or a schema revisit.

# 4. full corpus — resumes over the already-done audit calls; safe to re-run after a disconnect
ecvol featurize llm --dataset fincall --model-id Qwen/Qwen2.5-7B-Instruct \
    --engine vllm --max-model-len 65536 --yarn --root data
ecvol featurize llm --dataset maec    --model-id Qwen/Qwen2.5-7B-Instruct \
    --engine vllm --max-model-len 65536 --yarn --root data

# 5. outputs land at data/{dataset}/llm_features__{model}.parquet ON DRIVE (durable).
```

## Model panel

Core ladder (clean capability comparison, same family): **Qwen2.5-7B-Instruct →
Qwen2.5-32B-Instruct**, plus a **cross-family** check (`meta-llama/Llama-3.1-8B-Instruct`). The
best-κ model that clears **0.6** is the confirmatory feature set (T6.3); the rest are exploratory.

**Colab GPU-fit caveat (differs from OSC's A100-80GB):** on a single Colab A100-**40GB**,
7B-fp16 and Llama-3.1-8B-fp16 fit comfortably with the 65k context. **32B-fp16 does not fit
40 GB** — serve an **AWQ/GPTQ 4-bit** 32B checkpoint (e.g. `Qwen/Qwen2.5-32B-Instruct-AWQ`) and
audit that exact quant (reproducibility rule). **72B is not feasible on Colab** and is dropped
here (it was an OSC-only, 80 GB option) — revisit only if a paid API or larger GPU reappears.

## Cost / pacing

Flat **Colab Pro+ subscription**, no per-token budget (DECISIONS 2026-07-04). Use the monthly
compute-unit allotment; **no deadline**, so spread the panel across sessions/months as units
refill. Get a real per-model rate from the smoke test (step 2) before committing a long run —
`(sections/s)` × 9,443 corpus sections projects the wall-clock. vLLM on an A100-40GB does the
7B corpus in a handful of GPU-hours (prefill-bound on the long sections).
