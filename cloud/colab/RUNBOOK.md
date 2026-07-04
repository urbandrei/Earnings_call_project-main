# Colab run-from-scratch runbook — T6.2 LLM extraction

The complete, ordered procedure for the Colab Pro+ route, with every Colab-specific gotcha folded
in. Unlike OSC there is **one** environment (the Colab VM has internet + GPU + torch), so the
login/compute split that tripped up the OSC runbook does not exist here. The two things that do
bite are **persistence** (the VM disk is ephemeral → work off Drive) and **GPU class** (Pro+ does
not guarantee an A100).

---

## 0. Prerequisites — confirm Pro+ AND a usable GPU FIRST

1. A **Colab Pro+** subscription with compute units remaining (Runtime → *View resources*).
2. A GPU runtime: **Runtime → Change runtime type → A100** (fall back to L4). Then **Connect**.
3. Confirm what you actually got — this is the easy-to-miss step, because Pro+ silently hands out
   whatever is free:
   ```python
   !nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
   ```
   - **A100-40GB** → ideal; 7B/8B-fp16 at 65k context fit.
   - **L4-24GB** → ok for 7B; the 65k KV cache is tight (watch for OOM on the longest sections).
   - **T4-16GB** → **insufficient** for 7B-fp16 + long context. Factory-reset the runtime to
     re-roll the GPU (Runtime → *Disconnect and delete runtime* → reconnect), or use an AWQ 4-bit
     checkpoint. Do **not** just lower `--max-model-len` — that would desync the audit from the
     corpus (reproducibility rule).

---

## 1. Drive — put the repo + the two payloads there (one-time)

The Colab VM disk is wiped on every disconnect, so everything durable lives on Drive.

1. Create `MyDrive/ecvol/` and place the repo at
   `MyDrive/ecvol/Earnings_call_project-main/` — either `git clone` it (needs a GitHub PAT since
   the repo is private) from a Colab cell after mounting Drive, or upload a zip and unzip into
   that path.
2. Stage the **two gitignored parquets** `featurize llm` reads (everything else it needs —
   `data/splits/*.csv`, `data/coverage/fincall_llm_labels_rater1.csv` — is git-tracked and is
   already in the repo):
   - From your workstation, upload via **drive.google.com** → into
     `MyDrive/ecvol/Earnings_call_project-main/data/fincall/chunks.parquet` (138 MB) and
     `.../data/maec/chunks.parquet` (46 MB); **or**
   - use the Colab file browser (left sidebar → Files → the mounted Drive) to drag them in.
3. Sanity-check after staging (in a Colab cell, once Drive is mounted):
   ```python
   !ls -la /content/drive/MyDrive/ecvol/Earnings_call_project-main/data/fincall/chunks.parquet \
           /content/drive/MyDrive/ecvol/Earnings_call_project-main/data/maec/chunks.parquet
   ```

---

## 2. Open the launcher notebook

Open `cloud/colab/run.ipynb` in Colab (File → Open notebook → Google Drive → navigate to it, or
upload it). Run the cells top to bottom. Cells 1–2 mount Drive and install; the rest are the
smoke → κ-gate → corpus steps below. The commands are also copy-pasteable from here.

---

## 3. Mount Drive + install (once per session)

```python
from google.colab import drive; drive.mount('/content/drive')
```
```python
%cd /content/drive/MyDrive/ecvol/Earnings_call_project-main
!bash cloud/colab/setup.sh
```
`setup.sh` is idempotent (safe to re-run after a reconnect). Green = a version line, the GPU
name, `ecvol` help, and the two `chunks.parquet` listed. vLLM install is the slow part (~a few
min) and the **most likely thing to break** — see the gotcha index if it errors.

---

## 4. Smoke test — 3 calls (validates the vLLM/YaRN path)

```python
!ecvol featurize llm --dataset fincall --model-id Qwen/Qwen2.5-7B-Instruct \
    --engine vllm --max-model-len 65536 --yarn --limit 3 --root data
```
Green = `... N new sections ...` with no traceback. A `rope_scaling` / `hf_overrides` error =
the YaRN kwarg form needs a one-line tweak in `extract.VLLMEngine` — stop and report it. A CUDA
OOM here = the GPU is too small for the 65k policy (see §0 / gotchas).

Verify the v2 fields are present:
```python
import pandas as pd
d = pd.read_parquet('data/fincall/llm_features__Qwen__Qwen2.5-7B-Instruct.parquet')
print(d.columns.tolist()); print(len(d), 'rows')   # expect both management_optimism + quantitative_specificity
```

---

## 5. κ-GATE — the content gate that blocks the corpus

Extract exactly the 50 train-only audit calls **through the same engine** (`--audit-sample`),
then score them against the rater labels. Scaling to the corpus is blocked until this passes.

```python
!ecvol featurize llm --dataset fincall --model-id Qwen/Qwen2.5-7B-Instruct \
    --engine vllm --max-model-len 65536 --yarn --audit-sample --root data
!ecvol llm-kappa --sheet data/coverage/fincall_llm_labels_rater1.csv \
    --features data/fincall/llm_features__Qwen__Qwen2.5-7B-Instruct.parquet
```
- `GATE ... PASS` (κ>0.6 on the confirmatory core: `guidance_direction`, `hedging_intensity`,
  `surprise_mentions`) → go to §6.
- `FAIL`, or a borderline κ≈0.45–0.6 → **stop**. Try a stronger panel model, or re-block on the
  second rater before any Stage-5/Path-B claim (DECISIONS 2026-06-29). Single-annotator κ is
  always reported as "IAA pending".

---

## 6. Full corpus — once the gate is green

```python
!ecvol featurize llm --dataset fincall --model-id Qwen/Qwen2.5-7B-Instruct \
    --engine vllm --max-model-len 65536 --yarn --root data
!ecvol featurize llm --dataset maec    --model-id Qwen/Qwen2.5-7B-Instruct \
    --engine vllm --max-model-len 65536 --yarn --root data
```
The corpus run **resumes over the already-done audit calls** (same parquet, same model). If the
session times out mid-run, reconnect, re-run §3, and re-issue the command — it skips done rows.
Outputs land at `data/{dataset}/llm_features__{model}.parquet` on Drive (durable).

For the panel, repeat §5–6 per model (re-audit each — reproducibility rule). On A100-40GB use an
**AWQ** checkpoint for 32B; skip 72B (does not fit; see README).

---

## Quick gotcha index (Colab-specific)

| Symptom | Cause | Fix |
|---|---|---|
| Outputs vanish after a disconnect | wrote to the ephemeral VM disk, not Drive | `--root data` under the Drive repo path (§3 `%cd`); the parquet on Drive is the resume store |
| `python>=3.12` assert fails in `setup.sh` | Colab runtime is on an older Python | try a different runtime; or install into a py3.12 env — do not `--no-deps` install onto <3.12 |
| `pip install vllm` errors / torch version clash | vLLM pins a specific torch that fights Colab's preinstalled one | let vLLM win: it's installed first in `setup.sh`; if it still clashes, `pip install -q -U vllm` and restart the runtime once |
| CUDA OOM on the smoke test | GPU too small for 65k-context KV (T4, sometimes L4) | re-roll for an A100 (§0), or use an AWQ checkpoint — **never** silently lower `--max-model-len` for one run only |
| `rope_scaling` / `hf_overrides` TypeError | vLLM version's YaRN kwarg form differs | one-line tweak in `extract.VLLMEngine`; report it |
| Session dies after ~24h / on idle | Colab session caps + idle timeout | expected; resume (§6). No deadline, so multi-session is fine |
| `MISSING chunks.parquet` from `setup.sh` | payloads not staged to Drive | stage the two parquets (§1) |
| Model re-downloads every session | HF cache on the ephemeral disk | acceptable for 7B (~15 GB, a few min); for the panel, set `HF_HOME` to a Drive path to persist weights |
