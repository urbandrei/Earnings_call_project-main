# T6.2 local ETA probe — result (2026-06-24)

**Command:** `ecvol featurize llm-eta --dataset fincall --n-probe 50 --seed 0`
**Model:** Qwen/Qwen2.5-7B-Instruct, bitsandbytes nf4 4-bit, transformers + Outlines (greedy).
**GPU:** RTX 5060 Ti, 16 GB (Blackwell sm_120, torch 2.11+cu128).

## Outcome: local run is INFEASIBLE (not merely slow) → route to OSC

The model loads fine in 4-bit (~9.3 GB VRAM), but extraction **CUDA-OOMs** during the
long-context prefill of a single transcript section. Cause: SDPA attention is O(seq²) in the
sequence length and there is no flash-attention wheel on Windows, so a multi-thousand-token
section materializes an attention tensor that overflows the 16 GB card. Zero sections completed
before OOM. This is a feasibility wall, so the >20h routing rule (DECISIONS 2026-06-24) is moot:
the corpus run goes to **OSC** regardless.

## Why cloud is comfortable

Section token-length distribution (Qwen2.5 tokenizer, full corpus):

| corpus | sections | median | p90 | p99 | max | total input tokens |
|---|---|---|---|---|---|---|
| FinCall | 5,147 | 5,448 | 8,323 | 12,192 | 61,141 | 29M |
| MAEC | 4,296 | 1,840 | 4,625 | 7,488 | 11,137 | 9M |
| **total** | **9,443** | — | — | — | — | **~38M** |

On a cloud A100-80GB / H100 with vLLM (paged + flash attention), ~38M prefill tokens is on the
order of **a couple GPU-hours for the 7B**, scaling ~linearly with model size for the panel
(32B ≈ 3–4×). Output is tiny (constrained decode). The full panel lands **well under the $1000
budget**; get an exact per-model number from a short `--limit` sample job on OSC first.

## Open design call — the >32k-token section tail

Qwen2.5's native context is 32k; the FinCall max section is 61k tokens (only the extreme ~tail
exceeds 32k). Before the cloud run, decide:
- **Truncate** sections to the model context (e.g. `--max-input-tokens`/char cap), logging the
  affected count — simplest, negligible info loss on a handful of sections; or
- **Extend context** to ~64k via vLLM `--max-model-len 65536` + YaRN rope scaling — covers all,
  but long-context can slightly degrade quality.

(Whatever is chosen must be identical for a model's κ-audit sample and its corpus run.)
