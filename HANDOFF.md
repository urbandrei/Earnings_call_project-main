# HANDOFF.md — Human-pending queue

**Append-only.** This is the parallel to-do list for the **user** while the autonomous loop
([LOOP.md](LOOP.md)) keeps working on unblocked tasks. The loop writes an **Active** entry every
time it hits a hard-stop blocker (and marks the task `[!]` in [TASKS.md](TASKS.md)); the user
clears it whenever convenient and tells the loop, which then un-blocks that task.

Each entry: date · task ID · what the user must do · what unblocks when it's done.

---

## Active (action needed to unblock a task)

- **2026-06-24 · T6.2 — the corpus run goes to OSC (local OOM'd); needs OSC access + a context-policy call.**
  The local ETA probe proved the 16 GB GPU **can't** run extraction at full context (CUDA OOM on
  long-section prefill; see `data/coverage/llm_probe_report.md`). Everything for the cloud burst is
  built (`cloud/osc/`) and the $1000 spend is approved (DECISIONS 2026-06-24). To proceed:
  1. **Confirm your OSC allocation** — project/account code (`PASxxxx`) + cluster (recommend **Ascend
     A100-80GB** or **Cardinal H100**). Put the account in `cloud/osc/slurm/extract.sbatch`.
  2. **Build + stage on a login node:** `apptainer build cloud/osc/ecvol-llm.sif …` then
     `bash cloud/osc/stage.sh Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-32B-Instruct` (README has the
     full workflow). rsync the gitignored `data/{fincall,maec}/{calls,chunks}.parquet` + `data/splits/`.
  3. **Decide the >32k-token section policy** (design call): truncate-to-context vs YaRN-extend to
     ~64k (FinCall max section = 61k tokens). Must match between a model's κ-audit and its corpus run.
  - **What unblocks:** per-model corpus extraction (panel: 7B → 32B, cross-validated) → `ecvol
    llm-kappa` per model → T6.3. The κ-audit's 50-call extraction runs on OSC too (audit must match
    corpus), so the human labeling (below) can proceed in parallel.

- **2026-06-24 · T6.1 — read 20 calls + fill the labeling sheet, then sign off the schema.**
  The v1 LLM feature schema + rubric + prompts + reading tooling are built and committed; T6.1's
  acceptance test is **human** (two passes over 10 calls agree the schema applies), so the loop
  **pauses here**. To unblock:
  1. The pack is already generated at the **50-call audit size** (aligned with the κ-audit set):
     read the transcripts in `data/fincall/llm_reading/*.md` (regenerate any time with
     `ecvol featurize llm-audit-sample --dataset fincall --n 50 --seed 0`).
  2. Read the rubric `docs/llm_feature_rubric.md` and fill `data/coverage/fincall_llm_label_sheet.csv`
     (one row per call×section; "NA" cells are Q&A-only fields that don't apply to prepared remarks).
     For two-pass agreement, a second rater fills a copy of the 10-call subset.
  3. **Either sign off the v1 schema** ("schema signed off, continue") **or list rubric/field edits**
     you want — I'll revise and bump `PROMPT_VERSION` before extraction.
  - **What unblocks:** extraction runs against the **frozen** schema + prompt version on OSC (see
    the T6.2 entry above — local OOM'd); the filled sheet then feeds `ecvol llm-kappa` per model for
    the **κ>0.6** gate. Labeling can proceed in parallel with the OSC setup.
  - **Note:** all of Phase 6 (T6.2/T6.3) and Phase 7 T7.1 (API key) are human-gated, so the loop
    has no unblocked task to skip to — it genuinely waits on these.

### Resolved
- **2026-06-24 · PHASE-5 BOUNDARY CHECKPOINT — DONE (CI green).** Phase 5 (T5.1 fusion + T5.2
  Result Table 4) pushed; user confirmed GitHub Actions green. Loop advanced to Phase 6.
- **2026-06-19 · T3.1 section audit — DONE (30/30 correct).** Both operator-handoff and
  analyst-question boundaries accepted as correct Q&A-section starts.
- **2026-06-19 · §4 framing-gate decision — DONE (provisional Path B).** Adopt "rigorous
  re-examination" now; revisit/flip to Path A only if Phase-4 audio beats the floor surviving the
  identity controls. (DECISIONS 2026-06-19.)
- **2026-06-24 · §4 framing-gate revisit (post-audio) — DONE (Path B kept PROVISIONAL).** Audio is
  inert beyond past-vol (shuffle≈real even global), WavLM identity probe 76%, no gender disparity →
  Path A criterion not met; user chose to keep Path B provisional and revisit after Phase-5 fusion
  + Phase-6 LLM. (DECISIONS 2026-06-24.)

---

## Recurring (every cycle, not a blocker)

- **Push + confirm CI at each phase boundary.** The loop pauses after the last task of a phase
  and notifies you. Run `git push`, check GitHub Actions is green, then tell the loop "CI green,
  continue". (Agent cannot push or read CI — CLAUDE.md.)
- **Answer design calls.** The loop stops and asks on any non-trivial modeling/design choice
  (which model, pooling, ablations, "what counts as a win"). Reply and it resumes.

---

## Anticipated (FYI — not blocking yet, listed so you can prepare)

These come from the backlog scan; the loop will move each into **Active** when its task is reached.

- **Phase 4 · T4.3** — *(optional)* HuggingFace **gated-model license accept + `HF_TOKEN`** for
  `pyannote/speaker-diarization` (behind a config flag). Other audio/text models (WavLM,
  emotion2vec, BGE/GTE, FinBERT, Qwen) are ungated — no action needed for those.
- **Phase 4 · T4.3 / Phase 8 · T8.3** — possible/definite **cloud burst**: needs a budget +
  your go-ahead (DECISIONS.md entry). Definite for the Stage-6 QLoRA / audio-LLM experiments.
- **Phase 6 · T6.1 / T6.2 (and exploration TX1)** — **human reading + labeling**: design the
  LLM feature schema from ~20 calls, then a **50-call audit with κ>0.6** that blocks
  corpus-scale extraction. The loop builds the labeling tooling; the agreement numbers are yours.
- **Phase 7 · T7.1** — **EarningsCall/EarningsCast API key** (likely paid) + a **ToS review**
  before any bulk pull.
- **Phase 8 · T8.1** — **data-host account** (e.g. Zenodo / HF Datasets) to publish the released
  derived-feature archives.
- **Phase 8 · T8.2** — **advisor/co-author review**, venue choice, arXiv submission.
- **Phase 3 · T3.4** — the **§4 framing-gate review** (a decision, not a key): the identity
  controls' outcome triggers it and it requires your call either way.
