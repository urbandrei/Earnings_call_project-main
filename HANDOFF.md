# HANDOFF.md — Human-pending queue

**Append-only.** This is the parallel to-do list for the **user** while the autonomous loop
([LOOP.md](LOOP.md)) keeps working on unblocked tasks. The loop writes an **Active** entry every
time it hits a hard-stop blocker (and marks the task `[!]` in [TASKS.md](TASKS.md)); the user
clears it whenever convenient and tells the loop, which then un-blocks that task.

Each entry: date · task ID · what the user must do · what unblocks when it's done.

---

## Active (action needed to unblock a task)

- **2026-06-24 · PHASE-5 BOUNDARY CHECKPOINT (gate policy) — push + confirm CI.** Phase 5
  (T5.1 fusion + T5.2 Result Table 4) is complete; the loop **pauses here** until you push the
  local commits (Phase 5: fusion, grid, the two parallel-session commits paper/ + validate_results)
  and confirm GitHub Actions is green. Then reply "CI green, continue" and the loop starts
  **Phase 6 — LLM structured features**.
  - **⚠️ Heads-up — Phase 6 carries the first hard human-labor blocker:** T6.2 (and exploration
    TX1) require a **50-call human audit with κ>0.6** before corpus-scale LLM extraction, and T6.1
    needs you to read ~20 calls to co-design the feature schema. I'll build all the labeling tooling
    and pre-fill what I can, but the agreement numbers are yours — budget for a labeling session.
    T6.2 may also want a **local LLM** (Qwen2.5-7B-Instruct 4-bit via the GPU stack) — no new key,
    but a model download + a VRAM/throughput ETA gate like the audio runs.
  - *(Phase-4 checkpoint cleared: CI was confirmed green; `setup.md` is now gitignored.)*

### Resolved
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
