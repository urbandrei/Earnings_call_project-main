# HANDOFF.md — Human-pending queue

**Append-only.** This is the parallel to-do list for the **user** while the autonomous loop
([LOOP.md](LOOP.md)) keeps working on unblocked tasks. The loop writes an **Active** entry every
time it hits a hard-stop blocker (and marks the task `[!]` in [TASKS.md](TASKS.md)); the user
clears it whenever convenient and tells the loop, which then un-blocks that task.

Each entry: date · task ID · what the user must do · what unblocks when it's done.

---

## Active (action needed to unblock a task)

- **2026-06-24 · PHASE-4 BOUNDARY CHECKPOINT (gate policy) — push + confirm CI.** Phase 4
  (T4.1–T4.4, audio ladder) is complete; the loop **pauses here** until you push the local commits
  (~20 since the last push: Phase-3 + all of Phase 4 + the `paper/` scaffold) and confirm GitHub
  Actions is green. Then reply "CI green, continue" and the loop starts **Phase 5 (fusion +
  full ablation grid → Result Table 4, the main paper table)**. *Note:* CI installs only the `dev`
  group, so the new `gpu`/`audio` deps (torch/funasr/opensmile) are **not** in CI — the audio/GPU
  tests skip there by design; confirm the suite still goes green.
  - *Untracked, left for you:* `setup.md` (a generic project-governance bootstrap doc from a
    parallel session) is uncommitted at the repo root — commit it or move it out as you see fit;
    it's unrelated to the ecvol pipeline.

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
