# HANDOFF.md — Human-pending queue

**Append-only.** This is the parallel to-do list for the **user** while the autonomous loop
([LOOP.md](LOOP.md)) keeps working on unblocked tasks. The loop writes an **Active** entry every
time it hits a hard-stop blocker (and marks the task `[!]` in [TASKS.md](TASKS.md)); the user
clears it whenever convenient and tells the loop, which then un-blocks that task.

Each entry: date · task ID · what the user must do · what unblocks when it's done.

---

## Active (action needed to unblock a task)

*None.* The T3.1 section audit was completed 2026-06-19 (30/30 correct → gate cleared, T3.1
`[x]`). Next pause is the **T3.2 design call** (a decision, surfaced in-session, not a queued
chore). **Optional whenever convenient:** push the local commits (loop infra + T3.1×2) so CI
runs — not strictly required until the Phase-3 boundary (after T3.4) under the gate policy, but a
one-time push is worth it to confirm the new hooks/CI work end-to-end.

### Resolved
- **2026-06-19 · T3.1 section audit — DONE (30/30 correct).** Both operator-handoff and
  analyst-question boundaries accepted as correct Q&A-section starts.

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
