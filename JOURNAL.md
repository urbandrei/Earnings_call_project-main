# JOURNAL.md — Work Journal

**Append-only, newest entry last.** A chronological record of work performed on the project: what was done, what was found, what broke, what's next. The journal is the project's narrative memory — [TASKS.md](TASKS.md) says *what state* the work is in; this file says *how it got there*.

## What gets an entry

Append an entry at the end of every working session (or completed task, whichever is finer-grained). Include:

- **Date + scope** — what task(s)/topic the session touched (use task IDs where applicable).
- **Done** — work completed, with enough specifics that a future reader can retrace it (commands, files, run IDs).
- **Found** — results, measurements, surprises, dead ends. Negative findings are first-class: an experiment that failed is information.
- **Sources** — papers/links/model cards consulted (anything reusable should also land in DESIGN.md §13).
- **Next** — the intended next step, so any session can pick up where the last one stopped.

Keep entries factual and compact. Decisions that change the design belong in [DECISIONS.md](DECISIONS.md) (link the journal entry to them); the journal records the work and evidence around them.

---

## Log

### 2026-06-12 — Project review, literature research, design doc
- **Done:** Full review of the inherited legacy project (12 notebooks, Paper.pdf, papers/ library); established it is unrunnable (data lost with the `D:` drive, no environment spec). Three parallel web-research sweeps: (1) post-KeFVP SOTA and documented pitfalls, (2) open dataset options, (3) open-source model landscape for consumer GPUs. Wrote DESIGN.md (research questions, data/model/eval design, 9-phase plan, risk register).
- **Found:** Key literature facts that shaped the design — "Same Company, Same Signal" (arXiv 2412.18029): transcript models largely memorize ticker identity and training-free past-vol baselines beat them; financial-NLP reproducibility crisis (~14% of results reproduce, arXiv 2504.07274); FinCall-Surprise (arXiv 2510.03965, Apache-2.0, 2,688 calls 2019–2021 with full audio) is the best truly-open multimodal dataset; legacy +0.73%-vs-KeFVP result is single-seed and not defensible.
- **Sources:** consolidated in DESIGN.md §13 (R1–R27, D1–D8).
- **Next:** docs restructure, then T0.1 scaffolding.

### 2026-06-12 — Docs restructure
- **Done:** Renamed `Earnings_call_project-main/` → `legacy/` (read-only; deleted its stale CLAUDE.md); moved DESIGN.md to project root and removed `new/`; extracted the full task breakdown into TASKS.md (living tracker, 30 tasks T0.1–T8.3); wrote OLDWORK.md (legacy summary, marked outdated); wrote root CLAUDE.md (working principles, automation/testing discipline, exploration policy); moved the decision log out of DESIGN.md into DECISIONS.md; created this journal. Git deliberately not initialized yet — first subtask of T0.1.
- **Found:** Karpathy reference check: he never published a CLAUDE.md template; the viral file is Forrest Chang's distillation of Karpathy's Jan-2026 post on four agent failure modes (silent assumptions, code hypertrophy, collateral changes, no verifiable success criteria) — CLAUDE.md's principles built on those primary sources.
- **Next:** T0.1 — `git init`, .gitignore, initial commit, package skeleton, CI.
