# CLAUDE.md

## Project

Research project predicting post-earnings-call stock volatility from multimodal data (transcript text + call audio) using open-source LLMs and audio models, open data, and rigorously honest evaluation — targeting a publishable paper. This is a ground-up rework of an abandoned 2024 university project.

### Document map — read before working

| File | Role |
|---|---|
| [DESIGN.md](DESIGN.md) | **Source of truth.** Research questions, data/model/eval design, phase plan, risk register. Never silently edited — deviations go through DECISIONS.md first. |
| [TASKS.md](TASKS.md) | **Living tracker.** Every task (T0.1–T8.3) with acceptance tests. Update statuses/notes as part of finishing work. A task is done only when its acceptance test passes. |
| [DECISIONS.md](DECISIONS.md) | **Append-only decision log.** Every deviation from DESIGN.md, scope change, exploration promotion, spend approval, and gate outcome — dated, with rationale and rejected alternatives. |
| [JOURNAL.md](JOURNAL.md) | **Append-only work journal.** One entry per working session: done / found / sources / next. The project's narrative memory. |
| [OLDWORK.md](OLDWORK.md) | Summary of the legacy project. **Outdated — reference only.** Never extend or rerun it. |
| `legacy/` | The abandoned project's artifacts (notebooks, Paper.pdf, `papers/` literature). **Read-only.** The literature folder is genuinely useful. |

**Why this split:** DESIGN.md stays a stable contract; TASKS.md holds current state; DECISIONS.md explains *why* the contract changed; JOURNAL.md explains *how* the work actually went. Together they let any future session (human or agent) reconstruct both the state and the reasoning without re-deriving it.

## Core principles

1. **Think before coding; surface, don't assume.** If a task is ambiguous, state the interpretations and ask — or pick one and *say so explicitly*. If a simpler approach exists than the one requested, say that too. Silent assumptions are the top failure mode.
2. **Simplicity first.** Minimum code that solves the stated task. No speculative abstractions, no unrequested configurability, no "while I'm here" features. This repo aspires to nanoGPT-style readability: small, cohesive, hackable.
3. **Surgical changes.** Every changed line traces to the task at hand. Don't reformat, refactor, or "improve" untouched code. Small diffs are reviewable diffs.
4. **Goal-driven, verifiable execution.** Convert every task into a falsifiable success criterion before writing code — that's what the acceptance tests in TASKS.md are. "Fix the bug" means "write a test that reproduces it, then make it pass."

## Workflow discipline

- **Small, focused commits.** One logical change per commit, descriptive message, commit after each green test/acceptance check — the history must support rolling back any single step. Never bundle a refactor with a behavior change.
  - *Current state:* the git repo is **not yet initialized** — that is the first subtask of T0.1. Until then, make no destructive edits without keeping a copy.
- **Update TASKS.md as you work**, not afterwards: status when you start (`[~]`), subtask checkboxes as they land, notes + date when done (`[x]`).
- **Journal every session.** Append a [JOURNAL.md](JOURNAL.md) entry at the end of each working session (or completed task): date + scope, what was done, what was found (negative results included), sources consulted, intended next step. Rationale: work that isn't journaled gets re-derived or re-run later; the journal is what makes sessions resumable. Append-only — never rewrite past entries.
- **Log every decision.** Any deviation from DESIGN.md, scope change, exploration promotion, or spend approval gets a dated [DECISIONS.md](DECISIONS.md) entry — decision, rationale, alternatives rejected — *before or alongside* the change, never retroactively reconstructed. Reversals are new entries, not edits. Rationale: the design contract is only trustworthy if every divergence is traceable.
- **Acceptance test = definition of done.** If the test seems wrong, fix it via a DECISIONS.md entry first, then implement.
- **Sequencing matters.** Phases retire risk early (data → baselines → models; controls before expensive extraction). Don't jump ahead of an unmet gate.

## Automation & deep testing

- **Everything runs through the `ecvol` CLI.** No manual one-off steps, no "I ran this snippet in a notebook once" pipeline stages. If a human has to remember to do it, it will be forgotten — script it, make it idempotent and resumable.
- **Tests are the project's immune system.** Non-negotiable, CI-enforced:
  - leakage assertions on every split (temporal embargo, ticker-disjointness, no post-`as_of` data access),
  - unit tests for target computation against analytically known values,
  - byte-identical regeneration of all result tables from run artifacts (`ecvol report`),
  - deterministic feature re-extraction (cache hit = bit-identical).
- **The most dangerous bugs here are silent.** This project's known silent killers — check for them in every review:
  - data leakage across split boundaries (a target window crossing into test),
  - ticker-identity memorization masquerading as signal (see DESIGN.md §3.1),
  - silently dropped tickers/calls (every exclusion needs a reason code),
  - unseeded or single-seed results presented as findings,
  - features reading price data after the call's `as_of` timestamp.
- Reproducibility is a feature with tests: pinned lockfile, SHA-256 manifests for every data file, config hash + git SHA in every run artifact, seeds in configs.

## Exploration policy

This is an **explorative research project**. Novel techniques, frontier models, unusual ideas, and "weird concepts" are *encouraged* — the modeling ladder is a floor, not a ceiling. Rules that keep exploration honest:

- **Explore freely in `notebooks/` or scratch branches.** The pipeline (`src/ecvol`) only gains code through the normal task/commit discipline.
- **Document every experiment in [JOURNAL.md](JOURNAL.md)**, including dead ends: hypothesis → method → result → sources. A negative exploration result is information; an undocumented one is wasted compute.
- **Reference everything.** Borrowed techniques, papers, model cards, blog posts — cite with URLs (in the journal entry; durable references also go to DESIGN.md §13). No mystery methods.
- **Promotion requires a [DECISIONS.md](DECISIONS.md) entry.** An exploration that should change the design or add a task gets a dated entry there and, if needed, a new task ID in TASKS.md.
- **Label honestly.** Confirmatory vs. exploratory per DESIGN.md §7.5 — exploratory wins don't get promoted to headline claims without going through the protocol.
- Further literature research is always in scope: the field moves fast (see DESIGN.md §3); checking for newer models/datasets/findings before a phase starts is good practice, not scope creep.

## Environment

- Windows 11; PowerShell is the primary shell (POSIX bash also available). Paths use the project root `C:\Users\andre\Desktop\projects\Earnings_call_project-main\`.
- Compute: single consumer GPU (16–24 GB VRAM) — respect the per-component VRAM budget in DESIGN.md §8.3; components run sequentially, never co-resident. Cloud bursts are possible but require a DECISIONS.md entry with budget.
- Python via **uv** (lockfile committed) once T0.1 lands; configs are pydantic-validated YAML; `data/` and `artifacts/` are gitignored payloads tracked by committed manifests.
- Secrets (API keys) live in `.env`, never in code or commits.
