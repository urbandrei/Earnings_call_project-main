# DECISIONS.md — Project Decision Log

**Append-only.** This file is the single place where the project's design contract ([DESIGN.md](DESIGN.md)) is amended. DESIGN.md itself is never silently edited — if reality diverges from the design, the divergence is recorded here first, then (if needed) DESIGN.md is updated with a reference to the entry.

## What gets an entry

- Any deviation from, or amendment to, DESIGN.md (data sources, targets, models, protocols, gates).
- Scope changes: adding/removing/materially changing a task in [TASKS.md](TASKS.md); pulling anything from the out-of-scope list (DESIGN.md §3.6).
- Promotions: an exploration result (see CLAUDE.md exploration policy) becoming part of the pipeline or the paper.
- Spend approvals: cloud compute bursts, paid API usage — with budget and hypothesis.
- Gate outcomes: the §4 framing gate, Stage-6 go/no-go, acceptance-test changes.

## Entry format

```
YYYY-MM-DD — decision — rationale — alternatives rejected
```

One entry per decision; link supporting material (journal entries, run artifacts, papers) inline. Never edit or delete past entries — if a decision is reversed, append a new entry that says so and why.

---

## Log

- **2026-06-12 — Rework from scratch as a Python package; legacy notebooks retained read-only as reference.** Rationale: legacy code unrebuildable (no env pinning, data lost, 6-way copy-paste). Rejected: incremental notebook refactor.
- **2026-06-12 — Compute: start consumer GPU (16–24 GB), design for later cloud scaling.** User decision.
- **2026-06-12 — Data: existing open datasets first (FinCall-Surprise primary, MAEC secondary); fresh 2025–26 acquisition deferred to Phase 7 as post-cutoff holdout.** User decision after research. Rejected: fresh-dataset-as-centerpiece (months of data engineering before any result); existing-sets-only (loses the lookahead experiment).
- **2026-06-12 — Publication framing decided at a pre-registered gate after Phase 2/3 (dual-path, DESIGN.md §4).** User decision. Rejected: committing to positive-showcase or re-examination framing now.
- **2026-06-12 — Legacy EarningsCall (572-call) dataset demoted to one comparability table; never redistributed.** Rationale: license unclear, tiny, 2017-only, leaderboard indicted by [R14].
- **2026-06-12 — Per-sentence audio–text alignment excluded from v1.** Rationale: legacy pipeline's most fragile part; poor effort/value vs. section/turn pooling. Revisit gate: Stage 6.
- **2026-06-12 — Headline target variants: Δv and HAR-residual alongside level-v.** Rationale: mechanically removes ticker-identity signal (DESIGN.md §3.1); level-v kept for literature comparability.
- **2026-06-12 — Tooling: uv + pydantic-YAML configs + JSON manifests + parquet artifacts; no Hydra, no DVC, no wandb in v1.** Rationale: solo-researcher debuggability and zero service dependencies; revisit via this log if team grows.
- **2026-06-12 — Multi-task α-sweep procedure (legacy) permanently retired.** Rationale: selection overfitting (val-selected α then train+val retrain).
- **2026-06-12 — Docs restructured: DESIGN.md, TASKS.md, OLDWORK.md, CLAUDE.md all at project root; `new/` folder removed; code will live at root (`src/ecvol`…); legacy folder renamed `Earnings_call_project-main/` → `legacy/` (read-only).** Rationale: flat layout for day-to-day agentic work; legacy material isolated and clearly marked outdated.
- **2026-06-12 — Full task breakdown moved to TASKS.md as a living tracker (statuses, checkboxes, notes); DESIGN.md keeps the phase summary (§9). Task IDs stable across both; new/changed tasks require an entry here.** Rationale: separate the immutable design contract from mutable execution state.
- **2026-06-12 — Git repository not yet initialized; deferred to T0.1 (first subtasks: `git init`, .gitignore, initial commit). Small-focused-commit discipline documented in CLAUDE.md.** User decision. Rejected: initializing immediately (user prefers to start the repo with the scaffolding work).
- **2026-06-12 — Decision log moved out of DESIGN.md into this file (DECISIONS.md); work journal added as JOURNAL.md.** Rationale: DESIGN.md stays a stable design contract; mutable records (decisions, work history) live in dedicated append-only files. Rejected: keeping the log inline (DESIGN.md churn on every decision).
- **2026-06-12 — Git handling of `legacy/` (T0.1 subtask): track notebooks, README, and Paper.pdf (~2 MB); gitignore `legacy/papers/` (93 MB of third-party literature PDFs) and `PowerPoint.pptx` (7 MB).** Rationale: the tracked files are the useful, project-authored reference; redistributing publisher PDFs in a repo that may go public is a copyright problem and bloats every clone — the literature stays available locally and is cited in DESIGN.md §13. Rejected: ignoring `legacy/` entirely (loses the cross-check notebooks the targets task relies on); track-without-LFS of everything (100 MB clones, redistribution risk).
