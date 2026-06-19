# LOOP.md — Autonomous task-loop protocol

**This is the operating manual for the near-autonomous backlog loop.** When a self-paced
`/loop` is running against this project, the agent re-reads this file at the start of every
iteration and follows the protocol below. It is the single source of truth for *how the loop
behaves*; **what** to build still lives in [TASKS.md](TASKS.md) / [DESIGN.md](DESIGN.md), and
*why* anything deviated lives in [DECISIONS.md](DECISIONS.md).

The loop's job: advance the backlog one task at a time with minimal human intervention,
**pausing only when a human is genuinely required**, and surfacing a queue
([HANDOFF.md](HANDOFF.md)) of human-pending items the user can clear *in parallel*.

## How to start / stop / resume

- **Start:** the user runs `/loop follow LOOP.md` (self-paced/dynamic mode). The agent drives
  iterations with `ScheduleWakeup`; it does **not** rely on a Stop hook to continue.
- **Pause points (loop ends — no `ScheduleWakeup`):** a phase boundary, a design call, or a
  hard-stop blocker with no unblocked task left. The user restarts with `/loop follow LOOP.md`
  (or just "continue") once they've acted.
- **Resume after compaction:** the loop re-reads TASKS.md/HANDOFF.md each iteration, and the
  `SessionStart` hook re-injects the frontier — so the "what task am I on" thread survives
  auto-compaction without special handling.

## Per-iteration protocol

1. **Orient.** Re-read TASKS.md (frontier), HANDOFF.md (open blockers), and skim
   CLAUDE.md (discipline). Confirm no `[!]`-blocker was just resolved by the user.
2. **Select** the next task that is `[ ]`, **dependency-satisfied** (don't jump an unmet gate;
   CLAUDE.md "Sequencing matters"), and not `[!]`-blocked. If none remain unblocked →
   notify the user and **end the loop**.
3. **Design-call check (STOP and ask).** If finishing the task requires a non-trivial
   research/design choice — which model/checkpoint, pooling strategy, ablation set, what counts
   as a "win", any §7.5 confirmatory-vs-exploratory call, any §4 framing-gate input — **do not
   guess**. Write the options + a recommendation to the user, `PushNotification`, and **end the
   loop**. (See "What counts as a design call".)
4. **Implement** to the acceptance test (TDD per charter: falsifiable criterion first). Keep
   changes surgical. Long compute (GPU/CPU extraction, large pulls) → launch with
   `run_in_background`, then `ScheduleWakeup` to poll rather than blocking the turn.
5. **Local gate (must be green before any commit):**
   ```
   uv run ruff check .
   uv run ruff format --check .
   uv run pytest -q
   ```
   This is a faithful mirror of CI (`.github/workflows/ci.yml`). Red → fix; never commit red.
6. **Commit locally** (small, focused, one logical change, charter message format — see below).
   Then update governance as part of the same task:
   - TASKS.md: status `[x]`, subtask checkboxes, dated notes + artifact links.
   - JOURNAL.md: append a Done / Found / Sources / Next entry (append-only).
   - DECISIONS.md: append an entry for **any** deviation, default chosen, scope change, or spend
     — `PushNotification` on each DECISIONS entry.
   - Auto-memory: update if a durable, non-obvious fact was established.
7. **Blocker?** If the task cannot finish without a human (key/license/ToS/spend/labeling/
   framing-gate — see taxonomy): mark it `[!]` in TASKS.md, append a HANDOFF.md entry stating
   exactly what the user must do, `PushNotification`, and **skip to the next unblocked task**
   (back to step 2). Do not fake or stub past a real blocker.
8. **Phase boundary?** If that task was the **last in its phase**: `PushNotification`
   ("Phase N done — please push + confirm CI green"), and **end the loop**. Per the gate policy
   (DECISIONS 2026-06-18) the next phase does not start until the user confirms CI is green.
9. **Otherwise continue:** `ScheduleWakeup` with a delay sized to the work — ~270 s for short
   polls (stay in cache), 1200 s+ when idle or waiting on long compute — and loop.

## What counts as a design call (STOP and ask)

Stop for choices that shape **research validity or headline results**, e.g.:
- model / checkpoint / revision selection (embedding, sentiment, audio, LLM);
- pooling / aggregation / chunking strategy where alternatives are defensible;
- which ablations or controls define a comparison; the metric or threshold for a "win";
- promoting an exploration to confirmatory; any §4 framing-gate or §7.5 labeling decision.

Routine **engineering** choices (file layout, cache key format, test fixtures, obvious library
idioms) are *not* design calls — pick the sensible option, note it if non-obvious, and continue.

## Hard-stop blocker taxonomy (human required)

The loop **cannot** clear these itself — record in HANDOFF.md, notify, skip:
- **Accounts / keys / licenses:** HuggingFace gated-model accept + `HF_TOKEN` (T4.3 pyannote,
  optional/flagged); EarningsCall/EarningsCast API key + ToS (T7.1); data-host account for
  derived-feature archive upload (T8.1).
- **Spend / approval:** any cloud burst (possible T4.3, definite T8.3) — needs a DECISIONS.md
  entry with budget **and** the user's go-ahead. **Never auto-spend.**
- **Human labor:** schema co-design + 50-call κ>0.6 audits (T6.1/T6.2, TX1); advisor / venue /
  arXiv (T8.2); hand-checked acceptance samples (the agent builds the tooling; the *number*
  comes from the user).
- **Always blocked (enforced by `.claude/hooks/guardrail.ps1`):** `git push` (the user owns
  pushes + CI), and any write under `legacy/` (read-only).

## Commit message format (charter)

One logical change per commit. End every commit message with:
```
Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HmgQQmiASCAuGYtCtmt8nT
```

## Notification policy (DECISIONS 2026-06-18)

`PushNotification` on: (a) any blocker / hard-stop that needs the user, (b) every phase
completion, and (c) every DECISIONS.md entry. Keep messages one line, lead with the action.
