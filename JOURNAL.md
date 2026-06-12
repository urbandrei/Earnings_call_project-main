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

### 2026-06-12 — T0.1 package skeleton
- **Done:** Installed uv 0.11.21 (official installer → `~\.local\bin`). `git init -b main`; `.gitignore` (glob-based: `data/` except `data/manifests/`, `artifacts/`, caches, `.env`, legacy bulk binaries — see DECISIONS.md entry) + `.gitattributes` (LF normalization). Eight small commits: governance docs → repo config → legacy reference files → package skeleton → pre-commit → CI workflow → README → doc updates. Package: `pyproject.toml` (uv_build backend, Python pinned 3.12 via `.python-version`, lockfile committed), `src/ecvol/` subpackage layout per DESIGN.md §8.1 (config/data/features{text,audio,llm}/models/eval), Typer CLI with all seven verbs (`prices|targets|splits|featurize|train|evaluate|report`) as stubs exiting code 2, smoke tests (help lists every verb; stubs fail loudly). Ruff (E,F,I,UP,B; line 100; `legacy/` excluded) + pre-commit (ruff-check --fix, ruff-format) installed and green. CI workflow `.github/workflows/ci.yml`: `uv sync --locked` → ruff check → format check → pytest. Fresh-machine install tested once: cloned repo to `%TEMP%`, `uv sync --locked` + tests green from the lockfile alone; steps documented in README.md.
- **Found:** Python 3.14.4 is the system default — pinned 3.12 instead for ML-wheel compatibility (torch et al. lag new CPython releases). Ruff initially linted the legacy notebooks (395 KB of findings) → `extend-exclude = ["legacy"]`, consistent with their read-only status. PowerShell 5.1 `Set-Content -Encoding utf8` writes BOMs — generated files rewritten BOM-free.
- **Sources:** uv install/sync docs (docs.astral.sh/uv), astral-sh/ruff-pre-commit, astral-sh/setup-uv action.
- **Next:** User: create the GitHub repo and push (needs `gh auth login` or manual repo creation) to verify Actions green — the one open T0.1 acceptance item. Then T0.2 config system.

### 2026-06-12 — T0.1 closed
- **Done:** User created the GitHub repo (`urbandrei/Earnings_call_project-main`, private) and pushed; CI run on `main` green on first try (user-verified in the Actions tab — repo is private, so no API access from the agent without `gh`). T0.1 acceptance fully met → `[x]`.
- **Found:** Verifying Actions on a private repo from this machine needs an authenticated `gh`; worth installing (`winget install GitHub.cli` + `gh auth login`) before CI-dependent phases.
- **Next:** T0.2 config system (pydantic schemas, example YAML, config-hash function).

### 2026-06-12 — T0.2 config system
- **Done:** Added pydantic 2.13.4 + PyYAML 6.0.3 (`uv add`, lockfile committed). `src/ecvol/config/`: `schema.py` — `ExperimentConfig` with `DataConfig` (dataset literal), `TargetConfig` (variant level/delta/har_residual; horizons restricted to {3,7,15,30} per DESIGN.md §5.3), `SplitConfig` (temporal/ticker_disjoint/combined + embargo), `FeatureSpec` list (kind/extractor/model_id/revision/options — minimal now, extended when Phases 3–6 land), `ModelConfig` (name + `include_past_vol_covariates` for the §6 invariant + params dict), `EvalConfig` (metrics, bootstrap resamples); `extra="forbid"` on every model so typos fail loudly. `load.py` — `load_config` (ConfigError with file path + dotted field locations), `dump_config` (deterministic sorted YAML), `config_hash` (canonical JSON, sorted keys, compact separators → SHA-256). `configs/example.yaml` (5-seed GBDT on FinCall-Surprise temporal split). 11 tests in `tests/test_config.py` cover both acceptance cases plus round-trip stability and hash canonicalization. Three commits (deps / config system / docs).
- **Found:** Two design-derived validations added beyond the task spec: (1) cross-field check that `embargo_trading_days` ≥ longest horizon on temporal/combined splits — encodes the §5.4 leakage rule at config time; (2) uniqueness of seeds and horizons. Horizons are also sorted on resolve so the config hash is order-insensitive. Pydantic field named `model` does not clash with the `model_` protected namespace (no warning).
- **Next:** T0.3 run tracking & manifests (manifests.py, run-ID scheme + artifact writer, env fingerprint).
