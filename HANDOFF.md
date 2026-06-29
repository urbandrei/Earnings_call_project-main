# HANDOFF.md — Human-pending queue

**Append-only.** This is the parallel to-do list for the **user** while the autonomous loop
([LOOP.md](LOOP.md)) keeps working on unblocked tasks. The loop writes an **Active** entry every
time it hits a hard-stop blocker (and marks the task `[!]` in [TASKS.md](TASKS.md)); the user
clears it whenever convenient and tells the loop, which then un-blocks that task.

Each entry: date · task ID · what the user must do · what unblocks when it's done.

---

## Active (action needed to unblock a task)

- **2026-06-29 · T6.2 — rater 1 labels are in and ingested; the κ-gate is unblocked on the labels side. Second rater is now PAPER-STAGE, not an OSC blocker.**
  You delivered `ingest/Ratings_1.xlsx` (50-call audit set). It is ingested + validated →
  `data/coverage/fincall_llm_labels_rater1.csv` (97 rows, exact match to the frozen sample, all in
  range). Per DECISIONS 2026-06-29, the **κ>0.6 go/no-go for the OSC corpus run uses this single
  rater** — the second set buys inter-annotator agreement (a reviewer-expected, paper-stage number),
  not the compute decision. To finish the loop's side: nothing — `ecvol llm-kappa --sheet
  data/coverage/fincall_llm_labels_rater1.csv --features <model>.parquet` runs the moment OSC returns
  features. **Your remaining items:** (1) collect the **second rating set** when convenient (for IAA
  in the paper); (2) **a borderline model κ (≈0.45–0.6) re-blocks on rater 2** before any Stage-5/RQ3
  claim — only a clean pass makes rater 2 non-critical for the go/no-go; (3) the workbook's **Schema
  Feedback** sheet is the input to the T6.1 sign-off below — review it and either sign off v1 or list edits.
  - **What unblocks:** the content gate (`ecvol llm-kappa`) the instant OSC extraction lands; no
    further labeling is needed for the go/no-go.

- **2026-06-29 · T6.2 — the corpus run goes to OSC; package is turnkey, only OSC access + the operational steps remain.**
  Local can't run extraction (16 GB OOMs at full context; `data/coverage/llm_probe_report.md`).
  Everything for the cloud burst is built (`cloud/osc/`), the $1000 spend is approved (DECISIONS
  2026-06-24), the schema is frozen at v2 (signed off), and the **>32k context policy is resolved**
  (YaRN-extend to 65536, wired into `extract.sbatch` by default — DECISIONS 2026-06-29). Remaining is
  operational, all on OSC:
  1. **Confirm your OSC allocation** — project/account code (`PASxxxx`) + cluster (recommend **Ascend
     A100-80GB** or **Cardinal H100**). Put the account in `cloud/osc/slurm/extract.sbatch` (line 13).
  2. **Get code+data on OSC + build:** clone the repo; rsync the gitignored
     `data/{fincall,maec}/{calls,chunks}.parquet` + `data/splits/`; `module load apptainer` then
     `apptainer build cloud/osc/ecvol-llm.sif cloud/osc/apptainer/ecvol-llm.def`.
  3. **Stage weights + smoke-test + submit:** `bash cloud/osc/stage.sh Qwen/Qwen2.5-7B-Instruct
     Qwen/Qwen2.5-32B-Instruct`; run ONE `--limit` job first for a real ETA/cost; then the full panel
     (loop in README step 3b). (If you add gated Llama-3.1 to the panel: `huggingface-cli login` on the
     login node first. Qwen models are ungated.)
  - **What unblocks:** per-model corpus extraction (panel: 7B → 32B) → `ecvol llm-kappa --sheet
    data/coverage/fincall_llm_labels_rater1.csv --features …` per model (gate on confirmatory core) →
    T6.3. The κ-audit's 50 calls are extracted in the same corpus job (audit matches corpus).

### Resolved
- **2026-06-29 · T6.1 — v2 schema SIGNED OFF (user).** Schema + rubric + `PROMPT_VERSION="v2"`
  frozen; T6.1 → `[x]`. v2 added two exploratory fields (`management_optimism`,
  `quantitative_specificity`) from the rater's feedback + numeral-aware literature, and narrowed
  the κ-gate to the confirmatory core. OSC extraction may now build against the frozen schema.
  (DECISIONS 2026-06-29.) Remaining T6.2 blockers are operational only — see the active entries.
- **2026-06-29 · T6.2 — rater-1 labels ingested + validated** → `data/coverage/fincall_llm_labels_rater1.csv`
  (97 rows, exact match to the frozen sample). The κ-gate go/no-go runs on this single rater;
  second-rater IAA deferred to pre-publication (DECISIONS 2026-06-29). No further labeling needed
  for the OSC go/no-go.
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
