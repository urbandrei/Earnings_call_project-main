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
  range). Per DECISIONS 2026-06-29, the **κ>0.6 go/no-go for the corpus run uses this single
  rater** — the second set buys inter-annotator agreement (a reviewer-expected, paper-stage number),
  not the compute decision. To finish the loop's side: nothing — `ecvol llm-kappa --sheet
  data/coverage/fincall_llm_labels_rater1.csv --features <model>.parquet` runs the moment Colab
  returns features (compute rerouted OSC → Colab Pro+, 2026-07-04 — see the entry below). **Your remaining items:** (1) collect the **second rating set** when convenient (for IAA
  in the paper); (2) **a borderline model κ (≈0.45–0.6) re-blocks on rater 2** before any Stage-5/RQ3
  claim — only a clean pass makes rater 2 non-critical for the go/no-go; (3) the workbook's **Schema
  Feedback** sheet is the input to the T6.1 sign-off below — review it and either sign off v1 or list edits.
  - **What unblocks:** the content gate (`ecvol llm-kappa`) the instant OSC extraction lands; no
    further labeling is needed for the go/no-go.

- **2026-07-04 · T6.2 — corpus run REROUTED to Google Colab Pro+ (OSC access + funds lapsed); package is turnkey, only the Colab operational steps remain.**
  OSC is gone (deadline passed, funds unavailable), so the run moves to your **Colab Pro+**
  subscription — same vLLM + Outlines + YaRN engine, just on a Colab GPU (DECISIONS 2026-07-04,
  superseding the 2026-06-24 $1000 OSC spend, which was never used). Everything is built:
  `cloud/colab/` (README + RUNBOOK + `setup.sh` + `run.ipynb`), the `--audit-sample` gate flag, the
  frozen v2 schema, and the >32k YaRN policy. Remaining is operational, all on Colab:
  1. **Put the repo on Drive** at `MyDrive/ecvol/Earnings_call_project-main` (git-clone with a PAT,
     or upload a zip), and stage the **two gitignored payloads** it reads —
     `data/fincall/chunks.parquet` (138 MB) + `data/maec/chunks.parquet` (46 MB) — under its `data/`
     (drag-drop via drive.google.com or the Colab file browser). Splits + rater labels are git-tracked
     and arrive with the repo.
  2. **Attach a GPU runtime** (Runtime → Change runtime type → **A100**, fall back to L4; a T4 is too
     small — see RUNBOOK §0), then open `cloud/colab/run.ipynb` and run cells 1–2 (mount Drive +
     `setup.sh`).
  3. **Smoke → κ-gate → corpus** (notebook cells 4–6): a 3-call smoke test, then `--audit-sample` +
     `ecvol llm-kappa` (must PASS κ>0.6 before the corpus), then the full FinCall+MAEC run. Resumable
     across session timeouts (outputs on Drive). For the panel, use an **AWQ** 32B checkpoint on the
     40 GB GPU; 72B is dropped. (Gated Llama-3.1 needs `huggingface-cli login`; Qwen is ungated.)
  - **What unblocks:** per-model corpus extraction (panel: 7B → 32B-AWQ → Llama-3.1-8B) → `ecvol
    llm-kappa --sheet data/coverage/fincall_llm_labels_rater1.csv --features …` per model (gate on
    confirmatory core) → T6.3. `--audit-sample` extracts the κ-audit calls through the same engine as
    the corpus (audit matches corpus).

- **2026-08-09 · T6.2 — the κ>0.6 CONTENT GATE FAILED on Qwen2.5-7B-Q4_K_M; corpus extraction is blocked pending YOUR decision.**
  Good news first: **local extraction works.** The June "local is infeasible" verdict was about the
  `transformers`+bitsandbytes engine, not the card — a new `--engine llamacpp` runs the 7B over whole
  65k-token sections on your 16 GB GPU at **4.63 s/section** (FinCall ~6.6 h, both datasets ~12 h), and
  progress is resumable on Colab by running the identical GGUF. Three real defects were found and fixed
  on the way, two of which are **latent in the Colab/vLLM path as well** (see TASKS T6.2).
  Bad news: on the 50-call audit set the confirmatory core scored **guidance_direction κ=0.165,
  hedging_intensity κ=−0.050, surprise_mentions κ=0.222** — a clear fail, not a borderline one. I verified
  it is not a join bug, not a section-text bug, and not a scale offset, then **stopped and left the GPU
  idle** as you instructed. Full diagnosis: `data/coverage/llm_kappa_gate_report.md`.
  - **I ran two controls on the otherwise-idle GPU so this decision isn't a guess. Both came back negative:**
    - *Prompt anchoring?* **No.** A variant with sharpened anchors aimed at the exact failures moved
      κ by +0.01/+0.04/0.00 — nothing. Rewording the prompt will not fix this.
    - *4-bit quantization?* **No.** Q8_0 (near-lossless, and it fits locally at 13.5/16.3 GB) scored
      0.187 / −0.003 / 0.308 — better, but that closes ~10% of the gap to 0.6, not the gap.
  - **What I need from you — my recommendation is (1) first:**
    1. **A second rater on the same 50 calls.** This was deferred to paper stage assuming the gate
       would pass; a κ≈0 result promotes it to *the* diagnostic. It settles the only question the
       controls left open: if two humans agree with each other far better than the model agrees with
       either, the model is the problem and a bigger model is the fix; if two humans disagree
       comparably, the schema is the problem and **no model fixes it**. Every other option is a guess
       until this is answered.
    2. **Bigger model** (32B-AWQ on Colab, or Llama-3.1-8B locally) — justified once (1) says humans agree.
    3. **Rubric revisit (T6.1 → v3)** — justified once (1) says they don't. Note this means changing
       the anchors' *definitions*, not their wording; wording was already tested and did nothing.
  - **The concrete thing I'd look at first:** the rater marked 85 of 97 sections as hedging 0–1
    ("essentially no hedging") while the model saw moderate hedging almost everywhere. The rubric says
    "density of hedging language" without saying density *relative to what*. If a second rater also
    reads it as "almost no hedging in earnings calls", the anchor is fine and the model is wrong; if
    they read it the other way, the anchor is the bug.
  - **Also needs doing before the Colab route runs:** two of the three defects I fixed are in the
    *schema*, not my engine, so the vLLM/Outlines path still has them — most seriously, optional
    fields let the model skip `evidence` entirely and turn a skipped exploratory field into a silent
    `0` *rating*. I did not patch the vLLM path because there is no vLLM here to verify against.
    Details + the exact fix: `data/coverage/llm_kappa_gate_report.md`.
  - **What unblocks:** your pick → `featurize llm --audit-sample` + `ecvol llm-kappa` (~10 min per
    model on this machine) → on a pass, the corpus run is one command and finishes overnight (~12 h).

### Resolved
- **2026-07-04 · T6.2 — OSC route SUPERSEDED by Colab Pro+.** The OSC allocation + funds lapsed;
  the corpus compute moved to Colab Pro+ (DECISIONS 2026-07-04). The 2026-06-24 $1000 OSC spend
  approval is void (never used — no money spent). `cloud/osc/` is kept as reference; the live path
  is `cloud/colab/` (see the Active entry above). No OSC action is needed from you.
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
  *(T6.2's compute now runs on Colab Pro+, not a paid burst — DECISIONS 2026-07-04.)*
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
