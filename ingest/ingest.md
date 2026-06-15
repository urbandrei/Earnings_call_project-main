# `ingest/` — Breakdown & Project-Fit Analysis

**Created:** 2026-06-14 · **Author:** review pass over dropped-in resources · **Status:** analysis only — *nothing in this file edits the canonical contract docs* (`DESIGN.md`, `TASKS.md`, `DECISIONS.md`, `JOURNAL.md`). Proposals here are labelled as proposals; promoting any of them requires the normal `DECISIONS.md`/`TASKS.md` discipline.

---

## 0. What's in this folder, and what it is

| File | What it actually is |
|---|---|
| `Multi-modal Volatility .zip` → `MultimodalVolatility.html` + `images/image1–37.png` | A **collaborator's research log** (Google-Docs export). Structured as "Meeting 1–3" notes plus "Task 1–8" with task definition / report / **advisor feedback (red)**. Documents a *parallel* effort on the same problem this project tackles: predicting post-earnings-call volatility from transcript text + call audio. It already has a working pipeline and **headline results that beat KeFVP**. |
| `Undermind - …short-horizon stock volatility.pdf` | A **37-page AI-generated literature review** (Undermind), commissioned in the log's "Meeting 3". ~20 multimodal ECC-volatility papers with comparison tables, a foundational-citation analysis, and explicit design recommendations. Report dated 2025-11-22. |

**Provenance for this review:** treated as *a collaborator's work I am reviewing* — i.e. catalogued and compared **critically** against our `DESIGN.md` principles, **not** assumed to be merged into our pipeline.

### Bottom line up front

The collaborator has built something real and is ahead of us on *engineering* (a complete QA→cluster→audio-embedding→regression pipeline with multi-seed results). But their methodology **diverges sharply from our design contract** on exactly the axes `DESIGN.md` was written to protect:

- **Open vs. proprietary.** They run **GPT-5 / GPT-5-mini / GPT-5.2 / OpenAI `text-embedding-3-large`** for QA generation, clustering, and labelling. Our project is **open-weights-only** for reproducibility (`DESIGN.md` §6 Stage 5 pins `Qwen2.5-7B-Instruct`; §9 / risk register emphasise reproducibility).
- **Borrowed vs. computed labels.** They use **KeFVP's released volatility labels** (only 616 of 1,213 MAEC firms) instead of computing targets from raw prices. Our T1.2/T1.3 compute targets from Stooq with an after-hours rule + unit tests (`DESIGN.md` §5.3).
- **Missing the controls that are our whole thesis.** Their "Ours beats KeFVP" table has **no persistence/HAR-RV floor (§3.3), no ticker-identity controls (§3.1/§7.3), no ticker-disjoint split (§5.4), no DM/Holm significance (§7.2), no Δv / HAR-residual targets**. They *do* use **10 seeds with mean±std** (a genuine strength). They cite "Same Company, Same Signal" — our central threat-to-validity reference [R14] — but do **not** apply its controls to their own headline numbers.
- **New ideas worth harvesting.** Data-driven **QA-pair generation**, **QA topic clustering (k=35)**, and **Qwen2.5-Omni-7B audio embeddings conditioned on QA context** are not in our design. They map onto our Stage 5/6 and could become *exploratory* tasks — but must clear our §3.6 out-of-scope bar and the Stage-5/6 gates first.

The rest of this document walks both resources point-by-point, indexes all 37 figures, and ends with an alignment matrix, a conflicts list, and concrete (proposed-only) follow-ups.

---

## 1. Resource A — `MultimodalVolatility.html` (collaborator research log)

Walked in the document's own order.

### 1.1 Meeting 1 — Datasets & first paper survey

**Datasets they chose.**
- **MAEC** — "3443 earnings calls, period 2016–2017, 1213 tickers" ([github.com/Earnings-Call-Dataset/MAEC…](https://github.com/Earnings-Call-Dataset/MAEC-A-Multimodal-Aligned-Earnings-Conference-Call-Dataset-for-Financial-Risk-Prediction)). (Note the span differs from the MAEC paper's stated 2015–2018 / S&P 1500; their slice is 2016–2017.)
- **EC** (GeminiLn `EarningsCall_Dataset`) — "572 ECCs, S&P 500 companies in 2017, 242 tickers" ([github.com/GeminiLn/EarningsCall_Dataset](https://github.com/GeminiLn/EarningsCall_Dataset)). This is the Qin & Yang 2019 dataset.

> **How it fits us.** These are our **[D2] MAEC** (secondary) and **[D3] legacy EarningsCall** (one-comparability-table-only) datasets in `DESIGN.md` §5.1. **Key divergence:** their *primary* corpora are MAEC + EC; our *primary* is **FinCall-Surprise (2,688 calls, 2019–2021, Apache-2.0)**, which the collaborator does not use at all. Their EC reliance directly bumps into our `DECISIONS.md` 2026-06-12 ruling that the 572-call set is demoted to a single comparability table and *never redistributed* (unclear license, 2017-only, indicted by [R14]). Our datasets are also newer (2019–2021 vs 2016–2017), which matters for the §3.2 lookahead-bias story.

**Papers reviewed.**
- **Open-FinLLMs** (FinLLaMA / FinLLaVA, [arxiv 2408.11878](https://arxiv.org/abs/2408.11878)) — FinLLaMA pretrained/instruction-tuned on finance (MLM over ECCs, reports, academic text + math reasoning). Note from the log: **FinLLaMA is not actually released** on HF (only FinLLaVA), so its leaderboard claims aren't verifiable.
- **Multimodal Financial Foundation Models (MFFMs)** survey, [arxiv 2506.01973](https://www.arxiv.org/abs/2506.01973).
- **KeFVP** ([arxiv 2412.18029](https://arxiv.org/pdf/2412.18029)) — the baseline they chase. Two phases: (1) BERT MLM pretraining where each Financial-Metric sentence is concatenated with the metric's Wikipedia definition; (2) FVP using historical volatilities + ECC line embeddings in an **Autoformer** architecture. The collaborator's own critique: phase-1 is conceptually close to FinBERT, which already did this.

> **How it fits us.** **KeFVP is our [R6]** (EMNLP 2023 Findings) — in `DESIGN.md` it's "the SOTA reference the legacy project chased," now repositioned as a *control baseline*, not a target to beat. The collaborator treats KeFVP both as the **bar to beat** *and* as the **source of their labels** (see §1.5) — a circularity our design avoids by computing targets independently. The "concatenate metric definitions" idea is adjacent to our Stage-5 structured-feature philosophy but we get definitions implicitly via the LLM rather than via MLM pretraining.

**Their stated follow-up directions (Meeting 1):** (a) offline transforms on raw ECC transcripts to make them more informative; (b) audio-model research; (c) time-series data (historical vol, moving averages), signal/noise decomposition, Autoformer. → (a) ≈ our Stage 2/3 feature work; (c) ≈ the past-vol covariates baked into every model from Stage 2 up (`DESIGN.md` §6 invariant) and the HAR-RV baseline (Stage 0).

### 1.2 Meeting 2 — ECC structure & audio literature

**ECC anatomy** they note: boilerplate/legal → management remarks (CEO/CFO) → analyst **Q&A**. → matches our T3.1 sectioning goal ("section-detection precision >90%", no per-sentence alignment).

**Papers / techniques:**
- **ECC Analyzer** — builds a **RAG system over transcripts** with an expert-designed **Question Bank** to extract salient chunks, summarises hierarchically (chunk summaries → global summary, via GPT), and extracts **Wav2Vec** audio embeddings. The collaborator's contribution vs. ECC Analyzer: replace the *fixed* question bank with a **data-driven** auto-extracted set of Q&A pairs for broader coverage.
- **"Same Company, Same Signal"** — listed under reviewed papers.
- **Audio literature list:** HiFi, FullSubNet, "Glance and Gaze", SELM, Cacophony, **InstructSpeech**, **Qwen-Audio** (marked "TO BE COMPLETED").
- **InstructSpeech** detail: separates **acoustic vs. semantic** feature modules. Semantic tokens = HuBERT **layer-12** embeddings k-means–discretised (lower HuBERT layers = acoustic, higher = semantic); acoustic tokens = SoundStream (timbre, emotion, prosody). Proof that splitting acoustic/semantic helps downstream speech tasks.

> **How it fits us.** **"Same Company, Same Signal" is our [R14]** — the *central threat-to-validity* in `DESIGN.md` §3.1 (transcript models predominantly encode company identity, not content; training-free past-vol beats all transcript models). **Critical flag:** the collaborator *read and cites* [R14] but does not carry its controls into their headline results (§1.6). **ECC Analyzer is our [R8]** (§13, "reports 27.7% MSE reduction"); its RAG/question-bank idea is what the collaborator generalises into the QA pipeline (§1.3) and what our Stage 5 covers with **open** models + constrained JSON. **InstructSpeech's acoustic/semantic split** is intellectually adjacent to our Stage-3 choice of **eGeMAPS (interpretable paralinguistics)** + **WavLM** + **emotion2vec+** (`DESIGN.md` §6); we don't tokenise/discretise audio, we pool frozen embeddings — a simpler, consumer-GPU-friendly route.

### 1.3 Tasks 1–2 — QA generation & speaker-turn chunking

- **Task 1:** chunk ECC into **10-line windows w/ 2-line overlap**, prompt an LLM for Q&A pairs + a category + **source line numbers** (to align audio later). **Advisor feedback (red):** fixed-line chunking breaks speaker turns → **chunk by speaker turn instead**; apply **clustering** for categories.
- **Task 2:** extract speaker info from **audio filenames** as per-line metadata, then **chunk by speaker turn**. They generated QAs from **100 ECCs → 3,145 QA pairs** (≈30 QAs/ECC), then clustered. **Advisor feedback:** ~30 QAs/ECC is too costly; QA style too *specific* (e.g. "What role does SkyBlue technology play…?") rather than general.

> **How it fits us.** **Speaker-turn chunking** is a clean, directly-adoptable idea for **T3.1 (transcript normalization / speaker structure)** — and our FinCall identity work already extracts speaker metadata, so this is low-friction. The **QA-pair abstraction** itself is *not in our design*; it's closest to **Stage 5 / T6.1** (auditable structured features). The "~30 QAs/ECC too costly" and "too specific" feedback is exactly the kind of thing our **§3.6 out-of-scope discipline** and **Stage-5 human-audit gate (κ>0.6, T6.2)** would catch before scaling.

### 1.4 Tasks 3, 5 — QA clustering into a volatility topic taxonomy

Pipeline (EC first, then scaled to MAEC):
1. **Preliminary labels:** GPT-5 (low reasoning), batch 100, prompt = "Financial Volatility Topic Extraction" → one label per QA. On MAEC: **9,301 labels → 7,288 unique** after dedup.
2. **Merge:** embed the 7,288 labels with **OpenAI `text-embedding-3-large`**, **MiniBatchKMeans** (batch 512), test **k=20…100**, pick by **silhouette** → **chose k=35**.
3. **Representative label per cluster:** GPT-5.2 (medium reasoning).
4. **Classify** each QA into one of 35 labels: GPT-5-mini (medium reasoning).

The method is borrowed from a paper (label-generation → LLM label-merge → classify) that avoids pre-specifying k; the collaborator adapted it to bias candidate labels toward **volatility** concepts.

> **How it fits us.** This is genuinely novel relative to our design and **could become a Stage-5 exploratory feature**: "frequency of each of K volatility-topic categories per ECC" is one of their model inputs and is exactly the kind of **auditable, structured** feature `DESIGN.md` §6 Stage 5 / RQ3 favours over opaque embeddings. **But three caveats before it could be promoted:** (1) it's built entirely on **proprietary OpenAI models** — incompatible with our open-weights constraint without a Qwen re-implementation; (2) the **silhouette analysis (figure `image21`) actually peaks around k≈60–70, not 35** — k=35 looks chosen for interpretability, which is fine but should be stated honestly, not as "optimal"; (3) any topic taxonomy fit on the corpus must respect our split discipline (fit on **train only**, §5.4) to avoid leakage. Promotion would need a `DECISIONS.md` entry per the exploration policy.

### 1.5 Task 6 — Qwen2.5-Omni audio embeddings conditioned on QA

For each QA pair, generate a multimodal embedding capturing **(a) content** and **(b) delivery (emotion/tone)**:
- Model: **Qwen2.5-Omni-7B**, **4-bit NF4**, fp16 compute, double quantization.
- Architecture: **Thinker** (understanding) + **Talker** (generation). They take the **Thinker last hidden state**, shape `(B, L, 3584)`, and apply **masked mean pooling** (mask out PAD tokens via attention mask) → fixed-length 3584-d vector.
- Audio input = the spoken segment containing the answer (via `source_lines`); the **text prompt conditions the embedding** on topic + question + transcript and instructs focus on **acoustic/emotional** cues (tone, hesitation, confidence, pace, "volatility").
- Audio front-end primer they wrote down: 16 kHz → STFT **n_fft=400 (25 ms window)**, **hop 160 (10 ms)** → **128-mel** features → patch-merge (`temporal_patch_size=2`, `merge_size=2`) → **~40 ms/token** → max **300 s / ~7,500 audio tokens** (`n_samples=4,800,000`).

> **How it fits us.** This is a concrete recipe for our **Stage 6** ("audio-LLM scoring; Qwen2.5-Omni / Qwen2-Audio", `DESIGN.md` §6) — which in our design is **optional, cloud, and gated**: only runs *if* Stages 2–5 show DM-significant signal, and requires a `DECISIONS.md` budget entry. Their masked-mean-pool-the-Thinker approach is a reusable implementation detail and their **4-bit NF4 / 24 GB-friendly** setup matches our §8.3 VRAM budget. **Two cautions:** (1) our [R13] FinAudio note warns audio-LLMs struggle on long financial audio — their 300 s cap is consistent with that constraint; (2) a model that emits *"Neutral mood, male, aged 41 years"* (figure `image16`) is **encoding speaker gender/age**, which is precisely the **§3.5 gender-bias** confound — any adoption must carry the Stage-3/§3.5 gender-confound analysis.

### 1.6 Tasks 7–8 — Targets, experiments, ablation

**Task 7 — target labels.** They **reuse KeFVP's released price labels** rather than recomputing, arguing Yahoo adjusted prices drift over time so reusing released labels keeps consistency with prior work. Coverage limit they note: MAEC has **1,213 firms** but KeFVP's released price data covers only **616**.

> **How it fits us.** **Direct conflict with `DESIGN.md` §5.3** and `DECISIONS.md` (computed-targets decision). Our design computes log-RV at τ∈{3,7,15,30} from **Stooq adjusted close** with an explicit **after-hours timing rule**, unit-tested against synthetic known-RV values (**T1.3**), with **Δv** and **HAR-residual** identity-robust variants. Reusing a third party's labels (a) caps you at 616/1,213 firms — silently dropping ~half the corpus, which our "every exclusion needs a reason code" rule (CLAUDE.md) forbids; and (b) inherits whatever timing/after-hours conventions KeFVP used, unaudited. Their point that *Yahoo adjusted prices are non-stationary* is a real and useful warning — it reinforces our choice of **Stooq primary + Tiingo cross-check (>0.999 correlation gate, T1.2)** and our SHA-256 manifests for price snapshots over a live API like yfinance (explicitly rejected in §5.1).

**Task 8 — experiments.** Main table (MSE, datasets **EC / MAEC-15 / MAEC-16**, horizons MSE3/7/15/30 + avg):

| Model | EC MSE(avg) | MAEC-16 MSE(avg) |
|---|---|---|
| BiLSTM+ATT | 0.696 | 0.691 |
| **KeFVP (their main baseline)** | 0.204 | 0.318 |
| MDRM | 0.630 | 0.618 |
| HTML (text) | 0.514 | 0.579 |
| HTML (text+audio) | 0.487 | 0.556 |
| **Ours** | **0.188 ± 0.002** | **0.288 ± 0.005** |

Baselines taken from the KeFVP paper. **Ours = 10 seeds, mean ± std.**

**Ablation** (isolating audio's contribution; same downstream forecaster + multi-horizon historical-vol features, only the embedding changes):

| Variant | EC MSE(avg) | MAEC-16 MSE(avg) |
|---|---|---|
| Vols (LSTM only) | 0.227 ± 0.039 | — |
| Vols + text | 0.196 ± 0.005 | 0.295 ± 0.002 |
| Vols + text + audio (**Prompt 1**, generic emotion) | 0.189 ± 0.003 | 0.306 ± 0.003 |
| Vols + text + audio (**Prompt 2**, task-aware volatility) | **0.188 ± 0.002** | **0.288 ± 0.005** |

Their conclusion: audio helps **only with a task-aware prompt** (Prompt 2) — Prompt 1 *improves* EC but *degrades* MAEC-16. So the value of audio "depends strongly on how the multimodal model is guided during representation extraction."

> **How it fits us.** This maps onto our **Result Tables 1–5** plan and **§7** evaluation, and lets us score their rigour precisely:
> - **Present & good:** ≥10 seeds with mean±std (we require ≥5, §7.2); an explicit **vols-only** baseline (≈ our persistence/past-vol floor); a clean text-vs-+audio **ablation** (≈ our RQ2 / Stage-4 question); a prompt-design ablation showing prompts matter.
> - **Absent (the load-bearing parts of our design):** no **persistence/EWMA/HAR-RV/GARCH** floor (§3.3, Stage 0) — "Vols (LSTM)" is *learned* past-vol, not the training-free floor [R14] showed beats transcript models; no **ticker-identity controls** (§3.1/§7.3: ticker-only model, same-ticker shuffle, identity probe); no **ticker-disjoint or temporal×disjoint split** (§5.4) — temporal-only at best; no **Δv / HAR-residual** identity-robust targets (§5.3); no **DM tests / Holm correction** (§7.2). The headline "Ours 0.188 < KeFVP 0.204" is a **~8% relative MSE improvement at τ-avg** with std 0.002 — plausibly real, but *exactly* the kind of result [R14] argues can be ticker-identity memorisation until proven otherwise. Their own reading list contains [R14]; their evaluation doesn't apply it. **This gap is the single most important thing for the collaborator to close**, and it's the thing our project is built to do.
> - The **Prompt 1 helps EC but hurts MAEC-16** result is a useful caution: small multimodal gains that flip sign across datasets are weak evidence — our multi-dataset (FinCall + MAEC) + DM-significance design is meant to catch precisely this.

---

## 2. Resource B — `Undermind …pdf` (literature review)

A 37-page Undermind report (the "literature review by undermind AI" referenced in Meeting 3). Full query: predicting **very short-horizon** return volatility (realized **and implied**, event-window) from **text+audio** earnings-call ML/DL.

### 2.1 Key takeaway

A mature line of multimodal (text+audio) RV models exists at **3–30 day** horizons (anchored on **3-day** post-call RV), but **two things are essentially open**:
1. **Truly very-short-horizon** RV — intraday, call-start→close, explicit `[−1,+1]` event windows, next-session intraday RV. Only **"The Sound of Risk"** [their 11] has an explicit **1-day** RV target.
2. **Options-implied volatility** (or IV changes / RV-minus-IV) aligned to call timing — **no surveyed multimodal ECC paper does this.**

> **How it fits us.** Our `DESIGN.md` §5.3 targets are **multi-day RV at τ∈{3,7,15,30}** — squarely in the "mature" band, *not* in the open intraday/IV territory. This is a **strategic signal, not a defect**: it confirms our targets are comparable to the literature (good for Path-B "rigorous re-examination", §4), **and** it surfaces a **potential novelty axis** — adding a shorter (1-day or event-window) RV target, or an IV target, would move us into genuinely open space. That would be a **scope change requiring a `DECISIONS.md` entry** (it touches §5.3 and the §10 risk register: timestamp precision, microstructure noise, options-data sourcing).

### 2.2 The reference map, grouped as the report groups it

Cross-referenced against our `DESIGN.md` §13 ([R#]). **"✓ ours"** = already cited; **"＋new"** = not currently in our references.

**Foundational multimodal (the canonical setup: sentence-aligned text+audio, 3–30 d RV).**
- **MDRM** (Qin & Yang, ACL 2019) — RNN, GloVe + 27 Praat features. **✓ our [R1].**
- **HTML** (WWW 2020) — hierarchical Transformer, multi-task log-vol. **✓ our [R3].** (Undermind's own citation analysis ranks HTML the #1 most-cited foundational paper at 0.89 reference rate; MDRM #2 at 0.73.)
- **VolTAGE** (EMNLP 2020) — FinBERT text + inter-modal attention + **GCN over a stock-correlation graph**. **✓ our [R4].**

**Dataset / multitask.**
- **MAEC** (CIKM 2020). **✓ our [D2].**
- **Multimodal Multi-Task Risk Forecasting** (ACM MM 2020) — joint vol + price-movement. **＋new.**
- **NumHTML** (AAAI 2022) — numeral-aware hierarchical Transformer; returns main + vol auxiliary. **＋new.**

**LLM-centric / multi-source.**
- **ECC Analyzer** (ICAIF 2024 + arXiv). **✓ our [R8]** — and the direct ancestor of the collaborator's QA pipeline.
- **RiskLabs** (2024) — earnings call (LLM text + acoustic) **+ pre-call price time-series + firm news**, multi-task vol + 95% VaR. Insight: *LLMs best as feature extractors, not standalone forecasters.* **✓ our [R9].**

**Physics-informed audio.**
- **"The Sound of Risk"** (2025) — **PIAM** paralinguistic model + LLM text emotions mapped into a 3-D **Affective State** space (Tension/Stability/Arousal), ~150 call-level features, XGBoost; **explicit 1-day RV target**; up to **R²≈0.438 on 30-day** RV; finds **CFO instability in Q&A** strongly predictive; little power for CAR. **✓ our [R12].**

**Numeracy.**
- **NAM / ECNum** (CIKM 2021) and **GNAVol** (IJCNLP 2023) — numeral-type-aware models improving 3–7 d vol. **＋new (both).**

**Dialogue structure / denoising.**
- **Multi-Round Q&A Attention Network** (IJCAI 2020) — dialogue-aware attention over Q&A rounds. **＋new.**
- **DeFVP** (ICME 2024) — treats price/vol time-series and text as two modalities, with a **Differentiable Binary Selector** that picks which sentences feed the predictor (not all ECC text is informative). **＋new.**

**Fairness / robustness.**
- **Bias in Multimodal ECC Analysis** (NAACL 2021) — multimodal beats text-only on average but **amplifies male-vs-female error disparity**. **✓ ≈ our [R19] theme.**
- **DocFin** (EMNLP 2022) — adds semi-structured tables; **5–12%** vol/price gains, **>30% gender-bias reduction**. **✓ our [R20].**
- **AMA-LSTM** (2024) — adversarial multimodal LSTM, FinBERT+audio; audio carries more gender bias than text. **✓ our [R7].**

Undermind's **adjacency** table also surfaces **KeFVP** (✓ our [R6]), **"Same Company, Same Signal"** (✓ our [R14]), **ECHO-GL** (✓ our [R10]), and a "Language Modeling for the Future of Finance" survey — confirming our reference set already covers the spine of the field.

> **How it fits us.** Our §13 already cites the spine (**[R1]/[R3]/[R4]/[R6]/[R7]/[R8]/[R9]/[R10]/[R12]/[R14]/[R20]**). The review surfaces a handful of **genuinely new, citable references** worth adding to §13: **RiskLabs** multi-source fusion, **DeFVP** sentence-selection, **NAM/GNAVol** numeracy, **Multi-Round Q&A attention**, **NumHTML**, **Multimodal Multi-Task** — see §5.3 for the proposed (not-yet-logged) §13 addition.

### 2.3 The review's design recommendations (and how they land on us)

- **Targets:** keep n-day RV as benchmark **but** extend to intraday / event-window RV and **implied volatility** → our **novelty-axis** candidate (§2.1).
- **Modeling:** LLM + RAG (ECC Analyzer) + numeracy-aware modules (GNAVol/NAM) + dialogue/selection (DeFVP). → reinforces our **Stage 5 (LLM structured features)**; numeracy-aware handling is *not* in our design and could be a small Stage-5 enhancement (numbers like EPS surprises, guidance, margins are exactly the volatility-relevant tokens).
- **Audio:** move beyond fixed Praat features to physics-informed/robust paralinguistics (PIAM). → consistent with our **eGeMAPS → WavLM → emotion2vec+** ladder (§6 Stage 3).
- **Econometrics:** time-based splits & rolling windows; **factor-only / vol-history baselines**; firm/sector controls; market/sector vol + **VIX**; careful microstructure handling; evaluate on **log-vol scale**, **out-of-sample R²**, economic loss. → this is **our §7 verbatim, almost** — independent confirmation that our evaluation discipline is the right call. (We already do log-RV, R²_OOS vs persistence, ticker/quarter cluster bootstrap, DM/Holm.) The collaborator's pipeline (§1.6) implements *almost none* of these — a useful contrast.
- **Fairness/interpretability as first-class:** group-level error disparities, adversarial mitigation, affective + attention explanations. → our **§3.5 + §7.3** already bake this in.

---

## 3. Figure-by-figure index (all 37 PNGs)

Listed in the order they appear in the document. (Filenames are non-sequential because the export shuffles them.)

| # in doc | File | What it shows | Relevance to us |
|---|---|---|---|
| 1 | `image11.png` | Paper excerpt — ECC Analyzer "Question Bank" RAG: retrieve relevant chunks per question | Source idea behind collaborator's QA pipeline; our [R8] |
| 2 | `image34.png` | Paper excerpt — concatenate selected sentences → text embedding `T_t` (size 1024), eq (8) | Text-feature construction; cf. our Stage-2 pooled embeddings |
| 3 | `image19.png` | Paper excerpt — additive/weighted fusion + regression MSE loss, eqs (9–10) | Fusion math; cf. our Stage-4 gated/cross-attn fusion |
| 4 | `image25.png` | **Prompt** — Task-1 QA generation (2–4 pairs, category, `source_lines`, overlap context) | Early QA prompt; Stage-5/T6.1 territory |
| 5 | `image23.png` | Sample model text output (restructuring discussion) | QA/transcript sample |
| 6 | `image12.png` | **MAEC audio dir listing** — `…/20150721_MSFT/`: `features.csv`, per-sentence `*.mp3`, `text.txt` | Confirms MAEC per-sentence audio + speaker-coded filenames; T1.5 |
| 7 | `image3.png` | Sample ECC transcript turns (operator/management) | Corpus sample |
| 8 | `image27.png` | Sample ECC transcript turns (Q&A) | Corpus sample |
| 9 | `image5.png` | **Prompt** — QA generation (LangChain `ChatPromptTemplate`, volatility-relevant, `source_lines`) | Stage-5 prompt design |
| 10 | `image22.png` | **Sample generated QAs** with `source_lines` (Q2 EPS, FY16 guidance) | What their QA features look like; auditable-feature flavour |
| 11 | `image8.png` | **Prompt** — generate volatility **topic labels** for a QA batch (avoid company/industry names) | Clustering step 1; Stage-5 topic-feature idea |
| 12 | `image13.png` | **Sunburst** — candidate labels merged into clusters | The 35-cluster taxonomy (visual) |
| 13 | `image26.png` | Sample QAs as JSON (`al1_turn_QA`) with `source_lines` | Data-structure sample |
| 14 | `image9.png` | **Table** — representative cluster labels + sample sub-labels (Market Perf & Volatility, Revenue/Subscription, Tech Innovation, Competitive Dynamics, Supply Chain, Pricing) | The taxonomy content |
| 15 | `image30.png` | **Bar chart** — distribution of labelled QAs over 35 labels (200-sample): top = Market Perf & Volatility (17) … tail = Currency/Commodity (2) | Topic-frequency feature distribution |
| 16 | `image29.png` | **ECC Analyzer's "Table 3" Question Bank** (Focus Area / Item / Questions) | The *fixed* question bank they replace with data-driven QAs |
| 17 | `image35.png` | **Prompt** — Qwen audio **Prompt 1** (acoustic/emotional; system+user with `audio_path`) | Stage-6 audio-LLM recipe |
| 18 | `image32.png` | **Sample record** — SkyBlue tech QA: `label`, `audio_responses` with Qwen `model_response` describing confident/upbeat tone | Shows audio-LLM output conditioned on QA |
| 19 | `image16.png` | **Sample record** — Valero/VLP QA; Qwen says *"Neutral mood, male, aged 41 years"* | **§3.5 gender/age-leakage red flag** in audio features |
| 20 | `image2.png` | **Prompt** — Task-4 **general** QA generation (generalized conceptual Q, exclude entity names) | Improved QA prompt |
| 21 | `image37.png` | **Prompt** — refined general QA prompt (final Task-4 version) | Improved QA prompt v2 |
| 22 | `image28.png` | **OpenAI Batch JSONL** — `"model":"gpt-5","reasoning":{"effort":"low"}` rows | **Proprietary-LLM evidence**; conflicts with our open-weights rule |
| 23 | `image6.png` | Histogram — **EC** `num_lines` (572 calls) | Corpus profiling |
| 24 | `image10.png` | Histogram — **EC** `num_words` | Corpus profiling |
| 25 | `image36.png` | **Stats table — EC** (count 572; mean 157 lines / 3,306 words; max 522 / 9,514) | Confirms EC = 572-call Qin & Yang set ([D3]) |
| 26 | `image15.png` | Histogram — **MAEC** `num_lines` (3,443 calls) | Corpus profiling |
| 27 | `image4.png` | Histogram — **MAEC** `num_words` | Corpus profiling |
| 28 | `image17.png` | **Stats table — MAEC** (count 3,443; mean 116 lines / 2,213 words; max 496 / 9,081) | Confirms MAEC = 3,443 ([D2]); informs 100-line chunk choice |
| 29 | `image7.png` | **Prompt** — "Financial Volatility Topic Extraction" from *questions* (preliminary labels) | Clustering step 1 (MAEC variant) |
| 30 | `image21.png` | **Silhouette vs k** (subsampled embeddings); peaks ≈ k 60–70 (~0.135) | **They chose k=35 — a local, not global, optimum; honesty caveat** |
| 31 | `image14.png` | **Prompt** — select one representative label per cluster | Clustering step 3 |
| 32 | `image20.png` | **Sunburst** — final 35 clusters with sub-labels | Final taxonomy (visual) |
| 33 | `image33.png` | **Prompt** — classify each QA into exactly one of 35 labels | Clustering step 4 |
| 34 | `image18.png` | Sample general-style QAs with `label` + `source_lines` | Post-Task-4 QA quality |
| 35 | `image1.png` | **Final Qwen embedding records** — `qa_index`, `question`, `topic`, `embedding`: float32 array | The actual feature output fed to the regressor |
| 36 | `image24.png` | **Learning curve** — train/val MSE over 100 epochs (Vols+text, MAEC-15) | Downstream model training |
| 37 | `image31.png` | **Learning curve** — train/val MSE over 100 epochs (Vols+text+audio Prompt 1) | Downstream model training |

**Pattern summary:** the figures are ~13 prompt screenshots (QA gen × several iterations, clustering × 4 steps, audio × 1), ~6 sample data records/QAs, ~6 corpus-profiling plots/tables (EC & MAEC), 3 cluster-taxonomy visuals, 1 silhouette plot, 2 learning curves, 3 paper-excerpt equations, 1 MAEC dir listing, 1 batch-API JSONL, 1 reference question-bank table.

---

## 4. Synthesis — how it all fits our project

### 4.1 Alignment matrix (collaborator's component → our project)

| Collaborator component | Our nearest anchor | Relationship |
|---|---|---|
| MAEC (3,443) primary | `DESIGN.md` §5.1 [D2]; **T1.5** | Same dataset; we treat it **secondary**, FinCall primary |
| EC (572) primary | §5.1 [D3] | We **demote to one comparability table**, never redistribute |
| KeFVP as baseline | §3 [R6]; Stage-0/1 floor | We treat KeFVP as **a control**, not a target |
| **KeFVP released labels** | §5.3; **T1.2/T1.3** | **Conflict** — we **compute** targets from Stooq, unit-tested |
| Speaker-turn chunking | **T3.1** | **Adoptable** directly |
| Data-driven QA generation | §6 Stage 5; **T6.1** | New; candidate exploratory feature (open-model re-impl needed) |
| QA topic clustering (k=35) | §6 Stage 5 / RQ3; §3.6 | New; candidate "topic-frequency" structured feature |
| Qwen2.5-Omni audio embeddings | §6 **Stage 6** (gated); §8.3 VRAM | Concrete recipe for our optional cloud stage |
| Audio front-end primer (STFT/mel/tokens) | §6 Stage 3/6 | Reusable reference notes |
| 10-seed mean±std results | §7.2 (≥5 seeds) | **Aligned** — they exceed our floor |
| Text-vs-+audio ablation | §7.6 grid; RQ2; Result Table 4 | Same question; ours adds controls |
| Prompt-design ablation (P1 vs P2) | §6 Stage 6; §7.5 exploratory | Useful evidence prompts matter |
| Undermind lit review | §13 references; §3 lit map | Confirms our spine; adds ~6 new refs |
| Intraday / IV gap (Undermind) | §5.3 targets; §10 risk | **Novelty-axis candidate** (needs DECISIONS entry) |

### 4.2 Conflicts with `DESIGN.md` (explicit)

1. **Proprietary LLMs** (GPT-5 / GPT-5-mini / GPT-5.2 / OpenAI embeddings; figure `image28`) vs. our **open-weights-only** Stage-5 (`Qwen2.5-7B-Instruct`, §6) and reproducibility mandate (§9, risk #9). *Their whole QA+cluster pipeline would need a Qwen re-implementation to be usable by us.*
2. **Borrowed KeFVP labels** (616/1,213 firms) vs. **computed targets from raw prices** with after-hours rule + unit tests (§5.3, T1.2/T1.3); their approach also **silently drops ~half the firms** — violates our "every exclusion needs a reason code" rule.
3. **No identity controls / no Δv / no HAR-RV floor** vs. our §3.1/§3.3/§5.3/§7.3 — the core of our thesis. Their "Ours beats KeFVP" is *unguarded* against the [R14] ticker-identity critique they themselves cite.
4. **No ticker-disjoint (or temporal×disjoint) split** vs. §5.4 — temporal-only at best.
5. **No DM/Holm significance** vs. §7.2 — improvements reported as raw mean±std only.
6. **No post-LLM-cutoff lookahead test** vs. §3.2/§7.4 (our Phase 7) — and their 2016–2017 corpus is well inside every current model's training window, making lookahead risk *higher* than ours (2019–2021 + a 2025–2026 holdout).
7. **Audio features encode gender/age** (figure `image16`) with no fairness diagnostic vs. our §3.5 gender-confound analysis.

### 4.3 Adoptable ideas (candidate — not committed)

- **Speaker-turn chunking** → fold into **T3.1** transcript normalization (low-risk, aligns with our existing speaker metadata).
- **QA-topic-frequency features** (their K-category histogram per call) → an **exploratory Stage-5 feature** variant, re-implemented with **Qwen** + constrained JSON, fit on **train split only**.
- **Qwen2.5-Omni masked-mean-pool-the-Thinker** recipe → the concrete implementation for our **Stage 6** audio-LLM scoring, *if* the Stage-5 gate is cleared.
- **Prompt-design ablation** → evidence that **task-aware prompting** materially changes audio usefulness; informs our Stage-3/6 prompt choices and belongs in our §7.5 exploratory analyses.
- **Their Yahoo-price-drift warning** → independent support for our **Stooq + Tiingo + SHA-256 manifest** price design.
- **New literature** (RiskLabs, DeFVP, NAM/GNAVol, Multi-Round-QA, NumHTML, Multimodal-Multi-Task) → add to §13.

### 4.4 Proposed follow-ups (NOT yet logged — would need `DECISIONS.md`/`TASKS.md` entries)

> These are recommendations from this review only. Per CLAUDE.md, promoting any of them requires a dated `DECISIONS.md` entry (and a `TASKS.md` ID if it adds work). **No canonical doc is edited by this file.**

1. **[Reference upkeep]** Add RiskLabs, DeFVP, NAM, GNAVol, Multi-Round-Q&A-Attention, NumHTML, Multimodal-Multi-Task-Risk to `DESIGN.md` §13. *(Low-risk doc edit; still log it.)*
2. **[Exploration, Stage 5]** Evaluate **QA-topic-frequency features** (open-model re-impl) as an exploratory structured-feature set, fit on train-only, against our Stage-2 embeddings — under the existing Stage-5 human-audit gate (κ>0.6, T6.2).
3. **[Exploration, Stage 6]** Record the **Qwen2.5-Omni recipe** as the concrete Stage-6 audio-LLM plan (still gated on Stages 2–5 showing DM-significant signal + a budget entry).
4. **[Possible scope change]** Consider a **shorter-horizon (1-day / event-window) RV** and/or **implied-volatility** target as a novelty axis (Undermind's identified open gap). *This touches §5.3 and the §10 risk register (timestamp precision, microstructure noise, options-data sourcing) — a real scope decision, not a quick add.*
5. **[Collaboration feedback]** If we engage with the collaborator: the highest-value thing they could add is our **control suite** — a **training-free HAR-RV/persistence floor**, a **ticker-only baseline**, a **same-ticker transcript-shuffle**, and **Δv / HAR-residual** targets — to test whether "Ours beats KeFVP" survives the [R14] identity critique. That single change would tell us (and them) whether their 8% MSE gain is signal or memorisation.

---

## 5. Provenance & extraction notes

- Source files: `ingest/Multi-modal Volatility .zip`, `ingest/Undermind - …short-horizon stock volatility.pdf`.
- The zip was extracted to `ingest/_extracted/` (HTML + `images/`) so the 37 figures could be inspected; that folder can be deleted without losing anything — it's a verbatim copy of the zip.
- All 37 figures were opened and described directly (§3). Numbers quoted (MSE 0.188/0.204/0.288/0.318; k=35; 9,301→7,288 labels; 33,844 QAs / 5,742 calls / ~$50; 572 and 3,443 call counts; 616/1,213 firm coverage; Qwen 3584-d Thinker, 4-bit NF4; 16 kHz/25 ms/10 ms/128-mel/300 s) are transcribed from the HTML text and the figures.
