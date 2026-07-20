# Reference existence audit — 2026-07-19

**Scope:** every numbered reference in DESIGN.md §13 (R1–R32; the slide deck's name-checks are a
subset of these) plus the additional entries in `paper/refs.bib` (dataset paper, model cards,
tool repos; B1–B8 below). **Method:** five parallel verification agents, each required to fetch
the cited URL and confirm the page's title/authors/venue against the citation; where a
publisher blocked automated fetches (403), existence was confirmed via a second independent
index (dblp, RePEc, institutional repositories) and the working alternate URL is recorded.
No verdict relies on model memory. **Cross-check:** all 33 `refs.bib` keys are cited in the
LaTeX sources and every `\cite` key resolves — no dangling or unused entries.

**Headline: all 40 references exist. Zero fabricated or unlocatable citations. 5 entries carry
citation errors (wrong year or title) — errata listed at the bottom.**

## Verdicts

| Ref | Cited as | Verdict | Evidence (fetched title · first author · date · working URL) |
|---|---|---|---|
| R1 | Qin & Yang, ACL 2019 | VERIFIED | "What You Say and How You Say It Matters: Predicting Stock Volatility Using Verbal and Vocal Cues" · Yu Qin · ACL 2019 · https://aclanthology.org/P19-1038/ |
| R2 | MAEC, CIKM 2020 | VERIFIED | "MAEC: A Multimodal Aligned Earnings Conference Call Dataset for Financial Risk Prediction" · Jiazheng Li · CIKM '20 · ACM DL 403-blocks bots; DOI 10.1145/3340531.3412879 confirmed via https://kclpure.kcl.ac.uk/portal/en/publications/maec-a-multimodal-aligned-earnings-conference-call-dataset-for-fi/ and https://researchrepository.ucd.ie/handle/10197/12221 |
| R3 | HTML, WWW 2020 | VERIFIED | "HTML: Hierarchical Transformer-based Multi-task Learning for Volatility Prediction" · Linyi Yang · WWW 2020, pp. 441–451 · https://dl.acm.org/doi/10.1145/3366423.3380128 (GitHub repo also live) |
| R4 | VolTAGE, EMNLP 2020 | VERIFIED (title truncated) | Actual title ends "…with Graph Convolution Networks **for Earnings Calls**" · Ramit Sawhney · EMNLP 2020 main, pp. 8001–8013 · https://aclanthology.org/2020.emnlp-main.643/ |
| R5 | DialogueGAT, Findings EMNLP 2022 | VERIFIED | Title exact · Yunxin Sang · Findings of EMNLP 2022 · https://aclanthology.org/2022.findings-emnlp.117/ |
| R6 | KeFVP, Findings EMNLP 2023 | VERIFIED | Title exact · Hao Niu · Findings of EMNLP 2023 · https://aclanthology.org/2023.findings-emnlp.770/ (code https://github.com/hankniu01/KeFVP live) |
| R7 | AMA-LSTM, NAACL 2024 Industry | VERIFIED | Title exact · Shengkun Wang · arXiv 2407.18324; venue confirmed https://aclanthology.org/2024.naacl-industry.32/ |
| R8 | ECC Analyzer, ICAIF 2024 | VERIFIED (note) | ICAIF '24 title matches citation ("…Large Language Model**s**" vs "Model", trivial) · Yupeng Cao · https://dl.acm.org/doi/10.1145/3677052.3698689 — note: the arXiv v. (2404.18470) carries an older variant title ("Extract … Stock Performance Prediction") |
| R9 | RiskLabs, 2024 | VERIFIED | Title exact · Yupeng Cao · arXiv Apr 11 2024 (v2 May 2025) · https://arxiv.org/abs/2404.07452 |
| R10 | ECHO-GL, AAAI 2024 | VERIFIED | Title exact · Mengpu Liu · AAAI-24 Technical Tracks · https://ojs.aaai.org/index.php/AAAI/article/view/29305 |
| R11 | AT-FinGPT, FRL 2025 | VERIFIED | Title exact · Y. Liu · Finance Research Letters vol. 77, 2025 · ScienceDirect 403-blocks bots; confirmed via https://ideas.repec.org/a/eee/finlet/v77y2025ics1544612325002314.html |
| R12 | Sound of Risk, "2024" | **VERIFIED-MISMATCH** | Actual: submitted **Aug 26 2025**, full title ends "…Market Volatility **and Enhancing Market Interpretability**" · Xiaoliang Chen · https://arxiv.org/abs/2508.18653 |
| R13 | FinAudio, 2025 | VERIFIED | Title exact · Yupeng Cao · arXiv Mar 26 2025 · https://arxiv.org/abs/2503.20990 |
| R14 | Same Company, Same Signal, 2024 | VERIFIED | Title exact · Ding Yu · arXiv Dec 23 2024 · https://arxiv.org/abs/2412.18029 |
| R15 | "A Test of Lookahead Bias in LLM Forecasts", "2024" | **VERIFIED-MISMATCH** | Actual title: **"Detecting Lookahead Bias in LLM Forecasts"**, submitted **Dec 29 2025**; authors Gao/Jiang/Yan match · https://arxiv.org/abs/2512.23847 |
| R16 | DatedGPT, "2025" | **VERIFIED-MISMATCH** | Actual: submitted **Mar 12 2026** · Yutong Yan · title exact · https://arxiv.org/abs/2603.11838 |
| R17 | Corsi HAR-RV, JFEc 2009 | VERIFIED | "A Simple Approximate Long-Memory Model of Realized Volatility" · Fulvio Corsi · J. Financial Econometrics 7(2) 2009, pp. 174–196 · https://academic.oup.com/jfec/article-lookup/doi/10.1093/jjfinec/nbp001 |
| R18 | Finance-NLP survey, 2025 | VERIFIED (note) | Title exact · Nikita Tatarinov · arXiv Apr 9 2025 · https://arxiv.org/abs/2504.07274 — note: the "~14% exact reproduction" figure quoted in DESIGN §3.4 is not in the abstract; it rests on the paper body |
| R19 | Emo-bias, 2024 | VERIFIED | Title exact · Yi-Cheng Lin · arXiv Jun 7 2024 (INTERSPEECH 2024) · https://arxiv.org/abs/2406.05065 |
| R20 | DocFin, Findings EMNLP 2022 | VERIFIED | Title exact · Puneet Mathur · Findings of EMNLP 2022 · https://aclanthology.org/2022.findings-emnlp.139/ |
| R21 | MarketSenseAI 2.0, 2025 | VERIFIED | Title exact · George Fatouros · arXiv Feb 1 2025 · https://arxiv.org/abs/2502.00415 |
| R22 | P1GPT, 2025 | VERIFIED | Title exact (lowercase "a multi-agent…") · Chen-Che Lu · arXiv Oct 27 2025 · https://arxiv.org/abs/2510.23032 |
| R23 | AlphaAgents, 2025 | VERIFIED | Title exact · Tianjiao Zhao · arXiv Aug 15 2025 · https://arxiv.org/abs/2508.11152 |
| R24 | Kronos, 2025 | VERIFIED | Title exact · Yu Shi · arXiv Aug 2 2025 · https://arxiv.org/abs/2508.02739 |
| R25 | FinCast, CIKM 2025 | VERIFIED | Title exact · Zhuohang Zhu · arXiv Aug 27 2025; CIKM '25 confirmed via https://dl.acm.org/doi/10.1145/3746252.3761261 |
| R26 | Gu, Kelly & Xiu, RFS 2020 | VERIFIED | "Empirical Asset Pricing via Machine Learning" · Shihao Gu · RFS 33(5):2223–2273, 2020 · https://academic.oup.com/rfs/article/33/5/2223/5758276 |
| R27 | Diebold & Mariano, JBES 1995 | VERIFIED | "Comparing Predictive Accuracy" · Diebold & Mariano · JBES 13(3):253–263, 1995 · T&F 403-blocks bots; confirmed via https://ideas.repec.org/a/bes/jnlbes/v13y1995i3p253-63.html |
| R28 | Multi-Round Q&A, IJCAI 2020 | VERIFIED | Title exact · Zhen Ye · IJCAI 2020 (AI in FinTech track) · https://www.ijcai.org/proceedings/2020/631 |
| R29 | Multimodal Multi-Task, ACM MM 2020 | VERIFIED | Title exact · Ramit Sawhney · ACM MM 2020, pp. 456–465 · ACM DL 403-blocks bots; confirmed via https://vlgiitr.github.io/publication/mmtfrf/ |
| R30 | NAM/ECNum, CIKM 2021 | VERIFIED | "Distilling Numeral Information for Volatility Forecasting" · Chung-Chi Chen · CIKM 2021, pp. 2920–2924 · DOI resolves; confirmed via dblp |
| R31 | NumHTML, AAAI 2022 | VERIFIED | Title exact · Linyi Yang · arXiv Jan 5 2022, AAAI 2022 consistent · https://arxiv.org/abs/2201.01770 |
| R32 | GNAVol, IJCNLP-AACL 2023 | VERIFIED | Title exact · Ming-Xuan Shi · IJCNLP-AACL 2023 Short Papers · https://aclanthology.org/2023.ijcnlp-short.5/ |
| B1 | FinCall-Surprise dataset paper | **VERIFIED-MISMATCH** | arXiv 2510.03965 exists but actual title is **"FinCall-Surprise: A Large Scale Multi-modal Benchmark for Earning Surprise Prediction"** (bib has "An Earnings-Call Dataset with Audio, Transcripts, and Slides"); Dong Shu et al.; submitted **Oct 4 2025** · https://arxiv.org/abs/2510.03965 |
| B2 | Qwen2.5-7B-Instruct | VERIFIED | Model card live, owner Qwen, Apache-2.0 · https://huggingface.co/Qwen/Qwen2.5-7B-Instruct |
| B3 | BGE embeddings | VERIFIED (note) | bib cites BAAI/bge-large-en-v1.5 (live, MIT); the pipeline actually uses **BAAI/bge-m3**, which is also live · https://huggingface.co/BAAI/bge-m3 — bib should probably cite bge-m3 |
| B4 | FinBERT (ProsusAI) | VERIFIED | Model card live · https://huggingface.co/ProsusAI/finbert |
| B5 | WavLM-Large | VERIFIED | Model card live, owner Microsoft · https://huggingface.co/microsoft/wavlm-large |
| B6 | emotion2vec+ large | VERIFIED | Model card live · https://huggingface.co/emotion2vec/emotion2vec_plus_large |
| B7 | openSMILE python | VERIFIED | Repo live, eGeMAPS v01a/v01b/v02 supported, v2.6.0 (Jul 2025) · https://github.com/audeering/opensmile-python |
| B8 | Outlines | VERIFIED | Repo live ("Structured outputs for LLMs", dottxt-ai) · https://github.com/dottxt-ai/outlines |

## Errata to fix (all in DESIGN.md §13 and/or paper/refs.bib; papers themselves are real)

1. **R15** — title is wrong: "Detecting Lookahead Bias in LLM Forecasts" (not "A Test of Lookahead Bias in LLM Forecasts"); year is 2025, not 2024. Fix in DESIGN §13 + `lookahead2024` bib entry (key worth renaming at the same time).
2. **R16 (DatedGPT)** — year is 2026, not 2025. Fix in DESIGN §13 + `datedgpt2025` bib entry.
3. **R12 (Sound of Risk)** — year is 2025, not 2024; official title continues "…and Enhancing Market Interpretability". Fix in DESIGN §13 + `soundofrisk2024` bib entry.
4. **B1 (FinCall-Surprise)** — bib title is wrong: actual is "FinCall-Surprise: A Large Scale Multi-modal Benchmark for Earning Surprise Prediction"; year 2025. Fix `fincall2024` bib entry.
5. **R4 (VolTAGE)** — official title ends "…for Earnings Calls" (truncated in both DESIGN §13 and the bib). Cosmetic but should match the ACL Anthology record.

Minor notes, no action required: R8's arXiv version has a variant title (cite the ICAIF version, as we do); R18's 14% figure should be page-referenced when cited in the paper; B3 suggests re-pointing the BGE bib entry at bge-m3 to match the pipeline.

## Status: FIXES APPLIED (2026-07-19, same day)

All five errata above, plus the B3 BGE re-point, were applied to DESIGN.md §13 (R4/R12/R15/R16
lines + the §3.0 table year for R12) and `paper/refs.bib`. Bib keys whose year changed were
renamed (`soundofrisk2024→soundofrisk2025`, `lookahead2024→lookahead2025`,
`datedgpt2025→datedgpt2026`, `fincall2024→fincall2025`, `bge2023→bge2024`) and every `\cite`
site updated; the key cross-check passes (33/33 keys, no dangling or unused entries).
Remaining open note: page-reference R18's 14% figure at its citation site during Phase 8
writing.

**Slides check:** the deck names only Qin & Yang 2019, HTML, MAEC, KeFVP, ECC Analyzer, "Same Company, Same Signal" (2024), Corsi 2009, Diebold-Mariano, and the model/tool names — all verified above; no year or title stated in the slides is contradicted by the fetched records.
