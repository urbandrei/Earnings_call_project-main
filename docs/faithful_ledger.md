# Fidelity ledger — T6R.3 faithful-reproduction sprint

One section per paper. *Verbatim* = the authors' code runs unchanged; *patched* = a
forward-port change that does not alter numerics (listed with file and reason);
*substituted* = an input or component we rebuilt or replaced (labelled in every result
row); *reason-coded* = not reproducible, with the missing artefact named. Third-party
code lives under `D:\ecvol-data\raw\ref\repos\<name>` at the commits in
`data/manifests/repos.json`; patched copies are built at run time under `data/work/` and
are never committed.

## Same-Company-Same-Signal (Yu, Liu & He, Findings of ACL 2025) — `ecvol reproduce scss-tsmixer|scss-tmlp|scss`

| status | item |
|---|---|
| verbatim | `run.py`, `exp/`, `data_provider/`, `models/TSMixer.py`, `models/TMLP.py`, rolling masks in `DEC.csv`, seed 2021, `drop_last=True` on train/val |
| patched | `data_provider/data_loader.py`: drop the unused `from sktime.datasets import load_from_tsfile_to_dataframe` (removed upstream in sktime) |
| patched | `utils/reconfig_args.py`: run-name `rt(year|quarter)` → `rt(year-quarter)` (`|` is illegal in Windows paths) |
| patched | `utils/tools.py`: `np.Inf` → `np.inf` (NumPy 2); `matplotlib.pyplot` import → a `switch_backend` no-op stub (plotting is never reached) |
| checked out | the repository's 304 result CSVs carry `|` in their names and cannot be checked out on Windows; the tree was materialised file-by-file with `git show`, and the published per-cell MSEs were read from the file names (`raw/ref/scss/earnings_results_index.txt`) |
| **result** | **TSMixer, 76 cells: median \|ours − published\| = 1.6e-5, max 4.2e-3, 70/76 within 1e-3** (`result_table_6r_scss_tsmixer.csv`, run 20260909T054744Z) |
| substituted (name only) | TMLP embeddings fetched from the authors' Drive `Embeddings/openai` (`DEC.npz`, `DEC2RandomTicker.npz`, `DECRandomAll.npz`, each 1800 × 3072 float64 with the same ids); the Drive file `DEC2RandomTicker` is staged as the script's `DECRandomTicker` |
| **result** | **TMLP, 228 cells (3 embedding sets × 76): median \|ours − published\| 0.014, max 0.22 at single cells** (single seed 2021, GPU non-determinism); per-window means DEC 0.608/0.285/0.215/0.202 vs 0.587/0.283/0.211/0.205, RandomTicker 0.572/0.276/0.213/0.209 vs 0.565/0.273/0.204/0.200, RandomAll 0.646/0.315/0.259/0.249 vs 0.637/0.319/0.255/0.247 — the paper's finding (real ≈ ticker-shuffled embeddings, both below all-shuffled) reproduces (`result_table_6r_scss_tmlp.csv`, run 20260909T065848Z) |
| earlier | PEV / STPEV reproduced to 3 decimals (`ecvol reproduce scss`, 2026-09-03) |
| verbatim (S4) | `SS.ipynb` functions `build_aug_stpev_mean`, `run_pev_stpev`, `run_EC_MAEC`, exec-loaded from their cells unchanged, run on the shipped `dataset/EC/*.csv` and `dataset/MAEC/*.csv` exactly as the notebook's driver cells (MAEC: `day_earnings = time`); no patches needed under pandas 3.0.3 |
| published source (S4) | the comparison tables the notebook printed when the authors ran it (stored cell outputs), parsed — not hand-copied |
| **result (S4)** | **48/48 cells (EC, MAEC-15, MAEC-16 × PEV/STPEV/Aug_PEV/Aug_STPEV × τ) equal at the notebook's own 3-decimal rounding**; means Aug_PEV 0.367 / 0.283 / 0.229, Aug_STPEV 0.296 / 0.225 / 0.247 (`result_table_6r_scss_aug.csv`, run 20260916T163850Z) |
| audit notes (S4) | the augmented history is extra same-ticker earnings outside the benchmark's training set, filtered to `day_earnings < first test date` and to test tickers — so **Aug_PEV is a mean of the test tickers' per-ticker means**, i.e. it too uses test-set identity; windows reaching the first test date (weekday approximation): τ≤15 none, τ=30 EC 4/2195, MAEC-15 8/3192, MAEC-16 0/5033 — negligible |

## HTML (Yang et al., WWW 2020) — `ecvol reproduce html-faithful`

| status | item |
|---|---|
| verbatim | `SelfAttention`, `TransformerBlock`, `RTransformer` from `Model/Sentence-Level-Transformer/transformers/transformers_gpu.py`, exec-loaded from `class SelfAttention` onward; heads 2, depth 2, seq 520, mean pooling, Adam 2e-5, batch 4, clip 1.0, multi-task MSE loss, 10 epochs, α ∈ {0.1,…,1.0} on validation |
| patched | the 26 import lines above the classes (TensorFlow-1 `set_random_seed`, torchtext, matplotlib, a `transformer` package the repo never ships) are replaced by an injected namespace |
| **rewritten** | the driver `run_gpu.go`: as shipped it splits features, main labels and auxiliary labels with three independent, unseeded `train_test_split` calls (X and y decorrelated), seeds nothing, and its lr warm-up sets `opt.lr` (a no-op). Our driver keeps X/y aligned and seeds; three labelled conditions: `code` (random 70/10/20, dropout 0.0 as in the code, min-over-epochs on validation), `paper` (chronological 7:1:2, dropout 0.5 as in Table 1), `published_split` (VolTAGE split3) |
| substituted | inputs: 1024-d sentence vectors recomputed as BERT-WWM-Large layer −2 token means over `TextSequence.txt` lines (the repo's `Bert-As-A-Service-Readme.md` recipe; the original `.npy` is behind a dead Drive link); labels from the lineage's shipped files (VolTAGE `future_τ/past_τ`, KeFVP `future_Single_τ`) |
| **result (text)** | seeds 0–2, α on validation, test MSE at the best-validation epoch, 10 epochs: `code` (random 70/10/20, dropout 0) **0.660 / 0.336 / 0.253 / 0.160** (persistence 1.328 / 0.448 / 0.260 / 0.137); `paper` (chronological 7:1:2, dropout 0.5) **0.972 / 0.439 / 0.406 / 0.248** (persistence 1.496 / 0.563 / 0.335 / 0.199); `published_split` (VolTAGE split3, dropout 0.5) **1.231 / 1.122 / 1.063 / 1.479** ± 0.1–0.3 (persistence 1.485 / 0.570 / 0.340 / 0.202); published 1.175 / 0.372 / 0.153 / 0.133. Reading: under the published recipe (10 epochs, lr 2e-5, raw log-volatility targets, dropout 0.5) the model is still converging at epoch 10 (best epochs 7–9, seed spread 0.3) and never reaches persistence at τ ≥ 15 on any split; the only cell better than the paper is τ=3 on a random split with dropout 0 — the split every later paper inherited reproduces the τ=3 number and none of the others (`result_table_6r_html_faithful.csv`, run 20260909T080339Z) |
| **result (text+audio)** | 27 Praat features per sentence concatenated (1051-d, z-scored): `code` **0.658 / 0.347 / 0.244 / 0.165** vs published 0.845 / 0.349 / 0.251 / 0.158 — reproduces at τ ≥ 7 and beats the paper at τ=3; `paper` 0.876 / 0.520 / 0.291 / 0.239; `published_split` 2.70 / 1.20 / 1.44 / 1.41 (diverges; seed spread 0.1–0.6). Reading: the multimodal cell reproduces only under the random-split protocol the shipped code implements (`result_table_6r_html_faithful.csv`, run 20260909T174611Z) |
| our port | `ecvol reproduce html` (2026-09-03) additionally substitutes the encoder (BGE-M3 chunks) and `nn.MultiheadAttention`; its auxiliary label used `future_3` for every τ — **H4 (2026-09-09): now ln\|r\| on session +τ, the shipped `future_Single_τ` definition** |
| verified (labels) | against our EC price store, to 3 decimals: the shipped `future_Single_τ` (KeFVP) = ln\|r\| on session +τ; the shipped `future_τ` (VolTAGE/KeFVP split3) = ln √(Σ(r−r̄)²/τ) over sessions +1…+τ — i.e. the **mean-subtracted** formula (Sawhney's), not Qin & Yang's un-demeaned Eq. 1; our trading-day `v_post` differs by the demeaning only |

## KeFVP (Niu et al., Findings of EMNLP 2023) — `ecvol reproduce kefvp`

| status | item |
|---|---|
| verbatim | `kefvp/final_series_infer.py` (CondAutoformer, 200 epochs, lr 2e-4, wd 5e-2, mu 0.7, 10 repeats), `data_utils.py`, the shipped `price_data/` split + label files |
| patched | `/your/project/path/` and `/your/dataset/path/` placeholders → the build's `proj/` and `dataset/`; log filename `%H:%M:%S` → `%H-%M-%S` (Windows); `float(Series)` → `float(Series.iloc[0])` and `audio_path.values.astype(float64)` (pandas 3 dtype semantics); output dirs `preds_dir/text_dir/reg`, `log/<name>` created |
| patched (release defects) | `from data_utils import set_seed` — the function lives only in `pretrain/data_utils_pretrain_with_kg.py`: copied in verbatim; `transformers_model/__init__.py` star-imports `transformers_gpu.py`, which imports `CrossAttention`/`GraphConvolution` that no module defines: init emptied (the path uses only `transformers_model.modules`); `modules.py` lost its `class GraphChannelAttLayer(nn.Module):` header, so that class's `__init__`/`forward` had overridden `TransformerBlock`'s: header restored; `latent` (KumaGate/kumadist) never released and never instantiated on the CondAutoformer path: stubbed, raises if used; matplotlib/pylab: no-op stubs |
| noted | the audio branch is commented out in `CondInfer.forward` upstream, so the published model is text + price; the unshipped HuBERT audio pickle is replaced by zeros by the script's own `try/except` |
| pending (user) | EC headline run needs the released KePt-BERT-large embedding pickle (`text_embedding`, Drive `1F83bjiJKEpq_MYrc0lzQb9rOLgooz-5E`): the file is visible in the browser but `gdown`/`curl` are refused — HANDOFF |
| substituted | MAEC-15/16: `raw_bert_base_uncased` sentence embeddings regenerated with the authors' `generatePtmEmbeddings.py` (`bert-base-uncased` pooler output, 512 sentences × 768); patched: `Text.txt` → MAEC's `text.txt`, chunked encoding (64 sentences at a time, numerically identical), only the 2,165 split folders encoded |
| **result (MAEC)** | ten repeats × 200 epochs, `raw_bert_base_uncased` regenerated: **MAEC-15 0.419±0.007 / 0.185±0.002 / 0.123±0.005 / 0.086±0.002 vs published 0.418±0.012 / 0.187±0.003 / 0.122±0.003 / 0.087±0.002; MAEC-16 0.442±0.074 / 0.288±0.040 / 0.363±0.038 / 0.188±0.032 vs 0.445±0.064 / 0.279±0.044 / 0.303±0.036 / 0.177±0.033** — 7/8 cells within one published std, seed spreads match; the miss is the MAEC-16 τ=15 cell of a row that does not average to its stated mean (`result_table_6r_kefvp.csv`, run 20260911T044848Z) |
| `[colab]` | KePt adaptive pre-training (BERT, 60 epochs, ~5 h GPU, needs the `kept_dataset` Drive pickles) — not run locally |

## Sawhney et al. (ACM MM 2020) — `ecvol reproduce sawhney` (harness port)

| status | item |
|---|---|
| reason-coded (faithful) | unshipped `stock_data.csv`, `AllRetPrices/`, `so-cal.json`, `dictionary.csv`; TF 2.1 / Keras 2.3.1 / tensorflow-addons 0.8.3 + delta v1-graph layers; `text_feature_extraction.py` returns after the first record; no multi-task loss in the released code |
| ported | labels (`generate_price_vol_data.py`: mean-subtracted log σ over the τ sessions after/before the call, from our EC price store); split (`month*30+day` key, 60/20/20); text BiLSTM(100) 2 epochs best-val; SVR(rbf) grid on the 30-session `Past_Volatility_τ` vector; (α, 1−α) ensemble |
| substituted | GloVe 6B-300d sentence means over their shipped `union_vocab.csv` (no Mittens retrofit — W2); regex + scikit-learn stop words instead of NLTK; Keras recurrent dropout → input dropout; audio branch = the 27 Praat sentence features (their 26-d Praat+prosodic set is unshipped), aligned model ported from `aligned_audio_reg_model.py` (BiLSTM text, BiLSTM audio, 5-head attention Q=text/K=V=audio, BiLSTM, linear; 1 epoch, last-epoch weights) |
| labelled | the authors tune (α, β) on the **test** set; both `tuned_on=test` and `tuned_on=val` rows are reported |
| published | 0.601 / 0.308 / 0.181 / 0.119 (second-hand: KeFVP Table 2 "Ensemble(Text+Audio)"; the MM'20 PDF is not held locally) |
| **result** | finance SVR 0.712 / 0.370 / 0.218 / 0.154 (persistence 1.491 / 0.563 / 0.345 / 0.205); text BiLSTM after their 2 epochs 2.14 / 1.58 / 1.38 / 1.40 and aligned audio after their 1 epoch 6.51 / 5.30 / 5.37 / 5.13 (5 seeds) — both in the untrained regime; **3-way ensemble 0.639 / 0.298 / 0.175 / 0.102 test-tuned (α = 0, β ≈ 0.1) vs published 0.601 / 0.308 / 0.181 / 0.119**, 0.686 / 0.324 / 0.187 / 0.117 val-tuned (β ≈ 0.03): the untrained branch's near-constant output acts as an intercept on the SVR, which the test-tuned grid rewards — the published number is reachable with no learned text or audio signal; 547 EC calls (`result_table_6r_sawhney.csv`, run 20260909T071241Z) |

## DialogueGAT (Sang & Bao, Findings of EMNLP 2022) — `ecvol reproduce dialoguegat` (harness port)

| status | item |
|---|---|
| reason-coded (faithful) | `data/data_swd.pkl` unreleased; corpus = private SeekingAlpha re-scrape (~3,400 calls, 2015–2018, global named speakers); labels need CRSP; DGL has no wheels for torch 2.11 |
| ported | TextCNN (100 × kernels 3/4/5, max-pool, no ReLU) over frozen GloVe; utterance chain + utterance↔speaker graph; 5 GAT layers × 5 heads (head-mean, residual, dropout 0.1); context attention over speaker and utterance nodes; `v_past`; Adam 1e-5 / wd 1e-6 / batch 4 / ≤100 epochs / patience 5 / clip 15 / seed 1234; per-year chronological 70/10/20 |
| substituted | corpus FinCall 2019/2020/2021 (their 2015/2016/2017-18); speaker nodes role-typed (management/analyst/operator) instead of named persons — the paper's "random speaker embedding" ablation is the nearest published row; GloVe 6B instead of 840B; PyG `GATConv` instead of DGL; labels our trading-day `v_post/v_pre` |
| published (reference only) | τ=3/7/15: 2015 0.4530/0.3236/0.1898; 2016 0.4549/0.2884/0.1810; 2017-18 0.4090/0.2886/0.2036 |
| **result** | FinCall, seed 1234, τ=3/7/15/30: **2019 0.694/0.320/0.206/0.134** (persistence 1.429/0.566/0.289/0.139; best epochs 40/53/44/28); **2020 0.621/0.350/0.392/0.270** (persistence 1.409/0.335/0.184/0.101; every horizon stopped at epoch 0 — validation never improved after the first pass across the COVID-spring → autumn boundary); **2021 0.794/0.344/0.182/0.082** (persistence 1.301/0.465/0.225/0.118; epochs 3/20/22/43). Beats persistence in 9 of 12 cells, loses at 2020 τ≥7 (`result_table_6r_dialoguegat.csv`, run 20260909T090023Z) |

## Controls (T6R.4) — only the split changes

Protocol: DECISIONS 2026-09-16 (+ two same-day refinements). Every model keeps the configuration of its faithful/port run above; its anchor split is re-run in the same invocation and matches the T6R.3 numbers (bit-identical for HTML, Sawhney, SCSS; within 0.0014 for the PyG DialogueGAT port). `embargoed` = no train/val call whose 30-session target window reaches the first evaluation call (validation re-carved where it is chronological); `ticker_disjoint` = no evaluation ticker in train/val, compared against `anchor_heldout` (the anchor scored on the same held-out calls) wherever the controlled test set is a subset. MSE at τ = 3 / 7 / 15 / 30.

| model (data) | anchor | embargoed | ticker-disjoint (paired anchor) | reading |
|---|---|---|---|---|
| HTML text, faithful (EC; seeds 0–2) | code split 0.660 / 0.336 / 0.253 / 0.160 | 0.726 / 0.347 / 0.246 / 0.178 | 0.777 / 0.522 / 0.393 / 0.291 (vs code split; different test calls) | +18 / +55 / +55 / +82%; above persistence at τ ≥ 15 |
| HTML text+audio, faithful (EC) | 0.658 / 0.347 / 0.244 / 0.165 | 0.684 / 0.345 / 0.232 / 0.186 | 0.789 / 0.500 / 0.394 / 0.281 | +20 / +44 / +61 / +70% |
| Sawhney port, val-tuned 3-way ensemble (EC; 5 seeds) | their split 0.686 / 0.324 / 0.187 / 0.117 | 0.719 / 0.306 / 0.184 / 0.107 | 0.775 / 0.416 / 0.263 / 0.161 | +13 / +28 / +41 / +38%; worse than its own finance SVR at τ ≥ 7 |
| SCSS TMLP, authors' code (DEC; seed 2021, 19 quarters) | 0.608 / 0.285 / 0.215 / 0.202 | 0.585 / 0.287 / 0.216 / 0.208 | 0.660 / 0.354 / 0.280 / 0.267 vs 0.591 / 0.288 / 0.224 / 0.211 | +12 / +23 / +25 / +27%, worse in 16 / 17 / 17 / 18 of 19 quarters |
| SCSS TSMixer, authors' code (DEC; price only) | 0.561 / 0.252 / 0.210 / 0.257 | 0.570 / 0.251 / 0.205 / 0.232 | 0.556 / 0.254 / 0.194 / 0.233 vs 0.519 / 0.241 / 0.206 / 0.252 | mixed (worse in 14 / 13 / 8 / 10 of 19) |
| DialogueGAT port (FinCall per year; seed 1234) | see T6R.3 row | 2019 +11 / +14 / +10 / +23%; 2021 −7 / −6 / +52 / 0%; 2020 τ=30 diverges (1.187) | 2019 +15 / +7 / +18 / +8%; 2021 ≈ 0; 2020 mixed | inconclusive (one seed, 42–60 held-out calls/year, role-typed speakers) |
| KeFVP (MAEC-15/16; 3 repeats) | — | running | running | — |

Substitutions specific to the controls: SCSS and KeFVP masks/split files are rewritten outside the authors' code (row order preserved — TMLP indexes embeddings by row, KeFVP pairs its avg and single-day files by position); KeFVP runs 3 repeats instead of 10 (a patched loop bound, labelled); the held-out third is `CONTROL_SEED` 20260916.
