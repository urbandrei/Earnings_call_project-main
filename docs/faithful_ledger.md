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
| earlier | PEV / STPEV reproduced to 3 decimals (`ecvol reproduce scss`, 2026-09-03) |

## HTML (Yang et al., WWW 2020) — `ecvol reproduce html-faithful`

| status | item |
|---|---|
| verbatim | `SelfAttention`, `TransformerBlock`, `RTransformer` from `Model/Sentence-Level-Transformer/transformers/transformers_gpu.py`, exec-loaded from `class SelfAttention` onward; heads 2, depth 2, seq 520, mean pooling, Adam 2e-5, batch 4, clip 1.0, multi-task MSE loss, 10 epochs, α ∈ {0.1,…,1.0} on validation |
| patched | the 26 import lines above the classes (TensorFlow-1 `set_random_seed`, torchtext, matplotlib, a `transformer` package the repo never ships) are replaced by an injected namespace |
| **rewritten** | the driver `run_gpu.go`: as shipped it splits features, main labels and auxiliary labels with three independent, unseeded `train_test_split` calls (X and y decorrelated), seeds nothing, and its lr warm-up sets `opt.lr` (a no-op). Our driver keeps X/y aligned and seeds; three labelled conditions: `code` (random 70/10/20, dropout 0.0 as in the code, min-over-epochs on validation), `paper` (chronological 7:1:2, dropout 0.5 as in Table 1), `published_split` (VolTAGE split3) |
| substituted | inputs: 1024-d sentence vectors recomputed as BERT-WWM-Large layer −2 token means over `TextSequence.txt` lines (the repo's `Bert-As-A-Service-Readme.md` recipe; the original `.npy` is behind a dead Drive link); labels from the lineage's shipped files (VolTAGE `future_τ/past_τ`, KeFVP `future_Single_τ`) |
| pending | 27 Praat sentence features for the text+audio row (H3, shared with Sawhney W3) |
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
| `[colab]` | KePt adaptive pre-training (BERT, 60 epochs, ~5 h GPU, needs the `kept_dataset` Drive pickles) — not run locally |

## Sawhney et al. (ACM MM 2020) — `ecvol reproduce sawhney` (harness port)

| status | item |
|---|---|
| reason-coded (faithful) | unshipped `stock_data.csv`, `AllRetPrices/`, `so-cal.json`, `dictionary.csv`; TF 2.1 / Keras 2.3.1 / tensorflow-addons 0.8.3 + delta v1-graph layers; `text_feature_extraction.py` returns after the first record; no multi-task loss in the released code |
| ported | labels (`generate_price_vol_data.py`: mean-subtracted log σ over the τ sessions after/before the call, from our EC price store); split (`month*30+day` key, 60/20/20); text BiLSTM(100) 2 epochs best-val; SVR(rbf) grid on the 30-session `Past_Volatility_τ` vector; (α, 1−α) ensemble |
| substituted | GloVe 6B-300d sentence means over their shipped `union_vocab.csv` (no Mittens retrofit — W2); regex + scikit-learn stop words instead of NLTK; Keras recurrent dropout → input dropout; audio branch absent (W3) |
| labelled | the authors tune (α, β) on the **test** set; both `tuned_on=test` and `tuned_on=val` rows are reported |
| published | 0.601 / 0.308 / 0.181 / 0.119 (second-hand: KeFVP Table 2 "Ensemble(Text+Audio)"; the MM'20 PDF is not held locally) |
| **result** | finance SVR 0.712 / 0.370 / 0.218 / 0.154 (persistence 1.491 / 0.563 / 0.345 / 0.205); text BiLSTM after their 2 epochs 2.14 / 1.58 / 1.38 / 1.40 (5 seeds) — untrained regime; ensemble = finance (α = 0 on val and on test); 547 EC calls (`result_table_6r_sawhney.csv`, run 20260909T055052Z) |

## DialogueGAT (Sang & Bao, Findings of EMNLP 2022) — `ecvol reproduce dialoguegat` (harness port)

| status | item |
|---|---|
| reason-coded (faithful) | `data/data_swd.pkl` unreleased; corpus = private SeekingAlpha re-scrape (~3,400 calls, 2015–2018, global named speakers); labels need CRSP; DGL has no wheels for torch 2.11 |
| ported | TextCNN (100 × kernels 3/4/5, max-pool, no ReLU) over frozen GloVe; utterance chain + utterance↔speaker graph; 5 GAT layers × 5 heads (head-mean, residual, dropout 0.1); context attention over speaker and utterance nodes; `v_past`; Adam 1e-5 / wd 1e-6 / batch 4 / ≤100 epochs / patience 5 / clip 15 / seed 1234; per-year chronological 70/10/20 |
| substituted | corpus FinCall 2019/2020/2021 (their 2015/2016/2017-18); speaker nodes role-typed (management/analyst/operator) instead of named persons — the paper's "random speaker embedding" ablation is the nearest published row; GloVe 6B instead of 840B; PyG `GATConv` instead of DGL; labels our trading-day `v_post/v_pre` |
| published (reference only) | τ=3/7/15: 2015 0.4530/0.3236/0.1898; 2016 0.4549/0.2884/0.1810; 2017-18 0.4090/0.2886/0.2036 |
