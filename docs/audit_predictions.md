# `ecvol audit predictions` — identity controls for anyone's predictions

A model that beats the volatility baselines on a split where test companies also appear in
training may be reading *which company is speaking* rather than what was said. Our own
models did (paper §6), and so did every text model of the five papers we reproduced
(Table 6R-C). This command applies the controls that can be computed from a prediction
file alone — no model, features or retraining — so a result can be audited from the one
artefact its authors already have.

```
ecvol audit predictions my_predictions.csv            # → data/audits/my_predictions/report.{csv,md}
ecvol audit predictions preds.parquet --out ./audit   # anywhere (no run manifest outside the data root)
```

## Input: one row per (call, horizon)

| column | meaning |
|---|---|
| `call_id` | any unique id |
| `ticker` | the company (identity is what the controls test, so this must be the real issuer) |
| `date` | the call's as-of date, `YYYY-MM-DD` |
| `split` | `train`, `val` or `test` |
| `horizon` | forecast horizon in sessions (3, 7, 15, 30, …) |
| `y_true` | the target (needed on every row) |
| `y_pred` | the model's forecast (read on `test` rows) |
| `y_persistence` | *optional* — the past-volatility forecast for the same window, for the persistence floor |

Train/val rows are what the ticker-only and train-mean floors are fit on; they must be the
rows the model was trained on. The target convention (log volatility, mean-subtracted or
not, calendar vs trading days) is yours — every metric is relative to the same `y_true`.

## What the report contains (per horizon)

| check | metrics | what it tells you |
|---|---|---|
| `split_integrity` | % of test calls whose ticker is in train/val; business days between the last train/val call and the first test call; train/val calls whose horizon window reaches the test period | whether the split tests generalisation to new companies and whether training targets overlap the test period |
| `floors` | test MSE of the model vs persistence (if supplied), the **ticker-only** model (per-ticker train mean; §7.3 control 1) and the train mean; R²_OOS against each | whether the model beats a forecast that knows only the company |
| `seen_vs_unseen` | MSE on test calls whose ticker was in training vs not | the price of a new company |
| `prediction_shuffle` | MSE after swapping each test prediction with another test call of the *same* company / of a *different* company (mean of 50 permutations, on tickers with ≥ 2 test calls) | if a same-company swap changes nothing, the predictions are a function of the company, not the call (the prediction-space form of the transcript shuffle, §7.3 control 2) |
| `identity_share` | fraction of test-prediction variance that is between-company; the same for the target | how much of what the model outputs is a company constant |
| `significance` | Diebold–Mariano p-values vs each floor; ticker-clustered bootstrap CI on the model − ticker-only squared-error difference | whether any margin over the floors survives clustering by company |

`report.md` ends with a **Reading** per horizon. Its rules are mechanical and printed
with the numbers they fire on: ≥ 50 % seen companies, gap shorter than the horizon,
same-company swap within ±5 % while another company's costs > 10 %, ticker-only beaten
only if R²_OOS > 0 *and* DM p < 0.05.

## Worked example: Same-Company-Same-Signal's TMLP (the authors' code on DEC)

`data/audits/demo/` holds two frames built from the T6R.4 runs — TMLP's own test
predictions pooled over the four 2022 rolling quarters (four models; pooling gives each
company up to four test calls so the within-company controls can run), training rows from
the 2022-Q1 mask — and their reports (manifested runs `20260923T…-audit-predictions`).

| | their split, held-out third (`anchor_heldout`) | `ticker_disjoint` (same test calls, their history removed) |
|---|---|---|
| test calls whose company is in training | 100 % | 0 % |
| model MSE, τ = 3 / 7 / 15 / 30 | 0.651 / 0.340 / 0.215 / 0.138 | 0.733 / 0.380 / 0.283 / 0.205 |
| ticker-only MSE | 0.618 / 0.256 / 0.167 / 0.092 | = train mean (no company seen) |
| same-company prediction swap, ΔMSE | −4 / −5 / −5 / −6 % | −1 / −1 / −1 / −3 % |
| other-company prediction swap, ΔMSE | +18 / +21 / +40 / +55 % | +1 / +6 / +6 / +5 % |
| company share of prediction variance | 0.68–0.81 | 0.71–0.74 |

Reading: on the authors' split the model never beats the ticker-only floor (DM p 0.009–0.53),
a same-company swap of its predictions costs nothing while another company's costs up to
55 %, and three quarters of its prediction variance is a company constant — it predicts the
company. Once the company's history is removed, both swaps become nearly free: what remains
varies little across calls and does not beat the train mean either.

## Limits

- Prediction-space only: it cannot swap *inputs* or retrain. If you can, run the transcript
  shuffle on the model itself (`ecvol reproduce html-faithful --shuffle` is the template).
- The within-company controls need companies with ≥ 2 test calls; on a single-quarter test
  set pool several periods, as in the example.
- Persistence must be supplied by you (the tool does not have your price data); without it
  the floors are ticker-only and train mean.
- A short gap between train and test is flagged against the horizon in business days
  (weekday approximation, no holiday calendar).
