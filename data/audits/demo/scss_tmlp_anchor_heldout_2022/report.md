# Prediction audit — scss_tmlp_anchor_heldout_2022

Verdict rules are mechanical and stated inline; read every line against its n.

## Split
| horizon | 3 | 7 | 15 | 30 |
|---|---|---|---|---|
| test calls whose ticker is in train/val (%) | 100.0 | 100.0 | 100.0 | 100.0 |
| gap, last train/val → first test call (business days) | 19 | 19 | 19 | 19 |
| train/val target windows reaching the test period | 0 | 0 | 0 | 5 |

## Floors (test MSE)
| horizon | 3 | 7 | 15 | 30 |
|---|---|---|---|---|
| model | 0.651 | 0.340 | 0.215 | 0.138 |
| persistence (supplied) | 1.581 | 0.427 | 0.182 | 0.106 |
|   R²_OOS vs persistence | 0.588 | 0.205 | -0.181 | -0.303 |
|   DM p vs persistence | 0.000 | 0.106 | 0.242 | 0.114 |
| ticker-only (per-ticker train mean) | 0.618 | 0.256 | 0.167 | 0.092 |
|   R²_OOS vs ticker_only | -0.054 | -0.326 | -0.289 | -0.493 |
|   DM p vs ticker_only | 0.533 | 0.009 | 0.019 | 0.003 |
| train mean | 0.695 | 0.368 | 0.265 | 0.176 |
|   R²_OOS vs train_mean | 0.063 | 0.077 | 0.187 | 0.215 |
|   DM p vs train_mean | 0.325 | 0.294 | 0.031 | 0.038 |

## Identity
| horizon | 3 | 7 | 15 | 30 |
|---|---|---|---|---|
| MSE, seen tickers | 0.651 | 0.340 | 0.215 | 0.138 |
| MSE, unseen tickers | — | — | — | — |
| within-ticker prediction swap, ΔMSE (%) | -3.8 | -5.1 | -4.5 | -5.5 |
| other-ticker prediction swap, ΔMSE (%) | +17.7 | +20.9 | +40.2 | +55.3 |
| ticker share of prediction variance | 0.68 | 0.77 | 0.81 | 0.80 |
| ticker share of target variance | 0.44 | 0.58 | 0.63 | 0.73 |

## Reading
- **τ=3:** 100% of test calls come from companies in training — the split does not test generalisation to new companies; predictions are a function of the company: a same-company swap changes MSE by -3.8% while another company's costs +17.7% (the call-specific part of the prediction hurts); does not significantly beat the ticker-only model (R²_OOS -0.054, DM p 0.533); R²_OOS vs persistence +0.588 (DM p 0.000).
- **τ=7:** 100% of test calls come from companies in training — the split does not test generalisation to new companies; predictions are a function of the company: a same-company swap changes MSE by -5.1% while another company's costs +20.9% (the call-specific part of the prediction hurts); does not significantly beat the ticker-only model (R²_OOS -0.326, DM p 0.009); R²_OOS vs persistence +0.205 (DM p 0.106).
- **τ=15:** 100% of test calls come from companies in training — the split does not test generalisation to new companies; predictions are a function of the company: a same-company swap changes MSE by -4.5% while another company's costs +40.2% (the call-specific part of the prediction hurts); does not significantly beat the ticker-only model (R²_OOS -0.289, DM p 0.019); R²_OOS vs persistence -0.181 (DM p 0.242).
- **τ=30:** 100% of test calls come from companies in training — the split does not test generalisation to new companies; gap of 19 business days < horizon 30: training targets overlap the test period; predictions are a function of the company: a same-company swap changes MSE by -5.5% while another company's costs +55.3% (the call-specific part of the prediction hurts); does not significantly beat the ticker-only model (R²_OOS -0.493, DM p 0.003); R²_OOS vs persistence -0.303 (DM p 0.114).
