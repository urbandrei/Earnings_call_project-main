# Prediction audit — scss_tmlp_ticker_disjoint_2022

Verdict rules are mechanical and stated inline; read every line against its n.

## Split
| horizon | 3 | 7 | 15 | 30 |
|---|---|---|---|---|
| test calls whose ticker is in train/val (%) | 0.0 | 0.0 | 0.0 | 0.0 |
| gap, last train/val → first test call (business days) | 19 | 19 | 19 | 19 |
| train/val target windows reaching the test period | 0 | 0 | 0 | 5 |

## Floors (test MSE)
| horizon | 3 | 7 | 15 | 30 |
|---|---|---|---|---|
| model | 0.733 | 0.380 | 0.283 | 0.205 |
| persistence (supplied) | 1.581 | 0.427 | 0.182 | 0.106 |
|   R²_OOS vs persistence | 0.536 | 0.110 | -0.552 | -0.934 |
|   DM p vs persistence | 0.000 | 0.446 | 0.013 | 0.003 |
| ticker-only (per-ticker train mean) | 0.707 | 0.384 | 0.284 | 0.194 |
|   R²_OOS vs ticker_only | -0.037 | 0.009 | 0.005 | -0.055 |
|   DM p vs ticker_only | 0.588 | 0.904 | 0.946 | 0.475 |
| train mean | 0.707 | 0.384 | 0.284 | 0.194 |
|   R²_OOS vs train_mean | -0.037 | 0.009 | 0.005 | -0.055 |
|   DM p vs train_mean | 0.588 | 0.904 | 0.946 | 0.475 |

## Identity
| horizon | 3 | 7 | 15 | 30 |
|---|---|---|---|---|
| MSE, seen tickers | — | — | — | — |
| MSE, unseen tickers | 0.733 | 0.380 | 0.283 | 0.205 |
| within-ticker prediction swap, ΔMSE (%) | -1.3 | -1.2 | -1.3 | -2.6 |
| other-ticker prediction swap, ΔMSE (%) | +0.7 | +6.1 | +5.7 | +4.9 |
| ticker share of prediction variance | 0.71 | 0.73 | 0.74 | 0.74 |
| ticker share of target variance | 0.44 | 0.58 | 0.63 | 0.73 |

## Reading
- **τ=3:** neither swap costs much (same company -1.3%, other +0.7%): the differences between predictions carry little that matches the target; does not significantly beat the ticker-only model (R²_OOS -0.037, DM p 0.588); R²_OOS vs persistence +0.536 (DM p 0.000).
- **τ=7:** neither swap costs much (same company -1.2%, other +6.1%): the differences between predictions carry little that matches the target; does not significantly beat the ticker-only model (R²_OOS +0.009, DM p 0.904); R²_OOS vs persistence +0.110 (DM p 0.446).
- **τ=15:** neither swap costs much (same company -1.3%, other +5.7%): the differences between predictions carry little that matches the target; does not significantly beat the ticker-only model (R²_OOS +0.005, DM p 0.946); R²_OOS vs persistence -0.552 (DM p 0.013).
- **τ=30:** gap of 19 business days < horizon 30: training targets overlap the test period; neither swap costs much (same company -2.6%, other +4.9%): the differences between predictions carry little that matches the target; does not significantly beat the ticker-only model (R²_OOS -0.055, DM p 0.475); R²_OOS vs persistence -0.934 (DM p 0.003).
