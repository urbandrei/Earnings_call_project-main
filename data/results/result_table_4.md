# Result Table 4 — consolidated ablation grid (canonical model per stage)

`*` = Holm-significant vs Stage-1 (Δv family, across horizons per stage×split).

## fincall — temporal split — level-v — R²_OOS vs persistence (test) (n=483)

| Stage | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Persistence | 0.000 | 0.000 | 0.000 | 0.000 |
| Train mean | 0.361 | 0.141 | -0.367 | -1.048 |
| HAR-RV | 0.444 | 0.372 | 0.138 | -0.285 |
| GBDT (ticker FE) | 0.353 | 0.244 | -0.097 | -0.881 |
| Text (BGE+vol) | 0.435 | 0.339 | 0.034 | -0.454 |
| Audio (WavLM+eG+vol) | 0.367 | 0.259 | -0.119 | -0.763 |
| Fusion (stack) | 0.442 | 0.390 | 0.164 | -0.227 |

## fincall — ticker_disjoint split — level-v — R²_OOS vs persistence (test) (n=484)

| Stage | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Persistence | 0.000 | 0.000 | 0.000 | 0.000 |
| Train mean | 0.535 | 0.353 | 0.072 | 0.097 |
| HAR-RV | 0.570 | 0.491 | 0.349 | 0.345 |
| GBDT (ticker FE) | 0.501 | 0.440 | 0.292 | 0.242 |
| Text (BGE+vol) | 0.552 | 0.464 | 0.293 | 0.278 |
| Audio (WavLM+eG+vol) | 0.522 | 0.389 | 0.230 | 0.239 |
| Fusion (stack) | 0.556 | 0.466 | 0.319 | 0.365 |

## fincall — temporal split — Δv — R²_OOS vs persistence (test) (n=483)

| Stage | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Persistence | 0.000 | 0.000 | 0.000 | 0.000 |
| Train mean | 0.165 | 0.158 | 0.060 | 0.088 |
| HAR-RV | 0.444 | 0.372 | 0.138 | -0.285 |
| GBDT (ticker FE) | 0.340 | 0.249 | -0.126 | -0.948 |
| Text (BGE+vol) | 0.333 | 0.285 | 0.080* | -0.077* |
| Audio (WavLM+eG+vol) | 0.353 | 0.268 | 0.033* | 0.050* |
| Fusion (stack) | 0.406* | 0.347* | 0.066* | 0.100* |

## fincall — ticker_disjoint split — Δv — R²_OOS vs persistence (test) (n=484)

| Stage | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Persistence | 0.000 | 0.000 | 0.000 | 0.000 |
| Train mean | 0.200 | 0.201 | 0.132 | 0.035 |
| HAR-RV | 0.570 | 0.491 | 0.349 | 0.345 |
| GBDT (ticker FE) | 0.494 | 0.440 | 0.292 | 0.237 |
| Text (BGE+vol) | 0.399* | 0.375 | 0.256 | 0.301 |
| Audio (WavLM+eG+vol) | 0.442 | 0.364 | 0.241 | 0.217 |
| Fusion (stack) | 0.531 | 0.465 | 0.340 | 0.396* |

## maec — temporal split — level-v — R²_OOS vs persistence (test) (n=529)

| Stage | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Persistence | 0.000 | 0.000 | 0.000 | 0.000 |
| Train mean | 0.557 | 0.342 | 0.024 | -0.264 |
| HAR-RV | 0.603 | 0.486 | 0.321 | 0.221 |
| GBDT (ticker FE) | 0.563 | 0.429 | 0.240 | 0.113 |
| Text (BGE+vol) | 0.516 | 0.348 | 0.165 | 0.010 |
| Audio (WavLM+eG+vol) | — | — | — | — |
| Fusion (stack) | — | — | — | — |

## maec — ticker_disjoint split — level-v — R²_OOS vs persistence (test) (n=532)

| Stage | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Persistence | 0.000 | 0.000 | 0.000 | 0.000 |
| Train mean | 0.511 | 0.267 | 0.004 | -0.277 |
| HAR-RV | 0.574 | 0.447 | 0.332 | 0.183 |
| GBDT (ticker FE) | 0.483 | 0.339 | 0.221 | 0.091 |
| Text (BGE+vol) | 0.563 | 0.390 | 0.274 | 0.133 |
| Audio (WavLM+eG+vol) | — | — | — | — |
| Fusion (stack) | — | — | — | — |

## maec — temporal split — Δv — R²_OOS vs persistence (test) (n=529)

| Stage | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Persistence | 0.000 | 0.000 | 0.000 | 0.000 |
| Train mean | 0.254 | 0.251 | 0.187 | 0.090 |
| HAR-RV | 0.603 | 0.486 | 0.321 | 0.221 |
| GBDT (ticker FE) | 0.562 | 0.420 | 0.254 | 0.150 |
| Text (BGE+vol) | 0.438* | -0.015* | 0.233 | 0.152 |
| Audio (WavLM+eG+vol) | — | — | — | — |
| Fusion (stack) | — | — | — | — |

## maec — ticker_disjoint split — Δv — R²_OOS vs persistence (test) (n=532)

| Stage | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Persistence | 0.000 | 0.000 | 0.000 | 0.000 |
| Train mean | 0.265 | 0.254 | 0.210 | 0.105 |
| HAR-RV | 0.574 | 0.447 | 0.332 | 0.183 |
| GBDT (ticker FE) | 0.477 | 0.361 | 0.250 | 0.095 |
| Text (BGE+vol) | 0.453 | 0.340 | 0.235 | 0.093 |
| Audio (WavLM+eG+vol) | — | — | — | — |
| Fusion (stack) | — | — | — | — |

## fincall — temporal split — level-v — MSE (test) (n=483)

| Stage | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Persistence | 1.270 | 0.448 | 0.211 | 0.115 |
| Train mean | 0.811 | 0.384 | 0.288 | 0.235 |
| HAR-RV | 0.706 | 0.281 | 0.182 | 0.148 |
| GBDT (ticker FE) | 0.821 | 0.338 | 0.231 | 0.216 |
| Text (BGE+vol) | 0.718 | 0.296 | 0.204 | 0.167 |
| Audio (WavLM+eG+vol) | 0.804 | 0.332 | 0.236 | 0.203 |
| Fusion (stack) | 0.708 | 0.273 | 0.176 | 0.141 |

## fincall — ticker_disjoint split — level-v — MSE (test) (n=484)

| Stage | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Persistence | 1.368 | 0.477 | 0.248 | 0.262 |
| Train mean | 0.637 | 0.309 | 0.230 | 0.236 |
| HAR-RV | 0.589 | 0.243 | 0.162 | 0.171 |
| GBDT (ticker FE) | 0.683 | 0.267 | 0.176 | 0.198 |
| Text (BGE+vol) | 0.613 | 0.256 | 0.176 | 0.189 |
| Audio (WavLM+eG+vol) | 0.653 | 0.291 | 0.191 | 0.199 |
| Fusion (stack) | 0.607 | 0.255 | 0.169 | 0.166 |

## fincall — temporal split — Δv — MSE (test) (n=483)

| Stage | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Persistence | 1.270 | 0.448 | 0.211 | 0.115 |
| Train mean | 1.060 | 0.377 | 0.198 | 0.105 |
| HAR-RV | 0.706 | 0.281 | 0.182 | 0.148 |
| GBDT (ticker FE) | 0.838 | 0.336 | 0.237 | 0.224 |
| Text (BGE+vol) | 0.847 | 0.320 | 0.194* | 0.124* |
| Audio (WavLM+eG+vol) | 0.822 | 0.327 | 0.204* | 0.109* |
| Fusion (stack) | 0.755* | 0.292* | 0.197* | 0.103* |

## fincall — ticker_disjoint split — Δv — MSE (test) (n=484)

| Stage | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Persistence | 1.368 | 0.477 | 0.248 | 0.262 |
| Train mean | 1.095 | 0.381 | 0.216 | 0.253 |
| HAR-RV | 0.589 | 0.243 | 0.162 | 0.171 |
| GBDT (ticker FE) | 0.692 | 0.267 | 0.176 | 0.200 |
| Text (BGE+vol) | 0.822* | 0.298 | 0.185 | 0.183 |
| Audio (WavLM+eG+vol) | 0.764 | 0.304 | 0.188 | 0.205 |
| Fusion (stack) | 0.642 | 0.255 | 0.164 | 0.158* |

## maec — temporal split — level-v — MSE (test) (n=529)

| Stage | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Persistence | 1.774 | 0.712 | 0.356 | 0.201 |
| Train mean | 0.787 | 0.469 | 0.347 | 0.253 |
| HAR-RV | 0.705 | 0.366 | 0.242 | 0.156 |
| GBDT (ticker FE) | 0.775 | 0.406 | 0.270 | 0.178 |
| Text (BGE+vol) | 0.859 | 0.464 | 0.297 | 0.198 |
| Audio (WavLM+eG+vol) | — | — | — | — |
| Fusion (stack) | — | — | — | — |

## maec — ticker_disjoint split — level-v — MSE (test) (n=532)

| Stage | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Persistence | 1.709 | 0.632 | 0.326 | 0.161 |
| Train mean | 0.836 | 0.463 | 0.324 | 0.206 |
| HAR-RV | 0.730 | 0.350 | 0.218 | 0.132 |
| GBDT (ticker FE) | 0.884 | 0.417 | 0.254 | 0.147 |
| Text (BGE+vol) | 0.746 | 0.385 | 0.236 | 0.140 |
| Audio (WavLM+eG+vol) | — | — | — | — |
| Fusion (stack) | — | — | — | — |

## maec — temporal split — Δv — MSE (test) (n=529)

| Stage | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Persistence | 1.774 | 0.712 | 0.356 | 0.201 |
| Train mean | 1.323 | 0.533 | 0.289 | 0.183 |
| HAR-RV | 0.705 | 0.366 | 0.242 | 0.156 |
| GBDT (ticker FE) | 0.778 | 0.413 | 0.265 | 0.171 |
| Text (BGE+vol) | 0.997* | 0.723* | 0.273 | 0.170 |
| Audio (WavLM+eG+vol) | — | — | — | — |
| Fusion (stack) | — | — | — | — |

## maec — ticker_disjoint split — Δv — MSE (test) (n=532)

| Stage | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Persistence | 1.709 | 0.632 | 0.326 | 0.161 |
| Train mean | 1.256 | 0.471 | 0.257 | 0.144 |
| HAR-RV | 0.730 | 0.350 | 0.218 | 0.132 |
| GBDT (ticker FE) | 0.894 | 0.404 | 0.244 | 0.146 |
| Text (BGE+vol) | 0.934 | 0.417 | 0.249 | 0.146 |
| Audio (WavLM+eG+vol) | — | — | — | — |
| Fusion (stack) | — | — | — | — |

