# Result Table 2 — Stage-2 content heads (frozen text features)

`*` = DM-significant vs Stage-1 ticker-FE GBDT (p<0.05; the §7.5 headline comparison).

## fincall — temporal split — level-v — R²_OOS vs persistence (test) (n=483)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 0.430* | 0.322* | -0.007 | -0.539* |
| MLP (text) | 0.153* | -0.235* | -1.156* | -2.782* |
| Ridge (past-vol) | 0.407* | 0.314* | 0.079* | -0.382* |
| MLP (past-vol) | 0.439* | 0.352* | 0.116* | -0.459* |
| Ridge (text+vol) | 0.435* | 0.339* | 0.034* | -0.454* |
| MLP (text+vol) | 0.219* | -0.115* | -0.918* | -2.504* |

## fincall — ticker_disjoint split — level-v — R²_OOS vs persistence (test) (n=484)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 0.549* | 0.449 | 0.254 | 0.199 |
| MLP (text) | 0.177* | -0.484* | -1.491* | -1.354* |
| Ridge (past-vol) | 0.543 | 0.467 | 0.341 | 0.284 |
| MLP (past-vol) | 0.571* | 0.512* | 0.365* | 0.327 |
| Ridge (text+vol) | 0.552* | 0.464 | 0.293 | 0.278 |
| MLP (text+vol) | 0.209* | -0.348* | -1.277* | -1.166* |

## fincall — temporal split — Δv — R²_OOS vs persistence (test) (n=483)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 0.182* | 0.202 | 0.044* | -0.133* |
| MLP (text) | 0.013* | 0.069* | -0.253 | -1.202 |
| Ridge (past-vol) | 0.412* | 0.314* | 0.094* | 0.113* |
| MLP (past-vol) | 0.441* | 0.330* | 0.086* | -0.401* |
| Ridge (text+vol) | 0.333 | 0.285 | 0.080* | -0.077* |
| MLP (text+vol) | 0.275 | 0.222 | -0.108 | -0.855 |

## fincall — ticker_disjoint split — Δv — R²_OOS vs persistence (test) (n=484)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 0.195* | 0.217* | 0.195* | 0.169 |
| MLP (text) | 0.046* | -0.024* | -0.108* | -0.061* |
| Ridge (past-vol) | 0.552* | 0.469 | 0.345 | 0.318* |
| MLP (past-vol) | 0.577* | 0.508* | 0.381* | 0.356* |
| Ridge (text+vol) | 0.399* | 0.375 | 0.256 | 0.301 |
| MLP (text+vol) | 0.423* | 0.352* | 0.165* | 0.224 |

## maec — temporal split — level-v — R²_OOS vs persistence (test) (n=529)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 0.515 | 0.304* | 0.002* | -0.303* |
| MLP (text) | -0.174* | -1.142* | -2.602* | -5.867* |
| Ridge (past-vol) | 0.575 | 0.441 | 0.322* | 0.242* |
| MLP (past-vol) | 0.605* | 0.470 | 0.324* | 0.240* |
| Ridge (text+vol) | 0.516 | 0.348* | 0.165 | 0.010* |
| MLP (text+vol) | -0.116* | -1.036* | -1.853* | -4.494* |

## maec — ticker_disjoint split — level-v — R²_OOS vs persistence (test) (n=532)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 0.556* | 0.355 | 0.116 | -0.115* |
| MLP (text) | 0.233* | -0.523* | -1.583* | -3.321* |
| Ridge (past-vol) | 0.530 | 0.364 | 0.230 | 0.159 |
| MLP (past-vol) | 0.537 | 0.389 | 0.239 | 0.184 |
| Ridge (text+vol) | 0.563* | 0.390 | 0.274 | 0.133 |
| MLP (text+vol) | 0.103* | -0.670* | -1.556* | -3.281* |

## maec — temporal split — Δv — R²_OOS vs persistence (test) (n=529)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 0.247* | 0.148* | 0.191 | 0.098 |
| MLP (text) | 0.110* | 0.003* | -0.287* | -0.288* |
| Ridge (past-vol) | 0.581 | 0.444 | 0.323* | 0.242* |
| MLP (past-vol) | 0.615* | 0.494* | 0.349* | 0.258* |
| Ridge (text+vol) | 0.438* | -0.015* | 0.233 | 0.152 |
| MLP (text+vol) | 0.407* | 0.291* | 0.109* | -0.174* |

## maec — ticker_disjoint split — Δv — R²_OOS vs persistence (test) (n=532)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 0.276* | 0.267* | 0.221 | 0.113 |
| MLP (text) | 0.245* | 0.177* | 0.070* | -0.112* |
| Ridge (past-vol) | 0.530 | 0.364 | 0.230 | 0.159 |
| MLP (past-vol) | 0.559* | 0.436* | 0.318* | 0.182 |
| Ridge (text+vol) | 0.453 | 0.340 | 0.235 | 0.093 |
| MLP (text+vol) | 0.457 | 0.322 | 0.168 | -0.070* |

## fincall — temporal split — level-v — MSE (test) (n=483)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 0.724* | 0.304* | 0.212 | 0.177* |
| MLP (text) | 1.075* | 0.553* | 0.455* | 0.434* |
| Ridge (past-vol) | 0.752* | 0.307* | 0.194* | 0.159* |
| MLP (past-vol) | 0.713* | 0.290* | 0.186* | 0.168* |
| Ridge (text+vol) | 0.718* | 0.296* | 0.204* | 0.167* |
| MLP (text+vol) | 0.992* | 0.499* | 0.404* | 0.402* |

## fincall — ticker_disjoint split — level-v — MSE (test) (n=484)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 0.617* | 0.263 | 0.185 | 0.210 |
| MLP (text) | 1.125* | 0.708* | 0.619* | 0.616* |
| Ridge (past-vol) | 0.626 | 0.254 | 0.163 | 0.187 |
| MLP (past-vol) | 0.587* | 0.233* | 0.158* | 0.176 |
| Ridge (text+vol) | 0.613* | 0.256 | 0.176 | 0.189 |
| MLP (text+vol) | 1.082* | 0.643* | 0.565* | 0.567* |

## fincall — temporal split — Δv — MSE (test) (n=483)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 1.039* | 0.357 | 0.202* | 0.130* |
| MLP (text) | 1.253* | 0.417* | 0.264 | 0.253 |
| Ridge (past-vol) | 0.746* | 0.307* | 0.191* | 0.102* |
| MLP (past-vol) | 0.710* | 0.300* | 0.193* | 0.161* |
| Ridge (text+vol) | 0.847 | 0.320 | 0.194* | 0.124* |
| MLP (text+vol) | 0.921 | 0.348 | 0.234 | 0.213 |

## fincall — ticker_disjoint split — Δv — MSE (test) (n=484)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 1.101* | 0.374* | 0.200* | 0.217 |
| MLP (text) | 1.306* | 0.489* | 0.275* | 0.278* |
| Ridge (past-vol) | 0.613* | 0.254 | 0.163 | 0.178* |
| MLP (past-vol) | 0.578* | 0.235* | 0.154* | 0.168* |
| Ridge (text+vol) | 0.822* | 0.298 | 0.185 | 0.183 |
| MLP (text+vol) | 0.790* | 0.309* | 0.207* | 0.203 |

## maec — temporal split — level-v — MSE (test) (n=529)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 0.861 | 0.495* | 0.355* | 0.261* |
| MLP (text) | 2.082* | 1.525* | 1.281* | 1.377* |
| Ridge (past-vol) | 0.755 | 0.398 | 0.241* | 0.152* |
| MLP (past-vol) | 0.701* | 0.377 | 0.240* | 0.152* |
| Ridge (text+vol) | 0.859 | 0.464* | 0.297 | 0.198* |
| MLP (text+vol) | 1.981* | 1.450* | 1.015* | 1.102* |

## maec — ticker_disjoint split — level-v — MSE (test) (n=532)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 0.758* | 0.408 | 0.288 | 0.180* |
| MLP (text) | 1.310* | 0.962* | 0.841* | 0.697* |
| Ridge (past-vol) | 0.802 | 0.402 | 0.251 | 0.136 |
| MLP (past-vol) | 0.791 | 0.386 | 0.248 | 0.132 |
| Ridge (text+vol) | 0.746* | 0.385 | 0.236 | 0.140 |
| MLP (text+vol) | 1.532* | 1.055* | 0.833* | 0.690* |

## maec — temporal split — Δv — MSE (test) (n=529)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 1.336* | 0.607* | 0.288 | 0.181 |
| MLP (text) | 1.578* | 0.710* | 0.458* | 0.258* |
| Ridge (past-vol) | 0.744 | 0.396 | 0.241* | 0.152* |
| MLP (past-vol) | 0.684* | 0.360* | 0.231* | 0.149* |
| Ridge (text+vol) | 0.997* | 0.723* | 0.273 | 0.170 |
| MLP (text+vol) | 1.052* | 0.505* | 0.317* | 0.235* |

## maec — ticker_disjoint split — Δv — MSE (test) (n=532)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 1.237* | 0.463* | 0.254 | 0.143 |
| MLP (text) | 1.291* | 0.520* | 0.303* | 0.179* |
| Ridge (past-vol) | 0.802 | 0.402 | 0.251 | 0.136 |
| MLP (past-vol) | 0.753* | 0.357* | 0.222* | 0.132 |
| Ridge (text+vol) | 0.934 | 0.417 | 0.249 | 0.146 |
| MLP (text+vol) | 0.929 | 0.428 | 0.271 | 0.172* |

