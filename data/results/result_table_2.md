# Result Table 2 — Stage-2 content heads (frozen text features)

`*` = DM-significant vs Stage-1 ticker-FE GBDT (p<0.05; the §7.5 headline comparison).

## fincall — temporal split — level-v — R²_OOS vs persistence (test) (n=481)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 0.439* | 0.319 | 0.037 | -0.577 |
| MLP (text) | 0.147* | -0.241* | -1.101* | -2.633* |
| Ridge (past-vol) | 0.415* | -1.052 | -2.025 | -0.371* |
| MLP (past-vol) | 0.456* | -0.228 | -2.706 | -2.477 |
| Ridge (text+vol) | 0.448* | 0.328 | 0.045 | -1.091 |
| MLP (text+vol) | -2.092 | -7.189 | -18.384 | -26.901 |

## fincall — ticker_disjoint split — level-v — R²_OOS vs persistence (test) (n=484)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 0.416 | 0.194* | -0.047* | 0.112 |
| MLP (text) | -0.362* | -1.801* | -3.485* | -2.872* |
| Ridge (past-vol) | 0.432 | 0.306 | 0.191* | 0.224* |
| MLP (past-vol) | 0.482* | 0.365* | 0.243* | 0.212* |
| Ridge (text+vol) | 0.418 | 0.211 | -0.006 | 0.139 |
| MLP (text+vol) | -0.254* | -1.556* | -3.005* | -2.486* |

## fincall — temporal split — Δv — R²_OOS vs persistence (test) (n=481)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 0.028* | 0.030* | 0.015 | -0.144* |
| MLP (text) | -0.108* | -0.175* | -0.265* | -1.042 |
| Ridge (past-vol) | 0.415* | -1.053 | -2.858 | -0.667 |
| MLP (past-vol) | 0.390 | -1.017 | -2.249 | -0.760 |
| Ridge (text+vol) | -1.678 | -3.076 | -3.586 | -0.468 |
| MLP (text+vol) | -1.186 | -1.571 | -2.112 | -1.295 |

## fincall — ticker_disjoint split — Δv — R²_OOS vs persistence (test) (n=484)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | -0.044* | -0.026* | 0.008 | 0.171 |
| MLP (text) | -0.195* | -0.271* | -0.339* | -0.204* |
| Ridge (past-vol) | 0.451 | 0.306 | 0.191* | 0.224* |
| MLP (past-vol) | 0.492* | 0.363* | 0.230* | 0.249* |
| Ridge (text+vol) | 0.223* | 0.129* | 0.094 | 0.236 |
| MLP (text+vol) | 0.275* | 0.140* | -0.061* | 0.051 |

## maec — temporal split — level-v — R²_OOS vs persistence (test) (n=507)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 0.478* | 0.225* | -0.033* | -0.322* |
| MLP (text) | -0.748* | -1.753* | -3.272* | -6.016* |
| Ridge (past-vol) | 0.437 | 0.222* | 0.168 | 0.208 |
| MLP (past-vol) | 0.472* | 0.375 | 0.301* | 0.247* |
| Ridge (text+vol) | 0.482* | 0.279 | 0.132 | -0.013* |
| MLP (text+vol) | -0.405* | -1.460* | -2.977* | -5.058* |

## maec — ticker_disjoint split — level-v — R²_OOS vs persistence (test) (n=514)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 0.466* | 0.277 | 0.117* | -0.036* |
| MLP (text) | -0.115* | -0.956* | -1.583* | -2.635* |
| Ridge (past-vol) | 0.455* | 0.329 | 0.261 | 0.262 |
| MLP (past-vol) | 0.482* | 0.369* | 0.281 | 0.250 |
| Ridge (text+vol) | 0.451 | 0.299 | 0.186 | 0.090* |
| MLP (text+vol) | -0.077* | -0.884* | -1.418* | -2.992* |

## maec — temporal split — Δv — R²_OOS vs persistence (test) (n=507)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 0.017* | 0.013* | 0.047* | -0.074* |
| MLP (text) | -0.301* | -0.164* | -0.338* | -0.511* |
| Ridge (past-vol) | 0.455 | 0.313 | 0.244 | 0.212 |
| MLP (past-vol) | 0.497* | 0.395* | 0.323* | 0.261* |
| Ridge (text+vol) | -0.005* | -0.142* | 0.211 | 0.138 |
| MLP (text+vol) | 0.259* | 0.242* | 0.068* | -0.080* |

## maec — ticker_disjoint split — Δv — R²_OOS vs persistence (test) (n=514)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 0.049* | 0.056* | 0.048* | 0.047* |
| MLP (text) | -0.051* | -0.108* | -0.165* | -0.199* |
| Ridge (past-vol) | 0.455* | 0.329 | 0.261 | 0.262 |
| MLP (past-vol) | 0.489* | 0.389* | 0.328* | 0.294* |
| Ridge (text+vol) | 0.330 | 0.142* | 0.204 | 0.163 |
| MLP (text+vol) | 0.366 | 0.244 | 0.120* | -0.111* |

## fincall — temporal split — level-v — MSE (test) (n=481)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 0.659* | 0.285 | 0.200 | 0.176 |
| MLP (text) | 1.001* | 0.519* | 0.437* | 0.405* |
| Ridge (past-vol) | 0.687* | 0.859 | 0.629 | 0.153* |
| MLP (past-vol) | 0.639* | 0.514 | 0.771 | 0.387 |
| Ridge (text+vol) | 0.648* | 0.281 | 0.199 | 0.233 |
| MLP (text+vol) | 3.631 | 3.428 | 4.032 | 3.109 |

## fincall — ticker_disjoint split — level-v — MSE (test) (n=484)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 0.667 | 0.311* | 0.224* | 0.207 |
| MLP (text) | 1.556* | 1.080* | 0.958* | 0.903* |
| Ridge (past-vol) | 0.649 | 0.268 | 0.173* | 0.181* |
| MLP (past-vol) | 0.592* | 0.245* | 0.162* | 0.184* |
| Ridge (text+vol) | 0.665 | 0.304 | 0.215 | 0.201 |
| MLP (text+vol) | 1.432* | 0.986* | 0.856* | 0.813* |

## fincall — temporal split — Δv — MSE (test) (n=481)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 1.141* | 0.406* | 0.205 | 0.127* |
| MLP (text) | 1.302* | 0.492* | 0.263* | 0.228 |
| Ridge (past-vol) | 0.687* | 0.860 | 0.803 | 0.186 |
| MLP (past-vol) | 0.717 | 0.844 | 0.676 | 0.196 |
| Ridge (text+vol) | 3.145 | 1.706 | 0.954 | 0.164 |
| MLP (text+vol) | 2.567 | 1.076 | 0.647 | 0.256 |

## fincall — ticker_disjoint split — Δv — MSE (test) (n=484)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 1.193* | 0.396* | 0.212 | 0.193 |
| MLP (text) | 1.365* | 0.490* | 0.286* | 0.281* |
| Ridge (past-vol) | 0.627 | 0.268 | 0.173* | 0.181* |
| MLP (past-vol) | 0.580* | 0.246* | 0.164* | 0.175* |
| Ridge (text+vol) | 0.888* | 0.336* | 0.194 | 0.178 |
| MLP (text+vol) | 0.828* | 0.332* | 0.227* | 0.221 |

## maec — temporal split — level-v — MSE (test) (n=507)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 0.722* | 0.443* | 0.329* | 0.250* |
| MLP (text) | 2.416* | 1.573* | 1.359* | 1.325* |
| Ridge (past-vol) | 0.778 | 0.445* | 0.265 | 0.150 |
| MLP (past-vol) | 0.730* | 0.357 | 0.222* | 0.142* |
| Ridge (text+vol) | 0.716* | 0.412 | 0.276 | 0.191* |
| MLP (text+vol) | 1.943* | 1.405* | 1.266* | 1.144* |

## maec — ticker_disjoint split — level-v — MSE (test) (n=514)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 0.677* | 0.356 | 0.266* | 0.190* |
| MLP (text) | 1.413* | 0.963* | 0.779* | 0.668* |
| Ridge (past-vol) | 0.691* | 0.330 | 0.223 | 0.136 |
| MLP (past-vol) | 0.656* | 0.311* | 0.217 | 0.138 |
| Ridge (text+vol) | 0.695 | 0.345 | 0.246 | 0.167* |
| MLP (text+vol) | 1.365* | 0.927* | 0.729* | 0.734* |

## maec — temporal split — Δv — MSE (test) (n=507)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 1.358* | 0.564* | 0.303* | 0.203* |
| MLP (text) | 1.798* | 0.665* | 0.426* | 0.285* |
| Ridge (past-vol) | 0.753 | 0.392 | 0.241 | 0.149 |
| MLP (past-vol) | 0.695* | 0.346* | 0.215* | 0.140* |
| Ridge (text+vol) | 1.390* | 0.653* | 0.251 | 0.163 |
| MLP (text+vol) | 1.025* | 0.433* | 0.297* | 0.204* |

## maec — ticker_disjoint split — Δv — MSE (test) (n=514)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| Ridge (text) | 1.206* | 0.464* | 0.287* | 0.175* |
| MLP (text) | 1.332* | 0.546* | 0.351* | 0.220* |
| Ridge (past-vol) | 0.691* | 0.330 | 0.223 | 0.136 |
| MLP (past-vol) | 0.648* | 0.301* | 0.203* | 0.130* |
| Ridge (text+vol) | 0.850 | 0.422* | 0.240 | 0.154 |
| MLP (text+vol) | 0.804 | 0.372 | 0.265* | 0.204* |

