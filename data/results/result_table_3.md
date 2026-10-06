# Result Table 3 — Stage-3 audio heads (frozen audio features, FinCall)

`*` = DM-significant vs Stage-1 ticker-FE GBDT (p<0.05; the RQ1-audio floor).

## fincall — temporal split — level-v — R²_OOS vs persistence (test) (n=483)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| ridge eGeMAPS | 0.361 | 0.144* | -0.370* | -1.159* |
| mlp eGeMAPS | -10.085 | -20.164 | -38.738 | -61.570 |
| ridge eGeMAPS +vol | 0.374 | 0.260 | -0.067 | -0.677 |
| mlp eGeMAPS +vol | -7.831 | -17.515 | -37.949 | -50.038 |
| ridge WavLM | 0.357 | 0.084* | -0.507* | -1.106* |
| mlp WavLM | -0.072* | -0.808* | -2.505* | -5.253* |
| ridge WavLM +vol | 0.378 | 0.245 | -0.094 | -0.665* |
| mlp WavLM +vol | 0.016* | -0.582* | -1.882* | -4.053* |
| ridge emotion2vec+ | 0.354 | 0.152* | -0.320* | -0.967 |
| mlp emotion2vec+ | -0.270* | -1.582* | -3.692* | -6.470* |
| ridge emotion2vec+ +vol | 0.396 | 0.305* | 0.070* | -0.398* |
| mlp emotion2vec+ +vol | -0.313* | -1.580* | -3.559* | -6.632* |
| ridge WavLM+eGeMAPS | 0.354 | 0.139* | -0.411* | -1.114* |
| mlp WavLM+eGeMAPS | -1.847 | -5.053 | -9.675 | -14.344 |
| ridge WavLM+eGeMAPS +vol | 0.367 | 0.259 | -0.119 | -0.763 |
| mlp WavLM+eGeMAPS +vol | -2.005 | -5.365 | -10.910 | -16.691 |
| ridge WavLM+text +vol | 0.427* | 0.332* | -0.007 | -0.527* |
| mlp WavLM+text +vol | 0.213* | -0.050* | -0.823* | -2.165* |

## fincall — ticker_disjoint split — level-v — R²_OOS vs persistence (test) (n=484)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| ridge eGeMAPS | 0.535 | 0.320* | 0.073* | 0.060* |
| mlp eGeMAPS | -7.712 | -26.589 | -52.220 | -53.918 |
| ridge eGeMAPS +vol | 0.544 | 0.426 | 0.310 | 0.266 |
| mlp eGeMAPS +vol | -7.149 | -21.128 | -47.690 | -46.809 |
| ridge WavLM | 0.514 | 0.320* | -0.022* | -0.015* |
| mlp WavLM | -0.020* | -0.939* | -2.346* | -2.290* |
| ridge WavLM +vol | 0.521 | 0.401 | 0.243 | 0.247 |
| mlp WavLM +vol | -0.074* | -1.072* | -2.573* | -2.372* |
| ridge emotion2vec+ | 0.533 | 0.357* | 0.085* | 0.116* |
| mlp emotion2vec+ | -1.783 | -4.890 | -10.950 | -9.695 |
| ridge emotion2vec+ +vol | 0.540 | 0.463 | 0.335 | 0.283 |
| mlp emotion2vec+ +vol | -1.860 | -5.254 | -11.588 | -11.116 |
| ridge WavLM+eGeMAPS | 0.516 | 0.305* | -0.071* | -0.032* |
| mlp WavLM+eGeMAPS | -1.442 | -4.849 | -10.335 | -9.318 |
| ridge WavLM+eGeMAPS +vol | 0.522 | 0.389 | 0.230 | 0.239 |
| mlp WavLM+eGeMAPS +vol | -1.676 | -5.476 | -11.476 | -10.368 |
| ridge WavLM+text +vol | 0.539 | 0.448 | 0.276 | 0.272 |
| mlp WavLM+text +vol | 0.228* | -0.288* | -1.033* | -0.860* |

## fincall — temporal split — Δv — R²_OOS vs persistence (test) (n=483)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| ridge eGeMAPS | 0.163* | 0.156* | 0.059* | 0.079* |
| mlp eGeMAPS | -0.096* | -0.004* | -0.228 | -0.564 |
| ridge eGeMAPS +vol | 0.373 | 0.268 | 0.074* | 0.108* |
| mlp eGeMAPS +vol | 0.319 | -0.212 | -0.901 | -3.657 |
| ridge WavLM | 0.156* | 0.148* | -0.027 | -0.008* |
| mlp WavLM | 0.045* | -0.108* | -0.433* | -1.160 |
| ridge WavLM +vol | 0.357 | 0.262 | -0.044 | 0.052* |
| mlp WavLM +vol | 0.295 | 0.087* | -0.387* | -1.274* |
| ridge emotion2vec+ | 0.155* | 0.129* | 0.022* | 0.027* |
| mlp emotion2vec+ | 0.006* | -0.029* | -0.337* | -1.001 |
| ridge emotion2vec+ +vol | 0.403* | 0.307 | 0.095* | 0.064* |
| mlp emotion2vec+ +vol | 0.186* | 0.107* | -0.218 | -0.992 |
| ridge WavLM+eGeMAPS | 0.155* | 0.145* | 0.015* | -0.008* |
| mlp WavLM+eGeMAPS | -0.010* | -0.180* | -0.550* | -1.121 |
| ridge WavLM+eGeMAPS +vol | 0.353 | 0.268 | 0.033* | 0.050* |
| mlp WavLM+eGeMAPS +vol | 0.140 | -0.014* | -0.581* | -1.493* |
| ridge WavLM+text +vol | 0.311 | 0.237 | 0.050* | -0.102* |
| mlp WavLM+text +vol | 0.323 | 0.269 | -0.103 | -0.983 |

## fincall — ticker_disjoint split — Δv — R²_OOS vs persistence (test) (n=484)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| ridge eGeMAPS | 0.203* | 0.119* | 0.137* | 0.044* |
| mlp eGeMAPS | 0.093* | -0.037* | 0.038* | -0.044* |
| ridge eGeMAPS +vol | 0.524 | 0.431 | 0.291 | 0.286 |
| mlp eGeMAPS +vol | 0.328 | 0.169* | 0.179* | 0.189 |
| ridge WavLM | 0.178* | 0.192* | 0.131* | 0.026* |
| mlp WavLM | 0.026* | 0.011* | -0.140* | -0.198* |
| ridge WavLM +vol | 0.456 | 0.390 | 0.267 | 0.222 |
| mlp WavLM +vol | 0.424* | 0.346* | 0.139* | 0.060* |
| ridge emotion2vec+ | 0.181* | 0.185* | 0.122* | 0.047* |
| mlp emotion2vec+ | 0.077* | 0.032* | -0.188* | -0.210* |
| ridge emotion2vec+ +vol | 0.541* | 0.464 | 0.340 | 0.317 |
| mlp emotion2vec+ +vol | 0.431* | 0.323* | 0.123* | 0.029* |
| ridge WavLM+eGeMAPS | 0.181* | 0.194* | 0.136* | 0.110* |
| mlp WavLM+eGeMAPS | 0.033* | -0.007* | -0.131* | -0.194* |
| ridge WavLM+eGeMAPS +vol | 0.442 | 0.364* | 0.241 | 0.217 |
| mlp WavLM+eGeMAPS +vol | 0.327* | 0.154* | -0.152* | -0.196* |
| ridge WavLM+text +vol | 0.369* | 0.348* | 0.238 | 0.301 |
| mlp WavLM+text +vol | 0.404* | 0.315* | 0.128* | 0.193 |

## fincall — temporal split — level-v — MSE (test) (n=483)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| ridge eGeMAPS | 0.811 | 0.383* | 0.289* | 0.248* |
| mlp eGeMAPS | 14.075 | 9.472 | 8.379 | 7.186 |
| ridge eGeMAPS +vol | 0.795 | 0.331 | 0.225 | 0.193 |
| mlp eGeMAPS +vol | 11.213 | 8.287 | 8.213 | 5.862 |
| ridge WavLM | 0.816 | 0.410* | 0.318* | 0.242* |
| mlp WavLM | 1.361* | 0.809* | 0.739* | 0.718* |
| ridge WavLM +vol | 0.789 | 0.338 | 0.231 | 0.191* |
| mlp WavLM +vol | 1.250* | 0.708* | 0.608* | 0.580* |
| ridge emotion2vec+ | 0.821 | 0.380* | 0.278* | 0.226 |
| mlp emotion2vec+ | 1.612* | 1.156* | 0.989* | 0.858* |
| ridge emotion2vec+ +vol | 0.766 | 0.311* | 0.196* | 0.161* |
| mlp emotion2vec+ +vol | 1.667* | 1.154* | 0.961* | 0.877* |
| ridge WavLM+eGeMAPS | 0.820 | 0.386* | 0.298* | 0.243* |
| mlp WavLM+eGeMAPS | 3.615 | 2.709 | 2.251 | 1.762 |
| ridge WavLM+eGeMAPS +vol | 0.804 | 0.332 | 0.236 | 0.203 |
| mlp WavLM+eGeMAPS +vol | 3.815 | 2.849 | 2.511 | 2.032 |
| ridge WavLM+text +vol | 0.727* | 0.299* | 0.212 | 0.175* |
| mlp WavLM+text +vol | 1.000* | 0.470* | 0.384* | 0.364* |

## fincall — ticker_disjoint split — level-v — MSE (test) (n=484)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| ridge eGeMAPS | 0.636 | 0.325* | 0.230* | 0.246* |
| mlp eGeMAPS | 11.919 | 13.169 | 13.212 | 14.372 |
| ridge eGeMAPS +vol | 0.624 | 0.274 | 0.171 | 0.192 |
| mlp eGeMAPS +vol | 11.149 | 10.562 | 12.088 | 12.512 |
| ridge WavLM | 0.664 | 0.325* | 0.254* | 0.266* |
| mlp WavLM | 1.396* | 0.926* | 0.831* | 0.861* |
| ridge WavLM +vol | 0.655 | 0.286 | 0.188 | 0.197 |
| mlp WavLM +vol | 1.470* | 0.989* | 0.887* | 0.882* |
| ridge emotion2vec+ | 0.639 | 0.307* | 0.227* | 0.231* |
| mlp emotion2vec+ | 3.808 | 2.812 | 2.967 | 2.799 |
| ridge emotion2vec+ +vol | 0.630 | 0.256 | 0.165 | 0.188 |
| mlp emotion2vec+ +vol | 3.912 | 2.985 | 3.125 | 3.171 |
| ridge WavLM+eGeMAPS | 0.663 | 0.332* | 0.266* | 0.270* |
| mlp WavLM+eGeMAPS | 3.340 | 2.792 | 2.814 | 2.700 |
| ridge WavLM+eGeMAPS +vol | 0.653 | 0.291 | 0.191 | 0.199 |
| mlp WavLM+eGeMAPS +vol | 3.661 | 3.091 | 3.097 | 2.975 |
| ridge WavLM+text +vol | 0.631 | 0.263 | 0.180 | 0.191 |
| mlp WavLM+text +vol | 1.056* | 0.615* | 0.505* | 0.487* |

## fincall — temporal split — Δv — MSE (test) (n=483)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| ridge eGeMAPS | 1.062* | 0.378* | 0.198* | 0.106* |
| mlp eGeMAPS | 1.391* | 0.449* | 0.259 | 0.180 |
| ridge eGeMAPS +vol | 0.797 | 0.328 | 0.195* | 0.102* |
| mlp eGeMAPS +vol | 0.865 | 0.543 | 0.401 | 0.535 |
| ridge WavLM | 1.072* | 0.381* | 0.217 | 0.116* |
| mlp WavLM | 1.212* | 0.496* | 0.302* | 0.248 |
| ridge WavLM +vol | 0.817 | 0.330 | 0.220 | 0.109* |
| mlp WavLM +vol | 0.895 | 0.409* | 0.292* | 0.261* |
| ridge emotion2vec+ | 1.074* | 0.390* | 0.206* | 0.112* |
| mlp emotion2vec+ | 1.263* | 0.461* | 0.282* | 0.230 |
| ridge emotion2vec+ +vol | 0.757* | 0.310 | 0.191* | 0.108* |
| mlp emotion2vec+ +vol | 1.033* | 0.400* | 0.257 | 0.229 |
| ridge WavLM+eGeMAPS | 1.073* | 0.383* | 0.208* | 0.116* |
| mlp WavLM+eGeMAPS | 1.283* | 0.528* | 0.327* | 0.244 |
| ridge WavLM+eGeMAPS +vol | 0.822 | 0.327 | 0.204* | 0.109* |
| mlp WavLM+eGeMAPS +vol | 1.091 | 0.454* | 0.333* | 0.286* |
| ridge WavLM+text +vol | 0.875 | 0.342 | 0.200* | 0.127* |
| mlp WavLM+text +vol | 0.860 | 0.327 | 0.233 | 0.228 |

## fincall — ticker_disjoint split — Δv — MSE (test) (n=484)

| Model | τ=3 | τ=7 | τ=15 | τ=30 |
| --- | --- | --- | --- | --- |
| ridge eGeMAPS | 1.091* | 0.420* | 0.214* | 0.250* |
| mlp eGeMAPS | 1.241* | 0.495* | 0.239* | 0.273* |
| ridge eGeMAPS +vol | 0.652 | 0.272 | 0.176 | 0.187 |
| mlp eGeMAPS +vol | 0.920 | 0.397* | 0.204* | 0.212 |
| ridge WavLM | 1.125* | 0.386* | 0.216* | 0.255* |
| mlp WavLM | 1.333* | 0.472* | 0.283* | 0.313* |
| ridge WavLM +vol | 0.745 | 0.291 | 0.182 | 0.204 |
| mlp WavLM +vol | 0.788* | 0.312* | 0.214* | 0.246* |
| ridge emotion2vec+ | 1.121* | 0.389* | 0.218* | 0.249* |
| mlp emotion2vec+ | 1.263* | 0.462* | 0.295* | 0.317* |
| ridge emotion2vec+ +vol | 0.628* | 0.256 | 0.164 | 0.179 |
| mlp emotion2vec+ +vol | 0.778* | 0.323* | 0.218* | 0.254* |
| ridge WavLM+eGeMAPS | 1.120* | 0.385* | 0.214* | 0.233* |
| mlp WavLM+eGeMAPS | 1.323* | 0.481* | 0.281* | 0.312* |
| ridge WavLM+eGeMAPS +vol | 0.764 | 0.304* | 0.188 | 0.205 |
| mlp WavLM+eGeMAPS +vol | 0.921* | 0.404* | 0.286* | 0.313* |
| ridge WavLM+text +vol | 0.863* | 0.311* | 0.189 | 0.183 |
| mlp WavLM+text +vol | 0.815* | 0.327* | 0.217* | 0.211 |

