"""T6.2 kappa-gate diagnostics: is the failure a plumbing bug, a scale offset, or real disagreement?

Regenerates every number quoted in `data/coverage/llm_kappa_gate_report.md` beyond the headline
kappas (which come from `ecvol llm-kappa`): the label join check, per-field marginals, the
guidance/hedging crosstabs, the shifted-ordinal re-scoring that rules out a calibration offset,
and the raw-agreement/PABAK figures for the skewed binary field.

Read-only — touches nothing under data/ and needs no GPU.

Usage: uv run python notebooks/llm_kappa_diagnostics.py [features_parquet]
"""

from __future__ import annotations

import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from ecvol.features.llm.audit import _load_features, _load_labels

SHEET = "data/coverage/fincall_llm_labels_rater1.csv"
DEFAULT_FEAT = "data/fincall/llm_features__bartowski__Qwen2.5-7B-Instruct-GGUF_Q4_K_M.parquet"
FEAT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FEAT

lab = _load_labels(SHEET)
feat = _load_features(FEAT)

print("=== JOIN CHECK (rules out the boring explanation first) ===")
lk = set(zip(lab["call_id"], lab["section"], strict=False))
fk = set(zip(feat["call_id"], feat["section"], strict=False))
print(f"label rows={len(lab)} feature rows={len(feat)}")
print(f"matched={len(lk & fk)}  labels-not-in-features={len(lk - fk)}")

m = lab.merge(feat, on=["call_id", "section"], suffixes=("_h", "_m"))
print(f"merged rows={len(m)}")

print("\n=== MARGINALS + RAW AGREEMENT ===")
for f in ("guidance_direction", "hedging_intensity", "surprise_mentions",
          "qa_evasiveness", "analyst_tone"):  # fmt: skip
    h, mv = m[f + "_h"].astype(str), m[f + "_m"].astype(str)
    print(f"\n--- {f} ---")
    print("human:", dict(sorted(h.value_counts().items())))
    print("model:", dict(sorted(mv.value_counts().items())))
    print(f"raw agreement: {(h == mv).mean():.1%}")

print("\n=== guidance_direction confusion (rows=human, cols=model) ===")
print(pd.crosstab(m["guidance_direction_h"], m["guidance_direction_m"]))

print("\n=== hedging: is it just a scale offset? (if so, a shift would rescue kappa) ===")
h = m["hedging_intensity_h"].astype(int)
mv = m["hedging_intensity_m"].astype(int)
print(pd.crosstab(h, mv))
print(f"mean human={h.mean():.2f} model={mv.mean():.2f} offset={mv.mean() - h.mean():+.2f}")
for shift in (0, -1, -2):
    s = (mv + shift).clip(0, 4)
    k = cohen_kappa_score(h, s, weights="linear")
    print(f"  shift {shift:+d}: kappa={k:+.3f}  raw={(h == s).mean():.1%}")

print("\n=== analyst_tone / qa_evasiveness on Q&A rows only ===")
qa = m[m["section"] == "qa"]
for f in ("qa_evasiveness", "analyst_tone"):
    keep = qa[f + "_h"] != "NA"
    hh, mm = qa.loc[keep, f + "_h"].astype(int), qa.loc[keep, f + "_m"].astype(int)
    print(f"{f}: n={len(hh)} model distinct values={sorted(mm.unique())}")
    print(f"  human={dict(sorted(hh.value_counts().items()))}")
    print(f"  model={dict(sorted(mm.value_counts().items()))}")

print("\n=== surprise_mentions binarized: kappa vs raw agreement under skew ===")
hb = (m["surprise_mentions_h"].astype(int) > 0).astype(int)
mb = (m["surprise_mentions_m"].astype(int) > 0).astype(int)
print(pd.crosstab(hb, mb))
raw = (hb == mb).mean()
print(f"raw agreement={raw:.1%}  kappa={cohen_kappa_score(hb, mb):+.3f}  PABAK={2 * raw - 1:+.3f}")
print(f"human 'present' prevalence={hb.mean():.1%}  model={mb.mean():.1%}")
print(
    f"model misses {int(((hb == 1) & (mb == 0)).sum())} of {int(hb.sum())} human positives "
    "- a real detection failure, not only a metric artefact"
)
