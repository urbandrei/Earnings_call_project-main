"""How much of the kappa-gate arithmetic is signal? Bootstrap CIs on kappa and on differences.

Every comparison in `data/coverage/llm_kappa_gate_report.md` rests on kappa estimated from 97
rows (50 for the Q&A-only fields). At that size the standard error is large enough that
eyeballing a +0.14 swing as "better" is not defensible. This resamples **calls** (not rows —
a call's two sections are correlated, so rows are not independent) to put an interval on each
kappa and, for a pair of prediction sets scored against the same labels, on their difference.

A difference whose CI spans 0 is noise, and must be reported as such.

Read-only, CPU-only. Usage:
    uv run python notebooks/llm_kappa_uncertainty.py A.parquet [B.parquet]
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from ecvol.features.llm.audit import _load_features, _load_labels
from ecvol.features.llm.schema import CONFIRMATORY_FIELDS

SHEET = "data/coverage/fincall_llm_labels_rater1.csv"
FIELDS = ("guidance_direction", "hedging_intensity", "surprise_mentions",
          "qa_evasiveness", "analyst_tone")  # fmt: skip
N_BOOT = 2000
SEED = 0


def _kappa(field: str, h: pd.Series, m: pd.Series) -> float | None:
    keep = h != "NA"
    h, m = h[keep], m[keep]
    if len(h) < 2:
        return None
    if field == "guidance_direction":
        return cohen_kappa_score(h.astype(str), m.astype(str))
    if field == "surprise_mentions":
        return cohen_kappa_score((h.astype(int) > 0).astype(int), (m.astype(int) > 0).astype(int))
    return cohen_kappa_score(h.astype(int), m.astype(int), weights="linear")


def _merged(features_path: str) -> pd.DataFrame:
    return _load_labels(SHEET).merge(
        _load_features(features_path), on=["call_id", "section"], suffixes=("_h", "_m")
    )


def bootstrap(frames: dict[str, pd.DataFrame], n_boot: int = N_BOOT) -> None:
    """Paired call-level bootstrap: the same resampled calls score every prediction set."""
    calls = sorted(set.intersection(*(set(f["call_id"]) for f in frames.values())))
    rng = np.random.default_rng(SEED)
    by_call = {name: {c: g for c, g in f.groupby("call_id")} for name, f in frames.items()}

    draws: dict[str, dict[str, list[float]]] = {n: {f: [] for f in FIELDS} for n in frames}
    for _ in range(n_boot):
        picks = rng.choice(len(calls), size=len(calls), replace=True)
        sampled = [calls[i] for i in picks]
        for name, groups in by_call.items():
            boot = pd.concat([groups[c] for c in sampled], ignore_index=True)
            for f in FIELDS:
                k = _kappa(f, boot[f + "_h"], boot[f + "_m"])
                if k is not None and not np.isnan(k):
                    draws[name][f].append(k)

    names = list(frames)
    print(f"\ncalls={len(calls)}  bootstrap={n_boot} (resampling calls, not rows)\n")
    for f in FIELDS:
        tag = " [confirmatory]" if f in CONFIRMATORY_FIELDS else ""
        line = f"{f:24s}"
        for name in names:
            point = _kappa(f, frames[name][f + "_h"], frames[name][f + "_m"])
            lo, hi = np.percentile(draws[name][f], [2.5, 97.5])
            line += f"  {name}: {point:+.3f} [{lo:+.3f},{hi:+.3f}]"
        print(line + tag)

    if len(names) == 2:
        a, b = names
        print(f"\ndifference ({b} - {a}), 95% CI from the paired draws:")
        for f in FIELDS:
            da, db = np.array(draws[a][f]), np.array(draws[b][f])
            n = min(len(da), len(db))
            diff = db[:n] - da[:n]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            verdict = "NOISE (CI spans 0)" if lo <= 0 <= hi else "distinguishable"
            print(f"  {f:24s} {diff.mean():+.3f} [{lo:+.3f},{hi:+.3f}]  {verdict}")


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        base = "data/fincall/llm_features__bartowski__Qwen2.5-7B-Instruct-GGUF"
        paths = [f"{base}_Q4_K_M.parquet", f"{base}_Q8_0.parquet"]
    frames = {p.split("__")[-1].replace(".parquet", ""): _merged(p) for p in paths}
    bootstrap(frames)


if __name__ == "__main__":
    main()
