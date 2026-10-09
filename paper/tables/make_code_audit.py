"""Rewrite the rows of paper/tables/code_audit.tex from data/results/code_availability.csv.

Run from the repository root:  uv run python paper/tables/make_code_audit.py
Only the header comment (run id) and the tabular body between \\midrule and \\bottomrule
are replaced; the caption and the table environment are kept as they are.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "results" / "code_availability.csv"
TEX = Path(__file__).resolve().parent / "code_audit.tex"


def esc(s: str) -> str:
    s = "" if pd.isna(s) else str(s)
    for a, b in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("#", r"\#"),
    ):
        s = s.replace(a, b)
    return s


def probe(r: pd.Series) -> str:
    status = "" if pd.isna(r.probe_status) else str(r.probe_status)
    if status == "no_url":
        return esc(status)
    if pd.isna(r.pushed_at) or str(r.pushed_at) == "":
        return f"{esc(status)} ({r.probed_on})"
    files = "" if pd.isna(r.code_files) else f"{int(float(r.code_files))} code files, "
    return f"{esc(status)}, {files}pushed {r.pushed_at} ({r.probed_on})"


def run_id() -> str:
    runs = sorted((ROOT / "artifacts" / "runs").glob("*-audit-code-*"))
    for d in reversed(runs):
        j = json.loads((d / "run.json").read_text(encoding="utf-8"))
        if any("code_availability.csv" in str(o) for o in j.get("outputs", [])):
            return d.name
    return runs[-1].name if runs else "unknown"


df = pd.read_csv(CSV)
rows = [
    f"{esc(r.paper)} & {esc(r.venue)} & {probe(r)} & {esc(r.verdict)} & {esc(r.reason)} \\\\"
    for _, r in df.iterrows()
]
tex = TEX.read_text(encoding="utf-8")
tex = re.sub(
    r"(?m)^% Generated from data/results/code_availability\.csv \(run [^)]*\)\.",
    f"% Generated from data/results/code_availability.csv (run {run_id()}).",
    tex,
)
head, _, rest = tex.partition("\\midrule\n")
_, _, tail = rest.partition("\\bottomrule")
TEX.write_text(head + "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule" + tail, encoding="utf-8")
print(f"wrote {len(rows)} rows from {CSV.name} ({run_id()})")
