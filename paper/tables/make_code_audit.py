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


# Rendering of the audit rows: short name, bibliography key and venue per audit key. The CSV
# keeps the audit's own long names; the bibliography entries live in paper/refs.bib.
DISPLAY: dict[str, tuple[str, str, str]] = {
    "qin2019": ("What You Say and How You Say It Matters", "qin2019what", "ACL 2019"),
    "html2020": ("HTML", "yang2020html", "WWW 2020"),
    "sawhney2020": ("Multimodal financial forecasting", "sawhney2020mm", "ACM MM 2020"),
    "voltage2020": ("VolTAGE", "sawhney2020voltage", "EMNLP 2020"),
    "kefvp2023": ("KeFVP", "niu2023kefvp", "Findings of EMNLP 2023"),
    "dialoguegat2023": ("DialogueGAT", "sang2022dialoguegat", "Findings of EMNLP 2022"),
    "scss2025": ("Same Company, Same Signal", "scss2024", "Findings of ACL 2025"),
    "fintrust2023": ("FinTrust", "fintrust2023", "ACL 2023"),
    "soundofrisk2025": ("Sound of Risk", "soundofrisk2025", "arXiv 2025"),
    "defvp2024": ("DeFVP", "niu2024does", "ICME 2024"),
    "echogl2024": ("ECHO-GL", "echogl2024", "AAAI 2024"),
    "eccanalyzer2024": ("ECC Analyzer", "cao2024ecc", "ICAIF 2024"),
    "risklabs2024": ("RiskLabs", "cao2024risklabs", "arXiv 2024"),
    "numhtml2022": ("NumHTML", "yang2022numhtml", "AAAI 2022"),
    "gnavol": ("GNA-Vol", "shi2023gnavol", "IJCNLP-AACL 2023"),
    "amalstm": ("AMA-LSTM", "ama2024", "NAACL 2024 Industry"),
    "atfingpt": ("AT-FinGPT", "atfingpt2025", "Finance Research Letters 2025"),
}


def paper_cell(r: pd.Series) -> tuple[str, str]:
    """(name with citation, venue) for a row; falls back to the CSV text for unknown keys."""
    if r.key in DISPLAY:
        name, bib, venue = DISPLAY[r.key]
        return f"{esc(name)} \\citep{{{bib}}}", esc(venue)
    return esc(r.paper), esc(r.venue)


df = pd.read_csv(CSV)
rows = []
for _, r in df.iterrows():
    name, venue = paper_cell(r)
    rows.append(f"{name} & {venue} & {probe(r)} & {esc(r.verdict)} & {esc(r.reason)} \\\\")
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
