"""Probe why specific listed-company names fail SEC-table lookup."""

import json
from pathlib import Path

from ecvol.data.fincall_identity import _lookup_name, clean_candidate, load_sec_table, norm_name

ROOT = Path(__file__).resolve().parents[1]
sec = load_sec_table(ROOT / "data")

probes = [
    "Comerica",
    "Hologic",
    "ANSYS",
    "Sealed Air",
    "Interpublic Group",
    "Synovus",
    "The J. M. Smucker Company",
    "Wabtec",
    "C.H. Robinson",
    "J.B. Hunt",
    "Align Technologies",
    "Boston Properties",
    "Starbucks Coffee Company",
    "Acuity Brands",
    "Coterra Energy",
    "Lincoln Financial",
    "Uber",
    "Macy",
]
for p in probes:
    key = norm_name(clean_candidate(p))
    hit = _lookup_name(p, sec)
    in_table = key in sec
    print(f"{p!r:32s} key={key!r:28s} exact={in_table!s:5s} lookup={hit}")

raw = json.loads((ROOT / "data/raw/ref/company_tickers.json").read_text("utf-8"))
for needle in (
    "COMERICA",
    "HOLOGIC",
    "ANSYS",
    "SEALED",
    "INTERPUBLIC",
    "SMUCKER",
    "SYNOVUS",
    "ROBINSON",
    "HUNT TRANSPORT",
    "ALIGN",
    "STARBUCKS",
    "ACUITY",
    "COTERRA",
    "LINCOLN NATIONAL",
    "UBER",
    "MACY",
):
    rows = [(r["ticker"], r["title"]) for r in raw.values() if needle in r["title"].upper()]
    print(needle, "->", rows[:4])
