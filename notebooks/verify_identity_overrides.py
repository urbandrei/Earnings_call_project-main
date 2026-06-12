"""Verify fincall_name_overrides.csv against SEC sources.

For each override row: if the ticker still exists in company_tickers.json, the
CIK must match it exactly (so CIK-keyed evidence merges with the live table).
Otherwise the CIK must appear in EDGAR's all-registrants-ever name file
(cik-lookup-data.txt) — its registered names are printed for eyeball review.
"""

import csv
import json
import re
from pathlib import Path

from ecvol.data.fetch import _download
from ecvol.data.fincall_identity import SEC_HEADERS

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "raw" / "ref"

LOOKUP_URL = "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt"
lookup_path = REF / "cik-lookup-data.txt"
_download(LOOKUP_URL, lookup_path, headers=SEC_HEADERS)

by_ticker = {
    row["ticker"]: (str(row["cik_str"]), row["title"])
    for row in json.loads((REF / "company_tickers.json").read_text("utf-8")).values()
}

cik_names: dict[str, list[str]] = {}
with open(lookup_path, encoding="latin-1") as f:
    for line in f:
        m = re.match(r"^(.*):(\d{10}):$", line.rstrip("\n"))
        if m:
            cik_names.setdefault(str(int(m.group(2))), []).append(m.group(1))

problems = 0
with open(ROOT / "data" / "identity" / "fincall_name_overrides.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        alias, ticker, cik = row["alias"], row["ticker"], row["cik"]
        if ticker in by_ticker:
            live_cik, title = by_ticker[ticker]
            if live_cik != cik:
                print(
                    f"MISMATCH {alias!r}: ticker {ticker} is live with CIK {live_cik} "
                    f"({title}) but override says {cik}"
                )
                problems += 1
            else:
                print(f"ok-live  {alias!r:30s} {ticker:5s} {cik:8s} = {title}")
        elif cik in cik_names:
            names = "; ".join(cik_names[cik][:3])
            print(f"ok-hist  {alias!r:30s} {ticker:5s} {cik:8s} ~ {names}")
        else:
            print(f"MISSING  {alias!r}: CIK {cik} not in EDGAR registrant list")
            problems += 1

print(f"\nproblems: {problems}")
