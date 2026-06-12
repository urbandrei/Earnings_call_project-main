"""Residue analysis for T1.4 identity overrides: which company names go unmatched?

For every unresolved call, collect the name candidates the resolver saw
(greeting captures + PDF /Company) that failed SEC-table lookup, normalize them,
and print them by frequency — the worklist for the override CSV.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from ecvol.data.fincall_identity import (
    _greeting_names,
    _lookup_name,
    clean_candidate,
    load_sec_table,
    norm_name,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

sec, _single_ok = load_sec_table(DATA)
cache = json.loads((DATA / "raw" / "ref" / "fincall_pdf_signals.json").read_text(encoding="utf-8"))

unresolved = {}
with open(DATA / "identity" / "fincall_identity.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if not row["ticker"]:
            unresolved[row["call_id"]] = row

name_counts: Counter = Counter()
name_calls: dict[str, list[str]] = defaultdict(list)
name_raw: dict[str, Counter] = defaultdict(Counter)
calls_with_candidate: set[str] = set()

for year in (2019, 2020, 2021):
    data = json.loads((DATA / "raw" / "fincall" / f"transcripts_{year}.json").read_text("utf-8"))
    for call_id, rec in sorted(data.items()):
        if call_id not in unresolved:
            continue
        ppt_id = str(rec.get("ppt_id") or call_id)
        company_meta = cache.get(f"{year}/{ppt_id}", [None, None, None])[0]
        candidates = [name for name, _ in _greeting_names(rec["input"])]
        if company_meta:
            candidates.append(company_meta)
        for cand in candidates:
            if _lookup_name(cand, sec):
                continue  # matched but evidently not enough to resolve; skip here
            key = norm_name(clean_candidate(cand))
            if len(key) < 2:
                continue
            calls_with_candidate.add(call_id)
            if call_id not in name_calls[key]:
                name_calls[key].append(call_id)
            name_counts[key] += 1
            name_raw[key][clean_candidate(cand)] += 1

print(f"unresolved calls: {len(unresolved)}")
print(f"unresolved calls with >=1 unmatched name candidate: {len(calls_with_candidate)}")
print(f"unique unmatched normalized names: {len(name_counts)}\n")
for key, _n in name_counts.most_common():
    raws = "; ".join(r for r, _ in name_raw[key].most_common(2))
    print(f"{len(name_calls[key]):4d} calls  {key!r:45s} e.g. {raws}")
