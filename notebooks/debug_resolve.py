"""Verbose resolve trace for specific call IDs given on the command line."""

import json
import sys
from collections import Counter
from pathlib import Path

from ecvol.data.fincall_identity import (
    _greeting_names,
    _lookup_name,
    load_sec_table,
    phrase_mentions,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sec, single_ok = load_sec_table(DATA)
cik_to_ticker = {cik: t for t, cik in reversed(sec.values())}

transcripts = {}
for year in (2019, 2020, 2021):
    data = json.loads((DATA / "raw" / "fincall" / f"transcripts_{year}.json").read_text("utf-8"))
    transcripts.update({cid: rec["input"] for cid, rec in data.items()})

for cid in sys.argv[1:]:
    text = transcripts[cid]
    print(f"\n=== {cid} ===")
    scores: Counter = Counter(phrase_mentions(text, sec, single_ok))
    print("body:", {cik_to_ticker.get(c, c): n for c, n in scores.most_common(6)})
    for name, weight, idx in _greeting_names(text):
        hit = _lookup_name(name, sec)
        tick = cik_to_ticker.get(hit[1]) if hit else None
        print(f"greeting{idx} (w{weight}): {name!r} -> {hit} {tick or 'MISS'}")
