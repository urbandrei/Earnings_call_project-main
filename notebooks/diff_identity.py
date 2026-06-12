"""Inspect flips/losses between two identity CSV versions."""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

old_path, mode = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "lost"

old, new = {}, {}
for path, dest in ((Path(old_path), old), (DATA / "identity" / "fincall_identity.csv", new)):
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dest[row["call_id"]] = row

transcripts = {}
for year in (2019, 2020, 2021):
    data = json.loads((DATA / "raw" / "fincall" / f"transcripts_{year}.json").read_text("utf-8"))
    transcripts.update({cid: rec["input"] for cid, rec in data.items()})

if mode == "lost":
    cids = [c for c in new if not new[c]["ticker"] and old[c]["ticker"]]
elif mode == "flip":
    cids = [
        c
        for c in new
        if new[c]["ticker"] and old[c]["ticker"] and old[c]["ticker"] != new[c]["ticker"]
    ]
else:
    cids = [c for c in new if new[c]["ticker"] and not old[c]["ticker"]]

print(f"{mode}: {len(cids)} calls")
for cid in cids[:12]:
    o, n = old[cid], new[cid]
    head = " ".join(transcripts[cid][:400].split())
    print(
        f"\n--- {cid}: {o['ticker']}(s{o['score']}) -> {n['ticker'] or 'UNRESOLVED'}"
        f"(s{n['score']}, runner {n['runner_up']}:{n['runner_up_score']}) [{n['flags']}]"
    )
    print(f"    {head}")
