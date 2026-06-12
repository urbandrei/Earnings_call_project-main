"""Print seeded samples of each unresolved class with transcript heads + scores."""

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from ecvol.data.fincall_identity import load_sec_table, phrase_mentions

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8

sec, single_ok = load_sec_table(DATA)
rows = {}
with open(DATA / "identity" / "fincall_identity.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rows[row["call_id"]] = row

transcripts = {}
for year in (2019, 2020, 2021):
    data = json.loads((DATA / "raw" / "fincall" / f"transcripts_{year}.json").read_text("utf-8"))
    transcripts.update({cid: rec["input"] for cid, rec in data.items()})

by_class: dict[str, list[str]] = defaultdict(list)
for cid, row in sorted(rows.items()):
    for f in row["flags"].split(";"):
        if f.startswith("unresolved:"):
            by_class[f].append(cid)

rng = random.Random(0)
cik_to_ticker = {cik: t for t, cik in reversed(sec.values())}
for cls, cids in sorted(by_class.items()):
    print(f"\n{'=' * 20} {cls} ({len(cids)} calls) {'=' * 20}")
    for cid in rng.sample(cids, min(N, len(cids))):
        row = rows[cid]
        counts = phrase_mentions(transcripts[cid], sec, single_ok)
        top3 = ", ".join(f"{cik_to_ticker.get(c, c)}:{n}" for c, n in counts.most_common(3))
        head = " ".join(transcripts[cid][:500].split())
        print(
            f"\n--- {cid} (year {row['year']}, score {row['score']}, "
            f"runner {row['runner_up']}:{row['runner_up_score']}) body[{top3}]"
        )
        print(f"    {head}")
