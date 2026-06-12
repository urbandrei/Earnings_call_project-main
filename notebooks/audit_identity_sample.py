"""Draw the seeded audit sample for the identity accuracy gate (T1.4).

Strata: 40 random resolved earnings-type calls + 20 random calls that newly
resolved this session (vs the v1 CSV at commit 3699b31), which stress-test the
override table, greeting-dominance, and singles-counting logic. Prints each
call's resolved identity next to the transcript head for human verification;
verdicts are recorded in data/identity/fincall_identity_audit.csv.
"""

import csv
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BASELINE_COMMIT = "3699b31"

new = {}
with open(DATA / "identity" / "fincall_identity.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        new[row["call_id"]] = row

baseline_csv = subprocess.run(
    ["git", "show", f"{BASELINE_COMMIT}:data/identity/fincall_identity.csv"],
    capture_output=True,
    text=True,
    check=True,
    cwd=ROOT,
).stdout
old = {row["call_id"]: row for row in csv.DictReader(baseline_csv.splitlines())}

titles = {}
for r in json.loads((DATA / "raw" / "ref" / "company_tickers.json").read_text("utf-8")).values():
    titles.setdefault(str(r["cik_str"]), r["title"])
with open(DATA / "identity" / "fincall_name_overrides.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        titles.setdefault(row["cik"], f"[override] {row['alias']} ({row['note']})")

resolved_earnings = sorted(
    c for c, r in new.items() if r["ticker"] and r["call_type"] == "earnings"
)
gained = sorted(c for c, r in new.items() if r["ticker"] and not old[c]["ticker"])

rng = random.Random(0)
sample = [("random_earnings", c) for c in rng.sample(resolved_earnings, 40)]
chosen = {c for _, c in sample}
sample += [("gained", c) for c in rng.sample([c for c in gained if c not in chosen], 20)]

transcripts = {}
for year in (2019, 2020, 2021):
    data = json.loads((DATA / "raw" / "fincall" / f"transcripts_{year}.json").read_text("utf-8"))
    transcripts.update({c: r["input"] for c, r in data.items()})

for stratum, cid in sample:
    row = new[cid]
    head = " ".join(transcripts[cid][:450].split())
    print(
        f"--- {cid} [{stratum}] -> {row['ticker']} = {titles.get(row['cik'], '?')} "
        f"(score {row['score']}, type {row['call_type']}, date {row['date']})"
    )
    print(f"    {head}\n")

if "--csv" in sys.argv:
    out = DATA / "identity" / "fincall_identity_audit.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["call_id", "stratum", "ticker", "verdict", "note"])
        for stratum, cid in sample:
            w.writerow([cid, stratum, new[cid]["ticker"], "", ""])
    print(f"skeleton written: {out}")
