"""List all calls lost vs a baseline commit's identity CSV, with heads."""

import csv
import json
import subprocess
import sys

base = subprocess.run(
    ["git", "show", f"{sys.argv[1]}:data/identity/fincall_identity.csv"],
    capture_output=True,
    text=True,
).stdout
old = {r["call_id"]: r for r in csv.DictReader(base.splitlines())}
with open("data/identity/fincall_identity.csv", encoding="utf-8") as f:
    new = {r["call_id"]: r for r in csv.DictReader(f)}
lost = [c for c in new if not new[c]["ticker"] and old[c]["ticker"]]
transcripts = {}
for year in (2019, 2020, 2021):
    d = json.loads(open(f"data/raw/fincall/transcripts_{year}.json", encoding="utf-8").read())
    transcripts.update({c: r["input"] for c, r in d.items()})
for c in sorted(lost):
    head = " ".join(transcripts[c][:160].split())
    o = old[c]
    print(f"{c} {o['ticker']:6s}(s{o['score']}) {new[c]['call_type']:10s} | {head}")
