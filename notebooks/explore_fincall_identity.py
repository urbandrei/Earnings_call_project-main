"""Exploration: can FinCall-Surprise call identity (ticker + date) be reconstructed?

Context (DECISIONS.md 2026-06-12): the released dataset has no ticker/company/date
on any record. This script samples N seeded-random calls and tries to recover
identity from (a) slide-PDF metadata + title-page text, (b) transcript prose,
matching recovered names against SEC's company_tickers.json. It prints a per-call
table and summary hit rates. Findings go to JOURNAL.md; promotion of the approach
into src/ecvol requires a DECISIONS.md entry.

Run:  uv run --with pypdf python notebooks/explore_fincall_identity.py [N] [SEED]
"""

import json
import random
import re
import sys
import unicodedata
from pathlib import Path

import requests
from pypdf import PdfReader

RAW = Path("data/raw/fincall")
SCRATCH = Path("artifacts/scratch")
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
# SEC asks for a descriptive UA with contact info on all programmatic requests.
SEC_UA = "ecvol research project (andrei.roman.personal@gmail.com)"

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DATE_RE = re.compile(rf"({MONTHS})\s+(\d{{1,2}})\s*,?\s*(20\d\d)")
QUARTER_RE = re.compile(
    r"\b(?:(first|second|third|fourth)[ -]quarter"
    r"|([1-4])Q(\d\d)|Q([1-4])\s*(?:FY)?\s*(20\d\d|\d\d))",
    re.I,
)
# Operator greetings: "Welcome to the <name> First Quarter 2019 Earnings ..."
GREETING_RE = re.compile(
    r"[Ww]elcome,? (?:everyone,? )?to (?:the )?(.{3,60}?)(?:'s)?[ ,]+"
    r"(?:first|second|third|fourth|[1-4]Q|Q[1-4]|fiscal|full[- ]year"
    r"|\d{4}|earnings|year[- ]end|annual)",
)
LEGAL_SUFFIXES = re.compile(
    r"\b(incorporated|inc|corporation|corp|company|co|limited|ltd|plc|llc|lp|"
    r"group|holdings?|the|nv|sa|ag|se)\b\.?",
)


def norm_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^a-z0-9 ]", " ", name.lower())
    name = LEGAL_SUFFIXES.sub(" ", name)
    return re.sub(r"\s+", " ", name).strip()


def load_sec_table() -> dict[str, str]:
    """normalized company name -> ticker (largest-cap entry wins on collision)."""
    cache = SCRATCH / "company_tickers.json"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(SEC_TICKERS_URL, headers={"User-Agent": SEC_UA}, timeout=30)
        resp.raise_for_status()
        cache.write_bytes(resp.content)
    table = {}
    for row in json.loads(cache.read_text(encoding="utf-8")).values():
        key = norm_name(row["title"])
        if key and key not in table:  # file is ordered by market cap; keep first
            table[key] = row["ticker"]
    return table


QUARTER_WORDS = re.compile(
    r"\b(first|second|third|fourth|[1-4]Q\d*|Q[1-4]|fiscal|year[- ]?end|full[- ]year|annual|"
    r"quarter|earnings|results|investor|overview|review|conference|call|and|20\d\d)\b.*$",
    re.I,
)


def clean_candidate(name: str) -> str:
    name = re.sub(r"['’]s?\b", "", name)  # possessives: "ADP's" -> "ADP"
    name = QUARTER_WORDS.sub("", name)  # drop trailing "Fourth Quarter ..." tails
    return name.strip(" ,.-|")


def match_ticker(name: str, sec: dict[str, str]) -> str | None:
    import difflib

    key = norm_name(clean_candidate(name))
    if len(key) < 3 or key in ("the", "our", "company"):
        return None
    if key in sec:
        return sec[key]
    # fuzzy: high-cutoff close match on the normalized SEC name
    close = difflib.get_close_matches(key, sec.keys(), n=1, cutoff=0.92)
    if close:
        return sec[close[0]]
    # prefix containment, both directions, requiring most of the shorter string
    best = max(
        (
            k
            for k in sec
            if (k.startswith(key) and len(key) >= 0.6 * len(k))
            or (key.startswith(k) and len(k) >= 0.6 * len(key))
        ),
        key=len,
        default=None,
    )
    return sec[best] if best and len(best) >= 5 else None


def pdf_signals(call_id: str, year: int) -> dict:
    pdf = RAW / f"ppt_{year}" / f"{call_id}.pdf"
    out = {"pdf_company": None, "pdf_page1_head": None, "pdf_date": None, "pdf_created": None}
    if not pdf.exists():
        return out
    try:
        reader = PdfReader(str(pdf))
        meta = reader.metadata or {}
        out["pdf_company"] = str(meta.get("/Company") or "") or None
        created = str(meta.get("/CreationDate") or "")
        if m := re.match(r"D:(\d{4})(\d\d)(\d\d)", created):
            out["pdf_created"] = "-".join(m.groups())
        page1 = (reader.pages[0].extract_text() or "")[:600]
        lines = [ln.strip() for ln in page1.splitlines() if ln.strip()]
        out["pdf_page1_head"] = " | ".join(lines[:4])[:120] or None
        if m := DATE_RE.search(page1):
            month, day, yr = m.groups()
            out["pdf_date"] = f"{yr}-{m.group(1)[:3]}-{int(day):02d}"
    except Exception as exc:  # exploration: tolerate any malformed PDF, note it
        out["pdf_page1_head"] = f"<pdf error: {type(exc).__name__}>"
    return out


def transcript_signals(text: str) -> dict:
    head = text[:4000]
    greeting = GREETING_RE.search(head)
    quarter = QUARTER_RE.search(head)
    return {
        "greet_name": greeting.group(1).strip() if greeting else None,
        "quarter": quarter.group(0) if quarter else None,
    }


def main(n: int = 50, seed: int = 0) -> None:
    sec = load_sec_table()
    print(f"SEC table: {len(sec)} names\n")
    calls = []
    for year in (2019, 2020, 2021):
        data = json.loads((RAW / f"transcripts_{year}.json").read_text(encoding="utf-8"))
        calls += [(year, cid, rec) for cid, rec in data.items()]
    sample = random.Random(seed).sample(calls, n)

    rows, hits, date_hits = [], 0, 0
    for year, cid, rec in sample:
        sig = {**pdf_signals(rec.get("ppt_id") or cid, year), **transcript_signals(rec["input"])}
        candidates = [c for c in (sig["pdf_company"], sig["greet_name"]) if c]
        if sig["pdf_page1_head"] and not sig["pdf_page1_head"].startswith("<pdf error"):
            candidates += sig["pdf_page1_head"].split(" | ")[:2]
        ticker, source = None, None
        for cand in candidates:
            if t := match_ticker(cand, sec):
                ticker, source = t, cand[:40]
                break
        hits += ticker is not None
        date_hits += bool(sig["pdf_date"] or sig["pdf_created"])
        rows.append(
            (
                cid,
                year,
                ticker,
                source,
                sig["pdf_date"] or sig["pdf_created"],
                sig["quarter"],
                (sig["greet_name"] or "")[:38],
                (sig["pdf_company"] or "")[:30],
            )
        )

    print(
        f"{'call_id':>10} {'yr':>4} {'tick':>6} {'matched-on':40} {'date':>11} "
        "quarter / greeting / pdf_company"
    )
    for r in rows:
        print(
            f"{r[0]:>10} {r[1]:>4} {str(r[2]):>6} {str(r[3]):40} {str(r[4]):>11} "
            f"{r[5]} / {r[6]} / {r[7]}"
        )
    print(f"\nticker hit rate: {hits}/{n} ({100 * hits / n:.0f}%)")
    print(f"date signal (page-1 or pdf-created): {date_hits}/{n} ({100 * date_hits / n:.0f}%)")


if __name__ == "__main__":
    main(
        int(sys.argv[1]) if len(sys.argv) > 1 else 50, int(sys.argv[2]) if len(sys.argv) > 2 else 0
    )
