"""FinCall-Surprise identity reconstruction: call → (ticker, company, date, type) (T1.4).

The released dataset carries no ticker/company/date on any record (DECISIONS.md
2026-06-12). This module rebuilds identity from in-corpus evidence:

- company: capitalized phrases from the full transcript + slide-PDF `/Company`
  metadata + operator greeting, matched against SEC's company_tickers.json and
  scored; a call resolves only when one ticker clearly dominates.
- date: slide-PDF title-page date, falling back to the PDF creation stamp.
- call type: heuristic classification (earnings vs M&A/conference/sales/other),
  so non-earnings calls can later be excluded with reason codes, never silently.

Output is a deterministic CSV (committed, like split CSVs) written by
`ecvol data identity`. Known limitation (DECISIONS.md): the SEC table lists
current registrants only — companies delisted since 2021 resolve worse; the
unresolved residue is measured and reported, and goes to audit.
"""

import csv
import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from ecvol.data.fetch import _download

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_LICENSE = "Public domain (US government work)"

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
MONTH_NUM = {m: i + 1 for i, m in enumerate(MONTHS.split("|"))}
DATE_RE = re.compile(rf"({MONTHS})\s+(\d{{1,2}})\s*,?\s*(20\d\d)")

# Operator greeting: "Welcome to the <name> First Quarter 2019 Earnings ..."
GREETING_RE = re.compile(
    r"[Ww]elcome,? (?:everyone,? )?to (?:the )?(.{3,60}?)(?:'s)?[ ,]+"
    r"(?:first|second|third|fourth|[1-4]Q|Q[1-4]|fiscal|full[- ]year"
    r"|\d{4}|earnings|year[- ]end|annual)"
)
LEGAL_SUFFIXES = re.compile(
    r"\b(incorporated|inc|corporation|corp|company|companies|co|limited|ltd|plc|llc|lp|"
    r"group|holdings?|trust|the|nv|sa|ag|se)\b\.?"
)
# Trailing event words leaking into name candidates: "<name> Fourth Quarter ..."
EVENT_TAIL = re.compile(
    r"\b(first|second|third|fourth|[1-4]Q\d*|Q[1-4]|fiscal|year[- ]?end|full[- ]year|annual|"
    r"quarter|earnings|results|investor|overview|review|conference|call|webcast|and|20\d\d)\b.*$",
    re.I,
)
# Maximal runs of capitalized tokens (allowing &) anywhere in the transcript.
CAP_RUN_RE = re.compile(r"(?:[A-Z][\w.\-']*|&)(?:[ \t](?:[A-Z][\w.\-']*|&))*")

EARNINGS_RE = re.compile(
    r"(?:(?:first|second|third|fourth)[ -]quarter|[1-4]Q\d\d|Q[1-4]\b|earnings|quarterly|"
    r"(?:full[- ]year|year[- ]end|annual).{0,30}results)",
    re.I,
)
MA_RE = re.compile(r"\b(merger|acquisition|acquire|combination|transaction|tender offer)\b", re.I)
CONFERENCE_RE = re.compile(r"\b(forum|fireside|analyst at|tech conference|hosted by)\b", re.I)
SALES_RE = re.compile(r"\bmonthly sales\b", re.I)


@dataclass
class IdentityRow:
    call_id: str
    year: int
    ticker: str  # "" when unresolved
    cik: str
    company: str
    score: int
    runner_up: str
    runner_up_score: int
    date: str  # ISO yyyy-mm-dd, "" when unknown
    date_source: str  # pdf_page1 | pdf_created | ""
    call_type: str  # earnings | ma | conference | sales | unknown
    flags: str  # ;-separated quality flags


def norm_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"['’]s?\b", "", name)  # possessives before punctuation strip
    name = re.sub(r"[^a-z0-9 ]", " ", name.lower())
    name = LEGAL_SUFFIXES.sub(" ", name)
    return re.sub(r"\s+", " ", name).strip()


def clean_candidate(name: str) -> str:
    return EVENT_TAIL.sub("", name).strip(" ,.-|")


def load_sec_table(data_root: Path) -> dict[str, tuple[str, str]]:
    """normalized name -> (ticker, cik). Cached under data/raw/ref/, idempotent."""
    cache = data_root / "raw" / "ref" / "company_tickers.json"
    _download(SEC_TICKERS_URL, cache)
    table: dict[str, tuple[str, str]] = {}
    for row in json.loads(cache.read_text(encoding="utf-8")).values():
        key = norm_name(row["title"])
        if key and key not in table:  # ordered by market cap; first entry wins
            table[key] = (row["ticker"], str(row["cik_str"]))
    return table


def phrase_mentions(text: str, sec: dict[str, tuple[str, str]]) -> Counter:
    """Count SEC-table matches over contiguous sub-spans of capitalized runs."""
    counts: Counter = Counter()
    for run in CAP_RUN_RE.findall(text):
        tokens = run.split()
        if not 1 <= len(tokens) <= 12:
            continue
        spans = set()
        for i in range(len(tokens)):
            for j in range(i + 1, min(i + 7, len(tokens) + 1)):
                key = norm_name(" ".join(tokens[i:j]))
                if key in sec and key not in spans:
                    spans.add(key)
        for key in spans:
            # single-word names ("Target", "Gap") are weaker evidence per mention
            counts[sec[key][0]] += 1 if " " in key else 0.5
    return counts


def resolve_company(
    transcript: str,
    pdf_company: str | None,
    sec: dict[str, tuple[str, str]],
) -> tuple[str, str, int, str, int, list[str]]:
    """Score evidence; return (ticker, cik, score, runner_up, runner_score, flags)."""
    scores: Counter = Counter()
    flags: list[str] = []

    scores.update(phrase_mentions(transcript, sec))

    for source, boost, flag in (
        (pdf_company, 5, "pdf_company"),
        (_greeting_name(transcript), 3, "greeting"),
    ):
        if source:
            key = norm_name(clean_candidate(source))
            if key in sec:
                scores[sec[key][0]] += boost
                flags.append(f"match:{flag}")

    if not scores:
        return "", "", 0, "", 0, ["unresolved:no_candidates"]
    ranked = scores.most_common(2)
    top, top_score = ranked[0]
    second, second_score = ranked[1] if len(ranked) > 1 else ("", 0)
    if top_score < 3:
        return "", "", int(top_score), top, int(second_score), ["unresolved:weak_evidence"]
    if second_score * 2 > top_score:
        return "", "", int(top_score), top, int(second_score), ["unresolved:ambiguous"]
    cik = next(c for t, c in sec.values() if t == top)
    return top, cik, int(top_score), second, int(second_score), flags


def _greeting_name(transcript: str) -> str | None:
    m = GREETING_RE.search(transcript[:4000])
    return m.group(1).strip() if m else None


def extract_date(page1_text: str | None, created: str | None) -> tuple[str, str, list[str]]:
    """Pick the call date: title-page date first, creation stamp as fallback."""
    page1_date = ""
    if page1_text and (m := DATE_RE.search(page1_text)):
        month, day, year = m.groups()
        page1_date = f"{year}-{MONTH_NUM[month]:02d}-{int(day):02d}"
    if page1_date and created:
        gap = abs(_ordinal(page1_date) - _ordinal(created))
        flags = ["date_disagreement"] if gap > 45 else []
        return page1_date, "pdf_page1", flags
    if page1_date:
        return page1_date, "pdf_page1", []
    if created:
        return created, "pdf_created", ["date_from_creation_stamp"]
    return "", "", ["no_date"]


def _ordinal(iso: str) -> int:
    y, m, d = map(int, iso.split("-"))
    return y * 372 + m * 31 + d  # rough but monotonic; only used for gap checks


def classify_call(transcript_head: str, page1_text: str | None) -> str:
    blob = f"{transcript_head[:1500]} {page1_text or ''}"
    if SALES_RE.search(blob):
        return "sales"
    if EARNINGS_RE.search(blob):
        return "earnings"
    if MA_RE.search(blob):
        return "ma"
    if CONFERENCE_RE.search(blob):
        return "conference"
    return "unknown"


def _pdf_signals(pdf_path: Path) -> tuple[str | None, str | None, str | None]:
    """(company_meta, page1_text, created_iso) — tolerates malformed PDFs."""
    if not pdf_path.exists():
        return None, None, None
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(pdf_path))
        meta = reader.metadata or {}
        company = str(meta.get("/Company") or "") or None
        created = None
        if m := re.match(r"D:(\d{4})(\d\d)(\d\d)", str(meta.get("/CreationDate") or "")):
            created = "-".join(m.groups())
        page1 = (reader.pages[0].extract_text() or "")[:1500] or None
        return company, page1, created
    except Exception:
        return None, None, None


def build_identity(data_root: Path) -> tuple[Path, dict[str, int]]:
    """Resolve every FinCall call; write the committed identity CSV + return stats."""
    raw = data_root / "raw" / "fincall"
    sec = load_sec_table(data_root)
    rows: list[IdentityRow] = []
    for year in (2019, 2020, 2021):
        data = json.loads((raw / f"transcripts_{year}.json").read_text(encoding="utf-8"))
        for call_id, rec in sorted(data.items()):
            ppt_id = str(rec.get("ppt_id") or call_id)
            company_meta, page1, created = _pdf_signals(raw / f"ppt_{year}" / f"{ppt_id}.pdf")
            ticker, cik, score, runner, runner_score, flags = resolve_company(
                rec["input"], company_meta, sec
            )
            date, date_source, date_flags = extract_date(page1, created)
            rows.append(
                IdentityRow(
                    call_id=call_id,
                    year=year,
                    ticker=ticker,
                    cik=cik,
                    company=company_meta or "",
                    score=score,
                    runner_up=runner,
                    runner_up_score=runner_score,
                    date=date,
                    date_source=date_source,
                    call_type=classify_call(rec["input"], page1),
                    flags=";".join(flags + date_flags),
                )
            )

    out = data_root / "identity" / "fincall_identity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r.year, r.call_id))
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)

    stats = {
        "calls": len(rows),
        "resolved": sum(1 for r in rows if r.ticker),
        "dated": sum(1 for r in rows if r.date),
        "earnings": sum(1 for r in rows if r.call_type == "earnings"),
        "non_earnings": sum(1 for r in rows if r.call_type not in ("earnings", "unknown")),
        "unknown_type": sum(1 for r in rows if r.call_type == "unknown"),
    }
    return out, stats
