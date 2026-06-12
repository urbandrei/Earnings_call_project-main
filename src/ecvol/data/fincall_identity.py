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
# SEC requires a descriptive User-Agent with contact info on programmatic requests.
SEC_HEADERS = {"User-Agent": "ecvol research project (andrei.roman.personal@gmail.com)"}

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
MONTH_NUM = {m: i + 1 for i, m in enumerate(MONTHS.split("|"))}
DATE_RE = re.compile(rf"({MONTHS})\s+(\d{{1,2}})\s*,?\s*(20\d\d)")

# Host/greeting patterns naming the company near the start of the call, with the
# evidence weight each carries. The name capture must start uppercase; EVENT_TAIL
# cleanup strips trailing event words. Primary "welcome to" forms outweigh host/IR
# forms, which can name a sell-side host instead of the issuer.
GREETING_PATTERNS: list[tuple[re.Pattern, int]] = [
    # "Welcome to/joined the <name> First Quarter 2019 Earnings ..."
    (
        re.compile(
            r"(?:welcome,? (?:everyone,? )?(?:to )?|you've joined "
            r"|join(?:ing)? us (?:today )?(?:for|on) )(?:the )?"
            r"([A-Z].{2,60}?)(?:'s)?[ ,.]+"
            r"(?:first|second|third|fourth|[1-4]Q|Q[1-4]|fiscal|full[- ]year"
            r"|\d{4}|earnings|year[- ]end|annual|quarterly|special)",
            re.I,
        ),
        6,
    ),
    # "Welcome to the Fourth Quarter 2018 <name> Earnings Conference Call"
    # (greedy prefix anchors the capture to just before the event words)
    (
        re.compile(
            r"welcome,? (?:everyone,? )?(?:back )?to .{0,60}"
            r"(?:quarter|20\d\d|'\d\d|fiscal \d{4}|year[- ]end)\s+"
            r"([A-Z].{2,50}?)\s+(?:earnings|results|financial)",
            re.I,
        ),
        6,
    ),
    # "... Conference Call hosted by <name>." / "Teleconference for <name>."
    (
        re.compile(
            r"(?:hosted by|teleconference (?:for|of)|conference call (?:for|of))\s+"
            r"(?:the )?([A-Z][\w&.,' -]{2,60}?)\s*[.,;]",
        ),
        4,
    ),
    # "Investor Relations for <name>." / "Head of Investor Relations at <name>."
    (
        re.compile(
            r"[Ii]nvestor [Rr]elations (?:for|at|of)\s+(?:the )?([A-Z][\w&.,' -]{2,60}?)\s*[.,;]"
        ),
        4,
    ),
    # "... Earnings Call for <name>."
    (
        re.compile(r"(?:earnings|results) call for\s+(?:the )?([A-Z][\w&.,' -]{2,60}?)\s*[.,;]"),
        4,
    ),
    # "<name>'s Vice President of Investor Relations" — reversed IR form. The
    # possessive is mandatory and commas can't be crossed: the appositive person
    # form ("Zach Moxcey, Senior Vice President of ...") must fail here rather
    # than consume the slot with a person's name or a bare "Senior".
    (
        re.compile(
            r"([A-Z][\w&.' -]{2,60}?)'s (?:Senior |Executive )?"
            r"Vice President of Investor Relations"
        ),
        4,
    ),
    # "Welcome to the <name>, Inc. Conference Call" — no event word between the
    # name and "Conference/Earnings Call" (the dedicated alternative to widening
    # pattern 1, whose non-greedy capture would stop at prose like "today's
    # conference"). Event-word cleanup trims any quarter/year tail the greedy
    # capture drags along.
    (
        re.compile(
            r"welcome to (?:the )?([A-Z].{2,60}?)(?:'s)?[ ,.]+"
            r"(?:conference|earnings|quarterly) call",
            re.I,
        ),
        6,
    ),
]
LEGAL_SUFFIXES = re.compile(
    r"\b(incorporated|inc|corporation|corp|company|companies|co|limited|ltd|plc|llc|lp|"
    r"group|holdings?|the|nv|sa|ag|se)\b\.?"
)
_EVENT_WORDS = (
    rf"(?:first|second|third|fourth|[1-4]Q\d*|Q[1-4]|fiscal|year[- ]?end|full[- ]year|annual|"
    rf"quarter|quarterly|monthly|earnings|results|reports?|investor|overview|review|conference|"
    rf"call|webcast|business update|update|and|20\d\d|'\d\d|\d+(?:st|nd|rd|th)|{MONTHS})"
)
# Leading prose filler the greeting regexes can swallow before the actual name
# ("you to the Adobe...", "today's Bank of America..."); stripped iteratively.
FILLER_HEAD = re.compile(
    r"^(?:(?:you|to|the|this|that|our|your|todays?|thank|thanks|for|joining|all|of|"
    r"everyone|come|into|welcome|and|shareholders|ladies|gentlemen|good|morning|"
    r"afternoon|evening)(?:'s)?[ ,]+)+",
    re.I,
)
# Event words leaking into name candidates, trailing ("<name> Fourth Quarter ...")
# or leading ("Q2 2021 <name>").
EVENT_TAIL = re.compile(rf"\b{_EVENT_WORDS}\b.*$", re.I)
EVENT_HEAD = re.compile(rf"^(?:{_EVENT_WORDS}[ ,]+)+", re.I)
# Maximal runs of capitalized tokens (allowing &) anywhere in the transcript.
CAP_RUN_RE = re.compile(r"(?:[A-Z][\w.\-']*|&)(?:[ \t](?:[A-Z][\w.\-']*|&))*")

EARNINGS_RE = re.compile(
    r"(?:(?:first|second|third|fourth)[ -]quarter|[1-4]Q\d\d|Q[1-4]\b|earnings|quarterly|"
    r"(?:full[- ]year|year[- ]end|annual).{0,30}results)",
    re.I,
)
MA_RE = re.compile(r"\b(merger|acquisition|acquire|combination|transaction|tender offer)\b", re.I)
CONFERENCE_RE = re.compile(r"\b(forum|fireside|analyst at|tech conference|hosted by)\b", re.I)
# Markers strong enough to outrank earnings keywords: sell-side conference
# sessions quote quarters/earnings constantly, but a fireside chat or "our
# next company" introduction is never the issuer's own earnings call.
# ("investor conference call" is ordinary earnings-call phrasing — excluded.)
STRONG_CONF_RE = re.compile(
    # "fireside chat at <venue>" is housekeeping on real earnings calls
    # ("Adrian will be participating in a fireside chat at Cowen's...").
    r"fireside(?! chat at\b)|introduce our next company|analyst at |"
    r"my name is .{3,80}?analyst|"
    r"(?:healthcare|tech(?:nology)?|growth|industrial) conference|investor conference(?! call)",
    re.I,
)
MEETING_RE = re.compile(r"(?:annual|special) meeting of (?:the )?(?:share|stock)holders", re.I)
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
    call_type: str  # earnings | ma | conference | sales | meeting | unknown
    flags: str  # ;-separated quality flags


def norm_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    # Join apostrophe-s rather than stripping it: brand names like Kohl's/McDonald's
    # normalize to the SEC's spelling ("KOHLS CORP"); true possessives are handled
    # by the drop-variant in _lookup_name.
    name = re.sub(r"['’]", "", name)
    name = re.sub(r"/[A-Za-z]{2,3}/?\s*$", " ", name)  # EDGAR state tags: "CORP /MA/"
    name = re.sub(r"[^a-z0-9 ]", " ", name.lower())
    name = LEGAL_SUFFIXES.sub(" ", name)
    # Collapse whitespace *before* the single-letter join: punctuation removal
    # leaves double spaces ("C. H. Robinson" -> "c  h  robinson") that would
    # otherwise block the join and normalize dotted initials inconsistently.
    name = re.sub(r"\s+", " ", name).strip()
    return re.sub(r"\b([a-z]) (?=[a-z]\b)", r"\1", name)  # "u s bancorp" -> "us bancorp"


def _lookup_name(candidate: str, sec: dict[str, tuple[str, str]]) -> tuple[str, str] | None:
    """Greeting/PDF-side lookup: exact, possessive-dropped, then conservative fuzzy."""
    import difflib

    key = norm_name(clean_candidate(candidate))
    if len(key) < 2:
        return None
    if key in sec:
        return sec[key]
    dropped = norm_name(re.sub(r"['’]s\b", "", clean_candidate(candidate)))
    if dropped in sec:
        return sec[dropped]
    # A lone generic token gets no prefix/fuzzy attempt: "Financial" would
    # prefix-match "financial institutions", "Southern" -> "southern copper".
    # (Exact hits above still work — curated overrides may add such keys.)
    if " " not in key and key in GENERIC_TOKENS:
        return None
    # Prefix match: "Capital One" -> "capital one financial". The table preserves
    # the SEC file's market-cap order, so when several keys share the prefix
    # (Honeywell International / Honeywell Aerospace) the largest-cap entry wins —
    # the right bias for an earnings-call corpus. Guarded to substantial keys.
    if len(key) >= 5 or " " in key:
        for table_key, value in sec.items():
            if table_key.startswith(key + " "):
                return value
    # Fuzzy is for inflection-level misses ("Align Technologies" vs the SEC's
    # "Align Technology"); requiring an identical first token keeps it from
    # bridging genuinely different companies. Scan several candidates: the
    # closest match can fail the first-token guard ("palatin technologies")
    # while the right key still clears the cutoff.
    for close in difflib.get_close_matches(key, sec.keys(), n=5, cutoff=0.85):
        if close.split()[0] == key.split()[0]:
            return sec[close]
    return None


# Single tokens that survive suffix-stripping of real SEC titles but are far too
# generic to identify a company on their own ("INTERNATIONAL CORP" -> "international").
GENERIC_TOKENS = frozenset(
    "american international national general first united global standard federal "
    "central southern northern western eastern pacific atlantic continental "
    "industries enterprises resources technologies systems brands partners "
    "capital financial energy services solutions communications".split()
)


def clean_candidate(name: str) -> str:
    name = FILLER_HEAD.sub("", name)
    return EVENT_TAIL.sub("", EVENT_HEAD.sub("", name)).strip(" ,.-|")


def load_sec_table(data_root: Path) -> tuple[dict[str, tuple[str, str]], frozenset[str]]:
    """(normalized name -> (ticker, cik), keys safe for single-token body counts).

    Cached under data/raw/ref/, idempotent. Share classes collapse to one entry
    (first = largest-cap ticker per the file's ordering); keys that reduce to a
    single generic token are dropped entirely. The companion set holds genuine
    single-word company names plus curated override aliases — *not* the derived
    first/last-token brand aliases, which match prose far too easily in body
    text ("Turning" -> Turning Point Brands, "Watson" -> Willis Towers Watson)
    and exist for greeting/PDF lookup only.
    """
    cache = data_root / "raw" / "ref" / "company_tickers.json"
    _download(SEC_TICKERS_URL, cache, headers=SEC_HEADERS)
    table: dict[str, tuple[str, str]] = {}
    first_tokens: Counter = Counter()
    for row in json.loads(cache.read_text(encoding="utf-8")).values():
        key = norm_name(row["title"])
        if not key or (" " not in key and key in GENERIC_TOKENS):
            continue
        if key not in table:  # ordered by market cap; first entry wins
            table[key] = (row["ticker"], str(row["cik_str"]))
            if " " in key:
                first_tokens[key.split()[0]] += 1
    single_ok = {k for k in table if " " not in k}
    # Brand aliases from multi-word titles, usable by greeting/PDF matching only
    # (body counting requires multi-word keys): the first token ("Verizon
    # Communications" -> "verizon") and, more conservatively, the last token
    # ("W.W. Grainger" -> "grainger"), when distinctive and naming exactly one company.
    last_tokens: Counter = Counter(k.split()[-1] for k in table if " " in k)
    for key, value in list(table.items()):
        if " " not in key:
            continue
        first, last = key.split()[0], key.split()[-1]
        if (
            first_tokens[first] == 1
            and len(first) >= 5
            and first not in GENERIC_TOKENS
            and first not in table
        ):
            table[first] = value
        if (
            last_tokens[last] == 1
            and len(last) >= 6
            and last not in GENERIC_TOKENS
            and last not in table
        ):
            table[last] = value
    # Curated override aliases (committed CSV): brand acronyms ("ITW", "Citi")
    # and companies absent from the current registrant table — delisted/renamed
    # since the corpus era (Nordstrom, Kellogg, US Steel). CIKs are verified
    # against EDGAR (notebooks/verify_identity_overrides.py); rows win on key
    # collisions, and multi-word aliases also join body-mention counting.
    overrides = data_root / "identity" / "fincall_name_overrides.csv"
    if overrides.exists():
        with open(overrides, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = norm_name(row["alias"])
                table[key] = (row["ticker"], row["cik"])
                single_ok.add(key)
    return table, frozenset(single_ok)


def phrase_mentions(
    text: str, sec: dict[str, tuple[str, str]], single_ok: frozenset[str] = frozenset()
) -> Counter:
    """Count SEC-table matches (keyed by CIK, so share classes merge) over
    contiguous sub-spans of capitalized runs."""
    counts: Counter = Counter()
    singles: Counter = Counter()
    for run in CAP_RUN_RE.findall(text):
        tokens = run.split()
        if not 1 <= len(tokens) <= 12:
            continue
        # Each starting token contributes at most its longest match, so one
        # occurrence can't double-count via nested sub-spans; separate mentions
        # in the same run (sentences merged by abbreviation dots) still count.
        for i in range(len(tokens)):
            longest = None
            for j in range(i + 1, min(i + 7, len(tokens) + 1)):
                key = norm_name(" ".join(tokens[i:j]))
                if key not in sec:
                    continue
                # Single-token keys ("On", "Bill", "Target") match ordinary
                # prose too easily; only distinctive whole-name keys count,
                # tallied separately so their contribution is capped below.
                if " " in key or (
                    j - i == 1 and key in single_ok and len(key) >= 5 and key not in GENERIC_TOKENS
                ):
                    longest = key
            if longest:
                (counts if " " in longest else singles)[sec[longest][1]] += 1
    # Single-token brand evidence ("Allstate", "Costco") needs recurrence (an
    # issuer's name repeats; a one-off capitalized word is prose) and is capped
    # so a third-party brand can't outvote greetings or trip the margin guard.
    for cik, n in singles.items():
        if n >= 2:
            counts[cik] += min(n, 3)
    return counts


def resolve_company(
    transcript: str,
    pdf_company: str | None,
    sec: dict[str, tuple[str, str]],
    single_ok: frozenset[str] = frozenset(),
) -> tuple[str, str, int, str, int, list[str]]:
    """Score evidence; return (ticker, cik, score, runner_up, runner_score, flags)."""
    scores: Counter = Counter()  # keyed by CIK so share classes can't self-compete
    flags: list[str] = []

    scores.update(phrase_mentions(transcript, sec, single_ok))

    sources: list[tuple[str | None, int, str]] = [(pdf_company, 2, "pdf_company")]
    sources += [(name, weight, f"greeting{i}") for name, weight, i in _greeting_names(transcript)]
    boosted: set[str] = set()
    greet: Counter = Counter()  # greeting/PDF evidence alone, for the tie-break
    for source, boost, flag in sources:
        # One boost per evidence kind, first lookup hit wins.
        if flag not in boosted and source and (hit := _lookup_name(source, sec)):
            boosted.add(flag)
            scores[hit[1]] += boost
            greet[hit[1]] += boost
            flags.append(f"match:{flag}")

    cik_to_ticker = {cik: ticker for ticker, cik in reversed(sec.values())}
    if not scores:
        return "", "", 0, "", 0, ["unresolved:no_candidates"]
    ranked = scores.most_common(2)
    top, top_score = ranked[0]
    second, second_score = ranked[1] if len(ranked) > 1 else ("", 0)
    top_t, second_t = cik_to_ticker[top], cik_to_ticker.get(second, "")
    if top_score < 2:
        return "", "", int(top_score), top_t, int(second_score), ["unresolved:weak_evidence"]
    if top_score - second_score < 2:
        # Greeting dominance: greetings name the host; body mentions name
        # anyone (GE's call discusses Baker Hughes ten times, VF its own
        # spinoff). If one company's greeting evidence is strong and entirely
        # unchallenged by anyone else's, it wins the tie-break.
        g_ranked = greet.most_common(2)
        if (
            g_ranked
            and g_ranked[0][1] >= 6
            and g_ranked[0][1] - (g_ranked[1][1] if len(g_ranked) > 1 else 0) >= 6
        ):
            g_top = g_ranked[0][0]
            g_runner = next((c for c, _ in scores.most_common() if c != g_top), "")
            return (
                cik_to_ticker[g_top],
                g_top,
                int(scores[g_top]),
                cik_to_ticker.get(g_runner, ""),
                int(scores[g_runner]) if g_runner else 0,
                flags + ["greeting_dominant"],
            )
        return "", "", int(top_score), top_t, int(second_score), ["unresolved:ambiguous"]
    return top_t, top, int(top_score), second_t, int(second_score), flags


def _greeting_names(transcript: str) -> list[tuple[str, int, int]]:
    """First surviving (name, weight, pattern_index) capture per pattern.

    Junk captures (a generic operator line cleans down to nothing) are skipped
    so they can't shadow the host's real greeting later in the head — but the
    first capture that *looks* like a company consumes the pattern's slot even
    if the SEC lookup then misses: scanning on until something matches the
    table would let a later junk capture hijack a renamed/delisted issuer.
    """
    head = transcript[:4000]
    names: list[tuple[str, int, int]] = []
    for idx, (pattern, weight) in enumerate(GREETING_PATTERNS):
        for m in pattern.finditer(head):
            name = clean_candidate(m.group(1).strip())
            # The re.I patterns make their [A-Z] anchor case-blind; after
            # cleanup, require a proper-noun shape (second-char rule keeps
            # eBay-style brands) so prose fragments don't reach the lookup.
            if len(name) >= 2 and (name[0].isupper() or name[1].isupper()):
                names.append((name, weight, idx))
                break
    return names


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
    if MEETING_RE.search(blob):
        return "meeting"
    if STRONG_CONF_RE.search(blob):
        return "conference"
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


def _pdf_signals_cached(pdf_path: Path, cache: dict, cache_key: str):
    """PDF parsing dominates build time; signals are immutable, so cache them."""
    if cache_key not in cache:
        cache[cache_key] = list(_pdf_signals(pdf_path))
    return cache[cache_key]


def build_identity(data_root: Path) -> tuple[Path, dict[str, int]]:
    """Resolve every FinCall call; write the committed identity CSV + return stats."""
    raw = data_root / "raw" / "fincall"
    sec, single_ok = load_sec_table(data_root)
    cache_path = data_root / "raw" / "ref" / "fincall_pdf_signals.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    rows: list[IdentityRow] = []
    for year in (2019, 2020, 2021):
        data = json.loads((raw / f"transcripts_{year}.json").read_text(encoding="utf-8"))
        for call_id, rec in sorted(data.items()):
            ppt_id = str(rec.get("ppt_id") or call_id)
            company_meta, page1, created = _pdf_signals_cached(
                raw / f"ppt_{year}" / f"{ppt_id}.pdf", cache, f"{year}/{ppt_id}"
            )
            ticker, cik, score, runner, runner_score, flags = resolve_company(
                rec["input"], company_meta, sec, single_ok
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

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, sort_keys=True), encoding="utf-8")

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
