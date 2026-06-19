"""Transcript normalization: prepared-remarks vs Q&A sectioning + speaker-turn chunking (T3.1).

Turns the ingested ``transcript_json`` (a JSON list of ``{role, text}``) into per-call
structure used by every downstream text feature (T3.2+):

- a **prepared-remarks -> Q&A boundary**, detected with a deterministic heuristic — the
  operator turn carrying a Q&A cue ("question-and-answer session", "first question", ...),
  or, for FinCall, the first analyst turn (analysts speak only in Q&A). No model, no GPU,
  auditable (user decision 2026-06-18).
- **speaker-turn chunks** that NEVER split a turn across chunks (DECISIONS 2026-06-14): one
  chunk per turn, and a turn longer than ``max_words`` is split at sentence boundaries into
  sub-chunks carrying the same ``turn_idx`` (no content lost; T3.2 section-pools them back).

FinCall carries real roles (management/analyst/operator/...). MAEC has none — every turn is a
sentence with role ``"unknown"`` — so its boundary is detected from in-text cues only and roles
stay unavailable (best-effort, secondary dataset; user decision 2026-06-18).

Output per dataset: ``data/{dataset}/chunks.parquet`` (gitignored deterministic payload, one
row per chunk) + a committed manifest, plus committed coverage reports under ``data/coverage/``:
a ``{dataset}_sections.csv`` summary and a seeded ``{dataset}_section_audit.csv`` for the
human precision check (the >90% acceptance gate is human-verified — see HANDOFF.md).
"""

from __future__ import annotations

import csv
import io
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ecvol.data.manifests import make_entry, write_manifest

SECTION_PREPARED = "prepared_remarks"
SECTION_QA = "qa"

# Conservative word cap so a chunk stays under the 512-token window of the BGE/GTE family
# named in DESIGN §6 (T3.2). T3.2 re-runs `featurize sections --max-words N` if it pins a
# model with a different context window.
DEFAULT_MAX_WORDS = 320

DATASETS = ("fincall", "maec")

CHUNKS_SOURCE = "derived: ecvol featurize sections (T3.1)"
CHUNKS_LICENSE = "derived"

# Operator/host phrasings that open the Q&A. Matched case-insensitively anywhere in a turn.
_QA_CUE_PATTERNS = [
    r"question[- ]and[- ]answer",
    r"q\s*&\s*a\s+(?:session|portion)",
    r"floor\s+is\s+(?:now\s+)?open\s+for\s+question",
    r"open\s+(?:up\s+)?(?:the\s+\w+\s+)?(?:for|to)\s+question",
    r"(?:our|the)\s+first\s+question",
    r"first\s+question\s+(?:comes|is|will)",
    r"(?:like|want)\s+to\s+(?:open|turn|begin).{0,40}question",
    r"to\s+ask\s+a\s+question",
    r"press\s+star",
    r"\bstar\s+(?:one|1)\b",
    r"operator\s+instructions",
    r"we(?:'ll| will)\s+(?:now\s+)?(?:take|begin).{0,20}question",
]
_QA_CUE = re.compile("|".join(_QA_CUE_PATTERNS), re.IGNORECASE)

# Lightweight sentence splitter (no nltk dep): break after .!? when followed by an opener.
# Finance abbreviations cause minor over-splitting — harmless, as chunks just re-pack sentences.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9$])")

# Operator-cue turns within this many turns before the first analyst are treated as the Q&A
# intro (so the boundary includes the operator's "first question from..."); a cue far earlier
# is an intro-mention false positive and is ignored in favour of the first analyst turn.
_CUE_NEAR_ANALYST = 3


@dataclass
class SectionResult:
    sections: list[str]  # one label per input turn
    boundary_idx: int  # index of the first Q&A turn; == len(turns) when no Q&A
    has_qa: bool
    method: str  # "operator_cue" | "first_analyst" | "text_cue" | "none"
    corroborated: bool  # FinCall: both an operator cue and an analyst turn are present


def detect_sections(turns: list[dict]) -> SectionResult:
    """Label each turn prepared_remarks/qa via the deterministic boundary heuristic."""
    n = len(turns)
    if n == 0:
        return SectionResult([], 0, False, "none", False)
    roles = [t.get("role", "unknown") for t in turns]
    texts = [t.get("text", "") or "" for t in turns]
    has_roles = any(r not in (None, "unknown") for r in roles)
    op_cue_idx = next(
        (
            i
            for i, (r, tx) in enumerate(zip(roles, texts, strict=True))
            if r == "operator" and _QA_CUE.search(tx)
        ),
        None,
    )
    analyst_idx = next((i for i, r in enumerate(roles) if r == "analyst"), None)

    if has_roles and (op_cue_idx is not None or analyst_idx is not None):
        if analyst_idx is not None:
            near = op_cue_idx is not None and 0 <= analyst_idx - op_cue_idx <= _CUE_NEAR_ANALYST
            if near:
                boundary, method = op_cue_idx, "operator_cue"
            else:
                boundary, method = analyst_idx, "first_analyst"
            corroborated = op_cue_idx is not None
        else:
            boundary, method, corroborated = op_cue_idx, "operator_cue", False
        labels = [SECTION_PREPARED if i < boundary else SECTION_QA for i in range(n)]
        return SectionResult(labels, boundary, True, method, corroborated)

    # No roles (MAEC) or no role-based signal: fall back to in-text cues only.
    cue_idx = next((i for i, tx in enumerate(texts) if _QA_CUE.search(tx)), None)
    if cue_idx is not None:
        labels = [SECTION_PREPARED if i < cue_idx else SECTION_QA for i in range(n)]
        return SectionResult(labels, cue_idx, True, "text_cue", False)
    return SectionResult([SECTION_PREPARED] * n, n, False, "none", False)


@dataclass
class Chunk:
    section: str
    role: str
    turn_idx: int
    chunk_in_turn: int
    n_words: int
    n_chars: int
    oversize: bool  # a single sentence exceeded max_words and was kept whole
    text: str


def _words(s: str) -> int:
    return len(s.split())


def _split_sentences(text: str) -> list[str]:
    return [p for p in _SENT_SPLIT.split(text.strip()) if p]


def _mk_chunk(sec: str, role: str, ti: int, cin: int, sents: list[str], max_words: int) -> Chunk:
    text = " ".join(sents)
    nw = _words(text)
    return Chunk(sec, role, ti, cin, nw, len(text), nw > max_words, text)


def chunk_turns(
    turns: list[dict], sections: list[str], *, max_words: int = DEFAULT_MAX_WORDS
) -> list[Chunk]:
    """One chunk per turn; oversized turns split at sentence boundaries (never across turns)."""
    chunks: list[Chunk] = []
    for ti, (turn, sec) in enumerate(zip(turns, sections, strict=True)):
        role = turn.get("role", "unknown")
        text = (turn.get("text") or "").strip()
        if not text:
            continue
        if _words(text) <= max_words:
            chunks.append(Chunk(sec, role, ti, 0, _words(text), len(text), False, text))
            continue
        buf: list[str] = []
        bw = 0
        cin = 0
        for sent in _split_sentences(text) or [text]:
            sw = _words(sent)
            if buf and bw + sw > max_words:
                chunks.append(_mk_chunk(sec, role, ti, cin, buf, max_words))
                cin += 1
                buf, bw = [], 0
            buf.append(sent)
            bw += sw
            if sw > max_words and len(buf) == 1:  # lone over-long sentence: keep whole
                chunks.append(_mk_chunk(sec, role, ti, cin, buf, max_words))
                cin += 1
                buf, bw = [], 0
        if buf:
            chunks.append(_mk_chunk(sec, role, ti, cin, buf, max_words))
    return chunks


def write_chunks_parquet(rows: list[dict], path: Path, *, id_type: pa.DataType) -> None:
    """Deterministic parquet (sorted by call_id, turn_idx, chunk_in_turn) — T0.3 convention."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["call_id"], r["turn_idx"], r["chunk_in_turn"]))
    table = pa.table(
        {
            "call_id": pa.array([r["call_id"] for r in rows], id_type),
            "source": pa.array([r["source"] for r in rows], pa.string()),
            "section": pa.array([r["section"] for r in rows], pa.string()),
            "role": pa.array([r["role"] for r in rows], pa.string()),
            "turn_idx": pa.array([r["turn_idx"] for r in rows], pa.int64()),
            "chunk_in_turn": pa.array([r["chunk_in_turn"] for r in rows], pa.int64()),
            "n_words": pa.array([r["n_words"] for r in rows], pa.int64()),
            "n_chars": pa.array([r["n_chars"] for r in rows], pa.int64()),
            "oversize": pa.array([r["oversize"] for r in rows], pa.bool_()),
            "text": pa.array([r["text"] for r in rows], pa.string()),
        }
    )
    pq.write_table(table, path, compression="none", store_schema=True)


def _write_csv(header: list[str], rows: list[list], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    path.write_text(buf.getvalue(), encoding="utf-8")


def _snippet(text: str, n: int = 160) -> str:
    s = " ".join((text or "").split())
    return s[:n]


@dataclass
class SectionSummary:
    dataset: str
    n_calls: int  # rows in calls.parquet
    n_processed: int  # calls with >=1 turn
    n_no_turns: int
    calls_with_qa: int
    method_counts: dict[str, int]
    corroborated: int
    total_chunks: int
    oversize_chunks: int
    audit_path: str


def build_sections(
    root: Path,
    *,
    max_words: int = DEFAULT_MAX_WORDS,
    audit_n: int = 30,
    seed: int = 0,
) -> list[SectionSummary]:
    """Section + chunk each dataset → chunks.parquet + manifest + coverage reports (T3.1)."""
    summaries: list[SectionSummary] = []
    for dataset in DATASETS:
        calls_path = root / dataset / "calls.parquet"
        if not calls_path.is_file():
            continue
        id_type = pa.int64() if dataset == "fincall" else pa.string()
        df = pd.read_parquet(calls_path)

        chunk_rows: list[dict] = []
        per_call: list[dict] = []
        n_no_turns = 0
        for row in df.itertuples(index=False):
            turns = json.loads(row.transcript_json) if row.transcript_json else []
            if not turns:
                n_no_turns += 1
                continue
            res = detect_sections(turns)
            cks = chunk_turns(turns, res.sections, max_words=max_words)
            for c in cks:
                chunk_rows.append(
                    {
                        "call_id": row.call_id,
                        "source": row.source,
                        "section": c.section,
                        "role": c.role,
                        "turn_idx": c.turn_idx,
                        "chunk_in_turn": c.chunk_in_turn,
                        "n_words": c.n_words,
                        "n_chars": c.n_chars,
                        "oversize": c.oversize,
                        "text": c.text,
                    }
                )
            n_prepared = sum(1 for s in res.sections if s == SECTION_PREPARED)
            b_turn = turns[res.boundary_idx] if res.has_qa else None
            per_call.append(
                {
                    "call_id": row.call_id,
                    "ticker": row.ticker,
                    "n_turns": len(turns),
                    "boundary_idx": res.boundary_idx,
                    "has_qa": res.has_qa,
                    "method": res.method,
                    "corroborated": res.corroborated,
                    "n_prepared_turns": n_prepared,
                    "n_qa_turns": len(turns) - n_prepared,
                    "n_chunks": len(cks),
                    "boundary_role": b_turn["role"] if b_turn else "",
                    "boundary_text": _snippet(b_turn["text"]) if b_turn else "",
                }
            )

        # Payload + manifest.
        chunks_path = root / dataset / "chunks.parquet"
        write_chunks_parquet(chunk_rows, chunks_path, id_type=id_type)
        entry = make_entry(chunks_path, root, source_url=CHUNKS_SOURCE, license=CHUNKS_LICENSE)
        (root / "manifests").mkdir(parents=True, exist_ok=True)
        write_manifest([entry], root / "manifests" / f"{dataset}_chunks.json")

        # Committed coverage summary.
        method_counts: dict[str, int] = {}
        for pc in per_call:
            method_counts[pc["method"]] = method_counts.get(pc["method"], 0) + 1
        calls_with_qa = sum(1 for pc in per_call if pc["has_qa"])
        corroborated = sum(1 for pc in per_call if pc["corroborated"])
        oversize = sum(1 for r in chunk_rows if r["oversize"])
        cov = root / "coverage"
        summary_rows: list[tuple[str, object]] = [
            ("n_calls", len(df)),
            ("n_processed", len(per_call)),
            ("n_no_turns", n_no_turns),
            ("calls_with_qa", calls_with_qa),
            ("corroborated", corroborated),
            ("total_chunks", len(chunk_rows)),
            ("oversize_chunks", oversize),
            ("max_words", max_words),
        ] + [(f"method:{k}", v) for k, v in sorted(method_counts.items())]
        from ecvol.data.calls import write_metric_csv

        write_metric_csv(summary_rows, cov / f"{dataset}_sections.csv")

        # Seeded human-precision audit sample (the >90% gate; verified by hand — HANDOFF.md).
        ordered = sorted(per_call, key=lambda pc: str(pc["call_id"]))
        sample = random.Random(seed).sample(ordered, min(audit_n, len(ordered)))
        sample = sorted(sample, key=lambda pc: str(pc["call_id"]))
        audit_header = [
            "call_id",
            "ticker",
            "n_turns",
            "boundary_idx",
            "method",
            "corroborated",
            "boundary_role",
            "boundary_text",
            "correct_y_n",
        ]
        audit_rows = [
            [
                pc["call_id"],
                pc["ticker"],
                pc["n_turns"],
                pc["boundary_idx"],
                pc["method"],
                pc["corroborated"],
                pc["boundary_role"],
                pc["boundary_text"],
                "",
            ]
            for pc in sample
        ]
        audit_path = cov / f"{dataset}_section_audit.csv"
        _write_csv(audit_header, audit_rows, audit_path)

        summaries.append(
            SectionSummary(
                dataset=dataset,
                n_calls=len(df),
                n_processed=len(per_call),
                n_no_turns=n_no_turns,
                calls_with_qa=calls_with_qa,
                method_counts=method_counts,
                corroborated=corroborated,
                total_chunks=len(chunk_rows),
                oversize_chunks=oversize,
                audit_path=str(audit_path),
            )
        )
    return summaries
