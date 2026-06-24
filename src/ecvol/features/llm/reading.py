"""Reading-pack + labeling-sheet tooling for the T6.1 human schema pass (and T6.2 audit).

Produces the materials a human needs to (a) read N calls to validate/refine the feature
schema (T6.1) and (b) fill the rubric labels that the κ-audit scores model output against
(T6.2). The agent builds the tooling; the *numbers* (labels, agreement) are the human's.

**Leakage guard (mirrors TASKS.md TX1):** the sample is drawn from the **train split only**,
asserted at build time — no val/test/embargo call may inform the schema or taxonomy.

Outputs:
- ``data/{dataset}/llm_reading/{call_id}.md`` — rendered transcript (gitignored payload).
- ``data/coverage/{dataset}_llm_label_sheet.csv`` — one row per ``call × section`` with the
  applicable rubric-field columns blank for the human to fill (committed; no transcript text).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ecvol.features.llm.schema import LABEL_FIELDS, SECTIONS, applicable_fields

_SECTION_ORDER = {"prepared_remarks": 0, "qa": 1}
_SECTION_TITLE = {"prepared_remarks": "Prepared remarks", "qa": "Q&A"}
_NA = "NA"  # marks a field that does not apply to a section (Q&A-only field in prepared remarks)


@dataclass
class ReadingPack:
    dataset: str
    call_ids: list[str]
    reading_dir: Path
    sheet_path: Path


def _train_call_ids(root: Path, dataset: str) -> set[str]:
    """Call_ids in the train segment of the temporal split (the leakage-safe pool)."""
    split = pd.read_csv(root / "splits" / f"{dataset}_temporal.csv", dtype={"call_id": str})
    return set(split.loc[split["split"] == "train", "call_id"])


def _render_call(chunks: pd.DataFrame) -> str:
    """Markdown transcript for one call: sections in order, one block per speaker turn."""
    lines: list[str] = []
    for section in sorted(chunks["section"].unique(), key=lambda s: _SECTION_ORDER.get(s, 9)):
        lines.append(f"\n## {_SECTION_TITLE.get(section, section)}\n")
        sec = chunks[chunks["section"] == section].sort_values(["turn_idx", "chunk_in_turn"])
        for _turn_idx, turn in sec.groupby("turn_idx", sort=True):
            role = str(turn["role"].iloc[0])
            text = " ".join(str(t) for t in turn["text"])
            lines.append(f"**[{role}]** {text}\n")
    return "\n".join(lines)


def build_reading_pack(
    root: str | Path = "data",
    dataset: str = "fincall",
    n: int = 20,
    seed: int = 0,
) -> ReadingPack:
    """Sample ``n`` train-split calls, render transcripts, and emit a blank labeling sheet."""
    root = Path(root)
    calls = pd.read_parquet(root / dataset / "calls.parquet")
    chunks = pd.read_parquet(root / dataset / "chunks.parquet")

    train_ids = _train_call_ids(root, dataset)
    have_chunks = set(chunks["call_id"].astype(str))
    candidates = sorted(train_ids & have_chunks)
    if len(candidates) < n:
        raise ValueError(f"{dataset}: only {len(candidates)} train calls with chunks, need {n}")
    picked = sorted(random.Random(seed).sample(candidates, n))

    # Leakage assertion: every picked call must be train-only (never val/test/embargo).
    leaked = set(picked) - train_ids
    if leaked:
        raise AssertionError(f"reading pack leaked non-train calls: {sorted(leaked)}")

    reading_dir = root / dataset / "llm_reading"
    reading_dir.mkdir(parents=True, exist_ok=True)
    ticker_of = dict(zip(calls["call_id"].astype(str), calls["ticker"].astype(str), strict=False))

    sheet_rows: list[dict[str, str]] = []
    for call_id in picked:
        cc = chunks[chunks["call_id"].astype(str) == call_id]
        body = _render_call(cc)
        (reading_dir / f"{call_id}.md").write_text(
            f"# {call_id} ({ticker_of.get(call_id, '?')})\n{body}\n", encoding="utf-8"
        )
        for section in SECTIONS:
            if section not in set(cc["section"]):
                continue
            applicable = set(applicable_fields(section))
            row = {"call_id": call_id, "ticker": ticker_of.get(call_id, ""), "section": section}
            for field in LABEL_FIELDS:
                row[field] = "" if field in applicable else _NA
            sheet_rows.append(row)

    # "NA" marks Q&A-only fields in prepared remarks; readers must use keep_default_na=False
    # (else pandas coerces "NA" to NaN, losing the not-applicable marker).
    sheet = pd.DataFrame(sheet_rows, columns=["call_id", "ticker", "section", *LABEL_FIELDS])
    coverage = root / "coverage"
    coverage.mkdir(parents=True, exist_ok=True)
    sheet_path = coverage / f"{dataset}_llm_label_sheet.csv"
    sheet.to_csv(sheet_path, index=False)

    return ReadingPack(dataset, picked, reading_dir, sheet_path)
