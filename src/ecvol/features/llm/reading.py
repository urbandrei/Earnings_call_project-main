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


def sample_train_calls(root: str | Path, dataset: str, n: int, seed: int) -> list[str]:
    """Deterministic train-split-only sample of ``n`` call_ids that have chunks.

    Shared by the reading pack (T6.1), the audit sample, and the ETA probe (T6.2) so the
    human-labeled calls are exactly the model-extracted ones. **Leakage guard:** every
    returned call_id is asserted to be in the train split (never val/test/embargo).
    """
    root = Path(root)
    chunks = pd.read_parquet(root / dataset / "chunks.parquet")
    train_ids = _train_call_ids(root, dataset)
    have_chunks = set(chunks["call_id"].astype(str))
    candidates = sorted(train_ids & have_chunks)
    if len(candidates) < n:
        raise ValueError(f"{dataset}: only {len(candidates)} train calls with chunks, need {n}")
    picked = sorted(random.Random(seed).sample(candidates, n))
    leaked = set(picked) - train_ids
    if leaked:
        raise AssertionError(f"sample leaked non-train calls: {sorted(leaked)}")
    return picked


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

    picked = sample_train_calls(root, dataset, n, seed)  # train-only, leakage-asserted

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


# --- rater workbook ----------------------------------------------------------

# Human-facing section labels. `ratings.py` lowercases and maps these back; it does NOT accept
# the canonical `prepared_remarks` id, so a raw CSV→xlsx export fails on the first row. Emitting
# the workbook here closes that write/read asymmetry (rater 1's book had to be made by hand).
_SECTION_LABEL_OUT = {"prepared_remarks": "Prepared Remarks", "qa": "Q&A"}

_XLSX_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_XML_DECL = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
_XLSX_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


@dataclass
class RatingWorkbook:
    out_path: Path
    n_rows: int
    n_calls: int
    sheet_name: str


def _xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _col_ref(idx: int) -> str:
    """0-based column index → spreadsheet letters (0→A, 26→AA)."""
    letters = ""
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _sheet_xml(rows: list[list[str]]) -> str:
    """Worksheet XML with every cell as an inline string (no sharedStrings part needed)."""
    out = [f'{_XML_DECL}<worksheet xmlns="{_XLSX_MAIN}"><sheetData>']
    for r, row in enumerate(rows, start=1):
        cells = "".join(
            f'<c r="{_col_ref(c)}{r}" t="inlineStr"><is><t xml:space="preserve">'
            f"{_xml_escape(val)}</t></is></c>"
            for c, val in enumerate(row)
            if val != ""
        )
        out.append(f'<row r="{r}">{cells}</row>')
    out.append("</sheetData></worksheet>")
    return "".join(out)


def _write_xlsx(path: Path, sheet_name: str, rows: list[list[str]]) -> None:
    """Write a minimal single-sheet .xlsx readable by `ratings.ingest_ratings` and Excel."""
    import zipfile

    pkg_rel = "http://schemas.openxmlformats.org/package/2006/relationships"
    ct = "http://schemas.openxmlformats.org/package/2006/content-types"
    doc = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
    ws = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"
    sml = "application/vnd.openxmlformats-officedocument.spreadsheetml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            "[Content_Types].xml",
            f'{_XML_DECL}<Types xmlns="{ct}">'
            '<Default Extension="rels" ContentType="'
            'application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            f'<Override PartName="/xl/workbook.xml" ContentType="{sml}.sheet.main+xml"/>'
            f'<Override PartName="/xl/worksheets/sheet1.xml" ContentType="{sml}.worksheet+xml"/>'
            "</Types>",
        )
        z.writestr(
            "_rels/.rels",
            f'{_XML_DECL}<Relationships xmlns="{pkg_rel}">'
            f'<Relationship Id="rId1" Type="{doc}" Target="xl/workbook.xml"/></Relationships>',
        )
        z.writestr(
            "xl/workbook.xml",
            f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<workbook xmlns="{_XLSX_MAIN}" xmlns:r="{_XLSX_REL}"><sheets>'
            f'<sheet name="{_xml_escape(sheet_name)}" sheetId="1" r:id="rId1"/>'
            "</sheets></workbook>",
        )
        z.writestr(
            "xl/_rels/workbook.xml.rels",
            f'{_XML_DECL}<Relationships xmlns="{pkg_rel}">'
            f'<Relationship Id="rId1" Type="{ws}" Target="worksheets/sheet1.xml"/></Relationships>',
        )
        z.writestr("xl/worksheets/sheet1.xml", _sheet_xml(rows))


def build_rating_workbook(
    sheet_csv: str | Path,
    out_xlsx: str | Path,
    *,
    n_calls: int | None = None,
    seed: int = 0,
    shuffle: bool = True,
    sheet_name: str = "Ratings",
) -> RatingWorkbook:
    """Blank rater workbook built from the frozen label sheet — the deliverable a rater fills.

    Always derived from ``{dataset}_llm_label_sheet.csv`` so the rated rows cannot drift from
    the frozen audit sample. ``n_calls`` takes a deterministic subset (a blinded *partial*
    re-rate is a legitimate ceiling estimate; `ingest_ratings(allow_subset=True)` accepts it).
    ``shuffle`` randomises **call order** while keeping a call's sections together and ordered —
    a second pass by the same rater must not walk the original sequence, or the agreement number
    measures recall rather than reliability.
    """
    sheet = pd.read_csv(sheet_csv, keep_default_na=False, dtype=str)
    call_ids = list(dict.fromkeys(sheet["call_id"]))
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(call_ids)
    if n_calls is not None:
        if n_calls > len(call_ids):
            raise ValueError(f"n_calls {n_calls} > {len(call_ids)} calls in {sheet_csv}")
        call_ids = call_ids[:n_calls]

    order = {cid: i for i, cid in enumerate(call_ids)}
    sel = sheet[sheet["call_id"].isin(order)].copy()
    sel["_call"] = sel["call_id"].map(order)
    sel["_sec"] = sel["section"].map(_SECTION_ORDER)
    sel = sel.sort_values(["_call", "_sec"])

    header = ["call_id", "ticker", "section", *LABEL_FIELDS]
    rows = [header]
    for rec in sel.to_dict("records"):
        rows.append(
            [rec["call_id"], rec["ticker"], _SECTION_LABEL_OUT[rec["section"]]]
            + [rec[f] for f in LABEL_FIELDS]
        )
    out_xlsx = Path(out_xlsx)
    _write_xlsx(out_xlsx, sheet_name, rows)
    return RatingWorkbook(out_xlsx, len(rows) - 1, len(call_ids), sheet_name)
