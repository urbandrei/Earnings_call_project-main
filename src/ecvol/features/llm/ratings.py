"""Ingest a human rater's ``.xlsx`` workbook into the canonical κ-audit label sheet (T6.2).

A rater fills the ``Ratings`` worksheet of the workbook handed to them (one row per
``call × section``, schema columns + an evidence quote). This converts that sheet into the
flat CSV that :func:`ecvol.features.llm.audit.compute_kappa` consumes — same columns and
``NA`` markers as the blank sheet from ``ecvol featurize llm-audit-sample``.

Parsing is stdlib-only (``zipfile`` + ``xml``) so reading a rater deliverable adds no
dependency. The ingest is **validated against the frozen audit sample**: the rated
``(call_id, section)`` set must exactly match the reference sheet, and every value must be in
schema range — a misaligned or out-of-range workbook fails loudly rather than silently
scoring the wrong rows.
"""

from __future__ import annotations

import csv
import typing
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

from .schema import LABEL_FIELDS, SECTIONS, GuidanceDirection, applicable_fields

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PKG_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"

# Output schema (matches the blank sheet from build_reading_pack / llm-audit-sample).
OUT_COLUMNS = ("call_id", "ticker", "section", *LABEL_FIELDS)

# Human-facing section labels in the workbook → canonical section ids.
_SECTION_MAP = {
    "prepared remarks": "prepared_remarks",
    "qa": "qa",
    "q&a": "qa",
}

_GUIDANCE_VALUES = set(typing.get_args(GuidanceDirection))


@dataclass
class RatingsResult:
    """Outcome of one workbook ingest."""

    rater: str
    out_path: Path
    n_rows: int
    n_calls: int
    missing: list[tuple[str, str]] = field(default_factory=list)  # in reference, not rated
    extra: list[tuple[str, str]] = field(default_factory=list)  # rated, not in reference


def _read_shared_strings(z: zipfile.ZipFile) -> list[str]:
    try:
        raw = z.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(raw)
    return ["".join(t.text or "" for t in si.iter(_NS + "t")) for si in root.findall(_NS + "si")]


def _ratings_sheet_path(z: zipfile.ZipFile, sheet_name: str) -> str:
    """Resolve the worksheet XML path for the named sheet via the workbook relationships."""
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rid = None
    for s in wb.iter(_NS + "sheet"):
        if s.get("name") == sheet_name:
            rid = s.get(_REL_NS + "id")
            break
    if rid is None:
        names = [s.get("name") for s in wb.iter(_NS + "sheet")]
        raise ValueError(f"worksheet {sheet_name!r} not found; sheets present: {names}")
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    for r in rels.findall(_PKG_REL_NS + "Relationship"):
        if r.get("Id") == rid:
            return "xl/" + r.get("Target").lstrip("/").removeprefix("xl/")
    raise ValueError(f"relationship {rid!r} for sheet {sheet_name!r} not found in workbook rels")


def _col_letters(cell_ref: str) -> str:
    return "".join(ch for ch in cell_ref if ch.isalpha())


def _read_rows(z: zipfile.ZipFile, sheet_path: str, sst: list[str]) -> list[dict[str, str]]:
    """Read the worksheet as header-keyed dict rows (header = first non-empty row).

    Cells are matched to headers by spreadsheet column letter, so a sparse data row (cells
    omitted by the editor) still lands each value under the right header.
    """

    def cell_value(c: ET.Element) -> str:
        if c.get("t") == "inlineStr":  # <is><t>..</t></is> — strings stored in-cell
            return "".join(t.text or "" for t in c.iter(_NS + "t"))
        v = c.find(_NS + "v")
        if v is None or v.text is None:
            return ""
        return sst[int(v.text)] if c.get("t") == "s" else v.text

    sheet = ET.fromstring(z.read(sheet_path))
    data = sheet.find(_NS + "sheetData")
    col_to_header: dict[str, str] | None = None
    out: list[dict[str, str]] = []
    for row in data.findall(_NS + "row"):
        by_col = {_col_letters(c.get("r")): cell_value(c) for c in row.findall(_NS + "c")}
        if not any(str(v).strip() for v in by_col.values()):
            continue
        if col_to_header is None:
            col_to_header = {col: val.strip() for col, val in by_col.items() if val.strip()}
            continue
        record = {header: by_col.get(col, "") for col, header in col_to_header.items()}
        out.append(record)
    return out


def _normalize(record: dict[str, str]) -> dict[str, str]:
    raw_section = str(record.get("section", "")).strip().lower()
    section = _SECTION_MAP.get(raw_section)
    if section is None:
        raise ValueError(f"unknown section label {record.get('section')!r}; expected {SECTIONS}")
    out = {
        "call_id": str(record.get("call_id", "")).strip(),
        "ticker": str(record.get("ticker", "")).strip(),
        "section": section,
    }
    for f in LABEL_FIELDS:
        out[f] = str(record.get(f, "")).strip()
    return out


def _validate_values(rows: list[dict[str, str]]) -> None:
    """Every cell in schema range; ``NA`` only where the field does not apply to the section."""
    for r in rows:
        applies = applicable_fields(r["section"])
        for f in LABEL_FIELDS:
            val = r[f]
            if f not in applies:
                if val not in ("", "NA"):
                    raise ValueError(
                        f"{r['call_id']}/{r['section']}: {f}={val!r} but field is N/A here"
                    )
                continue
            if val in ("", "NA"):
                raise ValueError(f"{r['call_id']}/{r['section']}: {f} is blank/NA but required")
            if f == "guidance_direction":
                if val not in _GUIDANCE_VALUES:
                    raise ValueError(f"{r['call_id']}/{r['section']}: guidance {val!r} invalid")
            elif f == "surprise_mentions":
                if not val.isdigit() or not 0 <= int(val) <= 20:
                    raise ValueError(f"{r['call_id']}/{r['section']}: surprise {val!r} out of 0-20")
            else:  # ordinal 0-4
                if not val.isdigit() or not 0 <= int(val) <= 4:
                    raise ValueError(f"{r['call_id']}/{r['section']}: {f}={val!r} out of 0-4")


def _reference_keys(reference_sheet: str | Path) -> set[tuple[str, str]]:
    with open(reference_sheet, newline="") as fh:
        return {(row["call_id"], row["section"]) for row in csv.DictReader(fh)}


def ingest_ratings(
    xlsx_path: str | Path,
    out_path: str | Path,
    *,
    rater: str,
    reference_sheet: str | Path | None = None,
    sheet_name: str = "Ratings",
) -> RatingsResult:
    """Convert a rater workbook → canonical label CSV, validating against the frozen sample.

    ``reference_sheet`` (the blank sheet from ``llm-audit-sample``) freezes which
    ``(call_id, section)`` rows the audit covers. When given, the rated set must match it
    exactly — missing/extra rows are reported and raise.
    """
    with zipfile.ZipFile(xlsx_path) as z:
        sst = _read_shared_strings(z)
        sheet_path = _ratings_sheet_path(z, sheet_name)
        raw = _read_rows(z, sheet_path, sst)
    rows = [_normalize(r) for r in raw if str(r.get("call_id", "")).strip()]
    if not rows:
        raise ValueError(f"no rated rows found in {sheet_name!r} sheet of {xlsx_path}")
    _validate_values(rows)

    missing: list[tuple[str, str]] = []
    extra: list[tuple[str, str]] = []
    if reference_sheet is not None:
        ref = _reference_keys(reference_sheet)
        rated = {(r["call_id"], r["section"]) for r in rows}
        missing = sorted(ref - rated)
        extra = sorted(rated - ref)
        if missing or extra:
            raise ValueError(
                f"rated rows do not match frozen audit sample: "
                f"{len(missing)} missing, {len(extra)} extra (first missing={missing[:3]}, "
                f"first extra={extra[:3]})"
            )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(OUT_COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in OUT_COLUMNS})

    return RatingsResult(
        rater=rater,
        out_path=out_path,
        n_rows=len(rows),
        n_calls=len({r["call_id"] for r in rows}),
        missing=missing,
        extra=extra,
    )
