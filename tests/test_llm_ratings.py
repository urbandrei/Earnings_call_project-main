"""T6.2 ratings ingest: rater .xlsx → canonical label CSV, with alignment + range guards."""

import csv
import zipfile
from xml.etree import ElementTree as ET

import pytest

from ecvol.features.llm.ratings import OUT_COLUMNS, ingest_ratings

# One header + 2 calls × {Prepared remarks, Q&A}. Q&A-only fields are "NA" in prepared rows.
_HEADER = [
    "Transcript #",
    "call_id",
    "ticker",
    "section",
    "guidance_direction",
    "hedging_intensity",
    "qa_evasiveness",
    "analyst_tone",
    "surprise_mentions",
    "evidence (quote the text)",
]
_DATA = [
    ["1", "100", "AAA", "Prepared remarks", "raise", "1", "NA", "NA", "0", "q1"],
    ["1", "100", "AAA", "Q&A", "maintain", "2", "1", "3", "1", "q2"],
    ["2", "200", "BBB", "Prepared remarks", "none", "0", "NA", "NA", "2", "q3"],
    ["2", "200", "BBB", "Q&A", "lower", "1", "0", "2", "0", "q4"],
]


def _build_xlsx(path, rows, sheet_name="Ratings"):
    """Minimal inline-string .xlsx with a single named worksheet (no sharedStrings)."""
    main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    r_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

    def col_letter(i):
        s, n = "", i
        while True:
            s = chr(ord("A") + n % 26) + s
            n = n // 26 - 1
            if n < 0:
                return s

    sd = ET.Element(f"{{{main}}}worksheet")
    data = ET.SubElement(sd, f"{{{main}}}sheetData")
    for ri, row in enumerate(rows, start=1):
        re_ = ET.SubElement(data, f"{{{main}}}row", r=str(ri))
        for ci, val in enumerate(row):
            c = ET.SubElement(re_, f"{{{main}}}c", r=f"{col_letter(ci)}{ri}", t="inlineStr")
            is_ = ET.SubElement(c, f"{{{main}}}is")
            ET.SubElement(is_, f"{{{main}}}t").text = str(val)
    sheet_xml = ET.tostring(sd, encoding="unicode")

    workbook = (
        f'<?xml version="1.0"?><workbook xmlns="{main}" xmlns:r="{r_ns}">'
        f'<sheets><sheet name="{sheet_name}" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    wb_rels = (
        '<?xml version="1.0"?><Relationships '
        'xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/'
        '2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>'
    )
    content_types = (
        '<?xml version="1.0"?><Types '
        'xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.'
        'relationships+xml"/></Types>'
    )
    root_rels = (
        '<?xml version="1.0"?><Relationships '
        'xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/'
        '2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'
    )
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("xl/workbook.xml", workbook)
        z.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        z.writestr("xl/worksheets/sheet1.xml", '<?xml version="1.0"?>' + sheet_xml)


def _write_reference(path, keys):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["call_id", "ticker", "section"])
        for cid, sec in keys:
            w.writerow([cid, "X", sec])


def test_ingest_roundtrip_matches_schema(tmp_path):
    xlsx = tmp_path / "Ratings.xlsx"
    _build_xlsx(xlsx, [_HEADER, *_DATA])
    out = tmp_path / "labels_rater1.csv"
    res = ingest_ratings(xlsx, out, rater="rater1")
    assert res.n_rows == 4
    assert res.n_calls == 2
    with open(out, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys()) == list(OUT_COLUMNS)  # evidence/Transcript# dropped
    qa = next(r for r in rows if r["call_id"] == "100" and r["section"] == "qa")
    assert qa["guidance_direction"] == "maintain" and qa["analyst_tone"] == "3"
    prep = next(r for r in rows if r["call_id"] == "100" and r["section"] == "prepared_remarks")
    assert prep["qa_evasiveness"] == "NA"  # N/A marker preserved for κ-audit


def test_alignment_against_reference_passes(tmp_path):
    xlsx = tmp_path / "Ratings.xlsx"
    _build_xlsx(xlsx, [_HEADER, *_DATA])
    ref = tmp_path / "ref.csv"
    keys = [(r[1], "prepared_remarks" if r[3].startswith("Prep") else "qa") for r in _DATA]
    _write_reference(ref, keys)
    res = ingest_ratings(xlsx, tmp_path / "out.csv", rater="r1", reference_sheet=ref)
    assert not res.missing and not res.extra


def test_alignment_mismatch_raises(tmp_path):
    xlsx = tmp_path / "Ratings.xlsx"
    _build_xlsx(xlsx, [_HEADER, *_DATA])
    ref = tmp_path / "ref.csv"
    _write_reference(ref, [("100", "prepared_remarks"), ("999", "qa")])  # 999 not rated
    with pytest.raises(ValueError, match="do not match frozen audit sample"):
        ingest_ratings(xlsx, tmp_path / "out.csv", rater="r1", reference_sheet=ref)


def test_out_of_range_ordinal_raises(tmp_path):
    bad = [r[:] for r in _DATA]
    bad[1][5] = "7"  # hedging_intensity 7 > 4
    xlsx = tmp_path / "Ratings.xlsx"
    _build_xlsx(xlsx, [_HEADER, *bad])
    with pytest.raises(ValueError, match="out of 0-4"):
        ingest_ratings(xlsx, tmp_path / "out.csv", rater="r1")


def test_value_in_na_field_raises(tmp_path):
    bad = [r[:] for r in _DATA]
    bad[0][6] = "2"  # qa_evasiveness set in a prepared-remarks row (should be NA)
    xlsx = tmp_path / "Ratings.xlsx"
    _build_xlsx(xlsx, [_HEADER, *bad])
    with pytest.raises(ValueError, match="N/A here"):
        ingest_ratings(xlsx, tmp_path / "out.csv", rater="r1")


def test_unknown_sheet_name_raises(tmp_path):
    xlsx = tmp_path / "Ratings.xlsx"
    _build_xlsx(xlsx, [_HEADER, *_DATA], sheet_name="Ratings")
    with pytest.raises(ValueError, match="not found"):
        ingest_ratings(xlsx, tmp_path / "out.csv", rater="r1", sheet_name="Nope")


# --- rater workbook (the human deliverable) ----------------------------------


def _blank_sheet(tmp_path):
    """A frozen label sheet like `llm-audit-sample` writes: NA marks Q&A-only fields."""
    p = tmp_path / "fincall_llm_label_sheet.csv"
    p.write_text(
        "call_id,ticker,section,guidance_direction,hedging_intensity,qa_evasiveness,"
        "analyst_tone,surprise_mentions\n"
        "c1,AAA,prepared_remarks,,,NA,NA,\n"
        "c1,AAA,qa,,,,,\n"
        "c2,BBB,prepared_remarks,,,NA,NA,\n"
        "c2,BBB,qa,,,,,\n"
        "c3,CCC,qa,,,,,\n",
        encoding="utf-8",
    )
    return p


def test_rating_workbook_round_trips_through_the_real_ingester(tmp_path):
    """The whole point: what we hand a rater must come back through `ingest_ratings`."""
    from ecvol.features.llm.reading import build_rating_workbook

    sheet = _blank_sheet(tmp_path)
    xlsx = tmp_path / "Ratings_2.xlsx"
    res = build_rating_workbook(sheet, xlsx)
    assert res.n_rows == 5 and res.n_calls == 3

    # Fill it the way a rater would, then ingest with the frozen sheet as reference.
    filled = _fill_workbook(xlsx)
    out = ingest_ratings(filled, tmp_path / "labels.csv", rater="rater2", reference_sheet=sheet)
    assert out.n_rows == 5 and out.n_calls == 3 and not out.missing and not out.extra


def test_rating_workbook_uses_human_section_labels(tmp_path):
    """A raw CSV->xlsx export fails ingest: 'prepared_remarks' is not a label it maps."""
    from ecvol.features.llm.reading import build_rating_workbook

    xlsx = tmp_path / "wb.xlsx"
    build_rating_workbook(_blank_sheet(tmp_path), xlsx)
    text = zipfile.ZipFile(xlsx).read("xl/worksheets/sheet1.xml").decode()
    assert "Prepared Remarks" in text and "Q&amp;A" in text
    assert "prepared_remarks" not in text


def test_rating_workbook_preserves_na_markers(tmp_path):
    from ecvol.features.llm.reading import build_rating_workbook

    xlsx = tmp_path / "wb.xlsx"
    build_rating_workbook(_blank_sheet(tmp_path), xlsx)
    rows = _sheet_rows(xlsx)
    prepared = [r for r in rows if r["section"] == "Prepared Remarks"]
    assert prepared and all(r["qa_evasiveness"] == "NA" for r in prepared)


def test_rating_workbook_shuffles_calls_but_keeps_sections_together(tmp_path):
    """Blinding: a re-rate must not walk the original order, but a call's rows stay adjacent."""
    from ecvol.features.llm.reading import build_rating_workbook

    sheet = _blank_sheet(tmp_path)
    xlsx = tmp_path / "wb.xlsx"
    build_rating_workbook(sheet, xlsx, seed=3)
    order = [r["call_id"] for r in _sheet_rows(xlsx)]
    # each call's rows are contiguous
    for cid in set(order):
        idx = [i for i, c in enumerate(order) if c == cid]
        assert idx == list(range(idx[0], idx[0] + len(idx)))
    # deterministic for a given seed
    xlsx2 = tmp_path / "wb2.xlsx"
    build_rating_workbook(sheet, xlsx2, seed=3)
    assert [r["call_id"] for r in _sheet_rows(xlsx2)] == order


def test_rating_workbook_subset_and_partial_ingest(tmp_path):
    """A blinded partial re-rate must be ingestable — but only when asked for explicitly."""
    from ecvol.features.llm.reading import build_rating_workbook

    sheet = _blank_sheet(tmp_path)
    xlsx = tmp_path / "partial.xlsx"
    res = build_rating_workbook(sheet, xlsx, n_calls=2)
    assert res.n_calls == 2
    filled = _fill_workbook(xlsx)

    with pytest.raises(ValueError, match="allow_subset"):
        ingest_ratings(filled, tmp_path / "a.csv", rater="r2", reference_sheet=sheet)

    out = ingest_ratings(
        filled, tmp_path / "b.csv", rater="r2", reference_sheet=sheet, allow_subset=True
    )
    assert out.n_calls == 2 and out.missing and not out.extra


def test_ingest_still_rejects_extra_rows_even_with_allow_subset(tmp_path):
    """A row outside the frozen sample scores something never sampled — always fatal."""
    from ecvol.features.llm.reading import build_rating_workbook

    sheet = _blank_sheet(tmp_path)
    xlsx = tmp_path / "wb.xlsx"
    build_rating_workbook(sheet, xlsx)
    filled = _fill_workbook(xlsx, extra_call="c9")
    with pytest.raises(ValueError, match="extra"):
        ingest_ratings(
            filled, tmp_path / "c.csv", rater="r2", reference_sheet=sheet, allow_subset=True
        )


_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _sheet_rows(xlsx):
    """Header-keyed rows from a generated workbook (cells matched by column letter, as the
    real reader does — the writer omits empty cells, so position alone would misalign)."""
    root = ET.fromstring(zipfile.ZipFile(xlsx).read("xl/worksheets/sheet1.xml"))
    header, out = None, []
    for row in root.iter(_MAIN + "row"):
        by_col = {
            "".join(ch for ch in c.get("r") if ch.isalpha()): "".join(
                t.text or "" for t in c.iter(_MAIN + "t")
            )
            for c in row.findall(_MAIN + "c")
        }
        if header is None:
            header = {col: v for col, v in by_col.items() if v.strip()}
            continue
        out.append({h: by_col.get(col, "") for col, h in header.items()})
    return out


def _fill_workbook(xlsx, extra_call=None):
    """Fill a blank workbook the way a rater would, keeping NA where the field doesn't apply."""
    from ecvol.features.llm.reading import _write_xlsx

    header = [
        "call_id",
        "ticker",
        "section",
        "guidance_direction",
        "hedging_intensity",
        "qa_evasiveness",
        "analyst_tone",
        "surprise_mentions",
    ]
    values = {
        "guidance_direction": "maintain",
        "hedging_intensity": "1",
        "qa_evasiveness": "1",
        "analyst_tone": "2",
        "surprise_mentions": "0",
    }
    data = [header]
    for r in _sheet_rows(xlsx):
        row = dict(r)
        for field, val in values.items():
            if row.get(field) != "NA":
                row[field] = val
        data.append([row[h] for h in header])
    if extra_call:
        data.append([extra_call, "ZZZ", "Q&A", "none", "0", "0", "2", "0"])
    out = xlsx.with_name(xlsx.stem + "_filled.xlsx")
    _write_xlsx(out, "Ratings", data)
    return out
