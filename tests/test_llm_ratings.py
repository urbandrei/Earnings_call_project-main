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
