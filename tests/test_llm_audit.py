"""T6.2 κ-audit: per-field agreement + the κ>0.6 gate, on synthetic label/feature fixtures."""

import pandas as pd

from ecvol.features.llm.audit import compute_kappa, kappa_matrix, passes_gate
from ecvol.features.llm.schema import LABEL_FIELDS

# Eight rows: 4 calls × {prepared_remarks, qa}. Q&A-only fields are "NA" in prepared rows.
_CALLS = [f"c{i}" for i in range(4)]
_GUID = ["raise", "lower", "maintain", "none", "raise", "lower", "maintain", "none"]
_HEDGE = [0, 1, 2, 3, 4, 0, 1, 2]
_SURP = [0, 1, 0, 2, 3, 0, 1, 0]
_EVAS = [0, 1, 2, 3]  # qa rows only
_TONE = [4, 3, 2, 1]  # qa rows only


def _rows():
    rows = []
    qa_i = 0
    for ci, cid in enumerate(_CALLS):
        for sec in ("prepared_remarks", "qa"):
            k = ci * 2 + (0 if sec == "prepared_remarks" else 1)
            evas = _EVAS[qa_i] if sec == "qa" else "NA"
            tone = _TONE[qa_i] if sec == "qa" else "NA"
            if sec == "qa":
                qa_i += 1
            rows.append(
                {
                    "call_id": cid,
                    "ticker": "T",
                    "section": sec,
                    "guidance_direction": _GUID[k],
                    "hedging_intensity": _HEDGE[k],
                    "qa_evasiveness": evas,
                    "analyst_tone": tone,
                    "surprise_mentions": _SURP[k],
                }
            )
    return pd.DataFrame(rows)


def _write_sheet(path):
    _rows().to_csv(path, index=False)


def _write_features(path, mutate=None):
    df = _rows().copy()
    # prepared-remarks Q&A-only cells become the schema N/A floor (0), as the extractor emits
    for col in ("qa_evasiveness", "analyst_tone"):
        df.loc[df["section"] == "prepared_remarks", col] = 0
    df[["qa_evasiveness", "analyst_tone", "hedging_intensity", "surprise_mentions"]] = df[
        ["qa_evasiveness", "analyst_tone", "hedging_intensity", "surprise_mentions"]
    ].astype(int)
    df["model_id"] = "m"
    df["evidence"] = ""
    if mutate is not None:
        mutate(df)
    df.to_parquet(path)


def test_perfect_agreement_all_kappa_one_and_passes(tmp_path):
    _write_sheet(tmp_path / "sheet.csv")
    _write_features(tmp_path / "feat.parquet")
    k = compute_kappa(tmp_path / "sheet.csv", tmp_path / "feat.parquet")
    for field in LABEL_FIELDS:
        assert k[field]["kappa"] == 1.0, field
    assert passes_gate(k)


def test_disagreement_fails_gate(tmp_path):
    _write_sheet(tmp_path / "sheet.csv")
    # model predicts a constant guidance → ~0 agreement on the categorical field
    _write_features(
        tmp_path / "feat.parquet",
        mutate=lambda df: df.__setitem__("guidance_direction", "raise"),
    )
    k = compute_kappa(tmp_path / "sheet.csv", tmp_path / "feat.parquet")
    assert k["guidance_direction"]["kappa"] < 0.6
    assert not passes_gate(k)


def test_gate_ignores_weak_reported_fields_outside_confirmatory_core(tmp_path):
    # analyst_tone (a reported, non-confirmatory field) disagreeing must NOT fail the gate
    # as long as the confirmatory core (guidance/hedging/surprise) still clears κ>0.6.
    _write_sheet(tmp_path / "sheet.csv")
    _write_features(
        tmp_path / "feat.parquet",
        mutate=lambda df: df.__setitem__("analyst_tone", 0),  # wreck a reported field only
    )
    k = compute_kappa(tmp_path / "sheet.csv", tmp_path / "feat.parquet")
    assert k["analyst_tone"]["kappa"] < 0.6  # the weak field is scored + reported...
    assert passes_gate(k)  # ...but the confirmatory core still passes the gate


def test_qa_only_fields_scored_on_qa_rows_only(tmp_path):
    _write_sheet(tmp_path / "sheet.csv")
    _write_features(tmp_path / "feat.parquet")
    k = compute_kappa(tmp_path / "sheet.csv", tmp_path / "feat.parquet")
    # 4 qa rows carry evasiveness/analyst_tone; prepared "NA"/floor rows are excluded
    assert k["qa_evasiveness"]["n"] == 4
    assert k["analyst_tone"]["n"] == 4
    assert k["guidance_direction"]["n"] == 8  # applies to both sections


def test_kappa_matrix_model_vs_model(tmp_path):
    _write_features(tmp_path / "a.parquet")
    _write_features(tmp_path / "b.parquet")
    mat = kappa_matrix({"a": tmp_path / "a.parquet", "b": tmp_path / "b.parquet"})
    assert ("a", "b") in mat
    assert mat[("a", "b")]["guidance_direction"]["kappa"] == 1.0
