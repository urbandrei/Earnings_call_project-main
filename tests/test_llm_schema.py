"""T6.1 LLM feature schema, prompt drafts, and reading-pack tooling.

The acceptance gate for T6.1/T6.2 is human (two-pass schema agreement, then κ>0.6 on a
50-call audit — HANDOFF.md). These tests lock the engineering scaffolding that gate runs on:
the schema bounds, the per-section field applicability, prompt versioning, and — critically —
the train-only leakage guard on the reading-pack sampler.
"""

import pandas as pd
import pytest
from pydantic import ValidationError

from ecvol.features.llm import prompts as P
from ecvol.features.llm import schema as S
from ecvol.features.llm.reading import build_reading_pack

# --- schema ------------------------------------------------------------------


def test_schema_accepts_valid_and_rejects_out_of_range():
    ok = S.SectionFeatures(
        guidance_direction="raise",
        hedging_intensity=2,
        qa_evasiveness=3,
        surprise_mentions=1,
        analyst_tone=4,
        evidence="…",
    )
    assert ok.hedging_intensity == 2
    for bad in ({"hedging_intensity": 5}, {"qa_evasiveness": -1}, {"surprise_mentions": 21}):
        kw = dict(
            guidance_direction="none",
            hedging_intensity=0,
            qa_evasiveness=0,
            surprise_mentions=0,
            analyst_tone=0,
        )
        kw.update(bad)
        with pytest.raises(ValidationError):
            S.SectionFeatures(**kw)


def test_schema_rejects_unknown_field_and_bad_category():
    base = dict(
        guidance_direction="none",
        hedging_intensity=0,
        qa_evasiveness=0,
        surprise_mentions=0,
        analyst_tone=0,
    )
    with pytest.raises(ValidationError):
        S.SectionFeatures(**base, hallucinated=1)
    with pytest.raises(ValidationError):
        S.SectionFeatures(**{**base, "guidance_direction": "up"})


def test_applicable_fields_qa_only_excluded_in_prepared():
    prep = S.applicable_fields("prepared_remarks")
    qa = S.applicable_fields("qa")
    assert "qa_evasiveness" not in prep and "analyst_tone" not in prep
    assert set(qa) == set(S.LABEL_FIELDS)
    with pytest.raises(ValueError):
        S.applicable_fields("nonsense")


# --- prompts -----------------------------------------------------------------


def test_prompt_version_is_pinned():
    assert isinstance(P.PROMPT_VERSION, str) and P.PROMPT_VERSION


def test_prepared_prompt_omits_qa_only_anchors_and_flags_na():
    qa = P.build_user_prompt("qa", "some q and a text")
    prep = P.build_user_prompt("prepared_remarks", "scripted text")
    # Q&A-only rubric anchors appear in the Q&A prompt, not the prepared-remarks one.
    assert "dodge the question" in qa and "aggregate stance" in qa
    assert "dodge the question" not in prep and "aggregate stance" not in prep
    # but the prepared-remarks prompt still instructs the model to floor those fields to 0
    assert "set qa_evasiveness and analyst_tone to 0" in prep
    with pytest.raises(ValueError):
        P.build_user_prompt("nope", "x")


# --- reading pack (leakage guard) --------------------------------------------


def _seed_data(root, n_train=8, n_test=4):
    """Minimal calls/chunks/split fixtures: train + test calls, both with two sections."""
    (root / "fincall").mkdir(parents=True)
    (root / "splits").mkdir(parents=True)
    rows, chunk_rows, split_rows = [], [], []
    for i in range(n_train + n_test):
        cid = f"c{i:03d}"
        split = "train" if i < n_train else "test"
        rows.append({"call_id": cid, "ticker": f"T{i}"})
        split_rows.append(
            {"call_id": cid, "ticker": f"T{i}", "as_of": "2020-01-01", "split": split}
        )
        for sec in ("prepared_remarks", "qa"):
            for turn in range(2):
                chunk_rows.append(
                    {
                        "call_id": cid,
                        "source": "fincall",
                        "section": sec,
                        "role": "management" if sec == "prepared_remarks" else "analyst",
                        "turn_idx": turn,
                        "chunk_in_turn": 0,
                        "n_words": 3,
                        "n_chars": 10,
                        "oversize": False,
                        "text": f"{sec} turn {turn}",
                    }
                )
    pd.DataFrame(rows).to_parquet(root / "fincall" / "calls.parquet")
    pd.DataFrame(chunk_rows).to_parquet(root / "fincall" / "chunks.parquet")
    pd.DataFrame(split_rows).to_csv(root / "splits" / "fincall_temporal.csv", index=False)


def test_reading_pack_samples_train_only_and_writes_sheet(tmp_path):
    _seed_data(tmp_path)
    pack = build_reading_pack(tmp_path, "fincall", n=5, seed=0)

    split = pd.read_csv(tmp_path / "splits" / "fincall_temporal.csv", dtype={"call_id": str})
    train_ids = set(split.loc[split["split"] == "train", "call_id"])
    assert set(pack.call_ids) <= train_ids  # leakage guard: never a test call
    assert len(pack.call_ids) == 5

    for cid in pack.call_ids:
        assert (pack.reading_dir / f"{cid}.md").exists()

    # keep_default_na=False so the "NA" marker survives the round-trip (pandas would coerce it)
    sheet = pd.read_csv(pack.sheet_path, keep_default_na=False)
    assert list(sheet.columns) == ["call_id", "ticker", "section", *S.LABEL_FIELDS]
    prep = sheet[sheet["section"] == "prepared_remarks"]
    assert (prep["qa_evasiveness"] == "NA").all() and (prep["analyst_tone"] == "NA").all()
    # applicable cells are left blank ("") for the human to fill
    assert (prep["hedging_intensity"] == "").all()


def test_reading_pack_determinism_and_size_guard(tmp_path):
    _seed_data(tmp_path)
    a = build_reading_pack(tmp_path, "fincall", n=5, seed=0).call_ids
    b = build_reading_pack(tmp_path, "fincall", n=5, seed=0).call_ids
    assert a == b
    with pytest.raises(ValueError):
        build_reading_pack(tmp_path, "fincall", n=99, seed=0)
