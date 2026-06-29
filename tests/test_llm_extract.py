"""T6.2 extraction driver: section assembly, resumability, determinism, the train sampler.

These run without a GPU/model by injecting a fake engine (`engine_obj`); the real
transformers/vLLM engines are only constructed when no engine is supplied. The κ content
gate is exercised in test_llm_audit.py.
"""

import pandas as pd
import pytest

from ecvol.features.llm import extract as E
from ecvol.features.llm.reading import sample_train_calls
from ecvol.features.llm.schema import EXTRACTED_FIELDS


class _FakeEngine:
    """Returns a fixed schema-shaped dict; counts calls so we can assert resume skips work."""

    def __init__(self):
        self.calls = 0

    def generate(self, system, user):
        self.calls += 1
        return {
            "guidance_direction": "maintain",
            "hedging_intensity": 1,
            "qa_evasiveness": 0,
            "surprise_mentions": 0,
            "analyst_tone": 2,
            "management_optimism": 3,
            "quantitative_specificity": 2,
            "evidence": "x",
        }


def _seed_data(root, n_train=8, n_test=4):
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


def test_model_slug():
    assert E.model_slug("Qwen/Qwen2.5-7B-Instruct") == "Qwen__Qwen2.5-7B-Instruct"


def test_iter_section_inputs_concatenates_per_section(tmp_path):
    _seed_data(tmp_path)
    got = list(E.iter_section_inputs(tmp_path, "fincall", limit=1))
    sections = {s for _, s, _ in got}
    assert sections == {"prepared_remarks", "qa"}
    cid, _, text = got[0]
    assert isinstance(cid, str)
    assert "turn 0" in text and "turn 1" in text  # both turns concatenated in order


def test_build_llm_writes_and_is_resumable(tmp_path):
    _seed_data(tmp_path)
    fake = _FakeEngine()
    res = E.build_llm(tmp_path, "fincall", model_id="test/m", engine_obj=fake, limit=3)
    # 3 calls × 2 sections = 6 extraction units
    assert res.n_new == 6 and res.n_rows == 6 and fake.calls == 6
    assert res.out_path.exists()

    df = pd.read_parquet(res.out_path)
    expected = {"call_id", "section", "model_id", "prompt_version", *EXTRACTED_FIELDS}
    assert expected.issubset(df.columns)
    # v2 exploratory fields are extracted into the parquet alongside the labeled ones
    assert {"management_optimism", "quantitative_specificity"}.issubset(df.columns)
    assert (df["model_id"] == "test/m").all()
    first_bytes = res.out_path.read_bytes()

    # Resume: same call set → nothing new, no engine calls, byte-identical parquet.
    fake2 = _FakeEngine()
    res2 = E.build_llm(tmp_path, "fincall", model_id="test/m", engine_obj=fake2, limit=3)
    assert res2.n_new == 0 and res2.n_rows == 6 and fake2.calls == 0
    assert res2.out_path.read_bytes() == first_bytes


def test_build_llm_extends_on_more_calls(tmp_path):
    _seed_data(tmp_path)
    E.build_llm(tmp_path, "fincall", model_id="test/m", engine_obj=_FakeEngine(), limit=2)
    fake = _FakeEngine()
    res = E.build_llm(tmp_path, "fincall", model_id="test/m", engine_obj=fake, limit=4)
    assert res.n_new == 4 and res.n_rows == 8 and fake.calls == 4  # only the 2 new calls extracted


def test_total_sections(tmp_path):
    _seed_data(tmp_path)
    # 12 calls (8 train + 4 test) × 2 sections each
    assert E.total_sections(tmp_path, ("fincall",)) == 24


def test_sample_train_calls_train_only_and_deterministic(tmp_path):
    _seed_data(tmp_path)
    a = sample_train_calls(tmp_path, "fincall", 5, 0)
    b = sample_train_calls(tmp_path, "fincall", 5, 0)
    split = pd.read_csv(tmp_path / "splits" / "fincall_temporal.csv", dtype={"call_id": str})
    train_ids = set(split.loc[split["split"] == "train", "call_id"])
    assert a == b and set(a) <= train_ids and len(a) == 5
    with pytest.raises(ValueError):
        sample_train_calls(tmp_path, "fincall", 99, 0)


def test_build_engine_rejects_unknown():
    with pytest.raises(ValueError):
        E.build_engine("x/y", engine="nonsense")
