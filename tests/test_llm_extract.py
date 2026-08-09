"""T6.2 extraction driver: section assembly, resumability, determinism, the train sampler.

These run without a GPU/model by injecting a fake engine (`engine_obj`); the real
transformers/vLLM engines are only constructed when no engine is supplied. The κ content
gate is exercised in test_llm_audit.py.
"""

import json
import time
from datetime import timedelta

import pandas as pd
import pytest

from ecvol.features.llm import extract as E
from ecvol.features.llm.prompts import PROMPT_VERSION
from ecvol.features.llm.reading import sample_train_calls
from ecvol.features.llm.schema import EXTRACTED_FIELDS, SectionFeatures


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


def test_build_llm_audit_sample_restricts_to_sample(tmp_path):
    # The CLI's --audit-sample path: extract exactly the seeded audit calls (the κ-gate set)
    # through whatever engine runs the corpus (audit-matches-corpus rule), not the first-N calls.
    _seed_data(tmp_path)
    sample = sample_train_calls(tmp_path, "fincall", 3, 0)
    fake = _FakeEngine()
    res = E.build_llm(tmp_path, "fincall", model_id="test/m", engine_obj=fake, call_ids=sample)
    df = pd.read_parquet(res.out_path)
    assert set(df["call_id"]) == set(sample)  # only the sampled calls, both sections each
    assert res.n_new == len(sample) * 2 and fake.calls == len(sample) * 2


def test_featurize_llm_audit_sample_and_limit_mutually_exclusive():
    from typer.testing import CliRunner

    from ecvol.cli import app

    result = CliRunner().invoke(app, ["featurize", "llm", "--audit-sample", "--limit", "3"])
    assert result.exit_code != 0  # BadParameter before any model load


def test_build_engine_rejects_unknown():
    with pytest.raises(ValueError):
        E.build_engine("x/y", engine="nonsense")


def test_yarn_rope_scaling_config():
    # 64k over Qwen2.5's 32k native → factor 2.0, YaRN type, native recorded
    cfg = E.yarn_rope_scaling(65536)
    assert cfg == {
        "rope_type": "yarn",
        "factor": 2.0,
        "original_max_position_embeddings": 32768,
    }
    # not needed when the target is within native context → loud failure, not a silent no-op
    with pytest.raises(ValueError):
        E.yarn_rope_scaling(16384)


# --- llamacpp engine: grammar-safe schema, evidence clipping, deadline stop ---------------


def test_grammar_schema_replaces_int_bounds_with_enums():
    """Bounded ints must become explicit enums (llama.cpp's GBNF converter handles those)."""
    schema = E.grammar_json_schema()
    props = schema["properties"]
    assert props["hedging_intensity"]["enum"] == [0, 1, 2, 3, 4]
    assert props["surprise_mentions"]["enum"] == list(range(21))
    for name in ("hedging_intensity", "surprise_mentions"):
        assert "minimum" not in props[name] and "maximum" not in props[name]
    # The enum must be exactly the pydantic bound — a mismatch would let invalid values decode.
    assert props["guidance_direction"]["enum"] == ["raise", "maintain", "lower", "none"]


def test_grammar_schema_bounds_evidence_with_anchored_pattern():
    """llama.cpp rejects `maxLength` and needs '^...$'; an unbounded string decodes past the
    token budget and yields invalid JSON mid-string, so the bound must survive refactors."""
    ev = E.grammar_json_schema()["properties"]["evidence"]
    assert "maxLength" not in ev
    assert ev["pattern"].startswith("^") and ev["pattern"].endswith("$")
    assert f"{{0,{E.EVIDENCE_GRAMMAR_CHARS}}}" in ev["pattern"]


def test_evidence_grammar_bound_fits_schema_and_token_budget():
    """The grammar cap must stay inside the schema's own limit (else rows fail validation)."""
    schema_max = SectionFeatures.model_fields["evidence"].metadata[0].max_length
    assert E.EVIDENCE_GRAMMAR_CHARS <= schema_max
    # ~4 chars/token for English prose, plus the ratings and JSON scaffolding.
    assert E.EVIDENCE_GRAMMAR_CHARS / 4 + 100 < E.MAX_NEW_TOKENS


def test_grammar_schema_requires_every_field():
    """A defaulted field omitted by the model would silently become a rating (e.g. optimism=0)."""
    schema = E.grammar_json_schema()
    assert set(schema["required"]) == set(schema["properties"])
    for name in ("evidence", "management_optimism", "quantitative_specificity"):
        assert name in schema["required"]


def test_truncate_evidence_clips_to_schema_max():
    clipped = E._truncate_evidence({"evidence": "x" * 5000})
    assert len(clipped["evidence"]) == 2000
    assert E._truncate_evidence({"evidence": "short"})["evidence"] == "short"


def test_build_llm_stops_at_deadline_and_flushes(tmp_path):
    """A past deadline extracts nothing but still leaves a readable, resumable parquet."""
    _seed_data(tmp_path)
    eng = _FakeEngine()
    res = E.build_llm(
        tmp_path,
        "fincall",
        model_id="m",
        engine_obj=eng,
        deadline=time.time() - 1,
    )
    assert eng.calls == 0
    assert res.n_new == 0


def test_build_llm_deadline_none_processes_everything(tmp_path):
    _seed_data(tmp_path)
    eng = _FakeEngine()
    res = E.build_llm(tmp_path, "fincall", model_id="m", engine_obj=eng, deadline=None)
    assert res.n_new > 0


def test_build_engine_knows_llamacpp():
    with pytest.raises(ValueError, match="llamacpp"):
        E.build_engine("m", engine="nope")


def test_parse_stop_at_rolls_to_tomorrow_when_time_already_passed():
    """The overnight run's hard stop must never resolve to a time in the past (= instant exit)."""
    from datetime import datetime

    from ecvol.cli import _parse_stop_at

    now = datetime.now()
    past = (now.replace(microsecond=0) - timedelta(minutes=5)).strftime("%H:%M")
    assert _parse_stop_at(past) > time.time()

    future = (now + timedelta(minutes=30)).strftime("%H:%M")
    delta = _parse_stop_at(future) - time.time()
    assert 0 < delta <= 31 * 60


# --- run provenance sidecar --------------------------------------------------


def test_run_config_captures_decoding_settings():
    """Rows record only model/revision/prompt — the settings that change content live here."""
    cfg = E.run_config(
        "m",
        "rev1",
        "llamacpp",
        {"max_model_len": 65536, "rope_scaling": {"factor": 2.0}, "gguf_path": r"D:\x\W.gguf"},
    )
    assert cfg["max_model_len"] == 65536
    assert cfg["max_new_tokens"] == E.MAX_NEW_TOKENS
    assert cfg["evidence_grammar_chars"] == E.EVIDENCE_GRAMMAR_CHARS
    assert cfg["prompt_version"] == PROMPT_VERSION
    # machine-specific absolute paths must not leak into provenance
    assert cfg["weights_file"] == "W.gguf" and "D:" not in json.dumps(cfg)


def test_config_digest_changes_with_decoding_settings():
    base = E.run_config("m", "r", "llamacpp", {"max_model_len": 65536})
    same = E.run_config("m", "r", "llamacpp", {"max_model_len": 65536})
    diff = E.run_config("m", "r", "llamacpp", {"max_model_len": 32768})
    assert E.config_digest(base) == E.config_digest(same)
    assert E.config_digest(base) != E.config_digest(diff)


def test_sidecar_written_and_appends_across_resumes(tmp_path):
    _seed_data(tmp_path)
    E.build_llm(tmp_path, "fincall", model_id="test/m", engine_obj=_FakeEngine(), limit=2)
    res = E.build_llm(tmp_path, "fincall", model_id="test/m", engine_obj=_FakeEngine(), limit=4)
    side = res.out_path.with_suffix(".run.json")
    runs = json.loads(side.read_text())
    assert len(runs) == 2  # one record per invocation, so the parquet's history is auditable
    assert runs[0]["n_new_rows"] == 4 and runs[1]["n_new_rows"] == 4
    assert runs[-1]["n_rows_after"] == 8
    assert "git" in runs[0] and "env" in runs[0]
    assert E.mixed_config_digests(side) == [runs[0]["config_digest"]]  # one config → one digest


def test_mixed_config_digests_flags_a_parquet_built_under_two_configs(tmp_path):
    """The failure this exists to catch: rows pooled from different decoding settings."""
    _seed_data(tmp_path)
    res = E.build_llm(tmp_path, "fincall", model_id="test/m", engine_obj=_FakeEngine(), limit=2)
    side = res.out_path.with_suffix(".run.json")
    runs = json.loads(side.read_text())
    runs.append({**runs[0], "config_digest": "deadbeef0000"})
    side.write_text(json.dumps(runs))
    assert len(E.mixed_config_digests(side)) == 2
