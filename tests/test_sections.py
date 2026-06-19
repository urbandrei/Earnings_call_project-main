"""T3.1 transcript normalization: section detection, speaker-turn chunking, determinism.

Acceptance test (TASKS T3.1 / DESIGN §6) is section-detection precision >90% on a 30-call
hand-checked sample — that human check runs on the committed `*_section_audit.csv` (HANDOFF.md).
These tests lock the detection heuristic and the chunking invariants that feed it.
"""

import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ecvol.data.calls import CallRecord, write_calls_parquet
from ecvol.features.text import sections as S


def _turn(role, text):
    return {"role": role, "text": text}


# --- section detection -------------------------------------------------------


def test_operator_cue_boundary_fincall():
    turns = [
        _turn("management", "Welcome and thanks for joining our earnings call."),
        _turn("management", "Revenue grew twenty percent this quarter."),
        _turn("operator", "We will now begin the question-and-answer session. Our first question."),
        _turn("analyst", "Thanks for taking my question on margins."),
        _turn("management", "Good question, margins expanded."),
    ]
    res = S.detect_sections(turns)
    assert res.has_qa and res.method == "operator_cue"
    assert res.boundary_idx == 2
    assert res.corroborated
    assert res.sections == [S.SECTION_PREPARED] * 2 + [S.SECTION_QA] * 3
    # prepared remarks must contain no analyst turn (internal-consistency leakage check)
    prepared = [turns[i]["role"] for i, s in enumerate(res.sections) if s == S.SECTION_PREPARED]
    assert "analyst" not in prepared


def test_first_analyst_fallback_when_no_cue():
    turns = [
        _turn("management", "Prepared remarks here."),
        _turn("operator", "Thank you."),  # no cue
        _turn("analyst", "My question is about guidance."),
    ]
    res = S.detect_sections(turns)
    assert res.has_qa and res.method == "first_analyst" and res.boundary_idx == 2
    assert not res.corroborated


def test_no_qa_all_prepared():
    turns = [_turn("management", "Only prepared remarks, no Q&A in this transcript.")]
    res = S.detect_sections(turns)
    assert not res.has_qa and res.method == "none" and res.boundary_idx == 1
    assert res.sections == [S.SECTION_PREPARED]


def test_intro_mention_false_positive_guarded():
    # operator mentions Q&A in the intro (idx 0), but analysts don't start until idx 5
    turns = [
        _turn("operator", "After prepared remarks we will hold a question-and-answer session."),
        _turn("management", "Remark one."),
        _turn("management", "Remark two."),
        _turn("management", "Remark three."),
        _turn("management", "Remark four."),
        _turn("analyst", "Now my actual question."),
    ]
    res = S.detect_sections(turns)
    # cue is far (>3 turns) before the first analyst → boundary falls back to the analyst turn
    assert res.boundary_idx == 5 and res.method == "first_analyst"
    assert res.corroborated  # both signals present, just not adjacent


def test_maec_text_cue_no_roles():
    turns = [
        _turn("unknown", "Thank you all for joining today."),
        _turn("unknown", "We delivered solid results."),
        _turn("unknown", "Operator: our first question comes from a analyst."),
        _turn("unknown", "How are margins trending?"),
    ]
    res = S.detect_sections(turns)
    assert res.has_qa and res.method == "text_cue" and res.boundary_idx == 2


def test_empty_turns():
    res = S.detect_sections([])
    assert res.sections == [] and not res.has_qa


# --- chunking ----------------------------------------------------------------


def test_short_turns_one_chunk_each_never_merged():
    turns = [_turn("management", "First short turn."), _turn("management", "Second short turn.")]
    secs = [S.SECTION_PREPARED, S.SECTION_PREPARED]
    chunks = S.chunk_turns(turns, secs, max_words=50)
    # two consecutive same-role turns stay separate (never merged across turns)
    assert len(chunks) == 2
    assert [c.turn_idx for c in chunks] == [0, 1]
    assert all(c.chunk_in_turn == 0 for c in chunks)


def test_oversized_turn_sentence_split_preserves_content():
    sents = [f"Sentence number {i} has several words in it." for i in range(40)]
    turns = [_turn("management", " ".join(sents))]
    chunks = S.chunk_turns(turns, [S.SECTION_PREPARED], max_words=20)
    assert len(chunks) > 1
    assert all(c.turn_idx == 0 for c in chunks)  # never split across turns
    assert [c.chunk_in_turn for c in chunks] == list(range(len(chunks)))
    assert all(c.n_words <= 20 for c in chunks)  # none exceeds the cap (no lone long sentence)
    # every sentence survives somewhere
    joined = " ".join(c.text for c in chunks)
    for s in sents:
        assert s in joined


def test_single_long_sentence_kept_whole_and_flagged():
    long_sentence = "word " * 60  # 60 words, one sentence, no terminator
    turns = [_turn("management", long_sentence.strip())]
    chunks = S.chunk_turns(turns, [S.SECTION_PREPARED], max_words=20)
    assert len(chunks) == 1 and chunks[0].oversize and chunks[0].n_words == 60


def test_empty_text_turn_skipped():
    turns = [_turn("management", "   "), _turn("analyst", "Real question here.")]
    chunks = S.chunk_turns(turns, [S.SECTION_PREPARED, S.SECTION_QA], max_words=50)
    assert len(chunks) == 1 and chunks[0].turn_idx == 1


# --- build + determinism -----------------------------------------------------


def _toy_call(call_id, turns):
    return CallRecord(
        call_id=call_id,
        source="fincall",
        ticker="AAA",
        call_date="2020-01-15",
        time_known=False,
        assumed_after_hours=True,
        call_type="earnings",
        label=-1,
        n_turns=len(turns),
        n_chars=sum(len(t["text"]) for t in turns),
        transcript_json=json.dumps(turns),
        speaker_metadata="{}",
        audio_path="",
        audio_exists=False,
        audio_duration_sec=float("nan"),
        parsed=True,
        status="ok",
        reason="",
    )


def _build_toy(root: Path):
    turns_a = [
        _turn("management", "Welcome to the AAA earnings call."),
        _turn("operator", "We will now begin the question-and-answer session."),
        _turn("analyst", "Question about revenue?"),
    ]
    turns_b = [_turn("management", "Just prepared remarks for BBB, no questions today.")]
    write_calls_parquet(
        [_toy_call(1, turns_a), _toy_call(2, turns_b)], root / "fincall" / "calls.parquet"
    )


def test_build_sections_outputs_and_determinism(tmp_path: Path):
    root = tmp_path / "data"
    _build_toy(root)
    s1 = S.build_sections(root, audit_n=10, seed=0)
    assert len(s1) == 1 and s1[0].dataset == "fincall"
    assert s1[0].n_processed == 2 and s1[0].calls_with_qa == 1
    chunks_path = root / "fincall" / "chunks.parquet"
    assert chunks_path.is_file()
    assert (root / "manifests" / "fincall_chunks.json").is_file()
    assert (root / "coverage" / "fincall_sections.csv").is_file()
    assert (root / "coverage" / "fincall_section_audit.csv").is_file()
    b1 = chunks_path.read_bytes()
    S.build_sections(root, audit_n=10, seed=0)
    assert chunks_path.read_bytes() == b1  # byte-identical regeneration


def test_build_sections_skips_absent_dataset(tmp_path: Path):
    root = tmp_path / "data"
    _build_toy(root)  # only fincall present
    out = S.build_sections(root)
    assert [s.dataset for s in out] == ["fincall"]


def test_audit_restricted_to_cohort_with_context_columns(tmp_path: Path):
    root = tmp_path / "data"
    _build_toy(root)  # calls 1 (has Q&A) and 2 (no Q&A)
    # Only call 1 is in the cohort (has an ok target row).
    pq.write_table(
        pa.table(
            {
                "call_id": pa.array([1, 2], pa.int64()),
                "status": pa.array(["ok", "excluded"], pa.string()),
            }
        ),
        root / "fincall" / "targets.parquet",
    )
    S.build_sections(root, audit_n=30, seed=0)
    text = (root / "coverage" / "fincall_section_audit.csv").read_text(encoding="utf-8")
    rows = list(csv.DictReader(text.splitlines()))
    assert {r["call_id"] for r in rows} == {"1"}  # call 2 excluded from the sample
    for col in ("prev_text", "boundary_text", "next_text"):
        assert col in rows[0]
