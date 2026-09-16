"""T6R.3 DialogueGAT port: corpus encoding, per-year split, graph batching (no PyG needed)."""

import json

import numpy as np
import pandas as pd
import pytest

from ecvol.eval import port_dialoguegat as D


def test_turns_and_encoding_drop_empty_and_unknown_tokens():
    tj = json.dumps(
        [
            {"role": "management", "text": "Revenue grew strongly this quarter"},
            {"role": "analyst", "text": ""},
            {"role": "weird", "text": "zzz"},
        ]
    )
    w2i = {"revenue": 1, "grew": 2, "quarter": 3}
    ids, roles = D.encode_call(tj, w2i, stop={"this"})
    assert roles == ["management"] and ids.shape == (1, D.MAX_TOKENS)
    assert ids[0, :3].tolist() == [1, 2, 3] and ids[0, 3:].sum() == 0
    assert D.turns(tj)[1][0] == "unknown"


def test_year_split_is_chronological_70_10_20():
    calls = pd.DataFrame(
        {
            "call_id": [str(i) for i in range(20)],
            "ticker": ["T"] * 20,
            "call_date": [f"2019-{1 + i // 2:02d}-{10 + i % 2:02d}" for i in range(20)],
            "year": [2019] * 20,
        }
    )
    d = D.year_split(calls, 2019)
    assert d["split"].tolist() == ["train"] * 14 + ["val"] * 2 + ["test"] * 4
    assert d["call_date"].is_monotonic_increasing


def test_collate_builds_chain_and_speaker_edges():
    torch = pytest.importorskip("torch")
    items = [(np.ones((3, D.MAX_TOKENS), np.int64), ["management", "analyst", "management"])]
    p2gid = {r: i for i, r in enumerate(D.ROLES)}
    tok, ei, iu, pid, gid, ng = D.collate(items, p2gid, "cpu")
    assert tok.shape == (3, D.MAX_TOKENS) and ng == 1
    assert iu.tolist() == [True, True, True, False, False]  # 3 utterances + 2 speakers
    # chain: 2 undirected edges = 4 directed; speaker: 3 utterances × 2 directions = 6
    assert ei.shape == (2, 10)
    assert set(pid.tolist()) == {p2gid["analyst"], p2gid["management"]}
    assert torch.equal(gid, torch.zeros(5, dtype=torch.long))


def test_condition_split_rewrites_only_the_year_split():
    tickers = [f"T{i}" for i in range(6)]
    d = pd.DataFrame(
        {
            "ticker": tickers * 3,
            "call_date": ["2019-01-02"] * 6 + ["2019-06-03"] * 6 + ["2019-06-17"] * 6,
            "split": ["train"] * 6 + ["val"] * 6 + ["test"] * 6,
        }
    )
    assert D.condition_split(d, "their").tolist() == d["split"].tolist()
    a, t = D.condition_split(d, "anchor_heldout"), D.condition_split(d, "ticker_disjoint")
    assert (a == "test").sum() == 2 and ((a == "test") == (t == "test")).all()
    assert not set(d.loc[np.isin(t, ["train", "val"]), "ticker"]) & set(
        d.loc[t == "test", "ticker"]
    )
    e = D.condition_split(d, "embargoed")
    assert (e == "val").sum() == 0 and (e == "train").sum() == 6 and (e == "excluded").sum() == 6
