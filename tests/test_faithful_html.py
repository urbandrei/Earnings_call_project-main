"""T6R.3 HTML faithful: verbatim-class loader, aligned splits."""

import numpy as np
import pandas as pd
import pytest

from ecvol.eval import faithful_html as FH

STUB = """
import junk_that_does_not_exist
from transformer import util

class SelfAttention:
    pass

class RTransformer:
    def __init__(self, emb, heads, depth, seq_length, num_tokens, num_classes,
                 max_pool=True, dropout=0.0):
        self.emb, self.dropout, self.uses = emb, dropout, (util, d, mask_)
"""


def test_loader_execs_only_the_class_block():
    pytest.importorskip("torch")  # gpu dependency group; CI is torch-free
    RT = FH.load_authors_classes(STUB)
    m = RT(emb=4, heads=2, depth=2, seq_length=5, num_tokens=0, num_classes=1, dropout=0.5)
    assert (m.emb, m.dropout) == (4, 0.5) and m.uses[1]() in ("cuda", "cpu")


def test_chronological_split_is_7_1_2_and_ordered():
    ids = [f"c{i}" for i in range(20)]
    lab = pd.DataFrame(
        {
            "year": 2017,
            "month": [1 + (i % 12) for i in range(20)],
            "day": [1 + i for i in range(20)],
        },
        index=ids,
    )
    s = FH.chronological_split(lab, ids)
    assert s["split"].value_counts().to_dict() == {"train": 14, "val": 2, "test": 4}
    months = lab.loc[s["call_id"], "month"].to_numpy()
    assert (np.diff(months) >= 0).all()  # earliest first
    assert s["split"].tolist() == ["train"] * 14 + ["val"] * 2 + ["test"] * 4


def test_random_split_is_aligned_and_seeded():
    ids = [f"c{i}" for i in range(100)]
    a, b = FH.random_split(ids, 0), FH.random_split(ids, 0)
    assert a.equals(b) and set(a["call_id"]) == set(ids)
    assert a["split"].value_counts().to_dict() == {"train": 70, "val": 10, "test": 20}
    assert not a.equals(FH.random_split(ids, 1))


def test_pad_zero_fills_to_longest():
    out = FH._pad([np.ones((2, 3), np.float32), np.ones((4, 3), np.float32)])
    assert out.shape == (2, 4, 3) and out[0, 2:].sum() == 0 and out[1].sum() == 12


@pytest.mark.parametrize("cond", list(FH.CONDITIONS))
def test_conditions_have_dropout(cond):
    assert FH.CONDITIONS[cond] in (0.0, 0.5)


def test_controlled_conditions_change_only_the_split():
    # DECISIONS 2026-09-16: same hyperparameters as the `code` anchor, a committed split file
    for cond in ("embargoed", "ticker_disjoint"):
        assert FH.CONDITIONS[cond] == FH.CONDITIONS["code"]
        assert FH.SPLIT_FILES[cond].startswith("ec_")
    assert set(FH.SPLIT_FILES) == set(FH.CONDITIONS) - {"code", "paper"}


def test_shuffle_partners_within_same_ticker_and_global_other_ticker():
    tickers = np.array(["A", "A", "A", "B", "C", "C"])
    test_pos = np.array([0, 3, 4])
    w, w_ok = FH.shuffle_partners(tickers, test_pos, "within", seed=0)
    assert w_ok.tolist() == [True, False, True]  # B has no other call
    assert w[0] in (1, 2) and w[1] == 3 and w[2] == 5  # never itself; partner may be any split
    g, g_ok = FH.shuffle_partners(tickers, test_pos, "global", seed=0)
    assert g_ok.all() and all(tickers[p] != tickers[t] for p, t in zip(g, test_pos, strict=True))
    assert FH.shuffle_partners(tickers, test_pos, "within", seed=0)[0].tolist() == w.tolist()
