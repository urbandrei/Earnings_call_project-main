"""T6R.3 SCSS faithful: patch set and result-filename parsing."""

import pytest

from ecvol.eval import faithful_scss as FS

LOADER = "import x\nfrom sktime.datasets import load_from_tsfile_to_dataframe\nimport y\n"


def test_patches_apply_exactly_and_are_forward_ports_only():
    files = {
        "data_provider/data_loader.py": LOADER,
        "utils/reconfig_args.py": "s = f'_rt({args.test_year}|{args.test_quarter})'\n",
        "utils/tools.py": "import matplotlib.pyplot as plt\nself.best = np.Inf\n",
    }
    out = FS.apply_patches(files)
    assert out["data_provider/data_loader.py"] == "import x\nimport y\n"
    assert "|" not in out["utils/reconfig_args.py"]
    assert "np.inf" in out["utils/tools.py"]
    assert not out["utils/tools.py"].startswith("import matplotlib")
    ns: dict = {}
    exec(out["utils/tools.py"].split("self.best")[0], ns)  # noqa: S102 — the stub must import
    assert ns["plt"].switch_backend("agg") is None


def test_patch_target_missing_raises():
    with pytest.raises(AssertionError):
        FS.apply_patches({"utils/tools.py": "nothing here\n"})


@pytest.mark.parametrize(
    "name,win,year,q,mse",
    [
        ("mTSMixer_dDEC_seq22_win3_rt(2023|first)_0.670532.csv", 3, 2023, "first", 0.670532),
        ("mTMLP_dDEC_e(DEC)_win15_rt(2019-fourth)_0.218049.csv", 15, 2019, "fourth", 0.218049),
    ],
)
def test_parse_name(name, win, year, q, mse):
    rec = FS.parse_name(name)
    assert (rec["window"], rec["year"], rec["quarter"], rec["mse"]) == (win, year, q, mse)


def test_grid_has_76_cells_and_args_match_script():
    cells = [
        (w, y, q)
        for w in FS.WINDOWS
        for y in FS.YEARS
        for q in FS.QUARTERS
        if not (y == 2019 and q == "first")
    ]
    assert len(cells) == 76
    a = FS.tsmixer_args(7, 2021, "second")
    assert a[a.index("--prediction_window") + 1] == "7"
    assert a[a.index("--d_model") + 1] == "512"


PRINTED = """Overlapping Earnings per Ticker (OET): 19.775 (2195 / 111)

Volatility Prediction Error Comparison
============================================================
Method       | Mean     | 3-day    | 7-day    | 15-day   | 30-day
------------------------------------------------------------
PEV          | 0.399    | 0.743    | 0.389    | 0.262    | 0.201
Aug_STPEV    | 0.296    | 0.569    | 0.293    | 0.201    | 0.122
============================================================
"""


def test_parse_printed_table_drops_the_mean_column():
    t = FS.parse_printed_table(PRINTED)
    assert t == {"PEV": [0.743, 0.389, 0.262, 0.201], "Aug_STPEV": [0.569, 0.293, 0.201, 0.122]}


def _cell(src, text=""):
    outs = [{"output_type": "stream", "text": [text]}] if text else []
    return {"cell_type": "code", "source": [src], "outputs": outs}


def test_notebook_functions_load_verbatim_and_published_maps_by_file():
    nb = {
        "cells": [
            _cell("def build_aug_stpev_mean(e, h):\n    return pd.DataFrame({'a': [1]})\n"),
            _cell("def run_pev_stpev(e, w, a):\n    return np.float64(w)\n"),
            _cell("def run_EC_MAEC(e, a):\n    return 'ok'\n"),
            _cell(
                "x = pd.read_csv(f'{dataset_dir}/MAEC/MAEC16_earnings.csv')\n"
                "results = run_EC_MAEC(x, y)\n",
                PRINTED,
            ),
        ]
    }
    ns = FS.load_notebook_functions(nb)
    assert ns["run_pev_stpev"](None, 7, None) == 7.0 and ns["run_EC_MAEC"](0, 0) == "ok"
    assert FS.published_aug(nb) == {"maec16": FS.parse_printed_table(PRINTED)}


def test_history_window_crossings_counts_targets_reaching_the_test_period():
    import pandas as pd

    earnings = pd.DataFrame(
        {"cate": ["train", "test", "test"], "ticker": ["A", "A", "B"],
         "day_earnings": ["2017-01-02", "2017-03-06", "2017-03-08"]}  # Monday 6 March
    )  # fmt: skip
    history = pd.DataFrame(
        {"ticker": ["A", "A", "B", "C", "A"],
         "day_earnings": ["2017-02-27", "2017-03-01", "2017-01-02", "2017-03-03", "2017-03-06"]}
    )  # fmt: skip
    # used: A 02-27 (Mon), A 03-01 (Wed), B 01-02; C is not a test ticker, A 03-06 is not < first
    assert FS.history_window_crossings(earnings, history, 3) == (3, 1)  # Wed+3bd = Mon 03-06
    assert FS.history_window_crossings(earnings, history, 7) == (3, 2)  # Mon 02-27+7bd = 03-08


def _dec():
    import pandas as pd

    tickers = [f"T{i}" for i in range(6)]
    rows = []
    for t in tickers:
        for day, cate in (("2022-01-03", "train"), ("2022-03-25", "val"), ("2022-04-11", "test")):
            rows.append({"id": f"{t}_{day}", "ticker": t, "day_earnings": day,
                         "rolling_test_on_2022_second_cate": cate})  # fmt: skip
    return pd.DataFrame(rows)


def test_heldout_third_is_seeded_and_a_third():
    held = FS.heldout_tickers([f"T{i}" for i in range(6)] * 3)
    assert len(held) == 2 and held == FS.heldout_tickers([f"T{i}" for i in range(6)])


def test_control_masks_keep_rows_and_change_only_masks():
    dec = _dec()
    col = "rolling_test_on_2022_second_cate"
    held = FS.heldout_tickers(dec["ticker"])
    a = FS.control_masks(dec, "anchor_heldout")
    t = FS.control_masks(dec, "ticker_disjoint")
    for m in (a, t):
        assert m["id"].tolist() == dec["id"].tolist()  # row order = embedding order
        assert set(m.loc[m[col] == "test", "ticker"]) == held
    assert a.loc[a[col] == "test", "id"].tolist() == t.loc[t[col] == "test", "id"].tolist()
    assert not set(t.loc[t[col].isin(["train", "val"]), "ticker"]) & held
    assert set(a.loc[a[col].isin(["train", "val"]), "ticker"]) & held  # anchor keeps history
    e = FS.control_masks(dec, "embargoed")
    # 2022-03-25 + 30 business days reaches 2022-04-11; 2022-01-03 + 30 does not
    assert set(e.loc[e[col].isin(["train", "val"]), "day_earnings"]) == {"2022-01-03"}
    assert (e[col] == "test").sum() == 6
