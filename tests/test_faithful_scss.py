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
