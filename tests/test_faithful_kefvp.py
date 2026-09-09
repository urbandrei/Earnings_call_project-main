"""T6R.3 KeFVP faithful: patch functions are exact and leave no placeholders."""

import pytest

from ecvol.eval import faithful_kefvp as K

INFER = (
    'log = strftime("%Y-%m-%d_%H:%M:%S", localtime())\n'
    "p = '/your/project/path/log/'\nb = '/your/dataset/path/'\n"
    + "".join(
        f"x{t} = float(price_df[price_df.text_file_name == row['text_file_name']]"
        f"['future_label_{t}'])\n"
        for t in K.TAUS
    )
)


def test_patch_infer_replaces_everything():
    out = K.patch_infer(INFER, "C:/w/proj", "C:/w/dataset")
    assert "/your/" not in out and "%H-%M-%S" in out
    assert out.count(".iloc[0])") == 4 and "C:/w/proj/log/" in out and "C:/w/dataset/" in out


def test_patch_infer_requires_each_label_line_once():
    with pytest.raises(AssertionError):
        K.patch_infer(INFER.replace("future_label_30", "future_label_31"), "p", "d")


def test_infer_args_follow_the_shell_scripts():
    ec = K.infer_args("ec", 7, raw_data_path="r/")
    assert ec[ec.index("--text_indim") + 1] == "1024" and ec[ec.index("--duration") + 1] == "7"
    m = K.infer_args("16", 3, raw_data_path="r/")
    assert m[m.index("--audio_indim") + 1] == "29" and "--text_indim" not in m
    assert m[m.index("--text_embedding") + 1] == K.MAEC_EMBEDDING


def test_published_table_complete():
    assert set(K.PUBLISHED) == {"ec", "15", "16"}
    assert all(set(v) == set(K.TAUS) for v in K.PUBLISHED.values())
