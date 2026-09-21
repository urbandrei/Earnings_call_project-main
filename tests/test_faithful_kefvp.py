"""T6R.3 KeFVP faithful: patch functions are exact and leave no placeholders."""

import pandas as pd
import pytest

from ecvol.eval import faithful_kefvp as K

INFER = (
    'log = strftime("%Y-%m-%d_%H:%M:%S", localtime())\n'
    "p = '/your/project/path/log/'\nb = '/your/dataset/path/'\n"
    "                audio_matrix = audio_path.values\n"
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


def _maec_frames():
    import pandas as pd

    days = pd.bdate_range("2015-01-01", periods=60).strftime("%Y-%m-%d").tolist()
    parts = {"train": range(0, 42), "dev": range(42, 48), "test": range(48, 60)}
    frames = {}
    for kind in K.MAEC_KINDS:
        frames[kind] = {
            p: pd.DataFrame(
                {
                    "ticker": [f"T{i % 9}" for i in idx],
                    "time": [days[i] for i in idx],
                    "text_file_name": [f"c{i}" for i in idx],
                    "future_3": 0.1,
                    "past_27": 0.2,
                    "kind": kind,
                }
            )  # fmt: skip
            for p, idx in parts.items()
        }
    return frames


def test_maec_resplit_keeps_kinds_aligned_and_validation_alive():
    frames = _maec_frames()
    for cond in K.CONTROL_CONDITIONS:
        out = K.maec_resplit(frames, cond)
        for p in K.MAEC_PARTS:  # the script pairs avg and single rows by position
            names = [out[k][p]["text_file_name"].tolist() for k in K.MAEC_KINDS]
            assert names[0] == names[1] == names[2]
        assert len(out["avg_val"]["dev"]) > 0 and len(out["avg_val"]["test"]) > 0
    a, t = K.maec_resplit(frames, "anchor_heldout"), K.maec_resplit(frames, "ticker_disjoint")
    assert a["avg_val"]["test"].equals(t["avg_val"]["test"])
    fit = pd.concat([t["avg_val"]["train"], t["avg_val"]["dev"]])
    assert not set(fit["ticker"]) & set(t["avg_val"]["test"]["ticker"])
    e = K.maec_resplit(frames, "embargoed")["avg_val"]
    assert len(e["test"]) == 12 and e["dev"]["time"].max() < e["test"]["time"].min()


def test_patch_infer_repeats_hook():
    src = INFER + "    for i in range(10):\n"
    assert "range(10)" in K.patch_infer(src, "p", "d")
    assert "range(3)" in K.patch_infer(src, "p", "d", repeats=3)


def test_with_embedding_swaps_only_the_embedding_name():
    a = K.infer_args("15", 3, raw_data_path="r/")
    b = K._with_embedding(a, "raw_bert_base_uncased_maec15")
    assert b[b.index("--text_embedding") + 1] == "raw_bert_base_uncased_maec15"
    assert [x for x in a if x != K.MAEC_EMBEDDING] == [
        x for x in b if x != "raw_bert_base_uncased_maec15"
    ]


def test_persistence_column_counts_windows_backwards():
    assert [K.persistence_column(t) for t in K.TAUS] == ["past_27", "past_23", "past_15", "past_0"]


def test_patch_generator_picks_the_corpus_text_file():
    src = (
        "text_path = args.data_path + data_dir + '/Text.txt'   # For ec\n"
        "    data_list = os.listdir(args.data_path)\n    output_dct = {}\n    all_sent_num = []\n"
        "        with torch.no_grad():\n"
        "            model_out = model(input['input_ids'].cuda(), input['attention_mask'].cuda())\n"
    )
    assert "/text.txt'" in K.patch_generator(src, "d", "w.txt")
    ec = K.patch_generator(src, "d", "w.txt", maec=False)
    assert "/TextSequence.txt'" in ec and "_wanted" in ec and "_outs" in ec
