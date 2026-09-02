"""T8.1 release archives: member policy, determinism, manifest."""

import json
import zipfile

import pandas as pd

from ecvol import release as R


def _root(tmp_path):
    root = tmp_path / "data"
    for d in ("fincall", "splits", "identity", "coverage", "manifests"):
        (root / d).mkdir(parents=True)
    pd.DataFrame({"call_id": [1]}).to_parquet(root / "fincall" / "targets.parquet")
    pd.DataFrame({"call_id": [1]}).to_parquet(root / "fincall" / "text_embeddings.parquet")
    pd.DataFrame({"t": ["x"]}).to_parquet(root / "fincall" / "chunks.parquet")  # raw text
    (root / "fincall" / "calls.parquet").write_bytes(b"raw")  # raw transcripts
    (root / "splits" / "fincall_temporal.csv").write_text("call_id,split\n1,train\n")
    (root / "identity" / "fincall_identity.csv").write_text("call_id,ticker\n1,A\n")
    (root / "manifests" / "fincall_targets.json").write_text("[]\n")
    (tmp_path / "LICENSE-DATA.md").write_text("licence\n")
    return root


def test_members_exclude_raw_payloads(tmp_path):
    root = _root(tmp_path)
    names = [p.as_posix() for p in R.members(root, "fincall")]
    assert "fincall/targets.parquet" in names and "splits/fincall_temporal.csv" in names
    assert not any(n.endswith(("chunks.parquet", "calls.parquet")) for n in names)
    raw = ("chunks.parquet", "calls.parquet")
    assert all(not g.endswith(raw) for _, globs in R.RELEASED.values() for g in globs)


def test_release_is_deterministic_and_manifested(tmp_path):
    root = _root(tmp_path)
    m1 = R.build_release(root, tmp_path / "rel1", "abc1234", corpora=("fincall",))
    m2 = R.build_release(root, tmp_path / "rel2", "abc1234", corpora=("fincall",))
    a1 = json.loads(m1.read_text())["archives"][0]
    a2 = json.loads(m2.read_text())["archives"][0]
    assert a1["sha256"] == a2["sha256"]
    assert a1["archive"] == "ecvol-bench_fincall_abc1234.zip"
    with zipfile.ZipFile(tmp_path / "rel1" / a1["archive"]) as z:
        names = z.namelist()
        assert "LICENSE-DATA.md" in names and "fincall/targets.parquet" in names
        assert all(i.date_time == R.FIXED_TIME for i in z.infolist())
    assert {m["path"] for m in a1["members"]} == {p.as_posix() for p in R.members(root, "fincall")}
