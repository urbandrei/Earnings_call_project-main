"""Dataset fetch helpers: extraction idempotency, counts, CLI guards (T1.1).

Network downloads are not exercised here; these tests cover the local logic the
fetchers are built from (atomic extraction, call counting, manifest assembly).
"""

import json
import tarfile
import zipfile

from typer.testing import CliRunner

from ecvol.cli import app
from ecvol.data.fetch import (
    _extract_tarball,
    _extract_zip,
    count_fincall_calls,
    count_maec_calls,
)

runner = CliRunner()


def _make_zip(path, names, wrap_dir=None):
    with zipfile.ZipFile(path, "w") as zf:
        for name in names:
            arcname = f"{wrap_dir}/{name}" if wrap_dir else name
            zf.writestr(arcname, f"content of {name}")


def test_extract_zip_unwraps_single_top_dir(tmp_path):
    archive = tmp_path / "mp3_2019.zip"
    _make_zip(archive, ["a.mp3", "b.mp3"], wrap_dir="mp3_2019")
    dest = tmp_path / "mp3_2019"
    _extract_zip(archive, dest)
    assert sorted(p.name for p in dest.iterdir()) == ["a.mp3", "b.mp3"]


def test_extract_zip_flat_archive_and_idempotency(tmp_path):
    archive = tmp_path / "flat.zip"
    _make_zip(archive, ["x.mp3"])
    dest = tmp_path / "flat"
    _extract_zip(archive, dest)
    assert (dest / "x.mp3").is_file()
    marker = dest / "user_added.txt"
    marker.write_text("kept")
    _extract_zip(archive, dest)  # second call must be a no-op
    assert marker.read_text() == "kept"


def test_extract_zip_recovers_from_interrupted_extraction(tmp_path):
    archive = tmp_path / "data.zip"
    _make_zip(archive, ["a.mp3"], wrap_dir="data")
    stale = tmp_path / "data.partial"
    (stale / "junk").mkdir(parents=True)
    _extract_zip(archive, tmp_path / "data")
    assert (tmp_path / "data" / "a.mp3").is_file()
    assert not stale.exists()


def test_extract_tarball_strips_top_dir(tmp_path):
    src = tmp_path / "repo-abc123"
    (src / "MAEC_Dataset" / "20170101_AAPL").mkdir(parents=True)
    (src / "MAEC_Dataset" / "20170101_AAPL" / "text.txt").write_text("hello")
    archive = tmp_path / "repo.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(src, arcname="repo-abc123")
    dest = tmp_path / "repo"
    _extract_tarball(archive, dest)
    assert (dest / "MAEC_Dataset" / "20170101_AAPL" / "text.txt").read_text() == "hello"
    _extract_tarball(archive, dest)  # idempotent


def test_count_fincall_calls(tmp_path):
    raw = tmp_path / "fincall"
    raw.mkdir()
    (raw / "transcripts_2019.json").write_text(json.dumps({"c1": {}, "c2": {}}))
    (raw / "transcripts_2020.json").write_text(json.dumps({"c3": {}}))
    (raw / "mp3_2019").mkdir()
    (raw / "mp3_2019" / "c1.mp3").write_bytes(b"")
    counts = count_fincall_calls(raw)
    assert counts["transcripts_2019"] == 2
    assert counts["transcripts_2020"] == 1
    assert counts["calls_total"] == 3
    assert counts["mp3_files"] == 1
    assert "ppt_files" not in counts


def test_count_maec_calls(tmp_path):
    dataset = tmp_path / "maec" / "repo" / "MAEC_Dataset"
    for name, files in [
        ("20170101_AAPL", ["text.txt", "features.csv"]),
        ("20170102_MSFT", ["text.txt"]),
    ]:
        (dataset / name).mkdir(parents=True)
        for file in files:
            (dataset / name / file).write_text("x")
    counts = count_maec_calls(tmp_path / "maec")
    assert counts == {"calls_total": 2, "with_text": 2, "with_features": 1}


def test_cli_fetch_rejects_unknown_dataset(tmp_path):
    result = runner.invoke(app, ["data", "fetch", "nonsense", "--root", str(tmp_path)])
    assert result.exit_code == 2
    assert "unknown dataset" in result.output


def test_cli_spotcheck_fails_without_audio(tmp_path):
    result = runner.invoke(app, ["data", "spotcheck", "--root", str(tmp_path)])
    assert result.exit_code == 1
    assert "no .mp3 files" in result.output
