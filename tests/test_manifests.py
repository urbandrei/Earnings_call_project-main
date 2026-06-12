"""Manifest write/load/verify and `ecvol data verify` (T0.3 acceptance)."""

import hashlib

from typer.testing import CliRunner

from ecvol.cli import app
from ecvol.data.manifests import (
    load_manifest,
    make_entry,
    sha256_file,
    verify_manifest,
    write_manifest,
)

runner = CliRunner()


def _make_data_file(root, name, content=b"hello volatility"):
    file = root / name
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(content)
    return file


def _write_test_manifest(tmp_path, content=b"hello volatility"):
    data_root = tmp_path / "data"
    file = _make_data_file(data_root, "prices/AAPL.parquet", content)
    entry = make_entry(
        file,
        data_root,
        source_url="https://example.com/AAPL",
        license="CC0",
        retrieved_at="2026-06-12T00:00:00+00:00",
    )
    manifest_path = tmp_path / "prices.json"
    write_manifest([entry], manifest_path)
    return data_root, file, manifest_path


def test_sha256_file_matches_hashlib(tmp_path):
    file = _make_data_file(tmp_path, "blob.bin", b"\x00" * 3_000_000)
    assert sha256_file(file) == hashlib.sha256(b"\x00" * 3_000_000).hexdigest()


def test_roundtrip_and_clean_verify(tmp_path):
    data_root, _, manifest_path = _write_test_manifest(tmp_path)
    entries = load_manifest(manifest_path)
    assert [e.path for e in entries] == ["prices/AAPL.parquet"]
    assert verify_manifest(manifest_path, data_root) == []


def test_verify_detects_corruption(tmp_path):
    data_root, file, manifest_path = _write_test_manifest(tmp_path)
    file.write_bytes(b"corrupted!")
    problems = verify_manifest(manifest_path, data_root)
    assert len(problems) == 1
    assert "checksum mismatch" in problems[0]


def test_verify_detects_missing_file(tmp_path):
    data_root, file, manifest_path = _write_test_manifest(tmp_path)
    file.unlink()
    problems = verify_manifest(manifest_path, data_root)
    assert len(problems) == 1
    assert "missing" in problems[0]


def test_write_manifest_is_deterministic(tmp_path):
    data_root = tmp_path / "data"
    entries = [
        make_entry(
            _make_data_file(data_root, name, name.encode()),
            data_root,
            source_url="https://example.com",
            license="CC0",
            retrieved_at="2026-06-12T00:00:00+00:00",
        )
        for name in ["b.bin", "a.bin"]
    ]
    out1, out2 = tmp_path / "m1.json", tmp_path / "m2.json"
    write_manifest(entries, out1)
    write_manifest(list(reversed(entries)), out2)
    assert out1.read_bytes() == out2.read_bytes()


def test_cli_data_verify_clean_and_corrupt(tmp_path):
    data_root, file, manifest_path = _write_test_manifest(tmp_path)

    result = runner.invoke(app, ["data", "verify", str(manifest_path), "--root", str(data_root)])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output

    file.write_bytes(b"corrupted!")
    result = runner.invoke(app, ["data", "verify", str(manifest_path), "--root", str(data_root)])
    assert result.exit_code == 1
    assert "checksum mismatch" in result.output
