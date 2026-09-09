"""T6R.2 code-availability audit: curated rows regenerate offline; the probe is isolated."""

from pathlib import Path

from ecvol.eval import code_audit as C


def test_audit_offline_rows_and_verdicts(tmp_path: Path):
    t = C.run_code_audit(tmp_path, live=False)
    assert len(t) == len(C.PAPERS) and set(t["verdict"]) == {"runnable", "partial", "none"}
    assert (t[t["url"] == ""]["probe_status"] == "not_probed").all()
    assert t.set_index("key").loc["scss2025", "reproduced_here"].startswith("ecvol reproduce scss")
    assert (t["verdict"] == "runnable").sum() == 1  # only SCSS runs as released (T6R.3)
    assert (tmp_path / "results" / "code_availability.csv").is_file()


def test_probe_no_url_and_http_error(monkeypatch):
    assert C.probe("") == {"status": "no_url"}

    def boom(url, timeout=30):
        import urllib.error

        raise urllib.error.HTTPError(url, 409, "conflict", {}, None)

    monkeypatch.setattr(C, "_get", boom)
    assert C.probe("x/y") == {"status": "409"}
