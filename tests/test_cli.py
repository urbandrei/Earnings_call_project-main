"""Smoke tests for the CLI skeleton (T0.1 acceptance)."""

from typer.testing import CliRunner

from ecvol.cli import app

runner = CliRunner()

VERBS = ["prices", "targets", "splits", "featurize", "train", "evaluate", "report"]
STUB_VERBS = ["targets", "splits", "featurize", "train", "evaluate", "report"]  # prices: T1.2 done


def test_help_lists_all_verbs():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for verb in VERBS:
        assert verb in result.output


def test_stub_verbs_exit_nonzero():
    for verb in STUB_VERBS:
        result = runner.invoke(app, [verb])
        assert result.exit_code == 2, f"{verb} should be a stub"


def test_prices_subcommands_exposed():
    result = runner.invoke(app, ["prices", "--help"])
    assert result.exit_code == 0
    assert "pull" in result.output
    assert "crosscheck" in result.output
