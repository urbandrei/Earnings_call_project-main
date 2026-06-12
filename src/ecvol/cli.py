"""ecvol CLI — Typer app with one verb per pipeline stage.

Every verb is a stub until its task lands (see TASKS.md). The CLI contract
(idempotent, resumable, config-driven) is DESIGN.md §8.2.
"""

import typer

app = typer.Typer(
    no_args_is_help=True,
    help="Earnings-call volatility prediction pipeline.",
)


def _not_implemented(verb: str) -> None:
    typer.echo(f"ecvol {verb}: not implemented yet (see TASKS.md)", err=True)
    raise typer.Exit(code=2)


@app.command()
def prices() -> None:
    """Pull and cache adjusted daily OHLCV for the call universe (T1.2)."""
    _not_implemented("prices")


@app.command()
def targets() -> None:
    """Compute volatility targets per (call, horizon) (T1.3)."""
    _not_implemented("targets")


@app.command()
def splits() -> None:
    """Build leakage-proof temporal / ticker-disjoint splits (T1.6)."""
    _not_implemented("splits")


@app.command()
def featurize() -> None:
    """Extract and cache text / audio / LLM features (Phases 3-6)."""
    _not_implemented("featurize")


@app.command()
def train() -> None:
    """Train a model from a validated YAML config (Phases 2-5)."""
    _not_implemented("train")


@app.command()
def evaluate() -> None:
    """Evaluate trained models on committed splits (T2.2)."""
    _not_implemented("evaluate")


@app.command()
def report() -> None:
    """Regenerate result tables from run artifacts (T2.3)."""
    _not_implemented("report")
