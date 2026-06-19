"""ecvol CLI — Typer app with one verb per pipeline stage.

Every verb is a stub until its task lands (see TASKS.md). The CLI contract
(idempotent, resumable, config-driven) is DESIGN.md §8.2.
"""

from pathlib import Path

import typer

app = typer.Typer(
    no_args_is_help=True,
    help="Earnings-call volatility prediction pipeline.",
)

data_app = typer.Typer(no_args_is_help=True, help="Data acquisition & provenance (T0.3, T1.1).")
app.add_typer(data_app, name="data")


@data_app.command()
def fetch(
    dataset: str = typer.Argument(help="Dataset to mirror: fincall | maec | all."),
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    skip_drive: bool = typer.Option(
        False, help="FinCall only: skip the ~57 GB Google Drive payload (repo files only)."
    ),
) -> None:
    """Mirror datasets locally with checksummed manifests (T1.1). Idempotent/resumable."""
    from ecvol.data import fetch as f

    if dataset not in ("fincall", "maec", "all"):
        typer.echo(f"unknown dataset {dataset!r} (expected fincall | maec | all)", err=True)
        raise typer.Exit(code=2)
    if dataset in ("fincall", "all"):
        manifest = f.fetch_fincall(root, skip_drive=skip_drive)
        typer.echo(f"fincall mirrored; manifest: {manifest}")
        for key, value in f.count_fincall_calls(root / "raw" / "fincall").items():
            typer.echo(f"  {key}: {value}")
    if dataset in ("maec", "all"):
        manifest = f.fetch_maec(root)
        typer.echo(f"maec mirrored; manifest: {manifest}")
        for key, value in f.count_maec_calls(root / "raw" / "maec").items():
            typer.echo(f"  {key}: {value}")


@data_app.command()
def identity(
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
) -> None:
    """Reconstruct FinCall call identity (ticker/company/date/type) → committed CSV (T1.4)."""
    from ecvol.data.fincall_identity import build_identity

    out, stats = build_identity(root)
    typer.echo(f"identity table written: {out}")
    for key, value in stats.items():
        typer.echo(f"  {key}: {value}")


@data_app.command()
def ingest(
    dataset: str = typer.Argument(help="Dataset to normalize: fincall | maec."),
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    no_audio: bool = typer.Option(
        False, help="FinCall only: skip ffprobe audio-duration probing (durations left NaN)."
    ),
) -> None:
    """Normalize a dataset onto the common call schema → parquet + reports (T1.4/T1.5)."""
    if dataset not in ("fincall", "maec"):
        typer.echo(f"unknown dataset {dataset!r} (expected fincall | maec)", err=True)
        raise typer.Exit(code=2)
    if dataset == "fincall":
        from ecvol.data.fincall_ingest import ingest_fincall

        s = ingest_fincall(root, probe_audio=not no_audio)
        typer.echo(f"calls: {s.ok}/{s.total_calls} ok ({s.parsed} parsed)")
        typer.echo(f"audio: {s.audio_decoded}/{s.audio_present} decoded (of {s.total_calls} calls)")
        typer.echo(
            f"join: {s.earnings_joined}/{s.earnings_resolved} earnings-cohort calls "
            f"with >=1 target ({s.join_rate_pct}%)"
        )
        if s.reason_counts:
            typer.echo("exclusions: " + ", ".join(f"{k}={v}" for k, v in s.reason_counts.items()))
        typer.echo("calls: data/fincall/calls.parquet; reports: data/coverage/fincall_*.csv")
    else:
        from ecvol.data.maec_ingest import ingest_maec

        m = ingest_maec(root)
        typer.echo(f"calls: {m.ok}/{m.total_calls} ok ({m.parsed} parsed)")
        typer.echo(
            f"audio features: {m.calls_with_features}/{m.total_calls} calls "
            f"({m.total_sentences} sentences); raw audio: 0 (MAEC ships none)"
        )
        typer.echo(
            f"join: {m.joined}/{m.parsed} calls with >=1 target ({m.join_rate_pct}%); "
            f"missing-price tickers: {m.missing_price_tickers}"
        )
        if m.reason_counts:
            typer.echo("exclusions: " + ", ".join(f"{k}={v}" for k, v in m.reason_counts.items()))
        typer.echo("calls/targets: data/maec/*.parquet; reports: data/coverage/maec_*.csv")


@data_app.command()
def spotcheck(
    root: Path = typer.Option(Path("data/raw"), help="Tree to sample audio from."),  # noqa: B008
    n: int = typer.Option(50, help="Number of audio files to decode."),
    seed: int = typer.Option(0, help="Sampling seed."),
) -> None:
    """Decode a seeded-random sample of mirrored audio with ffmpeg (T1.1 acceptance)."""
    from ecvol.data.fetch import spotcheck_audio

    problems = spotcheck_audio(root, n=n, seed=seed)
    if problems:
        for problem in problems:
            typer.echo(problem, err=True)
        raise typer.Exit(code=1)
    typer.echo(f"spotcheck OK: {n} file(s) decoded cleanly (seed={seed})")


@data_app.command()
def verify(
    manifests: list[Path] = typer.Argument(  # noqa: B008
        None, help="Manifest JSON files (default: all of data/manifests/*.json)."
    ),
    root: Path = typer.Option(  # noqa: B008
        Path("data"), help="Directory the manifest paths are relative to."
    ),
) -> None:
    """Verify data files against their committed manifests (existence + SHA-256)."""
    from ecvol.data.manifests import verify_manifest

    if not manifests:
        manifests = sorted(Path("data/manifests").glob("*.json"))
        if not manifests:
            typer.echo("no manifests found under data/manifests/ — nothing to verify")
            return
    failed = False
    for manifest in manifests:
        if not manifest.is_file():
            typer.echo(f"{manifest}: manifest file not found", err=True)
            failed = True
            continue
        problems = verify_manifest(manifest, root)
        if problems:
            failed = True
            typer.echo(f"{manifest}: {len(problems)} problem(s)", err=True)
            for problem in problems:
                typer.echo(f"  {problem}", err=True)
        else:
            typer.echo(f"{manifest}: OK")
    if failed:
        raise typer.Exit(code=1)


def _not_implemented(verb: str) -> None:
    typer.echo(f"ecvol {verb}: not implemented yet (see TASKS.md)", err=True)
    raise typer.Exit(code=2)


prices_app = typer.Typer(no_args_is_help=True, help="Adjusted daily OHLCV ingestion (T1.2).")
app.add_typer(prices_app, name="prices")


@prices_app.command("pull")
def prices_pull(
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    refresh: bool = typer.Option(False, help="Re-download tickers even if cached."),
) -> None:
    """Pull adjusted daily OHLCV for the FinCall+MAEC universe → parquet + coverage (T1.2)."""
    from ecvol.data.prices import pull_prices

    summary = pull_prices(root, refresh=refresh)
    typer.echo(
        f"FinCall coverage: {summary.covered_fincall}/{summary.fincall_total} "
        f"({summary.fincall_coverage_pct}%)"
    )
    typer.echo(
        f"MAEC coverage:    {summary.covered_maec}/{summary.maec_total} "
        f"({summary.maec_coverage_pct}%)"
    )
    typer.echo(
        f"Combined:         {summary.covered_total}/{summary.universe_total} "
        f"({summary.combined_coverage_pct}%)"
    )
    typer.echo(f"missing: {len(summary.missing_tickers)}; gappy: {len(summary.low_completeness)}")
    typer.echo("coverage report: data/coverage/prices_coverage.csv")


@prices_app.command("crosscheck")
def prices_crosscheck(
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    fraction: float = typer.Option(0.05, help="Fraction of covered tickers to sample."),
    seed: int = typer.Option(0, help="Sampling seed."),
) -> None:
    """Cross-check the price pull against Tiingo on a random sample (DESIGN §5.2 gate)."""
    from ecvol.data.tiingo import (
        cross_check,
        load_api_key,
        load_documented_exceptions,
        sample_tickers,
        write_crosscheck_report,
    )

    key = load_api_key()
    if not key:
        typer.echo(
            "no Tiingo key — set TIINGO_API_KEY (env or .env). Free key: https://www.tiingo.com/",
            err=True,
        )
        raise typer.Exit(code=2)
    prices_dir = root / "prices"
    covered = sorted(p.stem for p in prices_dir.glob("*.parquet"))
    if not covered:
        typer.echo("no cached prices — run `ecvol prices pull` first", err=True)
        raise typer.Exit(code=2)
    coverage_dir = root / "coverage"
    documented = load_documented_exceptions(coverage_dir / "crosscheck_exceptions.csv")
    sample = sample_tickers(covered, fraction=fraction, seed=seed)
    result = cross_check(prices_dir, sample, key, documented=documented)
    for r in result.rows:
        note = f" — documented: {documented[r.ticker]}" if r.ticker in documented else ""
        typer.echo(f"  {r.ticker:8} corr={r.correlation} n={r.n_overlap} [{r.status}]{note}")
    report = write_crosscheck_report(result, documented, coverage_dir, fraction=fraction, seed=seed)
    typer.echo(
        f"sampled {result.n_sampled}; passed {result.n_passed}; min corr {result.min_correlation}"
    )
    typer.echo(f"report: {report}")
    if not result.gate_passed:
        typer.echo(
            "cross-check gate FAILED — undocumented sub-0.99 ticker(s): "
            + ", ".join(result.undocumented)
            + f"\ninvestigate, then add a reason to {coverage_dir / 'crosscheck_exceptions.csv'}",
            err=True,
        )
        raise typer.Exit(code=1)
    typer.echo("cross-check gate PASSED (corr>0.999, or documented exception)")


targets_app = typer.Typer(no_args_is_help=True, help="Volatility target computation (T1.3).")
app.add_typer(targets_app, name="targets")


@targets_app.command("build")
def targets_build(
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    horizons: str = typer.Option("3,7,15,30", help="Comma-separated trading-day horizons."),
) -> None:
    """Compute v_pre/v_post/Δv + HAR inputs per (call, horizon) → parquet + report (T1.3)."""
    from ecvol.data.targets import build_targets

    taus = tuple(int(h) for h in horizons.split(",") if h.strip())
    summary = build_targets(root, horizons=taus)
    typer.echo(
        f"calls: {summary.resolved_calls}/{summary.total_calls} resolved; "
        f"rows: {summary.ok_rows}/{summary.rows_total} ok"
    )
    typer.echo(
        f"join rate: {summary.calls_with_any_ok}/{summary.resolved_calls} "
        f"calls with ≥1 target ({summary.join_rate_pct}%)"
    )
    typer.echo("per-horizon ok: " + ", ".join(f"{h}d={n}" for h, n in summary.horizon_ok.items()))
    if summary.reason_counts:
        typer.echo("exclusions: " + ", ".join(f"{k}={v}" for k, v in summary.reason_counts.items()))
    typer.echo("targets: data/fincall/targets.parquet; report: data/coverage/targets_report.csv")


splits_app = typer.Typer(no_args_is_help=True, help="Leakage-proof split construction (T1.6).")
app.add_typer(splits_app, name="splits")


@splits_app.command("build")
def splits_build(
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    embargo: int = typer.Option(30, help="Trading-day embargo between temporal segments."),
    seed: int = typer.Option(0, help="Seed for the ticker-disjoint partition."),
) -> None:
    """Build temporal / ticker-disjoint / combined split CSVs per dataset (T1.6)."""
    from ecvol.data.splits import build_splits

    for s in build_splits(root, embargo=embargo, seed=seed):
        typer.echo(f"{s.dataset}: cohort={s.cohort} horizon={s.horizon} embargo={s.embargo}")
        for scheme, counts in s.scheme_counts.items():
            parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            typer.echo(f"  {scheme}: {parts}")
    typer.echo("splits: data/splits/<dataset>_<scheme>.csv (committed)")


@app.command()
def featurize() -> None:
    """Extract and cache text / audio / LLM features (Phases 3-6)."""
    _not_implemented("featurize")


@app.command()
def train() -> None:
    """Train a model from a validated YAML config (Phases 2-5)."""
    _not_implemented("train")


@app.command()
def evaluate(
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    seeds: str = typer.Option("0,1,2,3,4", help="Comma-separated seeds for the GBDT baseline."),
) -> None:
    """Run Stage-0/1 baselines → Result Table 1 + sanity gates (T2.2)."""
    from ecvol.eval.evaluate import run_evaluate

    seed_tuple = tuple(int(s) for s in seeds.split(",") if s.strip())
    s = run_evaluate(root, seeds=seed_tuple)
    for ds, frac in s.garch_convergence.items():
        flag = "OK" if frac > 0.95 else "BELOW 95% gate"
        typer.echo(f"GARCH convergence {ds}: {frac:.1%} [{flag}]")
    typer.echo("HAR vs persistence R2_OOS (tau=30, level-v, test):")
    for cell, r2 in s.gate_detail["har_r2_oos_vs_persistence"].items():
        typer.echo(f"  {cell}: {r2:+.4f}")
    if s.gate_detail["literal_pass"]:
        typer.echo("sanity gate: PASSED (HAR>persistence, FinCall temporal tau=30)")
    elif s.gate_detail["covid_regime_exception"]:
        typer.echo(
            "sanity gate: PASSED with documented COVID-regime exception "
            "(FinCall temporal tau=30 fails, but targets corroborated by FinCall "
            "ticker-disjoint + MAEC temporal; DECISIONS 2026-06-18)"
        )
    typer.echo("Result Table 1: data/results/result_table_1.csv")
    if not s.gate_passed:
        typer.echo("gate FAILED — halt and debug targets (DESIGN §6 Stage 0)", err=True)
        raise typer.Exit(code=1)


@app.command()
def report() -> None:
    """Regenerate result tables from run artifacts (T2.3)."""
    _not_implemented("report")
