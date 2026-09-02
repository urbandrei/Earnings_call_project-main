"""ecvol CLI — Typer app with one verb per pipeline stage.

Every verb is a stub until its task lands (see TASKS.md). The CLI contract
(idempotent, resumable, config-driven) is DESIGN.md §8.2.
"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import typer

# Several commands print non-ASCII (κ, em dashes). Python uses the console encoding for a
# terminal but falls back to the locale codepage (cp1252 here) when stdout is redirected to a
# file or pipe, where those characters raise UnicodeEncodeError — which killed an unattended
# run that logged to a file. Force UTF-8 on both streams.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

app = typer.Typer(
    no_args_is_help=True,
    help="Earnings-call volatility prediction pipeline.",
)

# configs/<command>.yaml is the default; a module-level singleton keeps ruff B008 quiet.
_CONFIG_OPT = typer.Option(None, help="Command config YAML (default: configs/<command>.yaml).")


def _parse_stop_at(hhmm: str) -> float:
    """``"19:30"`` → the next local-time epoch at that clock time (today, else tomorrow)."""
    try:
        hh, mm = (int(p) for p in hhmm.split(":"))
    except ValueError as exc:
        raise typer.BadParameter(f"--stop-at must be HH:MM, got {hhmm!r}") from exc
    now = datetime.now()
    stop = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if stop <= now:
        stop += timedelta(days=1)
    return stop.timestamp()


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
    dataset: str = typer.Argument(help="Dataset to normalize: fincall | maec | earnings25 | ec."),
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    no_audio: bool = typer.Option(
        False, help="FinCall only: skip ffprobe audio-duration probing (durations left NaN)."
    ),
) -> None:
    """Normalize a dataset onto the common call schema → parquet + reports (T1.4/T1.5/T7.1)."""
    if dataset not in ("fincall", "maec", "earnings25", "ec"):
        typer.echo(
            f"unknown dataset {dataset!r} (expected fincall | maec | earnings25 | ec)", err=True
        )
        raise typer.Exit(code=2)
    if dataset == "ec":
        from ecvol.data.ec_ingest import ingest_ec

        c = ingest_ec(root)
        typer.echo(
            f"calls: {c.ok}/{c.total_calls} ok; published split covers {c.published_split_calls}"
        )
        typer.echo(
            f"join: {c.joined}/{c.ok} ok calls with >=1 target ({c.join_rate_pct}%); "
            f"missing-price tickers: {c.missing_price_tickers}"
        )
        if c.reason_counts:
            typer.echo("exclusions: " + ", ".join(f"{k}={v}" for k, v in c.reason_counts.items()))
        typer.echo("calls/targets: data/ec/*.parquet; split: data/splits/ec_published.csv")
        return
    if dataset == "earnings25":
        from ecvol.data.earnings25_ingest import ingest_earnings25

        e = ingest_earnings25(root, probe_audio=not no_audio)
        typer.echo(f"calls: {e.ok}/{e.total_calls} admitted (months: {' '.join(e.months)})")
        typer.echo(
            f"join: {e.joined}/{e.ok} admitted calls with >=1 target ({e.join_rate_pct}%); "
            f"missing-price tickers: {e.missing_price_tickers}"
        )
        if e.reason_counts:
            typer.echo("exclusions: " + ", ".join(f"{k}={v}" for k, v in e.reason_counts.items()))
        if e.strata:
            typer.echo("bitrate strata: " + ", ".join(f"{k}={v}" for k, v in e.strata.items()))
        typer.echo("calls/targets: data/earnings25/*.parquet; reports: data/coverage/earnings25_*")
        return
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


def _pull_ec_prices(root: Path, refresh: bool) -> None:
    """EC tickers missing from the shared 2014–2022 archive (T6R.2) → data/prices/ + prices.json."""
    from ecvol.data.ec_ingest import TICKER_ALIASES, ec_tickers
    from ecvol.data.prices import END, START, pull_tickers

    tickers = ec_tickers(root)
    fetched, missing = pull_tickers(
        tickers,
        root / "prices",
        root,
        start=START,
        end=END,
        manifest_name="prices.json",
        refresh=refresh,
        aliases=TICKER_ALIASES,
    )
    typer.echo(
        f"ec prices: {len(tickers)} tickers; fetched {len(fetched)}; "
        f"missing {len(missing)}: {missing}"
    )


def _pull_earnings25_prices(root: Path, refresh: bool) -> None:
    """Earnings25's own price store (T7.1): the admitted tickers, 2025-06-01 → 2026-03-31."""
    from datetime import date

    from ecvol.collect.discovery import _cik_by_ticker
    from ecvol.data.earnings25_ingest import (
        DATASET_REL,
        PRICES_REL,
        build_records,
        membership_by_month,
        read_records,
    )
    from ecvol.data.fincall_identity import load_sec_table
    from ecvol.data.prices import pull_tickers

    records = read_records(root / DATASET_REL)
    months = sorted({r["extra_fields"]["ReleaseDate"][:7] for r in records})
    members, _ = membership_by_month(root, months)
    sec, _ = load_sec_table(root)
    calls = build_records(records, sec, members, _cik_by_ticker(root))
    tickers = sorted({c.record.ticker for c in calls if c.record.status == "ok"})
    fetched, missing = pull_tickers(
        tickers,
        root / PRICES_REL,
        root,
        start=date(2025, 6, 1),
        end=date(2026, 3, 31),
        manifest_name="prices_earnings25.json",
        refresh=refresh,
    )
    typer.echo(
        f"earnings25 prices: {len(tickers)} tickers; fetched {len(fetched)}; missing {missing}"
    )
    typer.echo(f"store: data/{PRICES_REL}/; manifest: data/manifests/prices_earnings25.json")


app.add_typer(prices_app, name="prices")


@prices_app.command("pull")
def prices_pull(
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    refresh: bool = typer.Option(False, help="Re-download tickers even if cached."),
    dataset: str = typer.Option(
        "fincall+maec",
        help="Universe: fincall+maec (T1.2 archive) | earnings25 (own store) | ec (archive).",
    ),
) -> None:
    """Pull adjusted daily OHLCV for the FinCall+MAEC universe → parquet + coverage (T1.2)."""
    from ecvol.data.prices import pull_prices

    if dataset == "earnings25":
        _pull_earnings25_prices(root, refresh)
        return
    if dataset == "ec":
        _pull_ec_prices(root, refresh)
        return
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
    """Compute v_pre/v_post/Δv + HAR inputs per (call, horizon), both conventions (T1.3, T9.1)."""
    from ecvol.data.targets import TARGET_FILES, build_targets

    taus = tuple(int(h) for h in horizons.split(",") if h.strip())
    summaries = build_targets(root, horizons=taus)
    for convention, summary in summaries.items():
        typer.echo(f"[{convention}-day horizons]")
        typer.echo(
            f"  calls: {summary.resolved_calls}/{summary.total_calls} resolved; "
            f"rows: {summary.ok_rows}/{summary.rows_total} ok"
        )
        typer.echo(
            f"  join rate: {summary.calls_with_any_ok}/{summary.resolved_calls} "
            f"calls with ≥1 target ({summary.join_rate_pct}%)"
        )
        typer.echo(
            "  per-horizon ok: " + ", ".join(f"{h}d={n}" for h, n in summary.horizon_ok.items())
        )
        if summary.reason_counts:
            typer.echo(
                "  exclusions: " + ", ".join(f"{k}={v}" for k, v in summary.reason_counts.items())
            )
        typer.echo(f"  targets: data/fincall/{TARGET_FILES[convention]}")
    typer.echo("reports: data/coverage/targets_report.csv, targets_calendar_report.csv")


@targets_app.command("compare")
def targets_compare(
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
) -> None:
    """Trading- vs calendar-day target delta per (dataset, horizon) → coverage CSV (T9.1)."""
    from ecvol.data.targets import convention_delta

    table = convention_delta(root)
    if table.empty:
        typer.echo("no dataset has both targets.parquet and targets_calendar.parquet", err=True)
        raise typer.Exit(code=1)
    out = root / "coverage" / "targets_convention_delta.csv"
    table.to_csv(out, index=False, lineterminator="\n", float_format="%.6f")
    for r in table.itertuples(index=False):
        typer.echo(
            f"{r.dataset} tau={r.horizon}: sessions {r.sessions_trading} vs "
            f"{r.sessions_calendar_mean:.1f} [{r.sessions_calendar_min}-{r.sessions_calendar_max}]"
            f"; ok {r.n_ok_trading}/{r.n_ok_calendar} (both {r.n_ok_both}); "
            f"corr v_post={r.corr_v_post:.3f} |diff|={r.mean_abs_diff_v_post:.3f}; "
            f"corr dv={r.corr_delta_v:.3f}"
        )
    typer.echo(f"delta report: {out}")


timing_app = typer.Typer(no_args_is_help=True, help="Call-timestamp retrofit (T9.2).")
app.add_typer(timing_app, name="timing")


@timing_app.command("build")
def timing_build(
    dataset: str = typer.Argument(help="Dataset: fincall | maec | earnings25."),
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    no_fetch: bool = typer.Option(False, help="Offline: use only cached EDGAR submissions."),
    limit: int = typer.Option(0, help="Only the first N resolved calls (0 = all; pilots)."),
) -> None:
    """Measure each call's information boundary (EDGAR 8-K › Earnings25 › DEC › fallback)."""
    from ecvol.data.timing import build_call_times, write_call_times

    rows, counts = build_call_times(root, dataset, fetch=not no_fetch, limit=(limit or None))
    out = root / "coverage" / f"{dataset}_timing.csv"
    write_call_times(rows, out)
    n = len(rows)
    typer.echo(f"{dataset}: {n} resolved calls")
    for tier, c in counts.items():
        typer.echo(f"  {tier}: {c} ({100 * c / n:.1f}%)" if n else f"  {tier}: 0")
    rules: dict[str, int] = {}
    for r in rows:
        rules[r.session_rule] = rules.get(r.session_rule, 0) + 1
    typer.echo("session rule: " + ", ".join(f"{k}={v}" for k, v in sorted(rules.items())))
    typer.echo(f"table: {out}")


@timing_app.command("targets")
def timing_targets(
    dataset: str = typer.Argument(help="Dataset: fincall | maec | earnings25."),
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
) -> None:
    """Measured-anchor target variant → data/{dataset}/targets_measured{,_calendar}.parquet."""
    from ecvol.data.timing import build_measured_targets

    s = build_measured_targets(root, dataset)
    typer.echo(
        f"{dataset}: {s['calls']} calls, {s['measured']} with a measured anchor; "
        f"{s['ok_rows']} ok trading rows → data/{dataset}/targets_measured.parquet"
    )


@timing_app.command("sensitivity")
def timing_sensitivity(
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Fallback vs measured day-0 anchors: Stage-0 baselines on the test segment (T9.2)."""
    from ecvol.data.timing import sensitivity
    from ecvol.tracking import resolve_command_config, write_command_run

    cfg = resolve_command_config("timing-sensitivity", config, None)
    table = sensitivity(root)
    shift = table[table["anchor"] == "shift"]
    for r in shift.itertuples():
        typer.echo(f"{r.dataset}: as_of moved for {100 * r.r2_oos:.1f}% of {r.n} calls")
    head = table[
        (table["anchor"] != "shift") & (table["target"] == "v") & (table["model"] == "har")
    ]
    for r in head.itertuples():
        typer.echo(
            f"  {r.dataset} {r.split} tau={r.horizon} HAR level-v [{r.anchor}]: "
            f"R2={r.r2_oos:+.3f} MSE={r.mse:.3f} (n={r.n})"
        )
    typer.echo("table: data/results/timing_sensitivity.csv")
    typer.echo(f"run artifact: {write_command_run(cfg, root)}")


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


featurize_app = typer.Typer(no_args_is_help=True, help="Text / audio / LLM features (Phases 3-6).")
app.add_typer(featurize_app, name="featurize")


@featurize_app.command("sections")
def featurize_sections(
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    max_words: int = typer.Option(320, help="Chunk word cap; oversized turns are sentence-split."),
    audit_n: int = typer.Option(30, help="Calls in the seeded section-precision audit sample."),
    seed: int = typer.Option(0, help="Audit-sample seed."),
) -> None:
    """Section transcripts (prepared vs Q&A) + speaker-turn chunk -> chunks.parquet (T3.1)."""
    from ecvol.features.text.sections import build_sections

    for s in build_sections(root, max_words=max_words, audit_n=audit_n, seed=seed):
        typer.echo(
            f"{s.dataset}: processed {s.n_processed}/{s.n_calls} "
            f"(no_turns {s.n_no_turns}); Q&A detected {s.calls_with_qa} "
            f"(corroborated {s.corroborated})"
        )
        methods = ", ".join(f"{k}={v}" for k, v in sorted(s.method_counts.items()))
        typer.echo(f"  methods: {methods}")
        typer.echo(
            f"  chunks: {s.total_chunks} (oversize {s.oversize_chunks}); audit: {s.audit_path}"
        )


@featurize_app.command("text")
def featurize_text(
    dataset: str = typer.Option("fincall", help="Dataset: fincall | maec."),
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    limit: int = typer.Option(0, help="Process only the first N calls (0 = full corpus)."),
    device: str = typer.Option("cuda", help="torch device (cuda | cpu)."),
    batch_size: int = typer.Option(32, help="Inference batch size."),
    weighted: bool = typer.Option(False, help="Use n_words-weighted embedding pooling."),
) -> None:
    """Stage-2 frozen text features: BGE-M3 embeddings + FinBERT + surface stats (T3.2)."""
    import time

    import pyarrow.parquet as pq

    from ecvol.features.text import embeddings, finbert, surface

    lim = limit or None
    total = pq.read_metadata(root / dataset / "chunks.parquet").num_rows

    t = time.perf_counter()
    n_surf = surface.build(root, dataset, limit=lim)
    t_surf = time.perf_counter() - t

    t = time.perf_counter()
    n_emb, new_emb = embeddings.build(
        root, dataset, limit=lim, device=device, batch_size=batch_size, weighted=weighted
    )
    t_emb = time.perf_counter() - t

    t = time.perf_counter()
    n_fin, new_fin = finbert.build(root, dataset, limit=lim, device=device, batch_size=batch_size)
    t_fin = time.perf_counter() - t

    typer.echo(f"{dataset}: {total} chunks total; device={device}")
    typer.echo(f"  surface:    {n_surf} rows in {t_surf:.1f}s")
    typer.echo(f"  embeddings: {n_emb} rows, {new_emb} chunks encoded in {t_emb:.1f}s")
    typer.echo(f"  finbert:    {n_fin} rows, {new_fin} chunks in {t_fin:.1f}s")
    if lim and new_emb and new_fin:
        r_emb, r_fin = new_emb / t_emb, new_fin / t_fin
        eta_min = (total / r_emb + total / r_fin) / 60
        typer.echo(f"  throughput: BGE-M3 {r_emb:.0f} ch/s, FinBERT {r_fin:.0f} ch/s")
        typer.echo(
            f"  full-corpus ETA (~{total} chunks): ~{eta_min:.1f} min "
            "(incl. one-time model load; conservative)"
        )


@featurize_app.command("llm-reading-pack")
def featurize_llm_reading_pack(
    dataset: str = typer.Option("fincall", help="Dataset: fincall | maec."),
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    n: int = typer.Option(20, help="Calls to sample (train split only) for human reading."),
    seed: int = typer.Option(0, help="Sample seed (deterministic)."),
) -> None:
    """Render N train-split calls + a blank rubric labeling sheet for the T6.1 human pass."""
    from ecvol.features.llm.reading import build_reading_pack

    pack = build_reading_pack(root, dataset, n=n, seed=seed)
    typer.echo(f"{dataset}: {len(pack.call_ids)} train-split calls (seed {seed})")
    typer.echo(f"  transcripts: {pack.reading_dir} (gitignored payload)")
    typer.echo(f"  labeling sheet: {pack.sheet_path}")
    typer.echo("  rubric: docs/llm_feature_rubric.md — fill the sheet, then sign off (HANDOFF)")


@featurize_app.command("llm-audit-sample")
def featurize_llm_audit_sample(
    dataset: str = typer.Option("fincall", help="Dataset: fincall | maec."),
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    n: int = typer.Option(50, help="Audit calls (train split only); the κ-gate sample size."),
    seed: int = typer.Option(0, help="Sample seed (deterministic; must match the probe)."),
) -> None:
    """Fix the 50-call train-only κ-audit sample + (re)write its labeling sheet (T6.2)."""
    from ecvol.features.llm.reading import build_reading_pack

    pack = build_reading_pack(root, dataset, n=n, seed=seed)
    typer.echo(f"{dataset}: {len(pack.call_ids)} train-split audit calls (seed {seed})")
    typer.echo(f"  transcripts: {pack.reading_dir} (gitignored payload)")
    typer.echo(f"  labeling sheet: {pack.sheet_path}")
    typer.echo("  fill this sheet, then `ecvol llm-kappa` once features are extracted")


@featurize_app.command("llm-rating-workbook")
def featurize_llm_rating_workbook(
    dataset: str = typer.Option("fincall", help="Dataset: fincall | maec."),
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    out: Path = typer.Option(  # noqa: B008
        Path("ingest/Ratings_2.xlsx"), help="Workbook to write (hand this to the rater)."
    ),
    n_calls: int = typer.Option(
        0, help="Rate only N calls (0 = all). A partial pass needs --allow-subset on ingest."
    ),
    seed: int = typer.Option(0, help="Seed for the call-order shuffle / subset."),
    no_shuffle: bool = typer.Option(
        False,
        help="Keep the frozen sheet's call order. Default shuffles it: a second pass by the "
        "same rater must not walk the original sequence or it measures recall, not reliability.",
    ),
    sheet_name: str = typer.Option("Ratings", help="Worksheet name (ingest expects 'Ratings')."),
) -> None:
    """Blank rater workbook (.xlsx) from the frozen label sheet — the human deliverable (T6.2)."""
    from ecvol.features.llm.reading import build_rating_workbook

    sheet_csv = root / "coverage" / f"{dataset}_llm_label_sheet.csv"
    if not sheet_csv.is_file():
        raise typer.BadParameter(f"frozen label sheet not found: {sheet_csv}")
    res = build_rating_workbook(
        sheet_csv,
        out,
        n_calls=(n_calls or None),
        seed=seed,
        shuffle=not no_shuffle,
        sheet_name=sheet_name,
    )
    typer.echo(f"workbook: {res.out_path}  ({res.n_rows} rows / {res.n_calls} calls)")
    typer.echo(f"  sheet: {res.sheet_name!r}; transcripts: {root / dataset / 'llm_reading'}")
    typer.echo("  rubric: docs/llm_feature_rubric.md")
    typer.echo("  BLIND: do not open the other raters' label CSVs or llm_features__*.parquet")
    typer.echo(
        f"  next: `ecvol featurize llm-ingest-ratings --xlsx {res.out_path} --rater rater2`"
        + (" --allow-subset" if n_calls else "")
    )


@featurize_app.command("llm-ingest-ratings")
def featurize_llm_ingest_ratings(
    xlsx: Path = typer.Option(..., help="Rater workbook (.xlsx) with a filled 'Ratings' sheet."),  # noqa: B008
    rater: str = typer.Option(..., help="Rater id (names the output, e.g. 'rater1')."),
    dataset: str = typer.Option("fincall", help="Dataset: fincall | maec."),
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    sheet_name: str = typer.Option("Ratings", help="Worksheet name holding the ratings."),
    allow_subset: bool = typer.Option(
        False,
        help="Accept a strict subset of the frozen audit rows (a deliberate partial re-rate, "
        "e.g. a blinded 20-call annotator-ceiling pass). Extra rows remain fatal.",
    ),
) -> None:
    """Ingest a human rater workbook → canonical κ-audit label CSV (validated vs. the sample)."""
    from ecvol.features.llm.ratings import ingest_ratings

    reference = root / "coverage" / f"{dataset}_llm_label_sheet.csv"
    out_path = root / "coverage" / f"{dataset}_llm_labels_{rater}.csv"
    res = ingest_ratings(
        xlsx,
        out_path,
        rater=rater,
        reference_sheet=reference if reference.exists() else None,
        sheet_name=sheet_name,
        allow_subset=allow_subset,
    )
    typer.echo(f"{dataset}: ingested {res.n_rows} rows / {res.n_calls} calls from rater {rater!r}")
    typer.echo(f"  labels: {res.out_path}")
    if reference.exists():
        typer.echo(f"  validated against frozen audit sample: {reference}")
    else:
        typer.echo("  WARNING: reference sample sheet not found — alignment NOT validated")
    typer.echo(
        f"  next: `ecvol llm-kappa --sheet {res.out_path} --features <llm_features.parquet>`"
    )


@featurize_app.command("llm-eta")
def featurize_llm_eta(
    dataset: str = typer.Option("fincall", help="Dataset whose train audit sample to probe."),
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    model_id: str = typer.Option("Qwen/Qwen2.5-7B-Instruct", help="HF model id."),
    revision: str = typer.Option("", help="Pinned HF commit (recommended; DESIGN §12)."),
    n_probe: int = typer.Option(50, help="Audit calls to extract for timing (aligns with sample)."),
    seed: int = typer.Option(0, help="Sample seed (must match llm-audit-sample)."),
    device: str = typer.Option("cuda", help="torch device (cuda | cpu)."),
    no_4bit: bool = typer.Option(False, help="Disable bitsandbytes 4-bit (use fp16)."),
) -> None:
    """Probe: extract the train audit sample, report rate + full-corpus ETA + VRAM (T6.2)."""
    from ecvol.features.llm.extract import build_llm, total_sections
    from ecvol.features.llm.reading import sample_train_calls

    call_ids = sample_train_calls(root, dataset, n_probe, seed)
    res = build_llm(
        root,
        dataset,
        model_id=model_id,
        revision=revision or None,
        engine="transformers",
        device=device,
        call_ids=call_ids,
        load_in_4bit=not no_4bit,
    )
    total = total_sections(root, ("fincall", "maec"))
    typer.echo(
        f"{dataset}: probed {res.n_new} new ({res.n_rows} total) sections in {res.secs:.1f}s"
    )
    typer.echo(f"  features: {res.out_path}")
    if res.n_new and res.secs > 0:
        rate = res.n_new / res.secs
        eta_h = total / rate / 3600
        typer.echo(f"  throughput: {rate:.3f} sections/s ({rate * 3600:.0f}/h)")
        typer.echo(
            f"  full-corpus ETA (~{total} sections, fincall+maec): ~{eta_h:.1f}h "
            f"on this GPU with {model_id} 4-bit"
        )
        typer.echo("  RULE: >20h here → route the corpus to OSC (cloud/osc/, DECISIONS spend gate)")
    try:
        import torch

        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1e9
            typer.echo(f"  peak VRAM: {peak:.1f} GB")
    except Exception:  # noqa: BLE001 - VRAM read is best-effort
        pass


@featurize_app.command("llm")
def featurize_llm(
    dataset: str = typer.Option("fincall", help="Dataset: fincall | maec."),
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    model_id: str = typer.Option("Qwen/Qwen2.5-7B-Instruct", help="HF model id."),
    revision: str = typer.Option("", help="Pinned HF commit (recommended; DESIGN §12)."),
    engine: str = typer.Option(
        "transformers", help="Inference engine: transformers | llamacpp | vllm."
    ),
    device: str = typer.Option("cuda", help="torch device (transformers engine)."),
    gguf_path: Path = typer.Option(  # noqa: B008
        None, help="llamacpp: path to the GGUF weights file (required for --engine llamacpp)."
    ),
    server_bin: Path = typer.Option(  # noqa: B008
        None, help="llamacpp: path to llama-server.exe."
    ),
    n_gpu_layers: int = typer.Option(99, help="llamacpp: layers offloaded to VRAM (99 = all)."),
    stop_at: str = typer.Option(
        "",
        help="Stop cleanly at this local wall-clock time (HH:MM, today; next day if already "
        "past). The current section finishes, the parquet is flushed, the engine is released.",
    ),
    limit: int = typer.Option(0, help="Process only the first N calls (0 = full corpus)."),
    audit_sample: bool = typer.Option(
        False,
        help="Restrict to the 50-call train-only audit sample (the κ-gate calls) instead of the "
        "corpus. Engine-agnostic, so the SAME engine that runs the corpus scores the gate "
        "(audit-matches-corpus rule, DECISIONS 2026-06-24). Run this + `ecvol llm-kappa` and "
        "clear κ>0.6 before the full run. Mutually exclusive with --limit.",
    ),
    audit_n: int = typer.Option(50, help="Audit-sample size (must match the labeling sheet)."),
    audit_seed: int = typer.Option(0, help="Audit-sample seed (must match llm-audit-sample)."),
    no_4bit: bool = typer.Option(False, help="Disable bitsandbytes 4-bit (transformers engine)."),
    max_model_len: int = typer.Option(
        0, help="vLLM context window (0 = model default; set ~65536 to cover the >32k tail)."
    ),
    yarn: bool = typer.Option(
        False,
        help="vLLM: enable YaRN rope-scaling to reach --max-model-len beyond the model's native "
        "context (the chosen >32k policy; DECISIONS 2026-06-29). Requires --max-model-len.",
    ),
    yarn_native: int = typer.Option(
        32768, help="Model native max context for the YaRN factor (Qwen2.5 = 32768)."
    ),
) -> None:
    """Constrained LLM structured-feature extraction → llm_features__{model}.parquet (T6.2)."""
    from ecvol.features.llm.extract import build_llm, yarn_rope_scaling

    if audit_sample and limit:
        raise typer.BadParameter("--audit-sample and --limit are mutually exclusive")
    call_ids = None
    if audit_sample:
        from ecvol.features.llm.reading import sample_train_calls

        call_ids = sample_train_calls(root, dataset, audit_n, audit_seed)
        typer.echo(f"audit sample: {len(call_ids)} train-only calls (seed={audit_seed}) — κ-gate")

    kwargs = {}
    if engine == "transformers":
        kwargs = {"device": device, "load_in_4bit": not no_4bit}
    elif engine in ("vllm", "llamacpp"):
        if max_model_len:
            kwargs["max_model_len"] = max_model_len
        if yarn:
            if not max_model_len:
                raise typer.BadParameter("--yarn requires --max-model-len (e.g. 65536)")
            kwargs["rope_scaling"] = yarn_rope_scaling(max_model_len, native=yarn_native)
        if engine == "llamacpp":
            if not gguf_path or not server_bin:
                raise typer.BadParameter("--engine llamacpp requires --gguf-path and --server-bin")
            kwargs |= {
                "gguf_path": gguf_path,
                "server_bin": server_bin,
                "n_gpu_layers": n_gpu_layers,
            }

    deadline = None
    if stop_at:
        deadline = _parse_stop_at(stop_at)
        typer.echo(f"stop-at: {time.strftime('%Y-%m-%d %H:%M %Z', time.localtime(deadline))}")

    res = build_llm(
        root,
        dataset,
        model_id=model_id,
        revision=revision or None,
        engine=engine,
        call_ids=call_ids,
        limit=(limit or None),
        deadline=deadline,
        **kwargs,
    )
    typer.echo(f"{dataset}: {res.n_new} new sections, {res.n_rows} total in {res.secs:.1f}s")
    typer.echo(f"  features: {res.out_path}")


audio_app = typer.Typer(no_args_is_help=True, help="Audio QC + features (Phase 4).")
app.add_typer(audio_app, name="audio")


@audio_app.command("qc")
def audio_qc(
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    limit: int = typer.Option(0, help="Process only the first N calls (0 = all)."),
    workers: int = typer.Option(8, help="Parallel ffmpeg workers."),
    dataset: str = typer.Option("fincall", help="Dataset with raw audio: fincall | earnings25."),
) -> None:
    """QC a dataset's audio + write its 16 kHz mono FLAC store (T4.1/T9.4; MAEC has no audio)."""
    from ecvol.features.audio.qc import build_qc, have_ffmpeg

    if not have_ffmpeg():
        typer.echo("ffmpeg/ffprobe not found on PATH", err=True)
        raise typer.Exit(code=2)
    s = build_qc(root, dataset, limit=(limit or None), workers=workers)
    typer.echo(f"audio QC: {s.decoded}/{s.n} decoded; store: {s.store_dir}")
    typer.echo(
        "flagged: " + (", ".join(f"{k}={v}" for k, v in sorted(s.flagged.items())) or "none")
    )
    typer.echo(f"report: data/coverage/{dataset}_audio_qc.csv")


@audio_app.command("qc-ref")
def audio_qc_ref(
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    n: int = typer.Option(3, help="Number of Earnings-21 reference files to fetch + QC."),
) -> None:
    """Validate the QC pipeline on known-good Earnings-21 samples (T4.1; needs --group gpu)."""
    from ecvol.features.audio.qc import validate_earnings21

    rows = validate_earnings21(root, n=n)
    for r in rows:
        typer.echo(
            f"  {r['call_id']}: decode={r['decode_ok']} sr={r['sample_rate']} "
            f"peak={r['peak_dbfs']:.1f}dB silence={r['silence_ratio']:.2f} "
            f"reason={r['reason'] or 'ok'}"
        )
    typer.echo("report: data/coverage/earnings21_qc_validation.csv")


@audio_app.command("egemaps")
def audio_egemaps(
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    limit: int = typer.Option(0, help="Process only the first N decoded calls (0 = all)."),
    workers: int = typer.Option(8, help="Parallel openSMILE workers."),
    dataset: str = typer.Option("fincall", help="Dataset: fincall | earnings25."),
) -> None:
    """Extract eGeMAPSv02 functionals (88) per call → parquet + summary (T4.2/T9.4)."""
    from ecvol.features.audio.egemaps import build_egemaps

    n, fails, features = build_egemaps(root, dataset, limit=(limit or None), workers=workers)
    typer.echo(f"eGeMAPS: {n} calls × {len(features)} features; failures: {fails}")
    typer.echo(f"output: data/{dataset}/audio_egemaps.parquet")
    typer.echo(f"summary: data/coverage/{dataset}_egemaps_summary.csv")


@audio_app.command("wavlm")
def audio_wavlm(
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    limit: int = typer.Option(0, help="Embed only the first N calls (use 50 for the ETA gate)."),
    device: str = typer.Option("cuda", help="torch device (cuda | cpu)."),
    fp16: bool = typer.Option(False, help="Half precision (faster; slightly non-deterministic)."),
    batch: int = typer.Option(4, help="Windows per forward pass (VRAM-bound)."),
    dataset: str = typer.Option("fincall", help="Dataset: fincall | earnings25."),
) -> None:
    """WavLM-Large per-call audio embeddings → parquet (T4.3; resumable). Run --limit 50 first."""
    import pandas as pd

    from ecvol.features.audio.wavlm import build_wavlm

    lim = limit or None
    total = int(pd.read_csv(root / "coverage" / f"{dataset}_audio_qc.csv")["decode_ok"].sum())
    n, n_new, secs = build_wavlm(root, dataset, limit=lim, device=device, fp16=fp16, batch=batch)
    typer.echo(
        f"WavLM: {n} calls embedded ({n_new} new in {secs:.0f}s); output: audio_wavlm.parquet"
    )
    if lim and n_new:
        rate = n_new / secs
        typer.echo(
            f"throughput: {rate:.2f} calls/s → full-corpus ETA (~{total} calls): "
            f"~{total / rate / 60:.0f} min (incl. one-time model load)"
        )


@audio_app.command("emotion2vec")
def audio_emotion2vec(
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    limit: int = typer.Option(0, help="Embed only the first N calls (use a small N for ETA)."),
    device: str = typer.Option("cuda", help="torch device (cuda | cpu)."),
    dataset: str = typer.Option("fincall", help="Dataset: fincall | earnings25."),
) -> None:
    """emotion2vec+ per-call audio embeddings → parquet (T4.3; resumable). Run a small --limit first."""  # noqa: E501
    import pandas as pd

    from ecvol.features.audio.emotion2vec import build_emotion2vec

    lim = limit or None
    total = int(pd.read_csv(root / "coverage" / f"{dataset}_audio_qc.csv")["decode_ok"].sum())
    n, n_new, secs = build_emotion2vec(root, dataset, limit=lim, device=device)
    typer.echo(f"emotion2vec+: {n} calls ({n_new} new in {secs:.0f}s) → audio_emotion2vec.parquet")
    if lim and n_new:
        rate = n_new / secs
        typer.echo(
            f"throughput: {rate:.2f} calls/s → full ETA (~{total}): ~{total / rate / 60:.0f} min"
        )


# Shared `--config` option for the result-producing commands (T9.3): the committed


@app.command()
def train() -> None:
    """Train a model from a validated YAML config (Phases 2-5)."""
    _not_implemented("train")


@app.command()
def evaluate(
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    seeds: str | None = typer.Option(None, help="Comma-separated seeds; overrides the config."),
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Run Stage-0/1 baselines → Result Table 1 + sanity gates (T2.2)."""
    from ecvol.eval.evaluate import run_evaluate
    from ecvol.tracking import resolve_command_config, write_command_run

    cfg = resolve_command_config("evaluate", config, seeds)
    s = run_evaluate(root, seeds=tuple(cfg.seeds))
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
    typer.echo(f"run artifact: {write_command_run(cfg, root)}")
    if not s.gate_passed:
        typer.echo("gate FAILED — halt and debug targets (DESIGN §6 Stage 0)", err=True)
        raise typer.Exit(code=1)


@app.command(name="evaluate-text")
def evaluate_text(
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    seeds: str | None = typer.Option(None, help="Comma-separated seeds; overrides the config."),
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Run Stage-2 content heads (ridge + MLP on text features) → Result Table 2 (T3.3)."""
    from ecvol.eval.stage2 import run_stage2
    from ecvol.tracking import resolve_command_config, write_command_run

    cfg = resolve_command_config("evaluate-text", config, seeds)
    table = run_stage2(root, seeds=tuple(cfg.seeds))
    typer.echo(f"Result Table 2: {len(table)} rows → data/results/result_table_2.csv")
    head = table[(table["target"] == "dv") & (table["segment"] == "test")]
    sig = head[head["dm_p_vs_stage1"] < 0.05]
    typer.echo(
        f"Δv test cells: {len(head)}; DM-significant vs Stage-1 (p<0.05): {len(sig)} "
        "(see `ecvol report` for the rendered tables)"
    )
    typer.echo(f"run artifact: {write_command_run(cfg, root)}")


@app.command()
def controls(
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    seeds: str | None = typer.Option(None, help="Comma-separated seeds; overrides the config."),
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Run the §7.3 identity-control suite (ticker-only, shuffle, probe) → control tables (T3.4)."""
    from ecvol.eval.controls import run_controls
    from ecvol.tracking import resolve_command_config, write_command_run

    cfg = resolve_command_config("controls", config, seeds)
    ctrl, probe = run_controls(root, seeds=tuple(cfg.seeds))
    typer.echo(f"Result Controls: {len(ctrl)} rows → data/results/result_controls.csv")
    for r in probe.itertuples():
        typer.echo(
            f"identity probe [{r.dataset}]: acc={r.probe_accuracy:.3f} "
            f"vs chance {r.chance:.4f} ({r.accuracy_over_chance:.0f}x), "
            f"{r.n_tickers} tickers / {r.n_calls} calls"
        )
    typer.echo(f"run artifact: {write_command_run(cfg, root)}")


@app.command(name="evaluate-audio")
def evaluate_audio(
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    seeds: str | None = typer.Option(None, help="Comma-separated seeds; overrides the config."),
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Stage-3 audio heads → Result Table 3 + identity probe + §3.5 gender analysis (T4.4)."""
    from ecvol.eval.audio_eval import run_audio_eval
    from ecvol.eval.stage3 import run_stage3
    from ecvol.tracking import resolve_command_config, write_command_run

    cfg = resolve_command_config("evaluate-audio", config, seeds)
    table = run_stage3(root, seeds=tuple(cfg.seeds))
    typer.echo(f"Result Table 3: {len(table)} rows → data/results/result_table_3.csv")
    probe, gender, shuffle = run_audio_eval(root)
    for r in probe.itertuples():
        typer.echo(
            f"identity probe [{r.embedding}]: acc={r.probe_accuracy:.3f} "
            f"vs chance {r.chance:.4f} ({r.accuracy_over_chance:.0f}x)"
        )
    g = dict(zip(gender["metric"], gender["value"], strict=True))
    typer.echo(
        f"gender (F0 proxy, coverage {g['f0_proxy_coverage']}): "
        f"MSE low={g['mse_low_pitch']} high={g['mse_high_pitch']}; "
        f"corr(F0,err2)={g['corr_f0_sq_error']}"
    )
    typer.echo(
        f"audio shuffle: {len(shuffle)} cells (real vs within/global) → "
        "data/results/audio_shuffle.csv"
    )
    typer.echo(f"run artifact: {write_command_run(cfg, root)}")


audit_app = typer.Typer(no_args_is_help=True, help="Benchmark substrate audit (T6R.1).")
app.add_typer(audit_app, name="audit")


@audit_app.command("substrate")
def audit_substrate(
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Split overlap, embargo, label defects and label diff for EC / MAEC-15 / MAEC-16 (5R)."""
    from ecvol.eval.substrate import run_substrate_audit
    from ecvol.tracking import resolve_command_config, write_command_run

    cfg = resolve_command_config("audit-substrate", config, None)
    table, labels = run_substrate_audit(root)
    for r in table[table["metric"] == "test_ticker_in_train_share"].itertuples():
        typer.echo(f"  {r.benchmark}: test-ticker-in-train {100 * float(r.value):.1f}%")
    for r in table[table["metric"].str.startswith("embargo_days")].itertuples():
        typer.echo(f"  {r.benchmark}: {r.metric} = {r.value} d")
    z = table[table["metric"] == "single_day_exact_zeros_all"].iloc[0]
    typer.echo(f"  EC single-day series: {z.numerator} exact zeros / {z.denominator} cells")
    for r in labels.itertuples():
        typer.echo(
            f"  {r.benchmark} tau={r.horizon}: joined {r.n_joined}/{r.n_shipped}, "
            f"corr={r.corr:.3f}, |diff|={r.mean_abs_diff:.3f}"
        )
    typer.echo("Result Table 5R: data/results/result_table_5r.csv (+ _labels.csv)")
    typer.echo(f"run artifact: {write_command_run(cfg, root)}")


reproduce_app = typer.Typer(no_args_is_help=True, help="Reproduce published models (T6R.2).")
app.add_typer(reproduce_app, name="reproduce")


@reproduce_app.command("html")
def reproduce_html(
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    seeds: str | None = typer.Option(None, help="Comma-separated seeds; overrides the config."),
    epochs: int = typer.Option(30, help="Training epochs per fit (paper: until convergence)."),
    config: Path | None = _CONFIG_OPT,
) -> None:
    """HTML (Yang et al. 2020) on EC: published split/labels, then our splits → Result Table 6R."""
    from ecvol.eval.reproduce import run_html_reproduction
    from ecvol.tracking import resolve_command_config, write_command_run

    cfg = resolve_command_config("reproduce-html", config, seeds)
    table = run_html_reproduction(root, seeds=tuple(cfg.seeds), epochs=epochs)
    for r in table[table["model"] == "html_text"].itertuples():
        pub = f" (published {r.published_mse:.3f})" if r.published_mse == r.published_mse else ""
        typer.echo(
            f"  {r.split:>15} labels={r.labels:<9} tau={r.horizon:<2} n_test={r.n_test:<3} "
            f"MSE={r.mse:.3f}±{r.mse_seed_std:.3f}{pub} R2vsPers={r.r2_oos_vs_persistence:+.3f}"
        )
    typer.echo("Result Table 6R: data/results/result_table_6r.csv")
    typer.echo(f"run artifact: {write_command_run(cfg, root)}")


@app.command(name="evaluate-audio-earnings25")
def evaluate_audio_earnings25(
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    seeds: str | None = typer.Option(None, help="Comma-separated seeds; overrides the config."),
    config: Path | None = _CONFIG_OPT,
) -> None:
    """T9.4: the frozen Stage-3 audio ladder on Earnings25, pooled + per bitrate stratum."""
    from ecvol.eval.earnings25_audio import run_earnings25_audio
    from ecvol.tracking import resolve_command_config, write_command_run

    cfg = resolve_command_config("evaluate-audio-earnings25", config, seeds)
    res = run_earnings25_audio(root, seeds=tuple(cfg.seeds))
    typer.echo(
        f"Result Table 3 (Earnings25): {len(res['table'])} rows → "
        "data/results/result_table_3_earnings25.csv"
    )
    for r in res["summary"].itertuples():
        typer.echo(
            f"  {r.stratum:>5} tau={r.horizon:<2} n_test={r.n_test:<3} "
            f"ridge WavLM+vol dv R2={r.r2_oos_vs_persistence:+.3f} "
            f"(DM vs HAR p={r.dm_p_vs_har:.3f}); "
            f"shuffle real={r.shuffle_real_r2:+.3f} global={r.shuffle_global_r2:+.3f}"
        )
    typer.echo(f"run artifact: {write_command_run(cfg, root)}")


@app.command(name="evaluate-fusion")
def evaluate_fusion(
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    seeds: str | None = typer.Option(None, help="Comma-separated seeds; overrides the config."),
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Stage-4 fusion heads (gated + late-fusion stack) → fusion rows for Result Table 4 (T5.1)."""
    from ecvol.eval.stage4 import run_stage4
    from ecvol.tracking import resolve_command_config, write_command_run

    cfg = resolve_command_config("evaluate-fusion", config, seeds)
    table = run_stage4(root, seeds=tuple(cfg.seeds))
    typer.echo(f"Stage-4 fusion: {len(table)} rows → data/results/result_table_4_fusion.csv")
    head = table[(table["target"] == "dv") & (table["segment"] == "test")]
    beat = head[(head["r2_oos"] > 0) & (head["dm_p_vs_stage1"] < 0.05)]
    typer.echo(f"Δv test cells beating Stage-1 (r2>0 & DM p<0.05): {len(beat)} of {len(head)}")
    typer.echo(f"run artifact: {write_command_run(cfg, root)}")


@app.command()
def grid(
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
    config: Path | None = _CONFIG_OPT,
) -> None:
    """Consolidate Stages 0-4 → Result Table 4 (main grid, Holm-corrected) + per-year (T5.2)."""
    from ecvol.eval.grid import run_grid
    from ecvol.tracking import resolve_command_config, write_command_run

    cfg = resolve_command_config("grid", config)
    table, peryear = run_grid(root)
    typer.echo(f"Result Table 4: {len(table)} rows → data/results/result_table_4.csv")
    sig = table[
        (table["target"] == "dv")
        & (table["segment"] == "test")
        & (table["holm_p_vs_stage1"] < 0.05)
    ]
    typer.echo(f"Δv test cells Holm-significant vs Stage-1: {len(sig)}")
    typer.echo(f"per-year breakdown: {len(peryear)} rows → data/results/result_table_4_peryear.csv")
    typer.echo(f"run artifact: {write_command_run(cfg, root)}")


@app.command()
def report(
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
) -> None:
    """Render result tables (Markdown + LaTeX) from provenance-verified CSVs (T2.3-T5.2, T9.3)."""
    from ecvol.eval.report import (
        write_reports,
        write_reports2,
        write_reports3,
        write_reports4,
    )
    from ecvol.tracking import ProvenanceError, verify_outputs

    present = [
        f"results/result_table_{i}.csv"
        for i in (1, 2, 3, 4)
        if (root / "results" / f"result_table_{i}.csv").is_file()
    ]
    try:
        matched = verify_outputs(root, present)
    except ProvenanceError as exc:
        typer.echo(f"provenance check FAILED: {exc}", err=True)
        raise typer.Exit(code=1) from None
    for rel, meta in matched.items():
        typer.echo(f"provenance OK: {rel} ← {meta['run_id']} ({meta['provenance']})")

    md_path, tex_path = write_reports(root)
    typer.echo(f"Table 1 markdown: {md_path}")
    typer.echo(f"Table 1 latex:    {tex_path}")
    if (root / "results" / "result_table_2.csv").is_file():
        md2, tex2 = write_reports2(root)
        typer.echo(f"Table 2 markdown: {md2}")
        typer.echo(f"Table 2 latex:    {tex2}")
    if (root / "results" / "result_table_3.csv").is_file():
        md3, tex3 = write_reports3(root)
        typer.echo(f"Table 3 markdown: {md3}")
        typer.echo(f"Table 3 latex:    {tex3}")
    if (root / "results" / "result_table_4.csv").is_file():
        md4, tex4 = write_reports4(root)
        typer.echo(f"Table 4 markdown: {md4}")
        typer.echo(f"Table 4 latex:    {tex4}")


runs_app = typer.Typer(no_args_is_help=True, help="Run manifests for result CSVs (T9.3).")
app.add_typer(runs_app, name="runs")


@runs_app.command("verify")
def runs_verify(
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
) -> None:
    """Check every present result CSV against artifacts/runs/*/run.json (exit 1 on mismatch)."""
    from ecvol.tracking import COMMAND_OUTPUTS, ProvenanceError, verify_outputs

    present = [rel for outs in COMMAND_OUTPUTS.values() for rel in outs if (root / rel).is_file()]
    try:
        matched = verify_outputs(root, present)
    except ProvenanceError as exc:
        typer.echo(f"provenance check FAILED: {exc}", err=True)
        raise typer.Exit(code=1) from None
    for rel, meta in matched.items():
        typer.echo(f"OK  {rel} ← {meta['run_id']} ({meta['provenance']})")
    typer.echo(f"{len(matched)} result file(s) verified")


@runs_app.command("backfill")
def runs_backfill(
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
) -> None:
    """Write provenance="backfill" manifests for result CSVs produced before T9.3 (one-off)."""
    from ecvol.tracking import (
        COMMAND_OUTPUTS,
        find_manifest,
        resolve_command_config,
        sha256_file,
        write_command_run,
    )

    n = 0
    for command, outs in COMMAND_OUTPUTS.items():
        present = [rel for rel in outs if (root / rel).is_file()]
        if not present:
            typer.echo(f"skip {command}: no outputs present")
            continue
        if all(find_manifest(rel, sha256_file(root / rel)) is not None for rel in present):
            typer.echo(f"skip {command}: already manifested")
            continue
        cfg = resolve_command_config(command)
        run_dir = write_command_run(cfg, root, present, provenance="backfill")
        typer.echo(f"backfilled {command}: {len(present)} file(s) → {run_dir}")
        n += 1
    typer.echo(f"{n} manifest(s) written")


@app.command("llm-kappa")
def llm_kappa(
    sheet: Path = typer.Option(..., help="Filled labeling sheet CSV (from llm-audit-sample)."),  # noqa: B008
    features: Path = typer.Option(..., help="llm_features__{model}.parquet to score."),  # noqa: B008
) -> None:
    """κ-audit: per-field model-vs-human agreement; gate κ>0.6 before corpus scale (T6.2)."""
    from ecvol.features.llm.audit import compute_kappa, passes_gate
    from ecvol.features.llm.schema import CONFIRMATORY_FIELDS

    k = compute_kappa(sheet, features)
    for field, v in k.items():
        kv = "n/a" if v["kappa"] is None else f"{v['kappa']:.3f}"
        tag = "  [confirmatory]" if field in CONFIRMATORY_FIELDS else "  [reported]"
        typer.echo(f"  {field:24s} κ={kv:>6s}  (n={v['n']}){tag}")
    typer.echo(f"GATE κ>0.6 (confirmatory core only): {'PASS' if passes_gate(k) else 'FAIL'}")


collect_app = typer.Typer(no_args_is_help=True, help="ecvol-live forward collection (TX4 pilot).")
app.add_typer(collect_app, name="collect")


@collect_app.command("discover")
def collect_discover(
    start: str = typer.Option(..., help="Calendar sweep start, YYYY-MM-DD."),
    end: str = typer.Option(..., help="Calendar sweep end, YYYY-MM-DD (inclusive)."),
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
) -> None:
    """Universe snapshot x earnings calendar -> live/discovery.csv (cached, resumable)."""
    from datetime import date as date_type

    from ecvol.collect.discovery import build_discovery, build_universe

    universe = build_universe(root)
    counts = universe.index_membership.value_counts()
    missing_cik = int(universe.cik.isna().sum())
    typer.echo(f"universe: {len(universe)} ({counts.to_dict()}); cik missing: {missing_cik}")
    discovery = build_discovery(root, date_type.fromisoformat(start), date_type.fromisoformat(end))
    typer.echo(f"discovery: {len(discovery)} candidate calls -> {root / 'live' / 'discovery.csv'}")
    typer.echo(f"  by index: {discovery.index_membership.value_counts().to_dict()}")
    typer.echo(f"  by timing: {discovery.timing.value_counts().to_dict()}")


@collect_app.command("sample")
def collect_sample(
    n_sp500: int = typer.Option(35, help="Pilot draws from the S&P 500 stratum."),
    n_sp400: int = typer.Option(15, help="Pilot draws from the S&P 400 stratum."),
    seed: int = typer.Option(20260813, help="Draw seed (recorded in the output)."),
    root: Path = typer.Option(Path("data"), help="Data root directory."),  # noqa: B008
) -> None:
    """Seeded stratified pilot sample from live/discovery.csv -> live/pilot_sample.csv."""
    import pandas as pd

    from ecvol.collect.discovery import sample_pilot

    discovery = pd.read_csv(root / "live" / "discovery.csv", dtype={"cik": str})
    sample = sample_pilot(discovery, n_sp500=n_sp500, n_sp400=n_sp400, seed=seed)
    sample["sample_seed"] = seed
    out = root / "live" / "pilot_sample.csv"
    sample.to_csv(out, index=False)
    typer.echo(f"pilot sample: {len(sample)} calls -> {out}")
    typer.echo(f"  dates {sample.call_date.min()} .. {sample.call_date.max()}")
