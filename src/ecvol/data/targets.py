"""Volatility target computation per (call, horizon) — DESIGN §5.3 (T1.3).

Computes, for each resolved FinCall call and each horizon τ ∈ {3,7,15,30}:
  v_post(τ) = ln( sqrt( (1/τ) · Σ_{t=1..τ} (r_t − r̄)² ) )   (log realized vol)
  v_pre(τ)  = same over the τ returns ending at day 0
  Δv(τ)     = v_post(τ) − v_pre(τ)
plus the HAR-RV inputs (realized variance over the last 1/5/22 sessions, as of
day 0). `r_t = (P_t − P_{t−1})/P_{t−1}` over adjusted closes; level log-RV, not
annualized (Qin & Yang convention, for literature comparability).

**Day-0 / after-hours rule (DESIGN §5.3).** `day0` is the last NYSE session whose
information is public *before* the post-call window. Call times-of-day are not yet
available (the identity CSV has dates only), so we use DESIGN §10 risk #7's
documented fallback — **assume after-hours** — recorded per row in
`assumed_after_hours`. `anchor_day0` accepts an optional `timestamp` so a later
T1.4 timestamp-extraction pass refines the anchor (≥16:00 ET → after-hours;
<09:30 ET → call date is day +1) and we rerun + sensitivity-check without rework.

Every (call, τ) yields exactly one row with a `status`/`reason` — never a silent
drop. The HAR-residual *target* (v_post − HAR_forecast) needs a train-split-only
HAR fit and is realized in Phase 2; this module emits only the HAR inputs.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ecvol.data import calendar as cal
from ecvol.data.manifests import make_entry, write_manifest
from ecvol.data.prices import load_close_series

HORIZONS = (3, 7, 15, 30)
HAR_WINDOWS = (1, 5, 22)  # daily / weekly / monthly realized-variance lookbacks
FINCALL_IDENTITY = "identity/fincall_identity.csv"
TARGETS_LICENSE = "Derived artifact — computed from price data (DESIGN §5.3); no external source"
TARGETS_SOURCE = "computed: ecvol targets build (DESIGN §5.3)"

NAN = float("nan")


# --- pure math ---------------------------------------------------------------


def realized_vol(returns: list[float]) -> float:
    """Log realized volatility of a return series: ln(sqrt(mean squared deviation)).

    Population variance (denominator = n, matching DESIGN §5.3's `(1/τ)·Σ`).
    Returns NaN for <2 points or a zero-variance (flat) window (`ln(0)`).
    """
    n = len(returns)
    if n < 2:
        return NAN
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / n
    if var <= 0.0:
        return NAN
    return math.log(math.sqrt(var))


def _last_session_on_or_before(day: date) -> date | None:
    window = cal.sessions_in_range(day - timedelta(days=10), day)
    return window[-1] if window else None


def _first_session_on_or_after(day: date) -> date | None:
    window = cal.sessions_in_range(day, day + timedelta(days=10))
    return window[0] if window else None


def anchor_day0(call_date: date, *, timestamp: datetime | None = None) -> date | None:
    """Day-0 anchor session for a call (after-hours rule, DESIGN §5.3).

    `timestamp is None` (the current fallback) → assume after-hours: day0 = last
    session on/before `call_date`. With a real timestamp: ≥16:00 ET stays
    after-hours; <09:30 ET makes the call date day +1, so day0 = the session
    before; intraday is treated as after-hours.
    """
    if timestamp is not None and (timestamp.hour, timestamp.minute) < (9, 30):
        reaction = _first_session_on_or_after(call_date)
        if reaction is None:
            return None
        prior = cal.sessions_in_range(reaction - timedelta(days=10), reaction - timedelta(days=1))
        return prior[-1] if prior else None
    return _last_session_on_or_before(call_date)


def session_offsets(day0: date, back: int, fwd: int) -> dict[int, date]:
    """Map integer session offset (day0 = 0) → session date, over a padded window."""
    lo = day0 - timedelta(days=back * 2 + 20)
    hi = day0 + timedelta(days=fwd * 2 + 20)
    sessions = cal.sessions_in_range(lo, hi)
    if day0 not in sessions:
        return {}
    z = sessions.index(day0)
    return {i - z: d for i, d in enumerate(sessions)}


# --- per-call computation ----------------------------------------------------


@dataclass
class TargetRow:
    call_id: int | str  # int for FinCall (numeric ids), str YYYYMMDD_TICKER for MAEC
    ticker: str
    call_type: str
    call_date: str
    as_of: str  # ISO date of day0; the information-rule boundary (close of this session)
    horizon: int
    v_pre: float
    v_post: float
    delta_v: float
    rv_daily: float
    rv_weekly: float
    rv_monthly: float
    assumed_after_hours: bool
    status: str  # "ok" | "excluded"
    reason: str


def _price_at(offsets: dict[int, date], close: dict[str, float], o: int) -> float | None:
    d = offsets.get(o)
    if d is None:
        return None
    return close.get(d.isoformat())


def _returns(
    offsets: dict[int, date], close: dict[str, float], lo: int, hi: int
) -> list[float] | None:
    """Daily simple returns for offsets (lo..hi); None if any price in [lo-1..hi] is missing."""
    prices = [_price_at(offsets, close, o) for o in range(lo - 1, hi + 1)]
    if any(p is None for p in prices) or len(prices) < 2:
        return None
    return [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]


def _har_inputs(offsets: dict[int, date], close: dict[str, float]) -> tuple[float, float, float]:
    out = []
    for w in HAR_WINDOWS:
        rets = _returns(offsets, close, 1 - w, 0)  # w returns ending at day 0
        out.append(sum(r * r for r in rets) / len(rets) if rets else NAN)
    return out[0], out[1], out[2]


def compute_call_targets(
    call: dict,
    close: dict[str, float],
    *,
    horizons: tuple[int, ...] = HORIZONS,
) -> list[TargetRow]:
    """One TargetRow per horizon for a single call. Never raises; encodes reasons."""
    call_id = call["call_id"]  # type preserved as given (FinCall int, MAEC str)
    ticker = str(call.get("ticker") or "").strip()
    call_type = str(call.get("call_type") or "").strip()
    call_date = str(call.get("date") or "").strip()
    timestamp = call.get("timestamp")
    assumed = timestamp is None

    def excluded(reason: str, as_of: str = "") -> list[TargetRow]:
        return [
            TargetRow(
                call_id,
                ticker,
                call_type,
                call_date,
                as_of,
                h,
                NAN,
                NAN,
                NAN,
                NAN,
                NAN,
                NAN,
                assumed,
                "excluded",
                reason,
            )
            for h in horizons
        ]

    if not ticker:
        return excluded("unresolved_ticker")
    try:
        cdate = date.fromisoformat(call_date)
    except ValueError:
        return excluded("invalid_date")
    if not (date(2000, 1, 1) <= cdate <= date(2030, 12, 31)):
        return excluded("invalid_date")

    day0 = anchor_day0(cdate, timestamp=timestamp)
    if day0 is None:
        return excluded("invalid_date")
    as_of = day0.isoformat()
    if not close:
        return excluded("no_price_data", as_of)

    maxh = max(horizons)
    offsets = session_offsets(day0, back=max(maxh, max(HAR_WINDOWS)), fwd=maxh)
    rv_daily, rv_weekly, rv_monthly = _har_inputs(offsets, close)

    rows: list[TargetRow] = []
    for h in horizons:
        pre_rets = _returns(offsets, close, 1 - h, 0)  # r_{-h+1..0}
        post_rets = _returns(offsets, close, 1, h)  # r_{1..h}
        v_pre = realized_vol(pre_rets) if pre_rets is not None else NAN
        v_post = realized_vol(post_rets) if post_rets is not None else NAN
        delta = v_post - v_pre if not (math.isnan(v_pre) or math.isnan(v_post)) else NAN

        if post_rets is None:
            status, reason = "excluded", "insufficient_post_history"
        elif pre_rets is None:
            status, reason = "excluded", "insufficient_pre_history"
        elif math.isnan(v_post):
            status, reason = "excluded", "zero_variance_post"
        elif math.isnan(v_pre):
            status, reason = "excluded", "zero_variance_pre"
        else:
            status, reason = "ok", ""

        rows.append(
            TargetRow(
                call_id,
                ticker,
                call_type,
                call_date,
                as_of,
                h,
                v_pre,
                v_post,
                delta,
                rv_daily,
                rv_weekly,
                rv_monthly,
                assumed,
                status,
                reason,
            )
        )
    return rows


# --- orchestration + artifacts -----------------------------------------------


@dataclass
class TargetSummary:
    total_calls: int
    resolved_calls: int
    rows_total: int
    ok_rows: int
    excluded_rows: int
    reason_counts: dict[str, int]
    horizon_ok: dict[int, int]
    calls_with_any_ok: int
    join_rate_pct: float


def _read_identity(root: Path) -> list[dict]:
    df = pd.read_csv(root / FINCALL_IDENTITY, dtype=str).fillna("")
    cols = {"call_id", "ticker", "date", "call_type"}
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f"{FINCALL_IDENTITY} missing columns: {sorted(missing)}")
    records = df[["call_id", "ticker", "date", "call_type"]].to_dict("records")
    for r in records:  # FinCall ids are numeric; coerce here so TargetRow.call_id is int
        r["call_id"] = int(r["call_id"])
    return records


def write_targets_parquet(
    rows: list[TargetRow], path: Path, *, id_type: pa.DataType | None = None
) -> None:
    """Write target rows as deterministic parquet (sorted by call_id, horizon)."""
    id_type = id_type or pa.int64()  # FinCall default; MAEC passes pa.string()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r.call_id, r.horizon))
    table = pa.table(
        {
            "call_id": pa.array([r.call_id for r in rows], id_type),
            "ticker": pa.array([r.ticker for r in rows], pa.string()),
            "call_type": pa.array([r.call_type for r in rows], pa.string()),
            "call_date": pa.array([r.call_date for r in rows], pa.string()),
            "as_of": pa.array([r.as_of for r in rows], pa.string()),
            "horizon": pa.array([r.horizon for r in rows], pa.int64()),
            "v_pre": pa.array([r.v_pre for r in rows], pa.float64()),
            "v_post": pa.array([r.v_post for r in rows], pa.float64()),
            "delta_v": pa.array([r.delta_v for r in rows], pa.float64()),
            "rv_daily": pa.array([r.rv_daily for r in rows], pa.float64()),
            "rv_weekly": pa.array([r.rv_weekly for r in rows], pa.float64()),
            "rv_monthly": pa.array([r.rv_monthly for r in rows], pa.float64()),
            "assumed_after_hours": pa.array([r.assumed_after_hours for r in rows], pa.bool_()),
            "status": pa.array([r.status for r in rows], pa.string()),
            "reason": pa.array([r.reason for r in rows], pa.string()),
        }
    )
    pq.write_table(table, path, compression="none", store_schema=True)


def _summarize(rows: list[TargetRow], total_calls: int, horizons: tuple[int, ...]) -> TargetSummary:
    resolved = {r.call_id for r in rows if r.ticker}
    ok = [r for r in rows if r.status == "ok"]
    reason_counts: dict[str, int] = {}
    for r in rows:
        if r.reason:
            reason_counts[r.reason] = reason_counts.get(r.reason, 0) + 1
    horizon_ok = {h: sum(1 for r in ok if r.horizon == h) for h in horizons}
    calls_any_ok = {r.call_id for r in ok}
    join_rate = round(100 * len(calls_any_ok) / len(resolved), 2) if resolved else 0.0
    return TargetSummary(
        total_calls=total_calls,
        resolved_calls=len(resolved),
        rows_total=len(rows),
        ok_rows=len(ok),
        excluded_rows=len(rows) - len(ok),
        reason_counts=dict(sorted(reason_counts.items())),
        horizon_ok=horizon_ok,
        calls_with_any_ok=len(calls_any_ok),
        join_rate_pct=join_rate,
    )


def write_targets_report(summary: TargetSummary, path: Path) -> None:
    """Committed human-readable summary (metric,value) — counts, per-horizon, join rate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    items: list[tuple[str, object]] = [
        ("total_calls", summary.total_calls),
        ("resolved_calls", summary.resolved_calls),
        ("rows_total", summary.rows_total),
        ("ok_rows", summary.ok_rows),
        ("excluded_rows", summary.excluded_rows),
        ("calls_with_any_ok", summary.calls_with_any_ok),
        ("join_rate_pct", summary.join_rate_pct),
    ]
    items += [(f"horizon_{h}_ok", n) for h, n in summary.horizon_ok.items()]
    items += [(f"reason:{k}", v) for k, v in summary.reason_counts.items()]
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["metric", "value"])
    writer.writerows(items)  # preserve int/float types (no pandas float coercion)
    path.write_text(buf.getvalue(), encoding="utf-8")


def build_targets(root: Path, *, horizons: tuple[int, ...] = HORIZONS) -> TargetSummary:
    """Compute targets for every resolved FinCall call; write parquet + report + manifest."""
    calls = _read_identity(root)
    prices_dir = root / "prices"
    close_cache: dict[str, dict[str, float]] = {}
    rows: list[TargetRow] = []
    for call in calls:
        ticker = str(call.get("ticker") or "").strip()
        if ticker and ticker not in close_cache:
            close_cache[ticker] = load_close_series(prices_dir, ticker)
        rows.extend(compute_call_targets(call, close_cache.get(ticker, {}), horizons=horizons))

    targets_path = root / "fincall" / "targets.parquet"
    write_targets_parquet(rows, targets_path)
    summary = _summarize(rows, total_calls=len(calls), horizons=horizons)
    write_targets_report(summary, root / "coverage" / "targets_report.csv")
    entry = make_entry(targets_path, root, source_url=TARGETS_SOURCE, license=TARGETS_LICENSE)
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    write_manifest([entry], root / "manifests" / "fincall_targets.json")
    return summary
