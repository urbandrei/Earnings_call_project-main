"""T1.6 split leakage assertions — run in CI forever (DESIGN §5.4.6).

The guarantees, on synthetic cohorts and on the committed real split CSVs:
- temporal: no target window crosses a boundary; ≥embargo-session gap between
  segments; strict calendar order train < val < test.
- ticker_disjoint: zero ticker intersection across train/val/test.
- combined: ticker-disjoint AND temporally ordered.
- determinism: same inputs → byte-identical CSVs.
"""

from pathlib import Path

import pandas as pd

from ecvol.data.splits import (
    DEFAULT_RATIOS,
    CallInfo,
    build_dataset_splits,
    combined_split,
    temporal_split,
    ticker_disjoint_split,
)

# --- synthetic cohort --------------------------------------------------------


def _synthetic(n: int = 300, tickers: int = 40) -> list[CallInfo]:
    # idx spaced (≈5 sessions/call) so the middle val band exceeds the embargo;
    # ticker cycles for grouping. Dense clustering is exercised on real data below.
    return [
        CallInfo(call_id=i, ticker=f"T{i % tickers:02d}", as_of="", idx=5 * i) for i in range(n)
    ]


def _by_id(cohort: list[CallInfo]) -> dict[object, CallInfo]:
    return {c.call_id: c for c in cohort}


def test_temporal_no_window_crosses_boundary_and_embargo_gap():
    cohort = _synthetic()
    by_id = _by_id(cohort)
    horizon, embargo = 30, 30
    assign = temporal_split(cohort, embargo=embargo, horizon=horizon)
    gap = max(embargo, horizon)

    def idxs(split):
        return [by_id[cid].idx for cid, s in assign.items() if s == split]

    train, val, test = idxs("train"), idxs("val"), idxs("test")
    assert train and val and test
    # Strict order with a ≥gap hole: a train call's whole target window precedes
    # the first val call's day0; likewise val → test.
    assert max(train) + horizon <= min(val)
    assert max(val) + horizon <= min(test)
    assert min(val) - max(train) >= gap
    assert min(test) - max(val) >= gap


def test_temporal_partitions_every_call():
    cohort = _synthetic()
    assign = temporal_split(cohort, embargo=30, horizon=30)
    assert set(assign) == {c.call_id for c in cohort}
    assert set(assign.values()) <= {"train", "val", "test", "embargo"}


def test_ticker_disjoint_zero_intersection():
    cohort = _synthetic()
    by_id = _by_id(cohort)
    assign = ticker_disjoint_split(cohort, seed=0)
    groups = {"train": set(), "val": set(), "test": set()}
    for cid, s in assign.items():
        groups[s].add(by_id[cid].ticker)
    assert groups["train"] & groups["val"] == set()
    assert groups["train"] & groups["test"] == set()
    assert groups["val"] & groups["test"] == set()
    # every call assigned to one of the three groups
    assert set(assign.values()) == {"train", "val", "test"}


def test_ticker_disjoint_deterministic_under_seed():
    cohort = _synthetic()
    assert ticker_disjoint_split(cohort, seed=0) == ticker_disjoint_split(cohort, seed=0)
    # a different seed generally yields a different partition
    assert ticker_disjoint_split(cohort, seed=0) != ticker_disjoint_split(cohort, seed=7)


def test_combined_is_disjoint_and_ordered():
    cohort = _synthetic()
    by_id = _by_id(cohort)
    temporal = temporal_split(cohort, embargo=30, horizon=30)
    ticker = ticker_disjoint_split(cohort, seed=0)
    combined = combined_split(temporal, ticker)

    groups = {"train": set(), "val": set(), "test": set()}
    idx_by_split = {"train": [], "val": [], "test": []}
    for cid, s in combined.items():
        if s in groups:
            groups[s].add(by_id[cid].ticker)
            idx_by_split[s].append(by_id[cid].idx)
    # ticker-disjoint preserved
    assert groups["train"] & groups["val"] == set()
    assert groups["train"] & groups["test"] == set()
    assert groups["val"] & groups["test"] == set()
    # temporally ordered where each split is non-empty
    if idx_by_split["train"] and idx_by_split["val"]:
        assert max(idx_by_split["train"]) < min(idx_by_split["val"])
    if idx_by_split["val"] and idx_by_split["test"]:
        assert max(idx_by_split["val"]) < min(idx_by_split["test"])


def test_combined_keeps_only_agreeing_calls():
    temporal = {1: "train", 2: "val", 3: "test", 4: "train"}
    ticker = {1: "train", 2: "test", 3: "test", 4: "val"}
    assert combined_split(temporal, ticker) == {
        1: "train",  # agree
        2: "excluded",  # val vs test
        3: "test",  # agree
        4: "excluded",  # train vs val
    }


# --- end-to-end on a built dataset -------------------------------------------


def _seed_targets(root: Path, dataset: str) -> None:
    """Minimal targets.parquet: 100 calls over distinct sessions, 10 tickers."""
    from datetime import date

    from ecvol.data import calendar as cal

    sessions = [d.isoformat() for d in cal.sessions_in_range(date(2020, 1, 2), date(2020, 12, 31))]
    rows = []
    for i in range(100):
        rows.append(
            {
                "call_id": 1000 + i,
                "ticker": f"TK{i % 10}",
                "as_of": sessions[i],
                "horizon": 30,
                "status": "ok",
            }
        )
    (root / dataset).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(root / dataset / "targets.parquet")


def test_build_dataset_splits_writes_committed_csvs_deterministically(tmp_path: Path):
    root = tmp_path / "data"
    _seed_targets(root, "fincall")
    s1 = build_dataset_splits(root, "fincall", embargo=30, seed=0)
    assert s1.cohort == 100 and s1.horizon == 30

    for scheme in ("temporal", "ticker_disjoint", "combined"):
        path = root / "splits" / f"fincall_{scheme}.csv"
        assert path.is_file()
        df = pd.read_csv(path)
        assert list(df.columns) == ["call_id", "ticker", "as_of", "split"]
        assert len(df) == 100  # every cohort call accounted for

    # ticker-disjoint really is disjoint on the written CSV
    td = pd.read_csv(root / "splits" / "fincall_ticker_disjoint.csv")
    grp = td.groupby("split")["ticker"].agg(set)
    assert grp["train"] & grp.get("val", set()) == set()
    assert grp["train"] & grp.get("test", set()) == set()

    # deterministic across reruns
    before = (root / "splits" / "fincall_temporal.csv").read_bytes()
    build_dataset_splits(root, "fincall", embargo=30, seed=0)
    assert (root / "splits" / "fincall_temporal.csv").read_bytes() == before


def test_ratios_constant_sums_to_one():
    assert abs(sum(DEFAULT_RATIOS) - 1.0) < 1e-9


# --- the committed real splits (CI-guarded forever; DESIGN §5.4.6) -----------

REPO_SPLITS = Path(__file__).resolve().parents[1] / "data" / "splits"


def _committed(dataset: str, scheme: str) -> pd.DataFrame | None:
    path = REPO_SPLITS / f"{dataset}_{scheme}.csv"
    return pd.read_csv(path, dtype={"call_id": str}) if path.is_file() else None


def _ordered_with_embargo(df: pd.DataFrame, embargo: int = 30) -> None:
    from datetime import date

    from ecvol.data import calendar as cal

    def span(split):
        d = df.loc[df["split"] == split, "as_of"]
        return (d.min(), d.max()) if len(d) else None

    order = [s for s in ("train", "val", "test") if span(s)]
    for earlier, later in zip(order, order[1:], strict=False):
        last_earlier = date.fromisoformat(span(earlier)[1])
        first_later = date.fromisoformat(span(later)[0])
        assert last_earlier < first_later, f"{earlier} not before {later}"
        # the embargo hole: ≥ `embargo` sessions strictly between the segments
        gap = cal.session_count(last_earlier, first_later) - 1
        assert gap >= embargo, f"{earlier}->{later} gap {gap} < embargo {embargo}"


def _ticker_disjoint(df: pd.DataFrame) -> None:
    groups = {s: set(g["ticker"]) for s, g in df.groupby("split")}
    seen = [groups.get(s, set()) for s in ("train", "val", "test")]
    assert seen[0] & seen[1] == set()
    assert seen[0] & seen[2] == set()
    assert seen[1] & seen[2] == set()


def test_committed_temporal_splits_leakage_proof():
    for dataset in ("fincall", "maec"):
        df = _committed(dataset, "temporal")
        if df is None:
            continue
        _ordered_with_embargo(df[df["split"].isin(["train", "val", "test"])])


def test_committed_ticker_disjoint_splits_have_no_overlap():
    for dataset in ("fincall", "maec"):
        df = _committed(dataset, "ticker_disjoint")
        if df is None:
            continue
        _ticker_disjoint(df)


def test_committed_combined_splits_disjoint_and_ordered():
    for dataset in ("fincall", "maec"):
        df = _committed(dataset, "combined")
        if df is None:
            continue
        keep = df[df["split"].isin(["train", "val", "test"])]
        _ticker_disjoint(keep)
        _ordered_with_embargo(keep)
