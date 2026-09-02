"""T6R.1 — substrate audit of the benchmarks this literature shares → Result Table 5R.

Every number here regenerates from the mirrored lineage files under
`raw/ref/lineage/` (KeFVP, VolTAGE, HTML repos, pinned to commit SHAs in
`raw/ref/lineage/lineage_manifest.json`) and from our own committed splits and
targets. No model is reproduced. Findings first established 2026-08-23
(`docs/advisor_redirection_2026-08.md` §5); this module formalises them so each
claim carries the file and the row/cell counts it was computed from.

Benchmarks: **EC** (Qin & Yang 2019; VolTAGE's `semi-supervised-gcn/gcn/data/
{train,val,test}_split3.csv`, 392/56/112 calls) and **MAEC-15 / MAEC-16** (KeFVP's
`price_data/maec/{15,16}/maec*_{train,dev,test}_price_label.csv`). Our own splits
(`splits/{dataset}_{scheme}.csv`) are the contrast rows.

Metrics per benchmark: calls per split; **test calls whose ticker appears in
train** (a literature-coverage claim — temporal splits always permit reuse, and
no paper in the lineage reports a ticker-disjoint condition); the **embargo** in
calendar days at each split boundary (last train date → first val date, last
val → first test); **sentinel zeros** in EC's single-day log-vol series
(`SeriesSingleDayVol3`, 30 future columns); the **interleaved binary labels** in
KeFVP's MAEC price-label files (`future_label_{3,7,15,30}` ∈ {0,1} among price
columns); and the **label-vs-our-targets diff** on MAEC, joined on
`text_file_name` = our `call_id`.

Outputs: `results/result_table_5r.csv` (long: benchmark, metric, value,
numerator, denominator, source) and `results/result_table_5r_labels.csv` (the
per-horizon MAEC label diff).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LINEAGE_REL = "raw/ref/lineage"
EC_SPLITS = {
    s: f"VolTAGE/semi-supervised-gcn/gcn/data/{s}_split3.csv" for s in ("train", "val", "test")
}
EC_SINGLE = {
    s: f"KeFVP/price_data/{s}_split_SeriesSingleDayVol3.csv" for s in ("train", "val", "test")
}
MAEC_PRICE = {
    (v, s): f"KeFVP/price_data/maec/{v}/maec{v}_{s}_price_label.csv"
    for v in ("15", "16")
    for s in ("train", "dev", "test")
}
MAEC_AVG = {v: f"KeFVP/price_data/maec/{v}/maec{v}_test_avg_val.csv" for v in ("15", "16")}
TAUS = (3, 7, 15, 30)


def _row(benchmark, metric, value, num=None, den=None, source=""):
    return {
        "benchmark": benchmark,
        "metric": metric,
        "value": value,
        "numerator": num,
        "denominator": den,
        "source": source,
    }


# --- generic split integrity ---------------------------------------------------


def split_integrity(
    frames: dict[str, pd.DataFrame], benchmark: str, source: str, *, temporal: bool = True
) -> list[dict]:
    """`frames` = {split: frame with ticker + date columns}; splits in temporal order.

    Embargo rows only make sense for temporally ordered splits (`temporal=True`).
    """
    order = [s for s in ("train", "val", "dev", "test") if s in frames]
    rows = []
    for s in order:
        rows.append(_row(benchmark, f"calls_{s}", int(len(frames[s])), source=source))
    train_tickers = set(frames[order[0]]["ticker"])
    test = frames[order[-1]]
    overlap = int(test["ticker"].isin(train_tickers).sum())
    rows.append(
        _row(
            benchmark,
            "test_ticker_in_train_share",
            round(overlap / len(test), 4) if len(test) else np.nan,
            overlap,
            int(len(test)),
            source,
        )
    )
    for a, b in zip(order, order[1:], strict=False):
        if temporal:
            gap = (frames[b]["date"].min() - frames[a]["date"].max()).days
            rows.append(_row(benchmark, f"embargo_days_{a}_to_{b}", int(gap), source=source))
        both = set(zip(frames[a]["ticker"], frames[a]["date"], strict=True)) & set(
            zip(frames[b]["ticker"], frames[b]["date"], strict=True)
        )
        rows.append(_row(benchmark, f"duplicate_calls_{a}_{b}", int(len(both)), source=source))
    return rows


def _ec_frames(lineage: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for s, rel in EC_SPLITS.items():
        d = pd.read_csv(lineage / rel)
        d["date"] = pd.to_datetime(d[["year", "month", "day"]]).dt.date
        out[s] = d
    return out


def _maec_frames(lineage: Path, v: str) -> dict[str, pd.DataFrame]:
    out = {}
    for s in ("train", "dev", "test"):
        d = pd.read_csv(lineage / MAEC_PRICE[(v, s)])
        d["date"] = pd.to_datetime(d["time"]).dt.date
        out[s] = d
    return out


# --- label defects -----------------------------------------------------------------


def ec_sentinel_zeros(lineage: Path) -> list[dict]:
    rows = []
    total_z = total_c = 0
    for s, rel in EC_SINGLE.items():
        d = pd.read_csv(lineage / rel)
        cols = [c for c in d.columns if c.startswith("future_Single_")]
        z = int((d[cols] == 0.0).sum().sum())
        c = int(d[cols].size)
        rows.append(_row("EC", f"single_day_exact_zeros_{s}", z, z, c, rel))
        total_z += z
        total_c += c
    rows.append(
        _row("EC", "single_day_exact_zeros_all", total_z, total_z, total_c, "SeriesSingleDayVol3")
    )
    # the headline averaged series carries none
    for s in ("train", "val", "test"):
        rel = f"KeFVP/price_data/{s}_split_Avg_Series_WITH_LOG.csv"
        d = pd.read_csv(lineage / rel)
        cols = [c for c in d.columns if c.startswith("future_")]
        rows.append(
            _row(
                "EC",
                f"avg_series_exact_zeros_{s}",
                int((d[cols] == 0.0).sum().sum()),
                None,
                int(d[cols].size),
                rel,
            )
        )
    return rows


def maec_interleaved_binary(lineage: Path, v: str) -> list[dict]:
    rows = []
    for s in ("train", "dev", "test"):
        rel = MAEC_PRICE[(v, s)]
        d = pd.read_csv(lineage / rel)
        fut = [c for c in d.columns if c.startswith("future_label_")]
        binary = [c for c in fut if set(d[c].dropna().unique()) <= {0, 1}]
        idx = sorted(int(c.split("_")[-1]) for c in binary)
        rows.append(
            _row(
                f"MAEC-{v}",
                f"binary_label_columns_{s}",
                ",".join(map(str, idx)),
                len(binary),
                len(fut),
                rel,
            )
        )
        rows.append(
            _row(f"MAEC-{v}", f"binary_columns_are_tau_{s}", int(idx == list(TAUS)), source=rel)
        )
    return rows


# --- label vs our targets -------------------------------------------------------


def maec_label_diff(root: Path, lineage: Path) -> pd.DataFrame:
    """KeFVP's shipped MAEC test labels vs our calendar-day v_post on the same calls."""
    ours = pd.read_parquet(
        root / "maec" / "targets_calendar.parquet",
        columns=["call_id", "horizon", "v_post", "status"],
    )
    ours = ours[ours["status"] == "ok"]
    out = []
    for v, rel in MAEC_AVG.items():
        lab = pd.read_csv(lineage / rel)
        for tau in TAUS:
            col = f"future_{tau}"
            if col not in lab.columns:
                continue
            j = lab[["text_file_name", col]].merge(
                ours[ours["horizon"] == tau][["call_id", "v_post"]],
                left_on="text_file_name",
                right_on="call_id",
            )
            out.append(
                {
                    "benchmark": f"MAEC-{v}",
                    "horizon": tau,
                    "n_shipped": int(len(lab)),
                    "n_joined": int(len(j)),
                    "join_share": round(len(j) / len(lab), 4) if len(lab) else np.nan,
                    "corr": float(j[col].corr(j["v_post"])) if len(j) > 2 else np.nan,
                    "mean_shipped": float(j[col].mean()) if len(j) else np.nan,
                    "mean_ours_calendar": float(j["v_post"].mean()) if len(j) else np.nan,
                    "mean_abs_diff": float((j[col] - j["v_post"]).abs().mean())
                    if len(j)
                    else np.nan,
                    "source": rel,
                }
            )
    return pd.DataFrame(out)


# --- our own splits as the contrast -------------------------------------------


def own_splits(root: Path) -> list[dict]:
    rows = []
    for dataset in ("fincall", "maec", "earnings25"):
        for scheme in ("temporal", "ticker_disjoint"):
            path = root / "splits" / f"{dataset}_{scheme}.csv"
            if not path.is_file():
                continue
            d = pd.read_csv(path, dtype={"call_id": str})
            d["date"] = pd.to_datetime(d["as_of"]).dt.date
            frames = {
                s: d[d["split"] == s] for s in ("train", "val", "test") if (d["split"] == s).any()
            }
            if "train" not in frames or "test" not in frames:
                continue
            rows += split_integrity(
                frames, f"ours:{dataset}:{scheme}", path.name, temporal=(scheme == "temporal")
            )
    return rows


# --- entry point -------------------------------------------------------------------


def run_substrate_audit(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    lineage = root / LINEAGE_REL
    rows: list[dict] = []
    rows += split_integrity(_ec_frames(lineage), "EC", "VolTAGE gcn/data/*_split3.csv")
    for v in ("15", "16"):
        rows += split_integrity(
            _maec_frames(lineage, v), f"MAEC-{v}", f"KeFVP maec{v}_*_price_label.csv"
        )
        rows += maec_interleaved_binary(lineage, v)
    rows += ec_sentinel_zeros(lineage)
    # the shared-files claim: byte-identical split labels across two repos
    import hashlib

    for name in ("test_split_SeriesSingleDayVol3.csv", "train_split_Avg_Series_WITH_LOG.csv"):
        a = hashlib.sha256((lineage / "KeFVP" / "price_data" / name).read_bytes()).hexdigest()
        b = hashlib.sha256(
            (lineage / "VolTAGE" / "cross-modal-attn" / name).read_bytes()
        ).hexdigest()
        rows.append(_row("EC", f"kefvp_voltage_identical:{name}", int(a == b), source=a[:12]))
    rows += own_splits(root)
    table = pd.DataFrame(rows)
    labels = maec_label_diff(root, lineage)
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "result_table_5r.csv", index=False, lineterminator="\n")
    labels.to_csv(out / "result_table_5r_labels.csv", index=False, lineterminator="\n")
    return table, labels
