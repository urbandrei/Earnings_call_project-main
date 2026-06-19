"""eGeMAPS paralinguistic features per call (T4.2, CPU via openSMILE).

Extracts the 88 eGeMAPSv02 *functionals* (F0, loudness, jitter/shimmer, HNR, spectral, MFCC
summaries) per call over the 16 kHz FLAC store from T4.1, using the `opensmile` package.
Deterministic; parallel across cores (openSMILE is CPU-cheap). Per-speaker-turn extraction is
deferred to T4.3 (needs diarization). Output: `data/fincall/audio_egemaps.parquet` (call_id + 88
features) + manifest + a committed distribution summary for the published-range sanity check.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

EGEMAPS_SOURCE = "derived: ecvol audio egemaps (T4.2)"
EGEMAPS_LICENSE = "derived; openSMILE eGeMAPSv02"

_SMILE = None  # per-process openSMILE instance (set by the pool initializer)


def _make_smile():
    import opensmile

    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def _init_worker():
    global _SMILE
    _SMILE = _make_smile()


def _extract(args):
    call_id, path = args
    try:
        row = _SMILE.process_file(path).iloc[0].to_dict()
        return call_id, row, ""
    except Exception as e:  # noqa: BLE001 — flag, never crash the batch
        return call_id, None, f"error:{type(e).__name__}"


def feature_names() -> list[str]:
    return list(_make_smile().feature_names)


def write_egemaps_parquet(df: pd.DataFrame, path: Path, features: list[str]) -> None:
    """Deterministic parquet (sorted by call_id; int64 call_id, float64 features)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("call_id").reset_index(drop=True)
    table = pa.table(
        {
            "call_id": pa.array(df["call_id"].to_numpy(), pa.int64()),
            **{f: pa.array(df[f].to_numpy(np.float64), pa.float64()) for f in features},
        }
    )
    pq.write_table(table, path, compression="none", store_schema=True)


def summarize(df: pd.DataFrame, features: list[str]) -> list[tuple[str, object]]:
    """Per-feature mean/std/min/max rows for the distribution sanity report."""
    rows: list[tuple[str, object]] = [("n_calls", len(df))]
    for f in features:
        col = df[f].to_numpy(np.float64)
        rows.append((f"{f}__mean", float(np.nanmean(col))))
        rows.append((f"{f}__std", float(np.nanstd(col))))
    return rows


CHUNK = 200  # flush the parquet every CHUNK new calls (resumable / crash-survivable)


def _pending(all_ids: list[int], done_ids: set[int]) -> list[int]:
    """Call ids still to extract (not already in the parquet) — keeps the run idempotent."""
    return [i for i in all_ids if i not in done_ids]


def build_egemaps(root: Path, *, limit: int | None = None, workers: int = 8):
    """Extract eGeMAPSv02 functionals for every decoded FinCall call; resumable + checkpointed.

    Re-reads any existing `audio_egemaps.parquet`, skips those calls, extracts the rest, and
    flushes the (sorted, deterministic) parquet every CHUNK calls — so a re-run resumes instantly
    and a crash loses at most CHUNK calls. Final parquet is identical regardless of chunking/order.
    """
    import sys

    from ecvol.data.calls import write_metric_csv
    from ecvol.data.manifests import make_entry, write_manifest

    qc = pd.read_csv(root / "coverage" / "fincall_audio_qc.csv")
    qc = qc[qc["decode_ok"]].reset_index(drop=True)
    if limit is not None:
        qc = qc.head(limit)
    store = root / "raw" / "audio_16k" / "fincall"
    features = feature_names()
    out = root / "fincall" / "audio_egemaps.parquet"

    rows: list[dict] = []
    done_ids: set[int] = set()
    if out.is_file():
        existing = pd.read_parquet(out)
        rows = existing.to_dict("records")
        done_ids = {int(r["call_id"]) for r in rows}

    all_ids = [int(c) for c in qc["call_id"]]
    todo = _pending(all_ids, done_ids)
    total = len(done_ids) + len(todo)
    print(f"eGeMAPS: {len(done_ids)} cached, {len(todo)} to extract", flush=True)

    jobs = [(cid, str(store / f"{cid}.flac")) for cid in todo]
    failures: list[tuple[int, str]] = []
    buf: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        for call_id, feat, err in ex.map(_extract, jobs):
            if feat is None:
                failures.append((call_id, err))
            else:
                buf.append({"call_id": call_id, **feat})
            if len(buf) >= CHUNK:
                rows.extend(buf)
                buf = []
                write_egemaps_parquet(pd.DataFrame(rows), out, features)
                print(f"  checkpoint: {len(rows)}/{total} done", file=sys.stderr, flush=True)
    rows.extend(buf)
    df = pd.DataFrame(rows)
    write_egemaps_parquet(df, out, features)

    (root / "manifests").mkdir(parents=True, exist_ok=True)
    write_manifest(
        [make_entry(out, root, source_url=EGEMAPS_SOURCE, license=EGEMAPS_LICENSE)],
        root / "manifests" / "fincall_audio_egemaps.json",
    )
    write_metric_csv(summarize(df, features), root / "coverage" / "fincall_egemaps_summary.csv")
    return len(df), len(failures), features
