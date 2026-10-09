"""examples/quickstart.py runs end to end on a tiny synthetic release archive (CPU, seconds)."""

from __future__ import annotations

import importlib.util
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

QUICKSTART = Path(__file__).resolve().parents[1] / "examples" / "quickstart.py"


def _load_quickstart():
    spec = importlib.util.spec_from_file_location("quickstart", QUICKSTART)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _synthetic_archive(path: Path, n_calls: int = 60, dim: int = 6) -> None:
    rng = np.random.default_rng(0)
    tickers = [f"T{i % 12:02d}" for i in range(n_calls)]
    dates = pd.date_range("2019-01-02", periods=n_calls, freq="7D")
    rows = []
    for i in range(n_calls):
        v_pre = rng.normal(-4.0, 0.5)
        for h in (3, 7, 15, 30):
            v_post = v_pre + rng.normal(0.1, 0.3)
            rows.append(
                {
                    "call_id": f"c{i}",
                    "ticker": tickers[i],
                    "as_of": dates[i].strftime("%Y-%m-%d"),
                    "horizon": h,
                    "v_pre": v_pre,
                    "v_post": v_post,
                    "delta_v": v_post - v_pre,
                    "rv_daily": abs(rng.normal(0, 1e-4)),
                    "rv_weekly": abs(rng.normal(0, 1e-4)),
                    "rv_monthly": abs(rng.normal(0, 1e-3)),
                    "status": "ok",
                }
            )
    targets = pd.DataFrame(rows)
    emb = pd.DataFrame(
        {
            "call_id": [f"c{i}" for i in range(n_calls)],
            "source": "fincall",
            "scope": "full",
            "vector": [rng.normal(size=dim).astype(np.float32) for _ in range(n_calls)],
        }
    )
    order = np.arange(n_calls)
    temporal = ["train"] * 36 + ["embargo"] * 4 + ["val"] * 8 + ["test"] * 12
    by_ticker = {
        t: ("train" if j < 8 else "val" if j < 10 else "test")
        for j, t in enumerate(sorted(set(tickers)))
    }
    disjoint = [by_ticker[t] for t in tickers]
    with zipfile.ZipFile(path, "w") as z:
        for name, df in (("demo/targets.parquet", targets), ("demo/text_embeddings.parquet", emb)):
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            z.writestr(name, buf.getvalue())
        for name, labels in (
            ("splits/demo_temporal.csv", temporal),
            ("splits/demo_ticker_disjoint.csv", disjoint),
        ):
            df = pd.DataFrame(
                {
                    "call_id": [f"c{i}" for i in order],
                    "ticker": tickers,
                    "as_of": dates,
                    "split": labels,
                }
            )
            z.writestr(name, df.to_csv(index=False))


def test_quickstart_runs_on_a_synthetic_archive(tmp_path: Path) -> None:
    archive = tmp_path / "ecvol-bench_demo_test.zip"
    _synthetic_archive(archive)
    qs = _load_quickstart()
    written = qs.run(archive, tmp_path / "out")
    assert set(written) == {
        f"demo_{s}_{v}_delta_v"
        for s in ("temporal", "ticker_disjoint")
        for v in ("pastvol", "text_pastvol")
    }
    for stem, report_dir in written.items():
        frame = pd.read_csv(tmp_path / "out" / f"{stem}.csv")
        assert {
            "call_id",
            "ticker",
            "date",
            "split",
            "horizon",
            "y_true",
            "y_pred",
            "y_persistence",
        } <= set(frame.columns)
        assert set(frame["split"]) <= {"train", "val", "test"}
        assert frame.loc[frame["split"] == "test", "y_pred"].notna().all()
        assert (report_dir / "report.csv").exists() and (report_dir / "report.md").exists()
