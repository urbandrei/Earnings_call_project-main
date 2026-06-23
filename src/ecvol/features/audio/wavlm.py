"""Frozen WavLM-Large per-call audio embeddings (T4.3, GPU).

Each 16 kHz FLAC (from T4.1) is split into non-overlapping 30 s windows; WavLM-Large
(`microsoft/wavlm-large`, 1024-d) encodes each window, the last hidden state is mean-pooled
over time → a per-window vector, then mean-pooled over windows → one per-call embedding.
fp32 + deterministic; resumable + checkpointed (skips cached call_ids, flushes every CHUNK) like
T4.2. emotion2vec+ is a separate second pass. Output: `data/fincall/audio_wavlm.parquet`.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

WAVLM_MODEL = "microsoft/wavlm-large"
WAVLM_DIM = 1024
SR = 16000
WINDOW = 30 * SR  # 30 s, non-overlapping
MIN_SAMPLES = SR  # skip trailing windows shorter than 1 s
CHUNK = 100  # checkpoint cadence (calls)
WAVLM_SOURCE = "derived: ecvol audio wavlm (T4.3)"
WAVLM_LICENSE = "derived; model microsoft/wavlm-large"


def set_deterministic(seed: int = 0) -> None:
    import torch

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def load_model(device: str = "cuda", *, fp16: bool = False):
    from transformers import AutoFeatureExtractor, AutoModel

    set_deterministic()
    fe = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL)
    model = AutoModel.from_pretrained(WAVLM_MODEL).to(device).eval()
    if fp16 and device != "cpu":
        model = model.half()
    return fe, model


def embed_file(
    fe, model, path: str, device: str, *, fp16: bool = False, batch: int = 4
) -> np.ndarray:
    """Per-call 1024-d embedding: mean over 30 s windows of the time-mean last hidden state.

    Full 30 s windows (uniform length, no padding) are batched `batch` at a time; the trailing
    short window runs alone. Optional fp16 for throughput (slightly non-deterministic).
    """
    import soundfile as sf
    import torch

    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    windows = [wav[i : i + WINDOW] for i in range(0, len(wav), WINDOW)]
    windows = [w for w in windows if len(w) >= MIN_SAMPLES]
    if not windows:
        return np.full(WAVLM_DIM, np.nan, dtype=np.float64)
    full = [w for w in windows if len(w) == WINDOW]
    rest = [w for w in windows if len(w) != WINDOW]

    def run(batch_list: list) -> np.ndarray:
        inp = fe(batch_list, sampling_rate=SR, return_tensors="pt").input_values.to(device)
        if fp16 and device != "cpu":
            inp = inp.half()
        hidden = model(inp).last_hidden_state  # (B, T, 1024)
        return hidden.float().mean(dim=1).cpu().numpy()  # (B, 1024)

    vecs = []
    with torch.no_grad():
        for j in range(0, len(full), batch):
            vecs.append(run(full[j : j + batch]))
        for w in rest:
            vecs.append(run([w]))
    return np.vstack(vecs).mean(axis=0).astype(np.float64)


def write_vector_parquet(rows: list[dict], path: Path) -> None:
    """Deterministic parquet (sorted by call_id): call_id:int64 + vector:list<double>."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: int(r["call_id"]))
    table = pa.table(
        {
            "call_id": pa.array([int(r["call_id"]) for r in rows], pa.int64()),
            "n_windows": pa.array([int(r["n_windows"]) for r in rows], pa.int64()),
            "vector": pa.array([list(r["vector"]) for r in rows], pa.list_(pa.float64())),
        }
    )
    pq.write_table(table, path, compression="none", store_schema=True)


def build_wavlm(
    root: Path,
    *,
    limit: int | None = None,
    device: str = "cuda",
    fp16: bool = False,
    batch: int = 4,
):
    """Embed every decoded FinCall call with WavLM-Large; resumable. Returns (n, n_new, secs)."""
    from ecvol.data.manifests import make_entry, write_manifest

    qc = pd.read_csv(root / "coverage" / "fincall_audio_qc.csv")
    qc = qc[qc["decode_ok"]].reset_index(drop=True)
    if limit is not None:
        qc = qc.head(limit)
    store = root / "raw" / "audio_16k" / "fincall"
    out = root / "fincall" / "audio_wavlm.parquet"

    rows: list[dict] = []
    done: set[int] = set()
    if out.is_file():
        ex = pd.read_parquet(out)
        rows = ex.to_dict("records")
        done = {int(r["call_id"]) for r in rows}
    todo = [int(c) for c in qc["call_id"] if int(c) not in done]
    print(f"WavLM: {len(done)} cached, {len(todo)} to embed", flush=True)

    fe, model = load_model(device, fp16=fp16) if todo else (None, None)
    t0 = time.perf_counter()
    n_new = 0
    for i, cid in enumerate(todo, 1):
        flac = store / f"{cid}.flac"
        vec = embed_file(fe, model, str(flac), device, fp16=fp16, batch=batch)
        rows.append({"call_id": cid, "n_windows": _n_windows(flac), "vector": vec})
        n_new += 1
        if i % CHUNK == 0:
            write_vector_parquet(rows, out)
            print(f"  checkpoint: {len(rows)} done ({i}/{len(todo)} new)", flush=True)
    secs = time.perf_counter() - t0
    write_vector_parquet(rows, out)
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    write_manifest(
        [make_entry(out, root, source_url=WAVLM_SOURCE, license=WAVLM_LICENSE)],
        root / "manifests" / "fincall_audio_wavlm.json",
    )
    return len(rows), n_new, secs


def _n_windows(path: Path) -> int:
    import soundfile as sf

    info = sf.info(str(path))
    return max(1, int(info.frames // WINDOW) + (1 if info.frames % WINDOW >= MIN_SAMPLES else 0))
