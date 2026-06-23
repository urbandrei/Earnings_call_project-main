"""Frozen emotion2vec+ per-call audio embeddings (T4.3 second pass, GPU).

`emotion2vec/emotion2vec_plus_large` (via funasr) produces an utterance-level embedding per
input. Each 16 kHz FLAC (from T4.1) is split into non-overlapping 30 s windows, embedded, and
mean-pooled over windows → one per-call vector (mirrors the WavLM path). Resumable + checkpointed
(skips cached call_ids). Output: `data/fincall/audio_emotion2vec.parquet`.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ecvol.features.audio.wavlm import MIN_SAMPLES, WINDOW, _n_windows, write_vector_parquet

E2V_MODEL = "iic/emotion2vec_plus_large"
E2V_DIM = 1024
CHUNK = 100
E2V_SOURCE = "derived: ecvol audio emotion2vec (T4.3)"
E2V_LICENSE = "derived; model emotion2vec/emotion2vec_plus_large"


def load_model(device: str = "cuda"):
    from funasr import AutoModel

    return AutoModel(model=E2V_MODEL, device=device, disable_update=True, disable_pbar=True)


def embed_file(model, path: str) -> np.ndarray:
    """Per-call embedding: mean over 30 s windows of emotion2vec+ utterance embeddings."""
    import soundfile as sf

    wav, _ = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    vecs = []
    for i in range(0, len(wav), WINDOW):
        ch = wav[i : i + WINDOW]
        if len(ch) < MIN_SAMPLES:
            continue
        res = model.generate(ch.astype(np.float32), granularity="utterance", extract_embedding=True)
        feats = np.asarray(res[0]["feats"], dtype=np.float64)
        vecs.append(feats)
    if not vecs:
        return np.full(E2V_DIM, np.nan, dtype=np.float64)
    return np.mean(vecs, axis=0).astype(np.float64)


def build_emotion2vec(root: Path, *, limit: int | None = None, device: str = "cuda"):
    """Embed every decoded FinCall call with emotion2vec+; resumable. Returns (n, n_new, secs)."""
    from ecvol.data.manifests import make_entry, write_manifest

    qc = pd.read_csv(root / "coverage" / "fincall_audio_qc.csv")
    qc = qc[qc["decode_ok"]].reset_index(drop=True)
    if limit is not None:
        qc = qc.head(limit)
    store = root / "raw" / "audio_16k" / "fincall"
    out = root / "fincall" / "audio_emotion2vec.parquet"

    rows: list[dict] = []
    done: set[int] = set()
    if out.is_file():
        ex = pd.read_parquet(out)
        rows = ex.to_dict("records")
        done = {int(r["call_id"]) for r in rows}
    todo = [int(c) for c in qc["call_id"] if int(c) not in done]
    print(f"emotion2vec+: {len(done)} cached, {len(todo)} to embed", flush=True)

    model = load_model(device) if todo else None
    t0 = time.perf_counter()
    n_new = 0
    for i, cid in enumerate(todo, 1):
        flac = store / f"{cid}.flac"
        vec = embed_file(model, str(flac))
        rows.append({"call_id": cid, "n_windows": _n_windows(flac), "vector": vec})
        n_new += 1
        if i % CHUNK == 0:
            write_vector_parquet(rows, out)
            print(f"  checkpoint: {len(rows)} done ({i}/{len(todo)} new)", flush=True)
    secs = time.perf_counter() - t0
    write_vector_parquet(rows, out)
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    write_manifest(
        [make_entry(out, root, source_url=E2V_SOURCE, license=E2V_LICENSE)],
        root / "manifests" / "fincall_audio_emotion2vec.json",
    )
    return len(rows), n_new, secs
