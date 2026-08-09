"""EXPLORATORY (T6.2 diagnostic): does 2x model scale move the kappa gate?

The gate failed on Qwen2.5-7B and the cheap levers did little (instruction anchoring ~0,
Q8_0 quantization small, few-shot small). The remaining model-side hypothesis is capability.

Design note — why this does not run at the frozen 65536/YaRN config: Qwen2.5-14B needs
~9.0 GB (Q4_K_M) + ~12.9 GB fp16 KV at 65k = far past this 16 GB card. Every workaround
(quantized KV, shorter context) differs from the 7B's config and would confound scale with
that difference. So BOTH models run here under one identical reduced-context config, and only
the 7B-vs-14B *difference* is interpreted. Every audit section fits whole at this context
(longest is ~15.7k tokens), so nothing is truncated — but because YaRN is off, these absolute
numbers are NOT the gate. The official gate number remains the 65k+YaRN 7B run.

Scoring goes through `audit.compute_kappa`, the same function `ecvol llm-kappa` uses, rather
than a reimplementation. Predictions are persisted under artifacts/diagnostics/ so the arms
can be compared with a paired bootstrap; nothing under data/ is touched.

Usage: uv run python notebooks/llm_model_scale_probe.py [ctx]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from ecvol.features.llm.audit import compute_kappa
from ecvol.features.llm.extract import (
    _OUTPUT_FIELDS,
    LlamaCppServerEngine,
    iter_section_inputs,
)
from ecvol.features.llm.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from ecvol.features.llm.reading import sample_train_calls

CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 20480
SERVER = r"D:\ecvol-data\tools\llamacpp\llama-server.exe"
MODELS = [
    ("Qwen2.5-7B-Q4_K_M", r"D:\ecvol-data\models\Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
    ("Qwen2.5-14B-Q4_K_M", r"D:\ecvol-data\models\Qwen2.5-14B-Instruct-Q4_K_M.gguf"),
]
SHEET = "data/coverage/fincall_llm_labels_rater1.csv"


def run_model(label: str, gguf: str, sections: list[tuple[str, str, str]], tmp: Path) -> dict:
    eng = LlamaCppServerEngine(
        label, gguf_path=gguf, server_bin=SERVER, max_model_len=CTX, rope_scaling=None
    )
    try:
        rows = []
        for i, (cid, sec, text) in enumerate(sections, 1):
            feat = eng.generate(SYSTEM_PROMPT, build_user_prompt(sec, text))
            rows.append(
                {
                    "call_id": cid,
                    "section": sec,
                    "model_id": label,
                    "revision": "",
                    "prompt_version": PROMPT_VERSION,
                    **feat,
                }
            )
            if i % 25 == 0:
                print(f"  {label}: {i}/{len(sections)}", flush=True)
    finally:
        eng.close()
    path = tmp / f"{label}.parquet"
    pd.DataFrame(rows, columns=_OUTPUT_FIELDS).to_parquet(path)
    k = compute_kappa(SHEET, path)  # the same scorer the gate uses
    out = {f: (None if v["kappa"] is None else round(v["kappa"], 3)) for f, v in k.items()}
    print(f"{label} (ctx={CTX}, no YaRN): {json.dumps(out)}", flush=True)
    return out


def main() -> None:
    call_ids = sample_train_calls("data", "fincall", 50, 0)
    sections = list(iter_section_inputs("data", "fincall", call_ids=call_ids))
    print(f"{len(sections)} audit sections; ctx={CTX}; YaRN off (see module docstring)")
    # Predictions are PERSISTED, not written to a temp dir: κ point estimates at n=97 carry a
    # standard error around 0.1, so any comparison drawn from them needs a paired bootstrap over
    # calls — and that must not require re-running the GPU.
    out_dir = Path("artifacts/diagnostics/scale_probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {label: run_model(label, gguf, sections, out_dir) for label, gguf in MODELS}
    print(f"predictions kept in {out_dir} for bootstrap analysis")
    a, b = results[MODELS[0][0]], results[MODELS[1][0]]
    delta = {f: (None if a[f] is None or b[f] is None else round(b[f] - a[f], 3)) for f in a}
    print("\n14B - 7B delta:", json.dumps(delta))


if __name__ == "__main__":
    main()
