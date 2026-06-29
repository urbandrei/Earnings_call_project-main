"""Constrained LLM structured-feature extraction (T6.2).

Runs a frozen instruction LLM over every ``(call, section)`` and decodes a schema-valid
``SectionFeatures`` JSON (Outlines JSON-schema-constrained decoding → 100% valid by
construction; the *content* gate is the κ-audit, `audit.py`). Engine-agnostic so the same
pipeline runs the **local** probe/corpus (transformers + bitsandbytes 4-bit, Windows) and the
**OSC cloud** corpus (vLLM, Linux) — only the weights+quant must match between a model's
audited 50 calls and its corpus run (see `docs/llm_feature_rubric.md` / DECISIONS 2026-06-24).

Mirrors the resumable audio extractors (`features/audio/wavlm.py`): the per-model output
parquet ``data/{dataset}/llm_features__{slug}.parquet`` IS the resume store — already-done
``(call_id, section)`` rows are skipped, so a run can be killed/requeued (OSC walltime) and
resumed. Decoding is greedy (`do_sample=False`) for reproducibility. Heavy deps (torch,
transformers, outlines) are imported lazily so the module + tests load without a GPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa

from ecvol.data.manifests import make_entry, write_manifest
from ecvol.features.text._common import write_feature_parquet

from .prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from .schema import EXTRACTED_FIELDS, SectionFeatures

CHUNK = 50  # flush the output parquet every N new rows (resumable checkpoint)
MAX_NEW_TOKENS = 640  # SectionFeatures incl. a ~2000-char evidence span fits comfortably
_SECTION_ORDER = {"prepared_remarks": 0, "qa": 1}
_OUTPUT_FIELDS = ["call_id", "section", "model_id", "revision", "prompt_version", *EXTRACTED_FIELDS]
_OUTPUT_FIELDS.append("evidence")


@dataclass
class BuildResult:
    n_rows: int  # total rows in the output parquet after the run
    n_new: int  # rows extracted this run
    secs: float  # wall-clock seconds spent extracting (model load excluded)
    out_path: Path


def model_slug(model_id: str) -> str:
    """Filesystem-safe slug for a HF model id (``org/name`` → ``org__name``)."""
    return model_id.replace("/", "__").replace(":", "_")


def iter_section_inputs(
    root: str | Path,
    dataset: str,
    *,
    call_ids: list[str] | None = None,
    limit: int | None = None,
):
    """Yield ``(call_id, section, text)`` — chunk text concatenated per section in turn order.

    ``call_ids`` restricts to a specific set (the audit sample); ``limit`` takes the first N
    calls (sorted). The two are mutually exclusive in practice; ``call_ids`` wins if both given.
    """
    root = Path(root)
    chunks = pd.read_parquet(root / dataset / "chunks.parquet")
    chunks["call_id"] = chunks["call_id"].astype(str)
    if call_ids is not None:
        wanted = set(map(str, call_ids))
        chunks = chunks[chunks["call_id"].isin(wanted)]
    elif limit is not None:
        keep = sorted(chunks["call_id"].unique())[:limit]
        chunks = chunks[chunks["call_id"].isin(keep)]
    for call_id, cc in chunks.groupby("call_id", sort=True):
        sections = sorted(cc["section"].unique(), key=lambda s: _SECTION_ORDER.get(s, 9))
        for section in sections:
            sec = cc[cc["section"] == section].sort_values(["turn_idx", "chunk_in_turn"])
            text = " ".join(str(t) for t in sec["text"])
            yield str(call_id), section, text


def total_sections(root: str | Path, datasets: tuple[str, ...]) -> int:
    """Count ``(call, section)`` pairs across datasets (for full-corpus ETA projection)."""
    root = Path(root)
    n = 0
    for ds in datasets:
        ch = pd.read_parquet(root / ds / "chunks.parquet", columns=["call_id", "section"])
        n += ch.drop_duplicates(["call_id", "section"]).shape[0]
    return n


# --- engines -----------------------------------------------------------------


class TransformersOutlinesEngine:
    """Local engine: transformers + bitsandbytes 4-bit (nf4) + Outlines constrained decoding."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cuda",
        revision: str | None = None,
        load_in_4bit: bool = True,
    ) -> None:
        import outlines
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        quant = None
        if load_in_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        hf = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            quantization_config=quant,
            device_map=device,
            dtype=torch.float16,
        )
        hf.eval()
        self.model = outlines.from_transformers(hf, self.tokenizer)
        self.generator = outlines.Generator(self.model, SectionFeatures)

    def generate(self, system: str, user: str) -> dict:
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        out = self.generator(prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        return SectionFeatures.model_validate_json(out).model_dump()


class VLLMEngine:
    """Cloud engine (OSC, Linux): vLLM offline + Outlines constrained decoding."""

    def __init__(self, model_id: str, *, revision: str | None = None, **llm_kwargs) -> None:
        import outlines
        from vllm import LLM

        llm = LLM(model=model_id, revision=revision, **llm_kwargs)
        self.tokenizer = llm.get_tokenizer()
        self.model = outlines.from_vllm_offline(llm)
        self.generator = outlines.Generator(self.model, SectionFeatures)

    def generate(self, system: str, user: str) -> dict:
        from vllm import SamplingParams

        prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        out = self.generator(prompt, SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0))
        return SectionFeatures.model_validate_json(out).model_dump()


def build_engine(model_id: str, *, engine: str = "transformers", **kwargs):
    """Construct the named engine. ``transformers`` (local) | ``vllm`` (cloud)."""
    if engine == "transformers":
        return TransformersOutlinesEngine(model_id, **kwargs)
    if engine == "vllm":
        return VLLMEngine(model_id, **kwargs)
    raise ValueError(f"unknown engine {engine!r}; expected 'transformers' or 'vllm'")


# --- driver ------------------------------------------------------------------


def _row(call_id: str, section: str, model_id: str, revision: str, feat: dict) -> dict:
    row = {
        "call_id": call_id,
        "section": section,
        "model_id": model_id,
        "revision": revision,
        "prompt_version": PROMPT_VERSION,
        "evidence": feat.get("evidence", ""),
    }
    for f in EXTRACTED_FIELDS:
        row[f] = feat[f]
    return row


def _flush(rows: list[dict], out_path: Path) -> None:
    df = pd.DataFrame(rows, columns=_OUTPUT_FIELDS)
    write_feature_parquet(df, out_path, id_type=pa.string(), sort_cols=["call_id", "section"])


def build_llm(
    root: str | Path,
    dataset: str,
    *,
    model_id: str,
    revision: str | None = None,
    engine: str = "transformers",
    device: str = "cuda",
    call_ids: list[str] | None = None,
    limit: int | None = None,
    engine_obj=None,
    **engine_kwargs,
) -> BuildResult:
    """Extract ``SectionFeatures`` for the requested calls; resumable, deterministic.

    The per-model parquet is the resume store: ``(call_id, section)`` rows already present are
    skipped. ``engine_obj`` injects a pre-built (or fake, for tests) engine; otherwise one is
    built lazily via ``build_engine`` (loads the model — GPU). Returns a ``BuildResult``.
    """
    root = Path(root)
    rev = revision or ""
    out_path = root / dataset / f"llm_features__{model_slug(model_id)}.parquet"

    rows: list[dict] = []
    done: set[tuple[str, str]] = set()
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        prev["call_id"] = prev["call_id"].astype(str)
        rows = prev[_OUTPUT_FIELDS].to_dict("records")
        done = {(r["call_id"], r["section"]) for r in rows}

    pending = [
        (c, s, t)
        for c, s, t in iter_section_inputs(root, dataset, call_ids=call_ids, limit=limit)
        if (c, s) not in done
    ]
    if not pending:
        if rows:
            _flush(rows, out_path)
        return BuildResult(len(rows), 0, 0.0, out_path)

    eng = engine_obj or build_engine(model_id, engine=engine, device=device, **engine_kwargs)

    t0 = time.perf_counter()
    n_new = 0
    for call_id, section, text in pending:
        feat = eng.generate(SYSTEM_PROMPT, build_user_prompt(section, text))
        rows.append(_row(call_id, section, model_id, rev, feat))
        n_new += 1
        if n_new % CHUNK == 0:
            _flush(rows, out_path)
    secs = time.perf_counter() - t0

    _flush(rows, out_path)
    src = f"derived: ecvol featurize llm (T6.2) model={model_id}@{rev} prompt={PROMPT_VERSION}"
    entry = make_entry(out_path, root, source_url=src, license="derived")
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    manifest = root / "manifests" / f"{dataset}_llm_features__{model_slug(model_id)}.json"
    write_manifest([entry], manifest)
    return BuildResult(len(rows), n_new, secs, out_path)
