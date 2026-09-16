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

import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath

import pandas as pd
import pyarrow as pa

from ecvol.data.manifests import make_entry, write_manifest
from ecvol.features.text._common import write_feature_parquet

from .prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from .schema import EXTRACTED_FIELDS, SectionFeatures

CHUNK = 50  # flush the output parquet every N new rows (resumable checkpoint)
# SectionFeatures + a full-length evidence span. The schema allows 2000 chars of quoted
# transcript, which is ~700 tokens on its own; with the ratings and JSON scaffolding, the old
# 640 truncated mid-string and the response failed to parse (observed 2026-08-09).
MAX_NEW_TOKENS = 1024
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


QWEN25_NATIVE_CONTEXT = 32768  # Qwen2.5 native max_position_embeddings


def yarn_rope_scaling(max_model_len: int, native: int = QWEN25_NATIVE_CONTEXT) -> dict:
    """YaRN rope-scaling config to extend a model's context to ``max_model_len``.

    The corpus has a tail of sections longer than Qwen2.5's 32k native context (FinCall max
    ~61k); the chosen policy (DECISIONS 2026-06-29) is to **extend, not truncate**, so every
    section is processed whole. Static YaRN slightly degrades short-context quality — accepted
    given the coverage gain. ``factor`` is the extension ratio over the native window.
    """
    if max_model_len <= native:
        raise ValueError(f"max_model_len {max_model_len} <= native {native}; YaRN not needed")
    return {
        "rope_type": "yarn",
        "factor": max_model_len / native,
        "original_max_position_embeddings": native,
    }


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

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        rope_scaling: dict | None = None,
        **llm_kwargs,
    ) -> None:
        import outlines
        from vllm import LLM

        if rope_scaling:
            # vLLM ≥0.6 takes HF model-config overrides (incl. YaRN rope scaling) via
            # ``hf_overrides``; older builds accepted a direct ``rope_scaling=`` arg. The OSC
            # container pins vllm-openai:latest → hf_overrides. ``max_model_len`` flows via
            # llm_kwargs (LLM accepts it natively).
            llm_kwargs.setdefault("hf_overrides", {})["rope_scaling"] = rope_scaling
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


# Grammar-level cap on the evidence quote, in characters. The schema allows 2000, but
# llama.cpp cannot compile a 2000-long bounded repetition (the grammar fails to parse), and an
# *unbounded* string lets the model quote until it hits the token budget and emits invalid JSON
# mid-string (observed on a long Q&A section, 2026-08-09). 600 chars (~100 words) is a solid
# audit excerpt, is well inside the schema's own limit so every row still validates, and — since
# evidence dominates decode time and is NOT a scored field (`audit.py` scores ratings only) —
# roughly triples corpus throughput.
EVIDENCE_GRAMMAR_CHARS = 600
_JSON_STRING_BODY = '[^"\\\\\\u0000-\\u001f]'  # no bare quote, backslash, or control char


def grammar_json_schema() -> dict:
    """``SectionFeatures`` JSON Schema rewritten to compile to a small GBNF grammar.

    llama.cpp converts a JSON Schema to a grammar, and two pydantic constructs compile badly:
    ``maxLength`` on a string becomes an N-way repetition rule (``evidence`` → 2000 of them),
    and integer ``minimum``/``maximum`` bounds have patchy converter support. Both are rewritten
    to *exactly equivalent* forms — a bounded int becomes an explicit enum of its allowed values,
    and ``maxLength`` is dropped here and enforced instead by `SectionFeatures` validation on the
    way back in (`_truncate_evidence`). The decoded object therefore satisfies the same contract
    as the Outlines engines; only the grammar's size changes.
    """
    schema = SectionFeatures.model_json_schema()
    for prop in schema.get("properties", {}).values():
        if prop.get("type") == "integer" and "minimum" in prop and "maximum" in prop:
            prop["enum"] = list(range(prop.pop("minimum"), prop.pop("maximum") + 1))
        prop.pop("maxLength", None)
    # llama.cpp rejects `maxLength` outright, so the length bound is expressed as an anchored
    # regex (which its converter does compile) — see EVIDENCE_GRAMMAR_CHARS.
    schema["properties"]["evidence"] = {
        "type": "string",
        "pattern": f"^{_JSON_STRING_BODY}{{0,{EVIDENCE_GRAMMAR_CHARS}}}$",
    }
    # Every field is REQUIRED in the grammar. pydantic omits defaulted fields from ``required``,
    # which lets the grammar close the object early — observed live: the model skipped the
    # trailing ``evidence`` on every section, and a skipped ``management_optimism`` would have
    # been filled by its default 0, i.e. a *non-rating* silently indistinguishable from a rated 0.
    # Requiring them changes no field, range, or rubric; it only forces the model to actually
    # answer what the prompt already asks for.
    schema["required"] = list(schema.get("properties", {}))
    return schema


def _truncate_evidence(obj: dict) -> dict:
    """Clip ``evidence`` to the schema's declared max so validation can't fail on length alone."""
    max_len = SectionFeatures.model_fields["evidence"].metadata[0].max_length
    if isinstance(obj.get("evidence"), str) and len(obj["evidence"]) > max_len:
        obj["evidence"] = obj["evidence"][:max_len]
    return obj


class LlamaCppServerEngine:
    """Local engine: llama.cpp ``llama-server`` (GGUF) + JSON-schema-constrained decoding.

    The transformers+bitsandbytes engine CUDA-OOMs on this project's long sections on a 16 GB
    card (`data/coverage/llm_probe_report.md`), so the *local* corpus path is llama.cpp, which
    chunks the prefill (``--ubatch-size``) and has a CUDA flash-attention kernel on Windows.
    The server is driven over HTTP rather than through Python bindings so no CUDA toolchain is
    needed to build anything (DECISIONS 2026-08-09).

    The engine owns the server process: it starts it, waits for ``/health``, and — because this
    runs unattended overnight — restarts it and retries once if the process dies mid-corpus.
    ``model_id`` is a provenance label only (``repo:QUANT``); the weights come from ``gguf_path``.
    """

    def __init__(
        self,
        model_id: str,
        *,
        gguf_path: str | Path,
        server_bin: str | Path,
        revision: str | None = None,
        max_model_len: int = 65536,
        rope_scaling: dict | None = None,
        n_gpu_layers: int = 99,
        ubatch: int = 512,
        host: str = "127.0.0.1",
        port: int = 8080,
        startup_timeout: float = 900.0,
        request_timeout: float = 3600.0,
    ) -> None:
        self.model_id = model_id
        self.gguf_path = Path(gguf_path)
        self.server_bin = Path(server_bin)
        self.max_model_len = max_model_len
        self.rope_scaling = rope_scaling
        self.n_gpu_layers = n_gpu_layers
        self.ubatch = ubatch
        self.url = f"http://{host}:{port}"
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.schema = grammar_json_schema()
        self.proc: subprocess.Popen | None = None
        self._start()

    # --- server lifecycle ---

    def _cmd(self) -> list[str]:
        cmd = [
            str(self.server_bin),
            "--model", str(self.gguf_path),
            "--ctx-size", str(self.max_model_len),
            "--n-gpu-layers", str(self.n_gpu_layers),
            "--ubatch-size", str(self.ubatch),
            "--parallel", "1",
            "--flash-attn", "on",
            "--seed", "0",
            "--no-webui",
            "--host", self.url.split("//")[1].split(":")[0],
            "--port", self.url.rsplit(":", 1)[1],
        ]  # fmt: skip
        if self.rope_scaling:
            # Same >32k policy as the vLLM engine (extend, don't truncate; DECISIONS 2026-06-29),
            # expressed in llama.cpp's flags instead of HF rope_scaling keys.
            cmd += [
                "--rope-scaling", "yarn",
                "--rope-scale", str(self.rope_scaling["factor"]),
                "--yarn-orig-ctx", str(self.rope_scaling["original_max_position_embeddings"]),
            ]  # fmt: skip
        return cmd

    def _start(self) -> None:
        self.proc = subprocess.Popen(
            self._cmd(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(self.server_bin.parent),
        )
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.proc.returncode}")
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=5) as r:  # noqa: S310
                    if r.status == 200:
                        return
            except (urllib.error.URLError, OSError, TimeoutError):
                time.sleep(2.0)
        self.close()
        raise RuntimeError(f"llama-server not healthy within {self.startup_timeout}s")

    def _restart(self) -> None:
        self.close()
        time.sleep(5.0)
        self._start()

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    # --- inference ---

    def _post(self, system: str, user: str) -> str:
        payload = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": 0.0,
            "top_k": 1,
            "seed": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "SectionFeatures", "schema": self.schema, "strict": True},
            },
        }
        req = urllib.request.Request(  # noqa: S310
            f"{self.url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.request_timeout) as r:  # noqa: S310
            body = json.load(r)
        return body["choices"][0]["message"]["content"]

    def generate(self, system: str, user: str) -> dict:
        try:
            out = self._post(system, user)
        except (urllib.error.URLError, OSError, TimeoutError, KeyError):
            # Server died or stalled (OOM on a pathological section, driver hiccup). One clean
            # restart + retry keeps an unattended overnight run alive; a second failure is real.
            self._restart()
            out = self._post(system, user)
        return SectionFeatures.model_validate(_truncate_evidence(json.loads(out))).model_dump()


def build_engine(model_id: str, *, engine: str = "transformers", **kwargs):
    """Construct the named engine. ``transformers`` | ``llamacpp`` (local) | ``vllm`` (cloud)."""
    if engine == "transformers":
        return TransformersOutlinesEngine(model_id, **kwargs)
    if engine == "llamacpp":
        return LlamaCppServerEngine(model_id, **kwargs)
    if engine == "vllm":
        return VLLMEngine(model_id, **kwargs)
    raise ValueError(f"unknown engine {engine!r}; expected 'transformers', 'llamacpp' or 'vllm'")


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


def run_config(model_id: str, revision: str, engine: str, engine_kwargs: dict) -> dict:
    """The decoding configuration that determines a row's content.

    The feature rows themselves record only ``model_id``/``revision``/``prompt_version``, so two
    runs differing in context window, YaRN, decode budget or evidence bound would be
    indistinguishable in the parquet — and this project changed exactly those mid-development.
    Captured here (and digested) so a parquet can be shown to be internally consistent.
    """
    cfg = {
        "model_id": model_id,
        "revision": revision,
        "prompt_version": PROMPT_VERSION,
        "engine": engine,
        "max_new_tokens": MAX_NEW_TOKENS,
        "evidence_grammar_chars": EVIDENCE_GRAMMAR_CHARS,
        "greedy": True,
        "max_model_len": engine_kwargs.get("max_model_len"),
        "rope_scaling": engine_kwargs.get("rope_scaling"),
        "n_gpu_layers": engine_kwargs.get("n_gpu_layers"),
        "load_in_4bit": engine_kwargs.get("load_in_4bit"),
        # basename only: the absolute path is machine-specific, the weights file is not
        "weights_file": (
            # PureWindowsPath splits on both `\` and `/`, so this holds on Linux CI too
            PureWindowsPath(engine_kwargs["gguf_path"]).name
            if engine_kwargs.get("gguf_path")
            else None
        ),
    }
    return {k: v for k, v in cfg.items() if v is not None}


def config_digest(cfg: dict) -> str:
    """Stable short hash of a run config — differing digests in one parquet = mixed rows."""
    import hashlib

    payload = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def _append_run_sidecar(out_path: Path, cfg: dict, n_new: int, n_rows: int, secs: float) -> Path:
    """Append this invocation to ``<features>.run.json`` (provenance for the parquet).

    A list, not a single record, because the parquet is a resume store built over many
    invocations: only the full list can show that every row was produced under one config.
    """
    from ecvol.tracking import env_fingerprint, git_info

    side = out_path.with_suffix(".run.json")
    runs = json.loads(side.read_text(encoding="utf-8")) if side.exists() else []
    runs.append(
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "config": cfg,
            "config_digest": config_digest(cfg),
            "n_new_rows": n_new,
            "n_rows_after": n_rows,
            "seconds": round(secs, 1),
            "git": git_info(),
            "env": env_fingerprint(),
        }
    )
    side.write_text(json.dumps(runs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return side


def mixed_config_digests(sidecar: str | Path) -> list[str]:
    """Distinct config digests in a sidecar. >1 means the parquet mixes configurations."""
    runs = json.loads(Path(sidecar).read_text(encoding="utf-8"))
    return sorted({r["config_digest"] for r in runs})


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
    deadline: float | None = None,
    **engine_kwargs,
) -> BuildResult:
    """Extract ``SectionFeatures`` for the requested calls; resumable, deterministic.

    The per-model parquet is the resume store: ``(call_id, section)`` rows already present are
    skipped. ``engine_obj`` injects a pre-built (or fake, for tests) engine; otherwise one is
    built lazily via ``build_engine`` (loads the model — GPU). Returns a ``BuildResult``.

    ``deadline`` (a ``time.time()`` epoch) stops the run cleanly at a wall-clock time: the loop
    finishes its current section, flushes, and returns. That is how a run that must release the
    GPU at a fixed hour ends without losing the rows extracted since the last periodic flush.
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

    # ``device`` is a torch concept — only the transformers engine takes it (vLLM and the
    # llama.cpp server manage placement themselves).
    if engine == "transformers":
        engine_kwargs["device"] = device
    eng = engine_obj or build_engine(model_id, engine=engine, **engine_kwargs)

    t0 = time.perf_counter()
    n_new = 0
    try:
        for call_id, section, text in pending:
            if deadline is not None and time.time() >= deadline:
                break
            feat = eng.generate(SYSTEM_PROMPT, build_user_prompt(section, text))
            rows.append(_row(call_id, section, model_id, rev, feat))
            n_new += 1
            if n_new % CHUNK == 0:
                _flush(rows, out_path)
    except KeyboardInterrupt:
        pass  # flush what we have below, then re-raise nothing — the parquet stays resumable
    finally:
        if engine_obj is None and hasattr(eng, "close"):
            eng.close()  # engines we built, we release (frees VRAM / kills the server process)
    secs = time.perf_counter() - t0

    _flush(rows, out_path)
    cfg = run_config(model_id, rev, engine, engine_kwargs)
    side = _append_run_sidecar(out_path, cfg, n_new, len(rows), secs)
    digests = mixed_config_digests(side)
    if len(digests) > 1:
        # Loud, not fatal: a config change mid-parquet is sometimes legitimate (resuming after
        # a bug fix), but rows produced under different decoding settings must never be
        # silently pooled into one feature table.
        print(
            f"WARNING: {out_path.name} now mixes {len(digests)} decoding configs "
            f"({', '.join(digests)}); see {side.name} — re-extract before using it as one table."
        )
    src = f"derived: ecvol featurize llm (T6.2) model={model_id}@{rev} prompt={PROMPT_VERSION}"
    entry = make_entry(out_path, root, source_url=src, license="derived")
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    manifest = root / "manifests" / f"{dataset}_llm_features__{model_slug(model_id)}.json"
    write_manifest([entry], manifest)
    return BuildResult(len(rows), n_new, secs, out_path)
