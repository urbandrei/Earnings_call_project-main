"""EXPLORATORY (T6.2 diagnostic): does *example-based* calibration rescue the κ gate?

The gate failed at κ≈0.2 (`data/coverage/llm_kappa_gate_report.md`). Two explanations were
already ruled out there: sharpened *instruction* anchoring (no effect) and 4-bit quantization
(no effect). This tests the remaining cheap model-side lever — showing the model worked
examples of the rater's own scale use, which is usually far more effective than describing it.

Design (the honesty-critical part):
- Exemplars are drawn from a held-IN subset of the 50 audit calls; κ is scored ONLY on the
  held-OUT calls. No call is ever both an exemplar and a scored row.
- Both arms (zero-shot v2, few-shot) are scored on the SAME held-out rows, so the comparison
  is like-for-like rather than against the 97-row number from the full sample.
- Writes nothing to data/. The frozen prompt, schema and PROMPT_VERSION are untouched — this
  is a measurement of what *would* be possible, not a change to the pipeline.

Caveat to carry into any writeup: exemplars come from rater 1, and the gate scores against
rater 1. Few-shot therefore partly fits this annotator's idiosyncrasy. That is legitimate for a
deployable configuration (the same exemplars would ship with the corpus run) but it weakens any
claim that the extraction is a *valid measure of the construct* rather than of this rater.

Usage: uv run python notebooks/llm_fewshot_calibration.py [gguf_path] [n_exemplar_calls]
"""

from __future__ import annotations

import json
import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from ecvol.features.llm.audit import _load_labels
from ecvol.features.llm.extract import (
    LlamaCppServerEngine,
    iter_section_inputs,
    yarn_rope_scaling,
)
from ecvol.features.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from ecvol.features.llm.reading import sample_train_calls

GGUF = sys.argv[1] if len(sys.argv) > 1 else r"D:\ecvol-data\models\Qwen2.5-7B-Instruct-Q4_K_M.gguf"
N_EXEMPLAR_CALLS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
SERVER = r"D:\ecvol-data\tools\llamacpp\llama-server.exe"
EXCERPT_CHARS = 900  # keeps ~10 exemplars well inside the context beside a full section

SHEET = "data/coverage/fincall_llm_labels_rater1.csv"
FIELDS = ("guidance_direction", "hedging_intensity", "surprise_mentions", "analyst_tone",
          "qa_evasiveness")  # fmt: skip


def build_exemplar_block(exemplars: list[tuple[str, str, str, dict]]) -> str:
    """Worked examples: a short excerpt plus the human rater's labels for that section."""
    parts = [
        "Worked examples of how an expert annotator applies this rubric. Match their scale use "
        "— note especially how sparingly they use the upper end.\n"
    ]
    for i, (_cid, section, text, labels) in enumerate(exemplars, 1):
        rated = ", ".join(f"{f}={labels[f]}" for f in FIELDS if labels.get(f) not in ("", "NA"))
        parts.append(
            f'EXAMPLE {i} ({section}):\n"{text[:EXCERPT_CHARS]}"\nExpert ratings: {rated}\n'
        )
    return "\n".join(parts)


def score(pred: pd.DataFrame, labels: pd.DataFrame, tag: str) -> dict:
    m = labels.merge(pred, on=["call_id", "section"], suffixes=("_h", "_m"))
    out = {}
    for f in FIELDS:
        h, mv = m[f + "_h"], m[f + "_m"]
        keep = h != "NA"
        h, mv = h[keep], mv[keep]
        if f == "guidance_direction":
            k = cohen_kappa_score(h.astype(str), mv.astype(str))
        elif f == "surprise_mentions":
            k = cohen_kappa_score((h.astype(int) > 0).astype(int), (mv.astype(int) > 0).astype(int))
        else:
            k = cohen_kappa_score(h.astype(int), mv.astype(int), weights="linear")
        out[f] = round(float(k), 3)
    print(f"{tag}: {json.dumps(out)}", flush=True)
    return out


def main() -> None:
    labels = _load_labels(SHEET)
    audit_calls = sample_train_calls("data", "fincall", 50, 0)
    exemplar_calls = set(audit_calls[:N_EXEMPLAR_CALLS])
    heldout_calls = set(audit_calls[N_EXEMPLAR_CALLS:])
    assert not (exemplar_calls & heldout_calls), "exemplar/held-out leak"

    sections = list(iter_section_inputs("data", "fincall", call_ids=list(audit_calls)))
    lab_by_key = {(r["call_id"], r["section"]): r for r in labels.to_dict("records")}

    exemplars = [
        (c, s, t, lab_by_key[(c, s)])
        for c, s, t in sections
        if c in exemplar_calls and (c, s) in lab_by_key
    ]
    heldout = [(c, s, t) for c, s, t in sections if c in heldout_calls]
    heldout_labels = labels[labels["call_id"].isin(heldout_calls)]
    print(
        f"{len(exemplars)} exemplar sections / {len(exemplar_calls)} calls; "
        f"{len(heldout)} scored sections / {len(heldout_calls)} calls",
        flush=True,
    )

    block = build_exemplar_block(exemplars)
    fewshot_system = SYSTEM_PROMPT + "\n\n" + block

    eng = LlamaCppServerEngine(
        "fewshot-probe",
        gguf_path=GGUF,
        server_bin=SERVER,
        max_model_len=65536,
        rope_scaling=yarn_rope_scaling(65536),
    )
    try:
        results = {}
        for tag, system in (("ZERO-SHOT v2", SYSTEM_PROMPT), ("FEW-SHOT    ", fewshot_system)):
            rows = []
            for i, (cid, sec, text) in enumerate(heldout, 1):
                feat = eng.generate(system, build_user_prompt(sec, text))
                rows.append({"call_id": cid, "section": sec, **feat})
                if i % 25 == 0:
                    print(f"  {tag}: {i}/{len(heldout)}", flush=True)
            results[tag] = score(pd.DataFrame(rows), heldout_labels, tag)
        a, b = results["ZERO-SHOT v2"], results["FEW-SHOT    "]
        print("\ndelta:", {k: round(b[k] - a[k], 3) for k in a})
    finally:
        eng.close()


if __name__ == "__main__":
    main()
