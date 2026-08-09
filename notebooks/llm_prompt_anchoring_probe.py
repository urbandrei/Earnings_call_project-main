"""EXPLORATORY (writes nothing to data/): would better rubric anchoring rescue the kappa gate?

Scores the frozen v2 prompt against an anchored variant on the same 97 audit sections, in
memory. Nothing here touches the canonical parquet or PROMPT_VERSION — this only tells us
whether "revisit the prompt" is a promising direction for the user to choose.
"""

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
SERVER = r"D:\ecvol-data\tools\llamacpp\llama-server.exe"

# The v2 system prompt plus explicit decision rules aimed at the three observed failures:
# hedging carries no signal, analyst_tone is near-constant at 2, guidance defaults to 'none'.
ANCHORED_SYSTEM = (
    SYSTEM_PROMPT + " Additional calibration rules. "
    "HEDGING: count actual hedge phrases ('we believe', 'approximately', 'we think', 'should', "
    "'may', 'hard to say', 'roughly'). 0 = essentially none; 1 = a handful in a long section; "
    "2 = hedges in most answers; 3 = nearly every claim qualified; 4 = pervasive. Long sections "
    "are NOT automatically more hedged - judge density, not raw count. "
    "GUIDANCE: if management states ANY forward-looking outlook (revenue, margin, EPS, volume) "
    "for a future period, you MUST classify it as raise/maintain/lower relative to the prior "
    "outlook. Use 'none' ONLY when no forward-looking outlook appears anywhere in the section. "
    "ANALYST TONE: use the full 0-4 range and differentiate. 2 is reserved for genuinely neutral "
    "question sets; use 1 when analysts press on weakness, 3 when they are complimentary. "
    "SURPRISE: count explicit surprise language ('surprised', 'unexpected', 'better/worse than "
    "expected', 'ahead of/behind plan'), including analyst phrasing."
)

labels = _load_labels("data/coverage/fincall_llm_labels_rater1.csv")
call_ids = set(sample_train_calls("data", "fincall", 50, 0))
sections = [
    (c, s, t) for c, s, t in iter_section_inputs("data", "fincall", call_ids=list(call_ids))
]
print(f"scoring {len(sections)} sections; model={GGUF.rsplit(chr(92), 1)[-1]}")

eng = LlamaCppServerEngine(
    "probe",
    gguf_path=GGUF,
    server_bin=SERVER,
    max_model_len=65536,
    rope_scaling=yarn_rope_scaling(65536),
)


def run(system, tag):
    rows = []
    for i, (cid, sec, text) in enumerate(sections, 1):
        feat = eng.generate(system, build_user_prompt(sec, text))
        rows.append({"call_id": cid, "section": sec, **feat})
        if i % 25 == 0:
            print(f"  {tag}: {i}/{len(sections)}", flush=True)
    return pd.DataFrame(rows)


def score(pred, tag):
    m = labels.merge(pred, on=["call_id", "section"], suffixes=("_h", "_m"))
    out = {}
    for f, kind in (
        ("guidance_direction", "cat"),
        ("hedging_intensity", "ord"),
        ("surprise_mentions", "count"),
        ("analyst_tone", "ord"),
        ("qa_evasiveness", "ord"),
    ):
        h, mv = m[f + "_h"], m[f + "_m"]
        keep = h != "NA"
        h, mv = h[keep], mv[keep]
        if kind == "cat":
            k = cohen_kappa_score(h.astype(str), mv.astype(str))
        elif kind == "ord":
            k = cohen_kappa_score(h.astype(int), mv.astype(int), weights="linear")
        else:
            k = cohen_kappa_score((h.astype(int) > 0).astype(int), (mv.astype(int) > 0).astype(int))
        out[f] = round(float(k), 3)
    print(f"{tag}: {json.dumps(out)}")
    return out


base = score(run(SYSTEM_PROMPT, "frozen-v2"), "FROZEN v2      ")
anch = score(run(ANCHORED_SYSTEM, "anchored"), "ANCHORED variant")
print("\ndelta:", {k: round(anch[k] - base[k], 3) for k in base})
eng.close()
