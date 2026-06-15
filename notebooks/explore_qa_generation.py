"""Exploration: open-weight QA generation on FinCall earnings calls (feasibility scout for TX1).

Question this answers: does an *open-weight* Qwen model produce usable volatility-relevant
Q&A pairs from FinCall transcripts, the way the prior team got from proprietary GPT-5?
(See ingest/ingest.md; DECISIONS.md 2026-06-14; TASKS.md TX1.)

Scope: FEASIBILITY ONLY — a few calls, eyeball the output. Not corpus scale, not the TX1
pipeline. No pipeline code, no committed deps, writes nothing to data/.

Env notes (2026-06-14): run with the interpreter that has CUDA torch + transformers
(global py3.14 here, NOT the project .venv which got a CPU torch from the scratch install).
Model: Qwen2.5-3B-Instruct in bf16 (~6 GB, fits the 16 GB RTX 5060 Ti without quantization)
used as a proxy for the DESIGN-pinned Qwen2.5-7B-Instruct (7B needs 4-bit/bitsandbytes, which
is fragile on this bleeding-edge Blackwell/torch stack — deferred to TX1 proper).
"""

import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ROOT / "data" / "identity" / "fincall_identity.csv"
RAW = ROOT / "data" / "raw" / "fincall"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
N_CALLS = 3
MAX_CHUNKS_PER_CALL = 3  # keep the scout cheap
CHUNK_TURNS = 8

# Open-weight QA-generation prompt, adapted from the prior team's "general/conceptual"
# prompt (ingest figures image2/image37) — generalized questions, evidence-grounded answers,
# source line numbers, strict JSON. No proprietary model.
SYSTEM = (
    "You are a financial analysis assistant. From an Earnings Conference Call (ECC) excerpt, "
    "extract concise Q&A pairs that capture statements relevant to potential stock volatility.\n"
    "Question rules: generalized and conceptual (a broad theme); exclude company/product/"
    "technology names; short and clear; avoid yes/no questions.\n"
    "Answer rules: specific, evidence-based, drawn verbatim from the excerpt; may include the "
    "entity names/numbers present in the source.\n"
    "Skip greetings, disclaimers, and operator/procedural text.\n"
    "Return STRICT JSON: a list of objects with keys "
    '"question", "answer", "category", "source_lines" ([start,end], 1-based). '
    "Return [] if nothing volatility-relevant is present."
)


def load_calls():
    """Pick the first N resolved earnings calls (with a ticker) from the identity table."""
    import csv

    rows = []
    with open(IDENTITY, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["call_type"] == "earnings" and r["ticker"]:
                rows.append(r)
            if len(rows) >= N_CALLS:
                break
    transcripts = {}
    for year in {r["year"] for r in rows}:
        with open(RAW / f"transcripts_{year}.json", encoding="utf-8") as f:
            transcripts[year] = json.load(f)
    out = []
    for r in rows:
        rec = transcripts[r["year"]].get(r["call_id"])
        if rec:
            out.append((r, rec["input"]))
    return out


def to_turns(text):
    """Approximate speaker-turn segmentation for FinCall (weak delineation: mostly newline
    paragraphs + inline 'Executives:'/'Operator:' markers). Real per-turn structure needs a
    better source — a finding for T3.1. Returns a list of (line_no, segment)."""
    # split on inline speaker markers while keeping paragraph (newline) boundaries
    parts = re.split(r"(?:\n+|(?<=[.?!])\s*(?=(?:Executives|Operator|Analyst|Q|A)\b:))", text)
    return [(i + 1, p.strip()) for i, p in enumerate(parts) if p.strip()]


def chunks(turns, size):
    for i in range(0, len(turns), size):
        block = turns[i : i + size]
        body = "\n".join(f"[{ln}] {seg}" for ln, seg in block)
        yield block[0][0], block[-1][0], body


def main():
    print(f"Loading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print(f"Loaded. VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB\n")

    for rec, text in load_calls():
        print("=" * 90)
        print(f"{rec['ticker']} {rec['company']} | {rec['date']} | call_id={rec['call_id']}")
        turns = to_turns(text)
        print(f"{len(turns)} segments (~speaker turns)\n")
        for ci, (lo, hi, body) in enumerate(chunks(turns, CHUNK_TURNS)):
            if ci >= MAX_CHUNKS_PER_CALL:
                break
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"ECC excerpt (lines numbered):\n{body}"},
            ]
            inputs = tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
            ).to(model.device)
            n_in = inputs["input_ids"].shape[1]
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            out = tok.decode(gen[0, n_in:], skip_special_tokens=True)
            print(f"--- chunk {ci} (lines {lo}-{hi}) ---")
            try:
                qas = json.loads(out[out.index("[") : out.rindex("]") + 1])
                for qa in qas:
                    print(f"  Q: {qa.get('question')}")
                    print(f"  A: {qa.get('answer')}")
                    print(f"  cat={qa.get('category')} src={qa.get('source_lines')}\n")
                if not qas:
                    print("  (empty — no volatility-relevant content)\n")
            except (ValueError, json.JSONDecodeError):
                print("  [unparseable JSON] raw:\n  " + out[:400].replace("\n", "\n  ") + "\n")


if __name__ == "__main__":
    main()
