#!/usr/bin/env bash
# T6.2 local corpus extraction — unattended supervisor (DECISIONS 2026-08-09).
#
# Runs `ecvol featurize llm` over FinCall then MAEC on the llama.cpp engine, restarting after a
# crash and stopping cleanly at a wall-clock deadline. The per-model parquet is the resume store,
# so a restart re-reads it and skips already-done (call_id, section) rows — no work is repeated
# and none is lost.
#
# Usage: bash cloud/local/run_corpus.sh [HH:MM]   (default stop 19:30 local)

set -uo pipefail
cd "$(dirname "$0")/../.."

STOP_HHMM="${1:-19:30}"
STOP_EPOCH=$(date -d "today $STOP_HHMM" +%s)
[ "$(date +%s)" -ge "$STOP_EPOCH" ] && STOP_EPOCH=$(date -d "tomorrow $STOP_HHMM" +%s)

MODEL_ID="bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M"
REVISION="8911e8a47f92bac19d6f5c64a2e2095bd2f7d031"
GGUF="D:/ecvol-data/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
SERVER="D:/ecvol-data/tools/llamacpp/llama-server.exe"
PARQUET_DIR="data"
MAX_ATTEMPTS=200
LOG="artifacts/llm_corpus_run.log"
mkdir -p artifacts

say() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# Row count IS the progress metric: if a restart adds no rows, retrying can only spin.
rows() {
  uv run python -c "
import sys, pathlib, pandas as pd
p = pathlib.Path('$PARQUET_DIR/$1/llm_features__bartowski__Qwen2.5-7B-Instruct-GGUF_Q4_K_M.parquet')
print(len(pd.read_parquet(p)) if p.exists() else 0)
" 2>/dev/null || echo 0
}

# A crashed run can leave the server holding the port, which makes every retry fail to bind.
reap_server() { taskkill //F //IM llama-server.exe >/dev/null 2>&1 || true; }

run_dataset() {
  local ds="$1" stalled=0 before after
  say "=== dataset $ds — start (stop at $STOP_HHMM) ==="
  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    if [ "$(date +%s)" -ge "$STOP_EPOCH" ]; then
      say "$ds: deadline reached — stopping"
      return 0
    fi
    before=$(rows "$ds")
    reap_server
    uv run ecvol featurize llm --dataset "$ds" \
      --model-id "$MODEL_ID" --revision "$REVISION" \
      --engine llamacpp --max-model-len 65536 --yarn \
      --gguf-path "$GGUF" --server-bin "$SERVER" \
      --stop-at "$STOP_HHMM" --root "$PARQUET_DIR" >>"$LOG" 2>&1
    local rc=$?
    after=$(rows "$ds")
    if [ "$rc" -eq 0 ]; then
      say "$ds: clean exit (rows $before -> $after)"
      return 0
    fi
    say "$ds: attempt $attempt failed rc=$rc (rows $before -> $after)"
    # Distinguish "crashed but made progress" (retry is useful) from "crashed making none"
    # (retrying just re-hits the same bad section) — bail rather than spin all night.
    if [ "$after" -le "$before" ]; then
      stalled=$((stalled + 1))
      if [ "$stalled" -ge 3 ]; then
        say "$ds: ABORT — 3 consecutive attempts made zero progress; needs a human/agent look"
        return 1
      fi
    else
      stalled=0
    fi
    sleep 30
  done
  say "$ds: ABORT — exhausted $MAX_ATTEMPTS attempts"
  return 1
}

say "supervisor start; deadline $(date -d "@$STOP_EPOCH" +'%Y-%m-%d %H:%M:%S')"
run_dataset fincall; fincall_rc=$?
say "fincall done rc=$fincall_rc, rows=$(rows fincall)"
run_dataset maec; maec_rc=$?
say "maec done rc=$maec_rc, rows=$(rows maec)"
reap_server
say "supervisor exit (fincall rc=$fincall_rc, maec rc=$maec_rc)"
