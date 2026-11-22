#!/usr/bin/env bash
# KV-pool x prefill-batch sweep for Hy4-flash TP2.
# Each cell restarts the pair, waits for the API, records the vLLM KV lines, runs a smoke test,
# then stops the pair.
set -euo pipefail
cd "$(dirname "$0")"
RECIPE=recipes/llm/hy4-flash-nvfp4-vllm-tp2.yaml
ORIG=$RECIPE.orig
NAME=hy4-flash-nvfp4-vllm-tp2
WORKER=${WORKER:-192.0.2.11}
OUT=${OUT:-$PWD/out/kv-batch}
mkdir -p "$OUT"
[ -f "$ORIG" ] || cp "$RECIPE" "$ORIG"

run_cell() {
  local kv=$1 batch=$2
  echo "===== KV=$kv BATCH=$batch ====="
  cp "$ORIG" "$RECIPE"
  if [ "$kv" = auto ]; then
    sed -i -E 's/ --kv-cache-memory-bytes [0-9]+G//' "$RECIPE"
  else
    sed -i -E "s/--kv-cache-memory-bytes [0-9]+G/--kv-cache-memory-bytes $kv/" "$RECIPE"
  fi
  sed -i -E "s/--max-num-batched-tokens [0-9]+/--max-num-batched-tokens $batch/" "$RECIPE"
  WORKER="$WORKER" ./run-recipe.sh llm/hy4-flash-nvfp4-vllm-tp2 --stop >/dev/null 2>&1 || true
  sleep 5
  WORKER="$WORKER" ./run-recipe.sh llm/hy4-flash-nvfp4-vllm-tp2
  local code=000
  for i in $(seq 1 90); do
    if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
      echo "head container exited"
      break
    fi
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:8000/v1/models || true)
    [ "$code" = 200 ] && break
    sleep 10
  done
  echo "health=$code"
  docker logs "$NAME" 2>&1 | grep -E "GPU KV cache size|Maximum concurrency|Available KV cache|kv_cache_memory_bytes|Initial free memory|gpu_memory_utilization|max_num_batched_tokens|Started server|ERROR|ValueError|OutOfMemory" | tail -40 | tee "$OUT/cell-$kv-$batch.log" || true
  free -g | head -2 | tee -a "$OUT/cell-$kv-$batch.log" || true
  if [ "$code" = 200 ]; then
    curl -s http://127.0.0.1:8000/v1/models > "$OUT/models-$kv-$batch.json" || true
    python3 "$HOME/data/hy4-flash-bench.py" --max-tokens 128 --effort no_think | tee "$OUT/smoke-$kv-$batch.txt" || true
  else
    docker logs "$NAME" > "$OUT/fail-$kv-$batch.log" 2>&1 || true
    ssh -o BatchMode=yes "$WORKER" "docker logs $NAME" > "$OUT/fail-worker-$kv-$batch.log" 2>&1 || true
  fi
  WORKER="$WORKER" ./run-recipe.sh llm/hy4-flash-nvfp4-vllm-tp2 --stop >/dev/null 2>&1 || true
  sleep 20
}

# 6G / 8192 is the published baseline already measured (578,345 tokens, 2.21x).
run_cell 10G 8192
run_cell 10G 4096
run_cell 12G 4096
run_cell 12G 2048
run_cell 14G 2048
run_cell 14G 1024
run_cell 16G 2048
run_cell 16G 1024
run_cell auto 8192
run_cell auto 2048
