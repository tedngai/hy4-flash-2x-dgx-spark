#!/usr/bin/env bash
# Start Hy4-flash NVFP4 on two DGX Sparks with TP2 and expert parallel.
#
#   ./start.sh --setup      first run: build the image, copy it to the worker,
#                           download the checkpoint on both Sparks, then start
#   ./start.sh              start in the background (the launcher is detached)
#   WORKER=<ssh-host> ./start.sh
#
# The first Spark (the head) serves the API on 127.0.0.1:8000. The worker is
# reached over ssh with key login and must have the same user and paths.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE="${RECIPE:-llm/hy4-flash-nvfp4-vllm-tp2}"
WORKER="${WORKER:-192.0.2.11}"
export WORKER

if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER" true 2>/dev/null; then
  echo "cannot ssh to the worker '$WORKER'; set WORKER=<ssh-host of the second Spark>" >&2
  exit 1
fi

if curl -sf --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
  if [[ "${FORCE:-0}" != "1" ]]; then
    echo "A server is already answering on port 8000." >&2
    echo "Stop it with ./stop.sh first, or set FORCE=1 to let the launcher decide." >&2
    exit 1
  fi
fi

SETUP=0
if [[ "${1:-}" == "--setup" ]]; then
  SETUP=1
  shift
fi

echo "recipe: $RECIPE"
echo "head: $(hostname)  worker: $WORKER"
if [[ "$SETUP" == "1" ]]; then
  exec "$ROOT/run-recipe.sh" "$RECIPE" --setup
fi
exec "$ROOT/run-recipe.sh" "$RECIPE"
