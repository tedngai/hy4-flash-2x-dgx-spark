#!/usr/bin/env bash
# Stop both ranks of the Hy4-flash TP2 recipe.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE="${RECIPE:-llm/hy4-flash-nvfp4-vllm-tp2}"
WORKER="${WORKER:-192.0.2.11}"
export WORKER

exec "$ROOT/run-recipe.sh" "$RECIPE" --stop
