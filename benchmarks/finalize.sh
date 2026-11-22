#!/usr/bin/env bash
# Rebuild the full matrix table from raw events and refresh the stable receipts.
set -euo pipefail
cd "$(dirname "$0")/.."

MATRIX="${1:-$(ls -d results/matrix-* | sort | tail -1)}"
python3 benchmarks/common_window.py "$MATRIX"
cp "$MATRIX/summary.json" results/hy4-flash-summary.json
cp "$MATRIX/TABLE.md" results/hy4-flash-table.md
cp results/launch.json "$MATRIX/launch.json"
python3 benchmarks/audit.py "$MATRIX"
echo "updated $MATRIX and results/hy4-flash-*"
