#!/usr/bin/env bash
# Generate per-candidate AU plots + cross-candidate comparison from Exported CSVs.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python analyze_aus.py --all "$@"
