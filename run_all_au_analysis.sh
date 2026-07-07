#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

PYTHON=.venv/bin/python
SCRIPT=analyze_aus.py

run_one() {
  local source="$1" year="$2" candidate="$3"
  $PYTHON "$SCRIPT" --source "$source" --year "$year" --candidate "$candidate"
}

run_one "Exported/Trump vs Clinton/Trump_clean_2016.mp4" 2016 Trump
run_one "Exported/Trump vs Clinton/Clinton_clean_2016.mp4" 2016 Clinton
run_one "Exported/Trump vs Harris/Trump_clean_2024.mp4" 2024 Trump
run_one "Exported/Trump vs Harris/Harris_clean_2024.mp4" 2024 Harris

echo "Done. Results in analysis/{2016,2024}/<Candidate>/"
