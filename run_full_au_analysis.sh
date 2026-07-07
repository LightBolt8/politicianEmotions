#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

PYTHON=.venv/bin/python
SCRIPT=analyze_aus.py
LOG=/tmp/au_analysis_full.log

run_one() {
  local source="$1" year="$2" candidate="$3"
  shift 3
  echo "=== [$year/$candidate] $(date) ===" | tee -a "$LOG"
  $PYTHON "$SCRIPT" --source "$source" --year "$year" --candidate "$candidate" --full "$@" 2>&1 | tee -a "$LOG"
}

echo "Starting full-video AU analysis at $(date)" | tee "$LOG"

echo "--- 2016 debate ---" | tee -a "$LOG"
run_one "Exported/Trump vs Clinton/Trump_clean_2016.mp4" 2016 Trump
run_one "Exported/Trump vs Clinton/Clinton_clean_2016.mp4" 2016 Clinton

echo "--- 2024 debate ---" | tee -a "$LOG"
run_one "Exported/Trump vs Harris/Trump_clean_2024.mp4" 2024 Trump
run_one "Exported/Trump vs Harris/Harris_clean_2024.mp4" 2024 Harris \
  --openface-csv "OpenFaceResults/Trump vs Harris/Harris_clean_2024/Harris_clean_2024.csv"

echo "Done at $(date). Results in analysis/{2016,2024}/<Candidate>/" | tee -a "$LOG"
