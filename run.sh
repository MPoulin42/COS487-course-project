#!/usr/bin/env bash
# Usage:
#   bash run.sh experiments DATA_DIR QUERIES_DIR TEAM [STUDY_PAPER_QREL STUDY_PARAGRAPH_QREL]
#   bash run.sh app         DATA_DIR [QUERIES_DIR TEAM]
set -e
mode=$1; data=$2
case "$mode" in
  experiments)
    args=(--data-dir "$data" --queries-dir "$3" --team "$4")
    [ -n "$5" ] && args+=(--study-qrels-paper "$5")
    [ -n "$6" ] && args+=(--study-qrels-paragraph "$6")
    python run_experiments.py "${args[@]}" ;;
  app)
    args=(--data-dir "$data")
    [ -n "$3" ] && args+=(--queries-dir "$3" --team "$4")
    python app.py "${args[@]}" ;;
  *) sed -n 2,4p "$0"; exit 1 ;;
esac
