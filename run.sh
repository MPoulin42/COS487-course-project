#!/usr/bin/env bash
# Usage:
#   bash run.sh part2       DATA_DIR QUERIES_DIR TEAM     # Part 2: fine-tune, encode, retrieve, evaluate (GPU)
#   bash run.sh experiments DATA_DIR QUERIES_DIR TEAM [STUDY_PAPER_QREL STUDY_PARAGRAPH_QREL]   # Part 1 runs
#   bash run.sh app         DATA_DIR [QUERIES_DIR TEAM]   # search interface (adds neural models if indexes exist)
#
# DATA_DIR    = folder of per-paper JSON files (on the CS servers: /mnt/netstore1_datasets/IR26/...)
# QUERIES_DIR = folder with Train.json, Study.json, Test.json and the Train_/Study_ qrel files
set -e
mode=$1; data=$2
case "$mode" in
  part2)
    queries=$3; team=$4
    # 1. cleaned corpus cache  2. training triples (Train split only)  3. fine-tuning
    # Finished steps are skipped on a re-run (delete their output to redo them); encoding resumes per shard.
    python -m neural.corpus --data-dir "$data"
    [ -f neural_cache/triples/train.jsonl ] || python -m neural.build_training_data --queries-dir "$queries"
    [ -d models/dense/final ]  || python -m neural.train --model dense  --out models/dense
    [ -d models/sparse/final ] || python -m neural.train --model sparse --out models/sparse
    # 4. encode the collection (resumable)  5. result files + evaluation vs Part 1 BM25
    python -m neural.encode --model dense  --checkpoint models/dense/final  --index indexes/dense
    python -m neural.encode --model sparse --checkpoint models/sparse/final --index indexes/sparse
    # Part 1 BM25 Study runs (the significance-test baseline) ship in "Retrieval Results.zip"
    python -c "import zipfile, os; z = zipfile.ZipFile('Retrieval Results.zip'); [z.extract(n) for n in z.namelist() if n.endswith('_study.tsv') and not os.path.exists(n)]"
    python run_part2.py --queries-dir "$queries" --team "$team" ;;
  experiments)
    args=(--data-dir "$data" --queries-dir "$3" --team "$4")
    [ -n "$5" ] && args+=(--study-qrels-paper "$5")
    [ -n "$6" ] && args+=(--study-qrels-paragraph "$6")
    python run_experiments.py "${args[@]}" ;;
  app)
    args=(--data-dir "$data")
    [ -n "$3" ] && args+=(--queries-dir "$3" --team "$4")
    python app.py "${args[@]}" ;;
  *) sed -n 2,9p "$0"; exit 1 ;;
esac
