#!/usr/bin/env python3
"""Generate every Part 1 result file and (optionally) evaluate the Study runs.

    python run_experiments.py --data-dir DATA --queries-dir QUERIES --team TEAM \
        [--study-qrels-paper P.tsv --study-qrels-paragraph G.tsv]

Writes <team>_{paper,paragraph}_{tfidf,bm25}_{study,test}.tsv into --out
(default "Retrieval Results") and prints average search time per query.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pipeline

MODELS = ("tfidf", "bm25")
LEVELS = ("paper", "paragraph")
SPLITS = {"study": "Study.json", "test": "Test.json"}


def write_run(path: Path, queries, level: str, model: str, run_id: str, k: int) -> float:
    """Write one 6-column TREC run file; return mean seconds per query."""
    elapsed = 0.0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for q in queries:
            t0 = time.perf_counter()
            hits = pipeline.api.search(q["query"], model=model, level=level, k=k)
            elapsed += time.perf_counter() - t0
            for h in hits:
                f.write(f"{q['query_id']}\tQ0\t{h['id']}\t{h['rank']}\t{h['score']:.6f}\t{run_id}\n")
    return elapsed / max(len(queries), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="folder with the per-paper JSON files")
    ap.add_argument("--queries-dir", required=True, help="folder with Study.json and Test.json")
    ap.add_argument("--team", required=True, help="team name used in every file name and run_id")
    ap.add_argument("--out", default="Retrieval Results")
    ap.add_argument("--cache", default="index_cache.pkl")
    ap.add_argument("--rebuild", action="store_true", help="ignore the index cache")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--study-qrels-paper")
    ap.add_argument("--study-qrels-paragraph")
    args = ap.parse_args()

    pipeline.build_system(args.data_dir, args.cache, args.rebuild)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    timings = {}
    for split, fname in SPLITS.items():
        queries = pipeline.load_queries(Path(args.queries_dir) / fname)
        for level in LEVELS:
            for model in MODELS:
                name = f"{args.team}_{level}_{model}_{split}"
                avg = write_run(out / f"{name}.tsv", queries, level, model, f"{args.team}_{level}_{model}", args.k)
                timings[name] = avg
                print(f"{name}.tsv  {len(queries)} queries  avg {avg * 1000:.1f} ms/query")
    Path("search_times.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")

    qrels = {"paper": args.study_qrels_paper, "paragraph": args.study_qrels_paragraph}
    for level, qrel in qrels.items():
        if not qrel:
            continue
        runs = [str(out / f"{args.team}_{level}_{m}_study.tsv") for m in MODELS]
        subprocess.run(
            [sys.executable, "eval_metrics.py", "--qrels", qrel, "--queries", str(Path(args.queries_dir) / "Study.json"),
             "--runs", *runs, "--summary-out", f"{level}_summary.csv", "--sig-out", f"{level}_sig.csv",
             "--per-query-out", f"{level}_per_query.csv"],
            check=True,
        )


if __name__ == "__main__":
    main()
