#!/usr/bin/env python3
"""Generate the Part 2 result files and evaluate them against Part 1's best model.

    python run_part2.py --queries-dir data --team NJMB \
        --dense-checkpoint models/dense/final  --dense-index indexes/dense \
        --sparse-checkpoint models/sparse/final --sparse-index indexes/sparse

Writes  <team>_{paper,paragraph}_{dense,sparse}_{study,test}.tsv  (6-column TREC, top 100)
into --out (default "Retrieval Results"), the mean search time per query to
search_times_part2.json, and, when Study qrels are found in --queries-dir, runs
eval_metrics.py on [Part 1 BM25, dense, sparse] for each level: every metric plus
paired t-tests and randomization tests between every pair of runs
(part2_{level}_summary.csv / _sig.csv / _per_query.csv).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from neural.retrievers import make_retriever

LEVELS = ("paper", "paragraph")
SPLITS = {"study": "Study.json", "test": "Test.json"}


def load_queries(path: Path) -> list[dict]:
    return [{"query_id": str(q["query_id"]), "query": q["query"]} for q in json.loads(path.read_text(encoding="utf-8"))]


def write_run(retriever, queries, level: str, path: Path, run_id: str, k: int, batch: int = 50) -> float:
    """Search every query (in small batches) and write the run; returns mean seconds per query."""
    elapsed = 0.0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for start in range(0, len(queries), batch):
            chunk = queries[start:start + batch]
            t0 = time.perf_counter()
            results = retriever.search_batch([q["query"] for q in chunk], level, k)
            elapsed += time.perf_counter() - t0
            for q, hits in zip(chunk, results):
                for h in hits:
                    f.write(f"{q['query_id']}\tQ0\t{h['id']}\t{h['rank']}\t{h['score']:.6f}\t{run_id}\n")
    return elapsed / max(len(queries), 1)


def single_query_latency(retriever, queries, level: str, n: int = 20) -> float:
    """Mean seconds for one query at a time (what a user of the live system waits)."""
    t0 = time.perf_counter()
    for q in queries[:n]:
        retriever.search(q["query"], level, 100)
    return (time.perf_counter() - t0) / max(min(n, len(queries)), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries-dir", required=True, help="folder with Study.json, Test.json and Study_*_qrel.tsv")
    ap.add_argument("--team", required=True)
    ap.add_argument("--out", default="Retrieval Results")
    ap.add_argument("--models", nargs="+", default=["dense", "sparse"], choices=["dense", "sparse"])
    ap.add_argument("--dense-checkpoint", default="models/dense/final")
    ap.add_argument("--dense-index", default="indexes/dense")
    ap.add_argument("--sparse-checkpoint", default="models/sparse/final")
    ap.add_argument("--sparse-index", default="indexes/sparse")
    ap.add_argument("--baseline", default="bm25", help="Part 1 model whose Study runs are compared against")
    ap.add_argument("--k", type=int, default=100)
    args = ap.parse_args()

    qdir, out = Path(args.queries_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    queries = {split: load_queries(qdir / f) for split, f in SPLITS.items()}
    times_path = Path("search_times_part2.json")
    timings = json.loads(times_path.read_text()) if times_path.exists() else {}

    for kind in args.models:
        ckpt, index = getattr(args, f"{kind}_checkpoint"), getattr(args, f"{kind}_index")
        print(f"Loading {kind} retriever ({ckpt}, {index})", flush=True)
        retriever = make_retriever(kind, ckpt, index, LEVELS)
        for split, qs in queries.items():
            for level in LEVELS:
                name = f"{args.team}_{level}_{kind}_{split}"
                avg = write_run(retriever, qs, level, out / f"{name}.tsv", f"{args.team}_{level}_{kind}", args.k)
                timings[name] = {"batched_s_per_query": avg}
                print(f"{name}.tsv  {len(qs)} queries  {avg * 1000:.1f} ms/query (batched)", flush=True)
        for level in LEVELS:
            timings[f"{args.team}_{level}_{kind}_single"] = {
                "s_per_query": single_query_latency(retriever, queries["study"], level),
                "device": retriever.device}
        del retriever
        times_path.write_text(json.dumps(timings, indent=2), encoding="utf-8")

    for level in LEVELS:
        qrel = qdir / f"Study_{level}_qrel.tsv"
        if not qrel.exists():
            continue
        runs = [out / f"{args.team}_{level}_{args.baseline}_study.tsv"] + [
            out / f"{args.team}_{level}_{m}_study.tsv" for m in ("dense", "sparse")]
        runs = [str(r) for r in runs if r.exists()]
        subprocess.run([sys.executable, str(Path(__file__).with_name("eval_metrics.py")), "--qrels", str(qrel), "--queries", str(qdir / "Study.json"),
                        "--runs", *runs, "--summary-out", f"part2_{level}_summary.csv",
                        "--sig-out", f"part2_{level}_sig.csv", "--per-query-out", f"part2_{level}_per_query.csv"],
                       check=True)


if __name__ == "__main__":
    main()
