#!/usr/bin/env python3
"""
eval_metrics.py - Evaluation for IR Project Part 1 (TF-IDF vs. BM25).

Inputs
    --qrels    Study qrels, TREC format, 4 columns, no header:  qid 0 docid relevance
               (Study_paper_qrel.tsv for paper runs, Study_paragraph_qrel.tsv for
               paragraph runs). relevance is 2 (fully relevant) or 1 (partially);
               any doc NOT listed in the qrels is treated as non-relevant (0).
    --queries  Study.json (list of {"query_id", "query", "label"}). Defines the 250
               Study queries that are evaluated and supplies the Easy/Medium/Hard
               labels for the per-difficulty breakdown. Optional but recommended.
    --runs     One or more result files, 6 columns:  qid Q0 docid rank score run_id
               Docids must match the qrels exactly: paper ids look like 2409.13695,
               paragraph ids like 2409.13695_8.

Metrics (per query, then averaged over queries)
    nDCG, nDCG@5
    MRR           lenient (rel >= 1) and strict (rel == 2)
    Precision@5   lenient and strict
    Precision@10  lenient and strict (spec gives no variant, so both are reported)

Significance tests between every pair of runs given (paired, per query):
    paired t-test and paired randomization test (both pure Python, no scipy).

Usage
    # paper level
    python eval_metrics.py --qrels Study_paper_qrel.tsv --queries Study.json \
        --runs teamName_paper_tfidf_study.tsv teamName_paper_bm25_study.tsv \
        --per-query-out paper_per_query.csv --summary-out paper_summary.csv \
        --sig-out paper_sig.csv

    # paragraph level
    python eval_metrics.py --qrels Study_paragraph_qrel.tsv --queries Study.json \
        --runs teamName_paragraph_tfidf_study.tsv teamName_paragraph_bm25_study.tsv \
        --per-query-out para_per_query.csv --summary-out para_summary.csv \
        --sig-out para_sig.csv

    Do not mix levels: paper runs go with the paper qrels, paragraph runs with the
    paragraph qrels.

Conventions (edit here if the instructor says otherwise)
    * Unjudged documents count as non-relevant.
    * Queries missing from a run score 0 on every metric.
    * Only queries in both the qrels and (if given) Study.json are evaluated.
    * nDCG uses linear gain (gain = rel); --exp-gain uses 2^rel - 1. The ideal
      ranking uses all judged docs for the query (as trec_eval does), truncated to
      the cutoff for nDCG@k.
    * Precision@k always divides by k, even if a run returns fewer than k results.
    * Queries with no relevant docs get nDCG = 0 (none occur in the Study qrels).
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict

METRICS = [
    "nDCG", "nDCG@5",
    "MRR_lenient", "MRR_strict",
    "P@5_lenient", "P@5_strict",
    "P@10_lenient", "P@10_strict",
]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_qrels_trec(path):
    """TREC qrels: qid 0 docid rel  (whitespace or tab separated)."""
    qrels = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 4:
                raise ValueError(f"{path}:{ln}: expected 4 columns, got {len(parts)}")
            qid, _, docid, rel = parts
            qrels[qid][docid] = int(float(rel))
    return dict(qrels)


def _first_key(d, names):
    for n in names:
        if n in d:
            return d[n]
    return None


def load_qrels(path):
    if path.lower().endswith(".json"):
        raise ValueError(
            f"{path}: --qrels expects a TREC qrels file (qid 0 docid relevance), not JSON. "
            "Study.json only holds the queries; pass it with --queries instead.")
    return load_qrels_trec(path)


def load_queries(path):
    """Study.json -> {qid(str): {"query": text, "label": Easy/Medium/Hard}}."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of query records")
    queries = {}
    for rec in data:
        qid = _first_key(rec, ["query_id", "qid", "id"])
        if qid is None:
            raise ValueError(f"{path}: record without query_id: {list(rec)}")
        queries[str(qid)] = {"query": rec.get("query", ""), "label": rec.get("label", "Unlabeled")}
    return queries


def load_run(path):
    """6-column TREC run: qid Q0 docid rank score run_id. Returns {qid: [docids in rank order]}."""
    rows = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 6:
                raise ValueError(f"{path}:{ln}: expected 6 columns (qid Q0 docid rank score run_id), got {len(parts)}")
            qid, q0, docid, rank, score, _ = parts
            if q0 != "Q0":
                print(f"WARNING {path}:{ln}: second column is '{q0}', expected 'Q0'", file=sys.stderr)
            rows[qid].append((int(rank), docid))
    run = {}
    for qid, items in rows.items():
        items.sort(key=lambda x: x[0])
        docs = [d for _, d in items]
        if len(set(docs)) != len(docs):
            print(f"WARNING {path}: query {qid} has duplicate docids", file=sys.stderr)
        if len(docs) > 100:
            print(f"WARNING {path}: query {qid} has {len(docs)} results (spec says top-100)", file=sys.stderr)
        run[qid] = docs
    return run


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _gain(rel, exp_gain):
    return (2 ** rel - 1) if exp_gain else rel


def _dcg(rels, exp_gain):
    return sum(_gain(r, exp_gain) / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg(ranked_rels, all_rels, k=None, exp_gain=False):
    ideal = sorted((r for r in all_rels if r > 0), reverse=True)
    if k is not None:
        ranked_rels, ideal = ranked_rels[:k], ideal[:k]
    idcg = _dcg(ideal, exp_gain)
    return _dcg(ranked_rels, exp_gain) / idcg if idcg > 0 else 0.0


def reciprocal_rank(ranked_rels, threshold):
    for i, r in enumerate(ranked_rels):
        if r >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(ranked_rels, k, threshold):
    """Denominator is always k (missing results count as non-relevant)."""
    return sum(1 for r in ranked_rels[:k] if r >= threshold) / k


def evaluate_query(ranking, judgments, exp_gain=False):
    rels = [judgments.get(d, 0) for d in ranking]
    all_rels = list(judgments.values())
    return {
        "nDCG": ndcg(rels, all_rels, None, exp_gain),
        "nDCG@5": ndcg(rels, all_rels, 5, exp_gain),
        "MRR_lenient": reciprocal_rank(rels, 1),
        "MRR_strict": reciprocal_rank(rels, 2),
        "P@5_lenient": precision_at_k(rels, 5, 1),
        "P@5_strict": precision_at_k(rels, 5, 2),
        "P@10_lenient": precision_at_k(rels, 10, 1),
        "P@10_strict": precision_at_k(rels, 10, 2),
    }


def evaluate_run(run, qrels, exp_gain=False):
    """Returns {qid: {metric: value}} for every qid in qrels."""
    out = {}
    for qid, judgments in qrels.items():
        out[qid] = evaluate_query(run.get(qid, []), judgments, exp_gain)
    return out


def mean_metrics(per_query):
    n = len(per_query)
    return {m: sum(q[m] for q in per_query.values()) / n for m in METRICS} if n else {m: 0.0 for m in METRICS}


def mean_by_label(per_query, queries):
    """{label: (n_queries, {metric: mean})} grouped by Study.json difficulty label."""
    groups = defaultdict(dict)
    for qid, vals in per_query.items():
        groups[queries.get(qid, {}).get("label", "Unlabeled")][qid] = vals
    order = {"Easy": 0, "Medium": 1, "Hard": 2}
    return {lab: (len(g), mean_metrics(g))
            for lab, g in sorted(groups.items(), key=lambda kv: order.get(kv[0], 9))}


# --------------------------------------------------------------------------
# Significance tests (pure Python)
# --------------------------------------------------------------------------
def _betacf(a, b, x, max_iter=300, eps=3e-14):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a, b, x):
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_bt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    bt = math.exp(ln_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def paired_ttest(diffs):
    """Two-sided paired t-test on per-query differences. Returns (t, p)."""
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan")
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    if var == 0:
        return (0.0, 1.0) if mean == 0 else (math.copysign(float("inf"), mean), 0.0)
    t = mean / math.sqrt(var / n)
    df = n - 1
    p = _betainc(df / 2.0, 0.5, df / (df + t * t))
    return t, p


def randomization_test(diffs, n_perm=10000, seed=13):
    """Two-sided paired randomization (sign-flip) test on mean difference."""
    n = len(diffs)
    if n == 0:
        return float("nan")
    observed = abs(sum(diffs) / n)
    if observed == 0:
        return 1.0
    rng = random.Random(seed)
    count = 0
    for _ in range(n_perm):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(s / n) >= observed - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


def compare_runs(name_a, pq_a, name_b, pq_b, n_perm):
    qids = sorted(set(pq_a) & set(pq_b))
    rows = []
    for m in METRICS:
        a = [pq_a[q][m] for q in qids]
        b = [pq_b[q][m] for q in qids]
        diffs = [x - y for x, y in zip(a, b)]
        t, p_t = paired_ttest(diffs)
        rows.append({
            "metric": m,
            "mean_A": sum(a) / len(a) if a else float("nan"),
            "mean_B": sum(b) / len(b) if b else float("nan"),
            "diff(A-B)": sum(diffs) / len(diffs) if diffs else float("nan"),
            "t": t,
            "p_ttest": p_t,
            "p_randomization": randomization_test(diffs, n_perm),
            "wins_A": sum(1 for d in diffs if d > 0),
            "wins_B": sum(1 for d in diffs if d < 0),
            "ties": sum(1 for d in diffs if d == 0),
            "n": len(diffs),
        })
    return rows


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def run_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def fmt_p(p):
    if p is None:
        return "n/a"
    if p != p:
        return "nan"
    return f"{p:.4f}" + ("*" if p < 0.05 else "")


def main():
    ap = argparse.ArgumentParser(description="IR Project Part 1 evaluation")
    ap.add_argument("--qrels", required=True, help="Study qrels (qid 0 docid rel), paper or paragraph level")
    ap.add_argument("--runs", required=True, nargs="+", help="one or more 6-column TREC run files")
    ap.add_argument("--queries", help="Study.json (queries + Easy/Medium/Hard labels)")
    ap.add_argument("--exp-gain", action="store_true", help="use 2^rel-1 gain for nDCG (default: linear)")
    ap.add_argument("--per-query-out", help="write per-query metrics to this CSV")
    ap.add_argument("--summary-out", help="write mean metrics to this CSV")
    ap.add_argument("--sig-out", help="write significance test results to this CSV")
    ap.add_argument("--perms", type=int, default=10000, help="randomization test permutations")
    args = ap.parse_args()

    qrels = load_qrels(args.qrels)
    print(f"Loaded qrels for {len(qrels)} queries from {args.qrels}")

    queries = {}
    if args.queries:
        queries = load_queries(args.queries)
        no_qrels = [q for q in queries if q not in qrels]
        extra = [q for q in qrels if q not in queries]
        print(f"Loaded {len(queries)} queries from {args.queries}")
        if no_qrels:
            print(f"WARNING: {len(no_qrels)} Study queries have no qrels and are skipped: {no_qrels[:5]}...",
                  file=sys.stderr)
        if extra:
            print(f"WARNING: {len(extra)} qrels queries are not in Study.json and are ignored: {extra[:5]}...",
                  file=sys.stderr)
        qrels = {q: qrels[q] for q in queries if q in qrels}
        if not qrels:
            sys.exit("ERROR: no overlap between qrels query ids and Study.json query_ids")

    results = {}
    for path in args.runs:
        run = load_run(path)
        missing = [q for q in qrels if q not in run]
        if missing:
            print(f"WARNING {path}: {len(missing)} qrels queries missing from run (scored 0): {missing[:5]}...",
                  file=sys.stderr)
        judged = {d for j in qrels.values() for d in j}
        returned = {d for q in qrels if q in run for d in run[q]}
        if returned and not (judged & returned):
            print(f"WARNING {path}: none of the returned docids appear in the qrels. "
                  f"Docid format mismatch (or paper/paragraph level mixed up)? "
                  f"run example: {sorted(returned)[:2]}, qrels example: {sorted(judged)[:2]}",
                  file=sys.stderr)
        results[run_name(path)] = evaluate_run(run, qrels, args.exp_gain)

    # Summary table
    width = max(len(n) for n in results) + 2
    print("\n=== Mean metrics ===")
    print("run".ljust(width) + "".join(m.rjust(14) for m in METRICS))
    summary_rows = []
    for name, pq in results.items():
        mm = mean_metrics(pq)
        print(name.ljust(width) + "".join(f"{mm[m]:14.4f}" for m in METRICS))
        summary_rows.append({"run": name, **mm})

    # Per-difficulty breakdown (needs --queries)
    if queries:
        print("\n=== Mean metrics by difficulty label ===")
        for name, pq in results.items():
            print(f"\n{name}")
            print("label (n)".ljust(16) + "".join(m.rjust(14) for m in METRICS))
            for lab, (n, mm) in mean_by_label(pq, queries).items():
                print(f"{lab} ({n})".ljust(16) + "".join(f"{mm[m]:14.4f}" for m in METRICS))

    # Significance
    sig_rows = []
    names = list(results)
    if len(names) >= 2:
        print("\n=== Significance tests (two-sided; * = p < 0.05) ===")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                print(f"\nA = {a}\nB = {b}")
                print(f"{'metric':<14}{'mean_A':>9}{'mean_B':>9}{'diff':>9}"
                      f"{'t-test p':>12}{'randomiz p':>13}{'A/B/tie':>12}")
                for r in compare_runs(a, results[a], b, results[b], args.perms):
                    print(f"{r['metric']:<14}{r['mean_A']:9.4f}{r['mean_B']:9.4f}{r['diff(A-B)']:9.4f}"
                          f"{fmt_p(r['p_ttest']):>12}{fmt_p(r['p_randomization']):>13}"
                          f"{str(r['wins_A']) + '/' + str(r['wins_B']) + '/' + str(r['ties']):>12}")
                    sig_rows.append({"run_A": a, "run_B": b, **r})

    # CSVs
    if args.per_query_out:
        with open(args.per_query_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["run", "qid", "label"] + METRICS)
            for name, pq in results.items():
                for qid in sorted(pq):
                    lab = queries.get(qid, {}).get("label", "")
                    w.writerow([name, qid, lab] + [f"{pq[qid][m]:.6f}" for m in METRICS])
        print(f"\nWrote per-query metrics to {args.per_query_out}")
    if args.summary_out:
        with open(args.summary_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["run"] + METRICS)
            w.writeheader()
            w.writerows(summary_rows)
        print(f"Wrote summary to {args.summary_out}")
    if args.sig_out and sig_rows:
        with open(args.sig_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sig_rows[0].keys()))
            w.writeheader()
            w.writerows(sig_rows)
        print(f"Wrote significance results to {args.sig_out}")


if __name__ == "__main__":
    main()