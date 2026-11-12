#!/usr/bin/env python3
"""Build paper/results_tables.tex from the evaluation CSVs and search_times.json.

Run from the repository root after run_experiments.py (with Study qrels):
    python paper/make_tables.py
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "results_tables.tex"
METRICS = ["nDCG", "nDCG@5", "MRR_lenient", "MRR_strict", "P@5_lenient", "P@5_strict", "P@10_lenient", "P@10_strict"]
HEAD = (r"Model & nDCG & nDCG@5 & MRR$_{\ge1}$ & MRR$_{2}$ & P@5$_{\ge1}$ & P@5$_{2}$ & P@10$_{\ge1}$ & P@10$_{2}$ \\")
SIG_NAMES = {"nDCG": "nDCG", "nDCG@5": "nDCG@5", "MRR_lenient": r"MRR$_{\ge1}$", "MRR_strict": r"MRR$_{2}$",
             "P@5_lenient": r"P@5$_{\ge1}$", "P@5_strict": r"P@5$_{2}$", "P@10_lenient": r"P@10$_{\ge1}$",
             "P@10_strict": r"P@10$_{2}$"}


def model_of(run):
    return "BM25" if "bm25" in run else "TF-IDF"


def read(name):
    with open(ROOT / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summary_table(level):
    rows = {model_of(r["run"]): {m: float(r[m]) for m in METRICS} for r in read(f"{level}_summary.csv")}
    best = {m: max(v[m] for v in rows.values()) for m in METRICS}
    lines = []
    for model in ("TF-IDF", "BM25"):
        cells = [(r"\textbf{%.4f}" if rows[model][m] == best[m] and len(set(v[m] for v in rows.values())) > 1
                  else "%.4f") % rows[model][m] for m in METRICS]
        lines.append(model + " & " + " & ".join(cells) + r" \\")
    return (r"\begin{table*}[t]\caption{%s-level results on the 250 Study queries (mean over queries; "
            r"$\ge1$ = Relevant or Partially Relevant counted as relevant, 2 = Relevant only). Best value in bold.}"
            r"\label{tab:%s-summary}" "\n"
            r"\small\begin{tabular}{lrrrrrrrr}\toprule" "\n" + HEAD + r"\midrule" "\n" + "\n".join(lines) + "\n"
            r"\bottomrule\end{tabular}\end{table*}" "\n") % (level.capitalize(), level)


def fmt_p(p):
    p = float(p)
    s = "$<$0.001" if p < 0.001 else "%.3f" % p
    return s + ("$^*$" if p < 0.05 else "")


def fmt_diff(d):
    return "0.0000" if abs(d) < 5e-5 else ("$+$%.4f" % d if d > 0 else "$-$%.4f" % -d)


def sig_table():
    lines = []
    for level in ("paper", "paragraph"):
        for r in read(f"{level}_sig.csv"):
            # CSV diff is run_A - run_B; report BM25 - TF-IDF
            diff = float(r["diff(A-B)"]) * (-1 if model_of(r["run_A"]) == "TF-IDF" else 1)
            lines.append(f"{level} & {SIG_NAMES[r['metric']]} & {fmt_diff(diff)} & {fmt_p(r['p_ttest'])} & "
                         f"{fmt_p(r['p_randomization'])} \\\\")
        lines.append(r"\midrule")
    lines.pop()
    return (r"\begin{table}[t]\caption{Paired significance tests, BM25 vs.\ TF-IDF on the 250 Study queries "
            r"(two-sided; $^*$ = $p<0.05$; randomization test with 10{,}000 sign flips). Diff = BM25 $-$ TF-IDF.}"
            r"\label{tab:sig}" "\n"
            r"\small\begin{tabular}{llrrr}\toprule Level & Metric & Diff & $p_{t}$ & $p_{rand}$ \\\midrule" "\n"
            + "\n".join(lines) + "\n" r"\bottomrule\end{tabular}\end{table}" "\n")


def difficulty_table():
    lines = []
    for level in ("paper", "paragraph"):
        groups = defaultdict(lambda: defaultdict(list))
        for r in read(f"{level}_per_query.csv"):
            groups[(model_of(r["run"]), r["label"])]["n"].append(1)
            for m in ("nDCG@5", "MRR_lenient", "P@5_lenient"):
                groups[(model_of(r["run"]), r["label"])][m].append(float(r[m]))
        for model in ("TF-IDF", "BM25"):
            for lab in ("Easy", "Medium", "Hard"):
                g = groups[(model, lab)]
                vals = " & ".join("%.3f" % (sum(g[m]) / len(g[m])) for m in ("nDCG@5", "MRR_lenient", "P@5_lenient"))
                lines.append(f"{level.capitalize()} / {model} & {lab} ({len(g['n'])}) & {vals} \\\\")
        lines.append(r"\midrule")
    lines.pop()
    return (r"\begin{table}[t]\caption{Mean metrics by query difficulty (Study queries; lenient variants; "
            r"number of queries in parentheses).}\label{tab:difficulty}" "\n"
            r"\small\begin{tabular}{llrrr}\toprule Level / Model & Difficulty & nDCG@5 & MRR$_{\ge1}$ & P@5$_{\ge1}$ \\"
            r"\midrule" "\n" + "\n".join(lines) + "\n" r"\bottomrule\end{tabular}\end{table}" "\n")


def time_table():
    t = json.loads((ROOT / "search_times.json").read_text(encoding="utf-8"))
    ms = lambda lvl, m, split: "%.1f" % (1000 * t[f"NJMB_{lvl}_{m}_{split}"])
    lines = [f"{lvl.capitalize()} & {ms(lvl, 'tfidf', 'study')} & {ms(lvl, 'bm25', 'study')} & "
             f"{ms(lvl, 'tfidf', 'test')} & {ms(lvl, 'bm25', 'test')} \\\\" for lvl in ("paper", "paragraph")]
    return (r"\begin{table}[t]\caption{Average search time per query (ms) for top-100 retrieval, including query "
            r"tokenization, excluding index loading, measured in the run that produced the submitted files. Single-threaded Python 3.13 on an Intel Core i7-9750H with 16\,GB RAM.}"
            r"\label{tab:time}" "\n"
            r"\small\begin{tabular}{lrrrr}\toprule & \multicolumn{2}{c}{Study} & \multicolumn{2}{c}{Test} \\"
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5} Level & TF-IDF & BM25 & TF-IDF & BM25 \\\midrule" "\n"
            + "\n".join(lines) + "\n" r"\bottomrule\end{tabular}\end{table}" "\n")


if __name__ == "__main__":
    OUT.write_text("% Generated by paper/make_tables.py -- do not edit by hand.\n" + summary_table("paper") + "\n"
                   + summary_table("paragraph") + "\n" + sig_table() + "\n" + difficulty_table() + "\n"
                   + time_table(), encoding="utf-8")
    print(f"wrote {OUT}")
