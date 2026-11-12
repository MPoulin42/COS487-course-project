#!/usr/bin/env python3
"""Build paper/results_tables.tex (Part 2) and copy the loss plots into paper/.

Reads, from the repository root (after run_part2.py and the Part 1 run_experiments.py):
    part2_{paper,paragraph}_{summary,sig,per_query}.csv   BM25 (Part 1 best) vs dense vs SPLADE, Study queries
    search_times.json, search_times_part2.json            per-query search times
    models/{dense,sparse}/loss_history.json, loss.pdf     fine-tuning logs and plots

    python paper/make_tables.py
"""

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = Path(__file__).resolve().parent
OUT = PAPER / "results_tables.tex"
METRICS = ["nDCG", "nDCG@5", "MRR_lenient", "MRR_strict", "P@5_lenient", "P@5_strict", "P@10_lenient", "P@10_strict"]
NAMES = {"nDCG": "nDCG", "nDCG@5": "nDCG@5", "MRR_lenient": r"MRR$_{\ge1}$", "MRR_strict": r"MRR$_{2}$",
         "P@5_lenient": r"P@5$_{\ge1}$", "P@5_strict": r"P@5$_{2}$", "P@10_lenient": r"P@10$_{\ge1}$",
         "P@10_strict": r"P@10$_{2}$"}
MODELS = ("bm25", "dense", "sparse")
LABEL = {"bm25": "BM25 (Part 1)", "dense": "Dense", "sparse": "SPLADE"}
PAIRS = (("dense", "bm25"), ("sparse", "bm25"), ("dense", "sparse"))  # reported as first - second


def model_of(run):
    return next(m for m in MODELS if f"_{m}_" in run)


def read(name):
    with open(ROOT / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summary_table():
    lines = []
    for level in ("paper", "paragraph"):
        rows = {model_of(r["run"]): {m: float(r[m]) for m in METRICS} for r in read(f"part2_{level}_summary.csv")}
        best = {m: max(v[m] for v in rows.values()) for m in METRICS}
        lines.append(r"\multicolumn{9}{l}{\textit{%s level}} \\" % level.capitalize())
        for model in MODELS:
            cells = [(r"\textbf{%.4f}" if rows[model][m] == best[m] else "%.4f") % rows[model][m] for m in METRICS]
            lines.append(LABEL[model] + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines.pop()
    head = "Model & " + " & ".join(NAMES[m] for m in METRICS) + r" \\"
    return (r"\begin{table*}[t]\caption{Results on the 250 Study queries (mean over queries; $\ge1$ = Relevant or "
            r"Partially Relevant counted as relevant, 2 = Relevant only). Best value per level in bold.}"
            r"\label{tab:summary}" "\n"
            r"\small\begin{tabular}{lrrrrrrrr}\toprule" "\n" + head + r"\midrule" "\n" + "\n".join(lines) + "\n"
            r"\bottomrule\end{tabular}\end{table*}" "\n")


def fmt_p(p):
    p = float(p)
    return ("$<$.001" if p < 0.001 else ("%.3f" % p).lstrip("0")) + ("$^*$" if p < 0.05 else "")


def fmt_diff(d):
    return "0.000" if abs(d) < 5e-4 else ("$+$%.3f" % d if d > 0 else "$-$%.3f" % -d)


def sig_table():
    lines = []
    for level in ("paper", "paragraph"):
        tests = {}
        for r in read(f"part2_{level}_sig.csv"):
            a, b = model_of(r["run_A"]), model_of(r["run_B"])
            tests[(a, b, r["metric"])] = (float(r["diff(A-B)"]), r["p_ttest"], r["p_randomization"])
            tests[(b, a, r["metric"])] = (-float(r["diff(A-B)"]), r["p_ttest"], r["p_randomization"])
        lines.append(r"\multicolumn{10}{l}{\textit{%s level}} \\" % level.capitalize())
        for m in METRICS:
            cells = []
            for a, b in PAIRS:
                d, pt, pr = tests[(a, b, m)]
                cells += [fmt_diff(d), fmt_p(pt), fmt_p(pr)]
            lines.append(NAMES[m] + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines.pop()
    groups = " & ".join(r"\multicolumn{3}{c}{%s $-$ %s}" % (LABEL[a].split(" ")[0], LABEL[b].split(" ")[0])
                        for a, b in PAIRS)
    rules = "".join(r"\cmidrule(lr){%d-%d}" % (2 + 3 * i, 4 + 3 * i) for i in range(3))
    return (r"\begin{table*}[t]\caption{Paired significance tests on the 250 Study queries: difference in means, "
            r"two-sided paired $t$-test $p_t$ and paired randomization test $p_r$ (10{,}000 sign flips). "
            r"$^*$ = $p<0.05$.}\label{tab:sig}" "\n"
            r"\small\begin{tabular}{lrrrrrrrrr}\toprule" "\n & " + groups + r" \\" + rules + "\n"
            r"Metric" + r" & Diff & $p_t$ & $p_r$" * 3 + r" \\\midrule" "\n" + "\n".join(lines) + "\n"
            r"\bottomrule\end{tabular}\end{table*}" "\n")


def difficulty_table():
    lines = []
    for level in ("paper", "paragraph"):
        groups = defaultdict(lambda: defaultdict(list))
        for r in read(f"part2_{level}_per_query.csv"):
            for m in ("nDCG@5", "MRR_lenient"):
                groups[(model_of(r["run"]), r["label"])][m].append(float(r[m]))
        for lab in ("Easy", "Medium", "Hard"):
            n = len(groups[("bm25", lab)]["nDCG@5"])
            cells = []
            for m in ("nDCG@5", "MRR_lenient"):
                vals = {mod: sum(groups[(mod, lab)][m]) / n for mod in MODELS}
                top = max(vals.values())
                cells += [(r"\textbf{%.3f}" if vals[mod] == top else "%.3f") % vals[mod] for mod in MODELS]
            lines.append(f"{level.capitalize()} & {lab} ({n}) & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines.pop()
    return (r"\begin{table}[t]\caption{Mean nDCG@5 and lenient MRR by query difficulty (Study queries; number of "
            r"queries in parentheses). B = BM25, D = Dense, S = SPLADE.}\label{tab:difficulty}" "\n"
            r"\small\setlength{\tabcolsep}{4pt}\begin{tabular}{llrrrrrr}\toprule"
            r" & & \multicolumn{3}{c}{nDCG@5} & \multicolumn{3}{c}{MRR$_{\ge1}$} \\"
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8} Level & Difficulty & B & D & S & B & D & S \\\midrule" "\n"
            + "\n".join(lines) + "\n" r"\bottomrule\end{tabular}\end{table}" "\n")


def time_table():
    t1 = json.loads((ROOT / "search_times.json").read_text(encoding="utf-8"))
    t2 = json.loads((ROOT / "search_times_part2.json").read_text(encoding="utf-8"))
    ms = lambda s: "%.1f" % (1000 * s)
    lines = []
    for lvl in ("paper", "paragraph"):
        lines.append(f"{lvl.capitalize()} & {ms(t1[f'NJMB_{lvl}_bm25_study'])} & "
                     f"{ms(t2[f'NJMB_{lvl}_dense_single']['s_per_query'])} & "
                     f"{ms(t2[f'NJMB_{lvl}_sparse_single']['s_per_query'])} & "
                     f"{ms(t2[f'NJMB_{lvl}_dense_study']['batched_s_per_query'])} & "
                     f"{ms(t2[f'NJMB_{lvl}_sparse_study']['batched_s_per_query'])} \\\\")
    return (r"\begin{table}[t]\caption{Average search time per Study query (ms) for top-100 retrieval, excluding "
            r"index loading. BM25: one query at a time, pure Python on a laptop CPU (Part 1). Neural models: query "
            r"encoding plus search on one NVIDIA RTX A5000, one query at a time and in batches of 50.}"
            r"\label{tab:time}" "\n"
            r"\small\setlength{\tabcolsep}{4pt}\begin{tabular}{lrrrrr}\toprule & BM25 & \multicolumn{2}{c}{Single query} & "
            r"\multicolumn{2}{c}{Batched} \\\cmidrule(lr){3-4}\cmidrule(lr){5-6}"
            r"Level & (CPU) & Dense & SPLADE & Dense & SPLADE \\\midrule" "\n"
            + "\n".join(lines) + "\n" r"\bottomrule\end{tabular}\end{table}" "\n")


def training_table():
    h = {m: json.loads((ROOT / f"models/{m}/loss_history.json").read_text(encoding="utf-8")) for m in ("dense", "sparse")}

    def best_step(m):
        ev = [x for x in h[m]["history"] if "eval_loss" in x]
        return min(ev, key=lambda x: x["eval_loss"])["step"]

    def first_eval(m):
        return next(x["eval_loss"] for x in h[m]["history"] if "eval_loss" in x)

    row = lambda name, f: f"{name} & {f('dense')} & {f('sparse')} \\\\"
    rows = [
        row("Base model", lambda m: r"\texttt{%s}" % h[m]["base_model"].split("/")[-1].replace("_", r"\_")),
        row("Batch size / learning rate", lambda m: "%d / %.0e" % (h[m]["batch_size"], h[m]["lr"])),
        row("Epochs / steps", lambda m: "%g / %d" % (h[m]["epochs"], round(h[m]["steps_per_epoch"] * h[m]["epochs"]))),
        row("Validation loss: untuned", lambda m: "%.3f" % first_eval(m)),
        row("Validation loss: best (step)", lambda m: "%.3f (%d)" % (h[m]["best_eval_loss"], best_step(m))),
    ]
    return (r"\begin{table}[t]\caption{Fine-tuning setup. Both models train on the same %s triples from 1{,}800 "
            r"Train queries and validate on %s triples from the 200 held-out Train queries; the checkpoint with the "
            r"lowest validation loss is kept.}\label{tab:training}" "\n"
            % (f"{h['dense']['train_triples']:,}".replace(",", "{,}"), f"{h['dense']['val_triples']:,}".replace(",", "{,}"))
            + r"\small\begin{tabular}{lrr}\toprule & Dense & SPLADE \\\midrule" "\n" + "\n".join(rows) + "\n"
            r"\bottomrule\end{tabular}\end{table}" "\n")


if __name__ == "__main__":
    for m in ("dense", "sparse"):
        shutil.copyfile(ROOT / f"models/{m}/loss.pdf", PAPER / f"loss_{m}.pdf")
    OUT.write_text("% Generated by paper/make_tables.py -- do not edit by hand.\n" + summary_table() + "\n"
                   + sig_table() + "\n" + difficulty_table() + "\n" + time_table() + "\n" + training_table(),
                   encoding="utf-8")
    print(f"wrote {OUT} and paper/loss_{{dense,sparse}}.pdf")
