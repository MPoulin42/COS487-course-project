#!/usr/bin/env python3
"""Search interface for the arXiv collection (paper and paragraph level).

Models: TF-IDF and BM25 (Part 1); fine-tuned dense bi-encoder and SPLADE (Part 2), loaded
when their model and index folders exist (see neural/).

    python app.py --data-dir DATA [--queries-dir QUERIES --team TEAM --best-model dense]

Pages: /  search,  /paper/<id>  full paper,  /annotate  Test-query qrel annotation (needs --queries-dir).
"""

from __future__ import annotations

import argparse
import html
import re
import time
from pathlib import Path

import json

from flask import Flask, abort, jsonify, redirect, render_template, request, url_for

import pipeline

app = Flask(__name__)
api = pipeline.api
CFG = {"queries_dir": None, "team": "team", "best_model": "bm25", "part": 2}
NEURAL: dict = {}  # "dense"/"sparse" -> neural.retrievers retriever, filled in main()
MODEL_LABELS = {"bm25": "BM25", "tfidf": "TF-IDF", "dense": "Dense", "sparse": "SPLADE"}


def search(q: str, model: str, level: str, k: int) -> list[dict]:
    """One entry point for the lexical (Part 1) and neural (Part 2) models."""
    if model in NEURAL:
        return NEURAL[model].search(q, level=level, k=k)
    return api.search(q, model=model, level=level, k=k)


def available_models() -> list[str]:
    return ["bm25", "tfidf"] + [m for m in ("dense", "sparse") if m in NEURAL]
PAGE_SIZE = 10
SNIPPET_CHARS = 420
_WORD = re.compile(r"[A-Za-z]+")
_BY_PAPER: dict[str, list[dict]] = {}  # paper_id -> its paragraph records, built on first paper view
# --annotate-only: just the papers whose paragraphs are being labelled, read straight from their JSON
# files, so the app runs in a few hundred MB instead of loading the full BM25/TF-IDF index.
_LOCAL: dict[str, dict] = {}


def _get_paper(paper_id: str) -> dict:
    return dict(_LOCAL[paper_id]) if _LOCAL else api.get_paper(paper_id)


def _get_paragraph(para_id: str) -> dict:
    if not _LOCAL:
        return api.get_paragraph(para_id)
    paper_id = para_id.rsplit("_", 1)[0]
    return {"para_id": para_id, "paper_id": paper_id, "text": _LOCAL[paper_id]["paragraphs"][para_id]}


def _tokenize(text: str) -> list[str]:
    return pipeline.tokenize(text) if _LOCAL else api.tokenize(text)


def load_annotation_papers(data_dir: str, run: Path) -> None:
    """Read only the papers that contain a top-1 paragraph in the best model's Test run."""
    paper_ids = {line.split()[2].rsplit("_", 1)[0] for line in run.read_text(encoding="utf-8").splitlines()
                 if len(line.split()) == 6 and line.split()[3] == "1"}
    for pid in paper_ids:
        rec = json.loads((Path(data_dir) / f"{pid}.json").read_text(encoding="utf-8"))
        _LOCAL[pid] = {"paper_id": pid, "title": str(rec.get("title") or ""),
                       "abstract": str(rec.get("abstract") or ""),
                       "paragraphs": pipeline._normalize_paragraphs(rec["paragraphs"], pid)}


# Inline/display LaTeX as it appears in the paper text; the pages render it with KaTeX (static/math.js).
_MATH = re.compile(r"\$\$.+?\$\$|\$[^$]+?\$|\\\(.+?\\\)|\\\[.+?\\\]", re.S)


def highlight(text: str, terms: set[str]) -> str:
    """HTML-escape text and wrap query-term matches in <mark>, leaving LaTeX math untouched
    (a <mark> inside a formula would split it and stop it from rendering)."""
    if not terms:
        return html.escape(text)
    out, last = [], 0
    for m in list(_MATH.finditer(text)) + [None]:
        prose = text[last:m.start() if m else len(text)]
        out.append(_WORD.sub(lambda w: f"<mark>{w.group(0)}</mark>" if w.group(0).lower() in terms else w.group(0),
                             html.escape(prose)))
        if m:
            out.append(html.escape(m.group(0)))
            last = m.end()
    return "".join(out)


def snippet(text: str, terms: set[str], width: int = SNIPPET_CHARS) -> str:
    """Window of text around the first query-term hit, widened so it never cuts a formula in half."""
    first = next((m.start() for m in _WORD.finditer(text) if m.group(0).lower() in terms), 0)
    start = max(0, first - width // 4)
    end = start + width
    for m in _MATH.finditer(text):
        if m.start() < start < m.end():
            start = m.start()
        if m.start() < end < m.end():
            end = m.end()
    return ("..." if start else "") + text[start:end] + ("..." if end < len(text) else "")


def arxiv_url(paper_id: str) -> str:
    return f"https://arxiv.org/abs/{paper_id}"


MONTHS = {"2408": "August 2024", "2409": "September 2024"}
LENGTHS = ("short", "medium", "long")
FILTER_FETCH = 1000  # candidates pulled before filtering, so filters still leave up to 100 results
META: dict = {"counts": {}, "cuts": (0, 0)}


def build_meta() -> None:
    """Paragraph count per paper and the tercile cut points used for the length filter."""
    counts: dict[str, int] = {}
    for rec in api._paragraph_records.values():
        counts[rec["paper_id"]] = counts.get(rec["paper_id"], 0) + 1
    ordered = sorted(counts.values())
    META["counts"] = counts
    META["cuts"] = (ordered[len(ordered) // 3], ordered[2 * len(ordered) // 3]) if ordered else (0, 0)


def length_bucket(paper_id: str) -> str:
    n = META["counts"].get(paper_id, 0)
    lo, hi = META["cuts"]
    return "short" if n <= lo else "medium" if n <= hi else "long"


def passes(paper_id: str, months: set[str], lengths: set[str]) -> bool:
    return (not months or paper_id[:4] in months) and (not lengths or length_bucket(paper_id) in lengths)


def run_query(q: str, model: str, level: str, k: int = 100,
              months: set[str] = frozenset(), lengths: set[str] = frozenset()) -> list[dict]:
    terms = set(api.tokenize(q))
    filtered = bool(months or lengths)
    rows = []
    for hit in search(q, model, level, FILTER_FETCH if filtered else k):
        if level == "paper":
            p = api.get_paper(hit["id"])
            body, para_id = p["abstract"], None
        else:
            para = api.get_paragraph(hit["id"])
            p = api.get_paper(para["paper_id"])
            body, para_id = para["text"], para["para_id"]
        if filtered and not passes(p["paper_id"], months, lengths):
            continue
        rows.append({
            "rank": len(rows) + 1, "score": round(hit["score"], 4), "paper_id": p["paper_id"], "para_id": para_id,
            "title": p["title"] or p["paper_id"], "snippet_html": highlight(snippet(body, terms), terms),
            "arxiv": arxiv_url(p["paper_id"]),
        })
        if len(rows) == k:
            break
    return rows


@app.route("/")
def index():
    if _LOCAL:
        return redirect(url_for("annotate"))
    lo, hi = META["cuts"]
    length_labels = {"short": f"Short (up to {lo} paragraphs)", "medium": f"Medium ({lo + 1}-{hi})",
                     "long": f"Long (over {hi})"}
    return render_template("index.html", page_size=PAGE_SIZE, annotate=bool(CFG["queries_dir"]),
                           months=MONTHS, length_labels=length_labels,
                           models=[(m, MODEL_LABELS[m]) for m in available_models()])


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    model = request.args.get("model", "bm25")
    level = request.args.get("level", "paper")
    if model not in available_models() or level not in ("paper", "paragraph"):
        abort(400)
    if not q:
        return jsonify(results=[], elapsed_ms=0, query_terms=[])
    months = {m for m in request.args.get("months", "").split(",") if m in MONTHS}
    lengths = {x for x in request.args.get("length", "").split(",") if x in LENGTHS}
    t0 = time.perf_counter()
    rows = run_query(q, model, level, months=months, lengths=lengths)
    return jsonify(results=rows, elapsed_ms=round((time.perf_counter() - t0) * 1000, 1), query_terms=api.tokenize(q))


@app.route("/paper/<paper_id>")
def paper_page(paper_id):
    try:
        paper = _get_paper(paper_id)
    except KeyError:
        abort(404)
    terms = set(_tokenize(request.args.get("q", "")))
    paras = [{"id": pid, "html": highlight(text, terms)} for pid, text in _paragraphs_of(paper_id)]
    return render_template("paper.html", paper=paper, paras=paras, arxiv=arxiv_url(paper_id),
                           focus=request.args.get("para", ""))


def _paragraphs_of(paper_id: str):
    """Paragraph (id, text) pairs of one paper in document order (IDs are <paper_id>_<n>)."""
    if _LOCAL:
        return list(_LOCAL[paper_id]["paragraphs"].items())
    if not _BY_PAPER:
        for r in api._paragraph_records.values():
            _BY_PAPER.setdefault(r["paper_id"], []).append(r)
        for recs in _BY_PAPER.values():
            recs.sort(key=lambda r: int(r["para_id"].rsplit("_", 1)[-1]) if r["para_id"].rsplit("_", 1)[-1].isdigit() else 0)
    return [(r["para_id"], r["text"]) for r in _BY_PAPER.get(paper_id, [])]


# ---------------------------------------------------------------------------
# Data annotation: top retrieved paragraph per Test query (best model of the current part)
# ---------------------------------------------------------------------------
def _qrel_path() -> Path:
    return Path(f"{CFG['team']}_qrel_part{CFG['part']}_test.tsv")


def _load_labels() -> dict[str, tuple[str, int]]:
    labels = {}
    if _qrel_path().exists():
        for line in _qrel_path().read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 4:
                labels[parts[0]] = (parts[2], int(parts[3]))
    return labels


_TOP1: dict[str, str | None] = {}


def _top_paragraph(qid: str, query: str) -> str | None:
    """Top-1 paragraph ID for a Test query.

    Taken from the submitted run file when present (instant, and exactly what was
    submitted); otherwise searched once and kept in memory.
    """
    if not _TOP1:
        run = Path("Retrieval Results") / f"{CFG['team']}_paragraph_{CFG['best_model']}_test.tsv"
        if run.exists():
            for line in run.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 6 and parts[3] == "1":
                    _TOP1[parts[0]] = parts[2]
    if qid not in _TOP1:
        hits = search(query, CFG["best_model"], "paragraph", 1)
        _TOP1[qid] = hits[0]["id"] if hits else None
    return _TOP1[qid]


@app.route("/annotate")
def annotate():
    if not CFG["queries_dir"]:
        abort(404)
    labels = _load_labels()
    items = []
    for q in pipeline.load_queries(Path(CFG["queries_dir"]) / "Test.json"):
        para_id = _top_paragraph(q["query_id"], q["query"])
        if para_id is None:
            continue
        para = _get_paragraph(para_id)
        label = labels.get(q["query_id"])
        items.append({"qid": q["query_id"], "query": q["query"], "para_id": para["para_id"],
                      "paper_id": para["paper_id"], "text": para["text"],
                      "label": label[1] if label and label[0] == para["para_id"] else None})
    done = sum(i["label"] is not None for i in items)
    return render_template("annotate.html", items=items, done=done, model=CFG["best_model"], path=_qrel_path().name)


@app.route("/api/annotate", methods=["POST"])
def save_annotation():
    body = request.get_json(force=True)
    qid, para_id, rel = str(body["qid"]), str(body["para_id"]), int(body["relevance"])
    if rel not in (0, 1, 2):
        abort(400)
    labels = _load_labels()
    labels[qid] = (para_id, rel)
    with _qrel_path().open("w", encoding="utf-8", newline="\n") as f:
        for q, (p, r) in labels.items():
            f.write(f"{q} 0 {p} {r}\n")
    return jsonify(ok=True, saved=len(labels))


def load_neural(args) -> None:
    """Load each Part 2 retriever whose fine-tuned model and index both exist."""
    from neural.retrievers import make_retriever

    for kind in ("dense", "sparse"):
        ckpt, index = getattr(args, f"{kind}_checkpoint"), getattr(args, f"{kind}_index")
        if Path(ckpt).exists() and (Path(index) / "paper" / "ids.json").exists():
            NEURAL[kind] = make_retriever(kind, ckpt, index)
            print(f"Loaded {kind} retriever on {NEURAL[kind].device}")
        else:
            print(f"Skipping {kind} model: {ckpt} or {index} not found")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache", default="index_cache.pkl")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--queries-dir", help="enables the /annotate page (needs Test.json)")
    ap.add_argument("--team", default="team")
    ap.add_argument("--best-model", default="bm25", choices=["bm25", "tfidf", "dense", "sparse"],
                    help="model whose top paragraph per Test query is annotated")
    ap.add_argument("--part", type=int, default=2, help="project part, used in the qrel file name")
    ap.add_argument("--dense-checkpoint", default="models/dense/final")
    ap.add_argument("--dense-index", default="indexes/dense")
    ap.add_argument("--sparse-checkpoint", default="models/sparse/final")
    ap.add_argument("--sparse-index", default="indexes/sparse")
    ap.add_argument("--no-neural", action="store_true", help="serve only the Part 1 models")
    ap.add_argument("--annotate-only", action="store_true",
                    help="serve only /annotate from the best model's Test run file; loads just the papers it "
                         "needs instead of the full index (for machines with little RAM)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    CFG.update(queries_dir=args.queries_dir, team=args.team, best_model=args.best_model, part=args.part)
    if args.annotate_only:
        best_run = Path("Retrieval Results") / f"{args.team}_paragraph_{args.best_model}_test.tsv"
        if not args.queries_dir or not best_run.exists():
            ap.error(f"--annotate-only needs --queries-dir and {best_run}")
        load_annotation_papers(args.data_dir, best_run)
        print(f"Annotate-only: loaded {len(_LOCAL)} papers from {best_run}")
        app.run(host=args.host, port=args.port, debug=False)
        return
    pipeline.build_system(args.data_dir, args.cache, args.rebuild)
    if not args.no_neural:
        load_neural(args)
    # Annotation only needs the best model's Test run file, so the model itself may be absent
    # (e.g. labelling locally after the neural runs were made on a GPU server).
    best_run = Path("Retrieval Results") / f"{args.team}_paragraph_{args.best_model}_test.tsv"
    if args.best_model not in available_models() and not (args.queries_dir and best_run.exists()):
        ap.error(f"--best-model {args.best_model} is not loaded and {best_run} does not exist")
    build_meta()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
