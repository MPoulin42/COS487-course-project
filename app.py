#!/usr/bin/env python3
"""Search interface for the arXiv collection (TF-IDF / BM25, paper and paragraph level).

    python app.py --data-dir DATA [--queries-dir QUERIES --team TEAM]

Pages: /  search,  /paper/<id>  full paper,  /annotate  Part 1 qrel annotation (needs --queries-dir).
"""

from __future__ import annotations

import argparse
import html
import re
import time
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request

import pipeline

app = Flask(__name__)
api = pipeline.api
CFG = {"queries_dir": None, "team": "team", "best_model": "bm25"}
PAGE_SIZE = 10
SNIPPET_CHARS = 420
_WORD = re.compile(r"[A-Za-z]+")
_BY_PAPER: dict[str, list[dict]] = {}  # paper_id -> its paragraph records, built on first paper view


def highlight(text: str, terms: set[str]) -> str:
    """HTML-escape text and wrap query-term matches in <mark>."""
    safe = html.escape(text)
    if not terms:
        return safe
    return _WORD.sub(lambda m: f"<mark>{m.group(0)}</mark>" if m.group(0).lower() in terms else m.group(0), safe)


def snippet(text: str, terms: set[str], width: int = SNIPPET_CHARS) -> str:
    """Window of text around the first query-term hit."""
    first = next((m.start() for m in _WORD.finditer(text) if m.group(0).lower() in terms), 0)
    start = max(0, first - width // 4)
    piece = text[start:start + width]
    return ("..." if start else "") + piece + ("..." if start + width < len(text) else "")


def arxiv_url(paper_id: str) -> str:
    return f"https://arxiv.org/abs/{paper_id}"


def run_query(q: str, model: str, level: str, k: int = 100) -> list[dict]:
    terms = set(api.tokenize(q))
    rows = []
    for hit in api.search(q, model=model, level=level, k=k):
        if level == "paper":
            p = api.get_paper(hit["id"])
            body, para_id = p["abstract"], None
        else:
            para = api.get_paragraph(hit["id"])
            p = api.get_paper(para["paper_id"])
            body, para_id = para["text"], para["para_id"]
        rows.append({
            "rank": hit["rank"], "score": round(hit["score"], 4), "paper_id": p["paper_id"], "para_id": para_id,
            "title": p["title"] or p["paper_id"], "snippet_html": highlight(snippet(body, terms), terms),
            "arxiv": arxiv_url(p["paper_id"]),
        })
    return rows


@app.route("/")
def index():
    return render_template("index.html", page_size=PAGE_SIZE, annotate=bool(CFG["queries_dir"]))


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    model = request.args.get("model", "bm25")
    level = request.args.get("level", "paper")
    if model not in ("bm25", "tfidf") or level not in ("paper", "paragraph"):
        abort(400)
    if not q:
        return jsonify(results=[], elapsed_ms=0, query_terms=[])
    t0 = time.perf_counter()
    rows = run_query(q, model, level)
    return jsonify(results=rows, elapsed_ms=round((time.perf_counter() - t0) * 1000, 1), query_terms=api.tokenize(q))


@app.route("/paper/<paper_id>")
def paper_page(paper_id):
    try:
        paper = api.get_paper(paper_id)
    except KeyError:
        abort(404)
    terms = set(api.tokenize(request.args.get("q", "")))
    paras = [{"id": pid, "html": highlight(text, terms)} for pid, text in _paragraphs_of(paper_id)]
    return render_template("paper.html", paper=paper, paras=paras, arxiv=arxiv_url(paper_id),
                           focus=request.args.get("para", ""))


def _paragraphs_of(paper_id: str):
    """Paragraph (id, text) pairs of one paper in document order (IDs are <paper_id>_<n>)."""
    if not _BY_PAPER:
        for r in api._paragraph_records.values():
            _BY_PAPER.setdefault(r["paper_id"], []).append(r)
        for recs in _BY_PAPER.values():
            recs.sort(key=lambda r: int(r["para_id"].rsplit("_", 1)[-1]) if r["para_id"].rsplit("_", 1)[-1].isdigit() else 0)
    return [(r["para_id"], r["text"]) for r in _BY_PAPER.get(paper_id, [])]


# ---------------------------------------------------------------------------
# Part 1 data annotation: top retrieved paragraph per Test query
# ---------------------------------------------------------------------------
def _qrel_path() -> Path:
    return Path(f"{CFG['team']}_qrel_part1_test.tsv")


def _load_labels() -> dict[str, tuple[str, int]]:
    labels = {}
    if _qrel_path().exists():
        for line in _qrel_path().read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 4:
                labels[parts[0]] = (parts[2], int(parts[3]))
    return labels


@app.route("/annotate")
def annotate():
    if not CFG["queries_dir"]:
        abort(404)
    labels = _load_labels()
    items = []
    for q in pipeline.load_queries(Path(CFG["queries_dir"]) / "Test.json"):
        hits = api.search(q["query"], model=CFG["best_model"], level="paragraph", k=1)
        if not hits:
            continue
        para = api.get_paragraph(hits[0]["id"])
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache", default="index_cache.pkl")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--queries-dir", help="enables the /annotate page (needs Test.json)")
    ap.add_argument("--team", default="team")
    ap.add_argument("--best-model", default="bm25", choices=["bm25", "tfidf"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    CFG.update(queries_dir=args.queries_dir, team=args.team, best_model=args.best_model)
    pipeline.build_system(args.data_dir, args.cache, args.rebuild)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
