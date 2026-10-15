"""Glue between T1 (tokenizer/indexing), T2 (scoring/search API) and the front ends.

Loads the per-paper JSON files, builds the paper- and paragraph-level collection
statistics, caches them on disk, and calls ``configure_retrieval`` once.
"""

from __future__ import annotations

import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

from t2_framework import CollectionStats, configure_retrieval
from t2_framework import retrieval as api

_MARKUP = re.compile(r"[<&]")
_t1 = None


def tokenize(text: str) -> List[str]:
    """T1's tokenizer (tokenizationandtfidf.tokenize).

    BeautifulSoup only changes text that contains markup or entities, so plain
    text skips it. The output is identical and much faster over the paragraphs.
    """
    global _t1
    if _t1 is None:
        import tokenizationandtfidf as t1

        _t1 = t1
    if _MARKUP.search(text):
        return _t1.tokenize(text)
    return [t for t in re.findall(r"\b[a-z]+\b", text.lower()) if t not in _t1.STOPWORDS]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _normalize_paragraphs(raw, paper_id: str) -> Dict[str, str]:
    if isinstance(raw, Mapping):
        return {str(k): str(v) for k, v in raw.items()}
    out: Dict[str, str] = {}
    for i, item in enumerate(raw or [], start=1):
        if isinstance(item, Mapping):
            pid = item.get("para_id") or item.get("paragraph_id") or item.get("id") or f"{paper_id}_{i}"
            out[str(pid)] = str(item.get("text", ""))
        else:
            out[f"{paper_id}_{i}"] = str(item)
    return out


def load_records(data_dir: str | Path) -> Dict[str, dict]:
    """Read every per-paper JSON under ``data_dir`` (a file holding a list of papers also works)."""
    records: Dict[str, dict] = {}
    for path in sorted(Path(data_dir).rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        for rec in payload if isinstance(payload, list) else [payload]:
            if not isinstance(rec, Mapping) or "paper_id" not in rec or "paragraphs" not in rec:
                continue
            pid = str(rec["paper_id"])
            records[pid] = {
                "paper_id": pid,
                "title": str(rec.get("title") or ""),
                "abstract": str(rec.get("abstract") or ""),
                "paragraphs": _normalize_paragraphs(rec["paragraphs"], pid),
            }
    if not records:
        raise FileNotFoundError(f"No paper JSON records (with paper_id and paragraphs) found under {data_dir}")
    return records


def build_collection(texts: Iterable[tuple]) -> CollectionStats:
    """texts: iterable of (doc_id, text). Returns raw-tf inverted index statistics."""
    index: Dict[str, Dict[str, int]] = defaultdict(dict)
    lengths: Dict[str, int] = {}
    for doc_id, text in texts:
        counts = Counter(tokenize(text))
        lengths[doc_id] = sum(counts.values())
        for term, tf in counts.items():
            index[term][doc_id] = tf
    n = len(lengths)
    return CollectionStats(
        inverted_index=dict(index),
        df={t: len(p) for t, p in index.items()},
        doc_lengths=lengths,
        num_docs=n,
        avg_doc_length=sum(lengths.values()) / n if n else 0.0,
    )


def build_system(data_dir: str | Path, cache: str | Path | None = None, rebuild: bool = False) -> None:
    """Index the collection (or load the cache) and configure the retrieval API."""
    cache = Path(cache) if cache else None
    if cache and cache.exists() and not rebuild:
        with cache.open("rb") as f:
            records, paper, paragraph = pickle.load(f)
        print(f"Loaded index cache {cache}")
    else:
        records = load_records(data_dir)
        print(f"Loaded {len(records)} papers; indexing...")
        # Paper level indexes title + abstract (T1's choice); paragraph level indexes each paragraph.
        paper = build_collection((pid, r["title"] + " " + r["abstract"]) for pid, r in records.items())
        paragraph = build_collection(
            (para_id, text) for r in records.values() for para_id, text in r["paragraphs"].items()
        )
        if cache:
            with cache.open("wb") as f:
                pickle.dump((records, paper, paragraph), f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Wrote index cache {cache}")
    configure_retrieval(
        paper_collection=paper,
        paragraph_collection=paragraph,
        paper_records=records,
        tokenizer=tokenize,
    )
    print(f"Ready: {paper.num_docs} papers, {paragraph.num_docs} paragraphs")


def load_queries(path: str | Path) -> List[dict]:
    """Query file -> [{"query_id", "query", "label"}]."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for rec in data:
        qid = next((rec[k] for k in ("query_id", "qid", "id") if k in rec), None)
        out.append({"query_id": str(qid), "query": rec.get("query", ""), "label": rec.get("label", "")})
    return out


__all__ = ["api", "build_system", "load_queries", "tokenize"]
