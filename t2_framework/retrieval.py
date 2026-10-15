"""Shared collection contract, query tokenization, and top-k retrieval API."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Mapping, Optional, Sequence, Tuple, List

from .scoring import Scorer, make_scorer

DocId = Hashable
Tokenizer = Callable[[str], Sequence[str]]


@dataclass
class CollectionStats:
    """
    Fields:
        inverted_index: term -> {document ID -> raw term frequency}
        df: term -> number of documents containing that term
        doc_lengths: document ID -> number of indexed tokens
        num_docs: number of documents in this collection
        avg_doc_length: average number of indexed tokens per document
        documents: optional document ID -> metadata/payload for T4 display
    """

    inverted_index: Mapping[str, Mapping[DocId, int]]
    df: Mapping[str, int]
    doc_lengths: Mapping[DocId, int]
    num_docs: int
    avg_doc_length: float
    documents: Optional[Mapping[DocId, object]] = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "CollectionStats":
        """FIXME with T1's collection keys.
        """
        required = ("inverted_index", "df", "doc_lengths", "num_docs", "avg_doc_length")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(
                "Collection export is missing required field(s): " + ", ".join(missing)
            )
        index = payload["inverted_index"]
        df = payload["df"]
        lengths = payload["doc_lengths"]
        n = payload["num_docs"]
        avgdl = payload["avg_doc_length"]
        documents = payload.get("documents")
        if not isinstance(index, Mapping) or not isinstance(df, Mapping) or not isinstance(lengths, Mapping):
            raise TypeError("index, df, and doc_lengths must be dictionaries/mappings")
        return cls(
            inverted_index=index,
            df=df,
            doc_lengths=lengths,
            num_docs=int(n),
            avg_doc_length=float(avgdl),
            documents=documents,
        )

    def validate(self) -> None:
        """Catch common handoff mistakes early."""
        if self.num_docs < 0:
            raise ValueError("num_docs cannot be negative")
        if self.avg_doc_length < 0:
            raise ValueError("avg_doc_length cannot be negative")
        if self.num_docs != len(self.doc_lengths):
            raise ValueError(
                f"num_docs={self.num_docs} but doc_lengths contains {len(self.doc_lengths)} IDs. "
                "Check whether the lengths mapping is complete and collection-specific."
            )
        for term, postings in self.inverted_index.items():
            if not isinstance(postings, Mapping):
                raise TypeError(f"Posting list for {term!r} must be a doc_id -> tf mapping")
            df_value = self.df.get(term)
            if df_value is None:
                raise ValueError(f"Term {term!r} exists in index but is missing from df")
            if int(df_value) < 0 or int(df_value) > self.num_docs:
                raise ValueError(f"Invalid df[{term!r}]={df_value}; expected 0..{self.num_docs}")
            for doc_id, tf in postings.items():
                if doc_id not in self.doc_lengths:
                    raise ValueError(f"Indexed doc {doc_id!r} is missing from doc_lengths")
                if int(tf) <= 0:
                    raise ValueError(f"Non-positive term frequency for {term!r}, {doc_id!r}: {tf}")


Result = Tuple[DocId, float]


def rank_scores(scores: Mapping[DocId, float], top_k: int = 100) -> List[Result]:
    """Sort scores descending with deterministic document-ID tie-breaking."""
    if top_k < 0:
        raise ValueError("top_k cannot be negative")
    if top_k == 0:
        return []
    # Heap selection gives the same order as a full sort (score desc, ID asc) without sorting every candidate.
    positive = ((doc_id, float(score)) for doc_id, score in scores.items() if score is not None and float(score) > 0.0)
    return heapq.nsmallest(top_k, positive, key=lambda pair: (-pair[1], str(pair[0])))


class SearchEngine:
    """Reusable API for T4 interactive search and T2 batch runs for T3."""

    def __init__(
        self,
        collection: CollectionStats,
        tokenizer: Tokenizer,
        *,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        collection.validate()
        if not callable(tokenizer):
            raise TypeError("incomplete tokenizer")
        self.collection = collection
        self.tokenizer = tokenizer
        self._scorers: Dict[str, Scorer] = {
            "tfidf": make_scorer("tfidf", collection),
            "bm25": make_scorer("bm25", collection, k1=bm25_k1, b=bm25_b),
        }

    def search(
        self,
        query: str,
        model: str = "bm25",
        top_k: int = 100,
    ) -> List[Result]:
        """Return [(doc_id, score), ...] ranked highest-to-lowest."""
        model_key = model.strip().lower().replace("-", "").replace("_", "")
        if model_key == "tfidf":
            scorer = self._scorers["tfidf"]
        elif model_key == "bm25":
            scorer = self._scorers["bm25"]
        else:
            raise ValueError(f"Unknown model {model!r}; choose 'tfidf' or 'bm25'.")
        query_terms = list(self.tokenizer(query))
        return rank_scores(scorer.score(query_terms), top_k=top_k)


# ---------------------------------------------------------------------------
# Public module-level API for T4
# ---------------------------------------------------------------------------


from collections.abc import Callable as _Callable, Mapping as _Mapping, Sequence as _Sequence
from typing import Any as _Any

_PublicTokenizer = _Callable[[str], _Sequence[str]]
_paper_engine: SearchEngine | None = None
_paragraph_engine: SearchEngine | None = None
_t1_tokenizer: _PublicTokenizer | None = None
_paper_records: Dict[str, Dict[str, str]] = {}
_paragraph_records: Dict[str, Dict[str, str]] = {}


def configure_retrieval(
    *,
    paper_collection: CollectionStats,
    paragraph_collection: CollectionStats,
    paper_records: _Mapping[str, _Mapping[str, _Any]],
    tokenizer: _PublicTokenizer,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
) -> None:
    """Configure retrieval once at application startup using T1's real data.

    Pass in prepared index/statistics objects instead of rebuilding them here.
    `paper_records` maps each paper ID to the per-paper JSON record containing
    `paper_id`, `title`, `abstract`, and a `paragraphs` mapping.

    Validate the IDs across the indexes and source records now, because a
    mismatch is an integration error and should fail during startup rather than
    return incomplete results to a user.
    """
    if not isinstance(paper_collection, CollectionStats):
        raise TypeError("paper_collection must be a CollectionStats instance")
    if not isinstance(paragraph_collection, CollectionStats):
        raise TypeError("paragraph_collection must be a CollectionStats instance")
    if not callable(tokenizer):
        raise TypeError("tokenizer must be T1's callable tokenize(text) function")
    paper_collection.validate()
    paragraph_collection.validate()

    papers: Dict[str, Dict[str, str]] = {}
    paragraphs: Dict[str, Dict[str, str]] = {}
    for map_id, record in paper_records.items():
        if not isinstance(map_id, str):
            raise TypeError(f"Paper record keys must be strings; received {type(map_id).__name__}")
        if not isinstance(record, _Mapping):
            raise TypeError(f"Paper record {map_id!r} must be a mapping")
        for field in ("paper_id", "title", "abstract", "paragraphs"):
            if field not in record:
                raise ValueError(f"Paper record {map_id!r} is missing required field {field!r}")
        paper_id = record["paper_id"]
        if not isinstance(paper_id, str) or paper_id != map_id:
            raise ValueError(
                f"Paper record key {map_id!r} must exactly match string paper_id; got {paper_id!r}"
            )
        if not isinstance(record["title"], str) or not isinstance(record["abstract"], str):
            raise TypeError(f"Paper {paper_id!r} title and abstract must be strings")
        if not isinstance(record["paragraphs"], _Mapping):
            raise TypeError(f"Paper {paper_id!r} paragraphs must map paragraph IDs to text")
        if paper_id not in paper_collection.doc_lengths:
            raise ValueError(f"Paper {paper_id!r} has metadata but is missing from paper index/doc_lengths")

        papers[paper_id] = {
            "paper_id": paper_id,
            "title": record["title"],
            "abstract": record["abstract"],
        }
        for para_id, para_text in record["paragraphs"].items():
            if not isinstance(para_id, str):
                raise TypeError(f"Paragraph IDs in paper {paper_id!r} must be strings")
            if not isinstance(para_text, str):
                raise TypeError(f"Paragraph {para_id!r} text must be a string")
            if para_id in paragraphs:
                raise ValueError(f"Duplicate paragraph ID {para_id!r} across papers")
            if para_id not in paragraph_collection.doc_lengths:
                raise ValueError(f"Paragraph {para_id!r} has text but is missing from paragraph index/doc_lengths")
            paragraphs[para_id] = {
                "para_id": para_id,
                "paper_id": paper_id,
                "text": para_text,
            }

    indexed_papers = {str(doc_id) for doc_id in paper_collection.doc_lengths}
    indexed_paragraphs = {str(doc_id) for doc_id in paragraph_collection.doc_lengths}
    if indexed_papers != set(papers):
        missing = sorted(indexed_papers - set(papers))
        extra = sorted(set(papers) - indexed_papers)
        raise ValueError(f"Paper metadata/index IDs differ. Missing metadata={missing[:5]}, extra metadata={extra[:5]}")
    if indexed_paragraphs != set(paragraphs):
        missing = sorted(indexed_paragraphs - set(paragraphs))
        extra = sorted(set(paragraphs) - indexed_paragraphs)
        raise ValueError(f"Paragraph text/index IDs differ. Missing text={missing[:5]}, extra text={extra[:5]}")

    # Build locally first; failed setup never leaves a half-configured API.
    paper_engine = SearchEngine(paper_collection, tokenizer=tokenizer, bm25_k1=bm25_k1, bm25_b=bm25_b)
    paragraph_engine = SearchEngine(paragraph_collection, tokenizer=tokenizer, bm25_k1=bm25_k1, bm25_b=bm25_b)
    global _paper_engine, _paragraph_engine, _t1_tokenizer, _paper_records, _paragraph_records
    _paper_engine = paper_engine
    _paragraph_engine = paragraph_engine
    _t1_tokenizer = tokenizer
    _paper_records = papers
    _paragraph_records = paragraphs


def _require_configured() -> None:
    if _paper_engine is None or _paragraph_engine is None or _t1_tokenizer is None:
        raise RuntimeError(
            "Retrieval is not configured. Call configure_retrieval(...) once at startup "
            "with T1's paper/paragraph collections, source paper records, and tokenizer."
        )


def tokenize(text: str) -> List[str]:
    """Use T1's tokenizer so query terms match terms in the saved indexes."""
    _require_configured()
    if not isinstance(text, str):
        raise TypeError(f"tokenize(text) expected str, received {type(text).__name__}")
    assert _t1_tokenizer is not None
    tokens = _t1_tokenizer(text)
    if isinstance(tokens, (str, bytes)) or not isinstance(tokens, _Sequence):
        raise TypeError("T1 tokenizer must return a sequence of string tokens")
    if any(not isinstance(token, str) for token in tokens):
        raise TypeError("T1 tokenizer returned a non-string token")
    return list(tokens)


def search(
    query: str,
    model: str = "bm25",
    level: str = "paper",
    k: int = 100,
) -> List[Dict[str, _Any]]:
    """T4-facing query API: return up to k hits as id/score/rank dictionaries.

    The default k is 100 to match the assignment's retrieval requirement.
    Request fewer results explicitly, e.g. k=10 for a compact UI.
    """
    _require_configured()
    if not isinstance(query, str):
        raise TypeError(f"query must be str, received {type(query).__name__}")
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError("k must be an integer")
    if k < 0:
        raise ValueError("k cannot be negative")
    if not isinstance(level, str):
        raise TypeError("level must be 'paper' or 'paragraph'")
    if not isinstance(model, str):
        raise TypeError("model must be 'bm25' or 'tfidf'")

    level_key = level.strip().lower()
    model_key = model.strip().lower()
    engines = {"paper": _paper_engine, "paragraph": _paragraph_engine}
    if level_key not in engines:
        raise ValueError(f"Unknown level {level!r}; expected 'paper' or 'paragraph'")
    if model_key not in {"bm25", "tfidf"}:
        raise ValueError(f"Unknown model {model!r}; expected 'bm25' or 'tfidf'")
    engine = engines[level_key]
    assert engine is not None
    pairs = engine.search(query, model=model_key, top_k=k)
    return [
        {"id": str(doc_id), "score": float(score), "rank": rank}
        for rank, (doc_id, score) in enumerate(pairs, start=1)
    ]


def get_paper(paper_id: str) -> Dict[str, str]:
    """Return the paper fields T4 displays beside search results."""
    _require_configured()
    if not isinstance(paper_id, str):
        raise TypeError(f"paper_id must be str, received {type(paper_id).__name__}")
    try:
        return dict(_paper_records[paper_id])
    except KeyError as exc:
        raise KeyError(f"Unknown paper_id {paper_id!r}; no paper metadata was loaded") from exc


def get_paragraph(para_id: str) -> Dict[str, str]:
    """Return the paragraph text and parent paper ID for T4's result snippet."""
    _require_configured()
    if not isinstance(para_id, str):
        raise TypeError(f"para_id must be str, received {type(para_id).__name__}")
    try:
        return dict(_paragraph_records[para_id])
    except KeyError as exc:
        raise KeyError(f"Unknown para_id {para_id!r}; no paragraph text was loaded") from exc
