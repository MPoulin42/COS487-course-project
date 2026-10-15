"""T2 scoring and ranking layer (TF-IDF, BM25) plus the public search API."""

from .retrieval import CollectionStats, SearchEngine, configure_retrieval, get_paper, get_paragraph, search, tokenize

__all__ = [
    "CollectionStats",
    "SearchEngine",
    "configure_retrieval",
    "get_paper",
    "get_paragraph",
    "search",
    "tokenize",
]
