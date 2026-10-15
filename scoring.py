"""Manual TF-IDF and BM25 scoring implementations.

This module deliberately does not use scikit-learn, PyTerrier, rank-bm25, or
other ready-made retrieval/scoring implementations.

Expected inverted index format:
    inverted_index[term][doc_id] = term frequency in that document

Expected statistics:
    df[term] = number of documents containing the term
    doc_lengths[doc_id] = number of indexed tokens in the document
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from collections import Counter
from typing import Dict, Hashable, Mapping, Sequence

DocId = Hashable


class Scorer(ABC):
    """Common interface implemented by all ranking models."""

    @abstractmethod
    def score(self, query_terms: Sequence[str]) -> Dict[DocId, float]:
        """Return a mapping of candidate document IDs to relevance scores."""
        raise NotImplementedError


class TFIDFScorer(Scorer):
    """Log-weighted TF-IDF with cosine normalization.

    Weighting follows the supplied class-demo convention:
        tf_weight = 1 + log10(tf)      (tf > 0)
        idf = log10(N / df)
        weight = tf_weight * idf

    Query and document vectors are L2-normalized, and the dot product of their
    TF-IDF weights is returned as cosine similarity. Terms with df == N have
    IDF 0 and therefore contribute no score.
    """

    def __init__(
        self,
        inverted_index: Mapping[str, Mapping[DocId, int]],
        df: Mapping[str, int],
        num_docs: int,
    ) -> None:
        if num_docs < 0:
            raise ValueError("num_docs cannot be negative")
        self.inverted_index = inverted_index
        self.df = df
        self.num_docs = int(num_docs)
        self._idf_cache: Dict[str, float] = {}
        self._doc_norms: Dict[DocId, float] = {}
        self._build_document_norms()

    def _idf(self, term: str) -> float:
        if term in self._idf_cache:
            return self._idf_cache[term]
        term_df = self.df.get(term, 0)
        if self.num_docs <= 0 or term_df <= 0 or term_df > self.num_docs:
            value = 0.0
        else:
            value = math.log10(self.num_docs / term_df)
        self._idf_cache[term] = value
        return value

    @staticmethod
    def _tf_weight(tf: int) -> float:
        return 1.0 + math.log10(tf) if tf > 0 else 0.0

    def _build_document_norms(self) -> None:
        squared_norms: Dict[DocId, float] = {}
        for term, postings in self.inverted_index.items():
            idf = self._idf(term)
            if idf == 0.0:
                continue
            for doc_id, raw_tf in postings.items():
                tf = int(raw_tf)
                if tf <= 0:
                    continue
                weight = self._tf_weight(tf) * idf
                squared_norms[doc_id] = squared_norms.get(doc_id, 0.0) + weight * weight
        self._doc_norms = {
            doc_id: math.sqrt(total) for doc_id, total in squared_norms.items()
        }

    def score(self, query_terms: Sequence[str]) -> Dict[DocId, float]:
        if self.num_docs == 0:
            return {}
        query_tf = Counter(term for term in query_terms if term)
        if not query_tf:
            return {}

        query_weights: Dict[str, float] = {}
        for term, tf in query_tf.items():
            idf = self._idf(term)
            if idf > 0.0:
                query_weights[term] = self._tf_weight(tf) * idf
        if not query_weights:
            return {}

        query_norm = math.sqrt(sum(weight * weight for weight in query_weights.values()))
        if query_norm == 0.0:
            return {}

        dot_products: Dict[DocId, float] = {}
        for term, query_weight in query_weights.items():
            postings = self.inverted_index.get(term, {})
            for doc_id, raw_tf in postings.items():
                tf = int(raw_tf)
                if tf <= 0:
                    continue
                doc_weight = self._tf_weight(tf) * self._idf(term)
                dot_products[doc_id] = dot_products.get(doc_id, 0.0) + query_weight * doc_weight

        scores: Dict[DocId, float] = {}
        for doc_id, dot in dot_products.items():
            doc_norm = self._doc_norms.get(doc_id, 0.0)
            if doc_norm > 0.0:
                scores[doc_id] = dot / (query_norm * doc_norm)
        return scores


class BM25Scorer(Scorer):
    """Okapi BM25 scorer with document-length normalization.

    Uses the common positive Robertson-Sparck Jones IDF variant:
        idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

    Per-term document contribution:
        idf(t) * tf * (k1 + 1) /
        (tf + k1 * (1 - b + b * doc_length / avg_doc_length))

    Query terms are counted once per distinct term by default. This is a common
    BM25 choice and avoids accidental over-weighting when a term is repeated in
    a natural-language query. Query tokenization itself is supplied by the
    caller so it can match T1's preprocessing.
    """

    def __init__(
        self,
        inverted_index: Mapping[str, Mapping[DocId, int]],
        df: Mapping[str, int],
        doc_lengths: Mapping[DocId, int],
        num_docs: int,
        avg_doc_length: float,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if num_docs < 0:
            raise ValueError("num_docs cannot be negative")
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be in [0, 1]")
        if avg_doc_length < 0:
            raise ValueError("avg_doc_length cannot be negative")
        self.inverted_index = inverted_index
        self.df = df
        self.doc_lengths = doc_lengths
        self.num_docs = int(num_docs)
        self.avg_doc_length = float(avg_doc_length)
        self.k1 = float(k1)
        self.b = float(b)

    def _idf(self, term: str) -> float:
        term_df = int(self.df.get(term, 0))
        if self.num_docs <= 0 or term_df < 0 or term_df > self.num_docs:
            return 0.0
        return math.log1p(
            (self.num_docs - term_df + 0.5) / (term_df + 0.5)
        )

    def score(self, query_terms: Sequence[str]) -> Dict[DocId, float]:
        if self.num_docs == 0 or self.avg_doc_length <= 0.0:
            return {}
        unique_query_terms = dict.fromkeys(term for term in query_terms if term)
        scores: Dict[DocId, float] = {}

        for term in unique_query_terms:
            postings = self.inverted_index.get(term, {})
            if not postings:
                continue
            idf = self._idf(term)
            if idf == 0.0:
                continue
            for doc_id, raw_tf in postings.items():
                tf = float(raw_tf)
                if tf <= 0.0:
                    continue
                if doc_id not in self.doc_lengths:
                    raise KeyError(
                        f"Document {doc_id!r} appears in the index but has no entry in doc_lengths"
                    )
                length = float(self.doc_lengths[doc_id])
                norm = self.k1 * (
                    1.0 - self.b + self.b * length / self.avg_doc_length
                )
                contribution = idf * (tf * (self.k1 + 1.0)) / (tf + norm)
                scores[doc_id] = scores.get(doc_id, 0.0) + contribution
        return scores


def make_scorer(model: str, collection, *, k1: float = 1.5, b: float = 0.75) -> Scorer:
    """Build a scorer using a CollectionStats-like object.

    The collection is expected to expose: inverted_index, df, num_docs,
    doc_lengths, and avg_doc_length.
    """
    model_key = model.strip().lower().replace("-", "").replace("_", "")
    if model_key in {"tfidf", "tfidfcosine", "vector-space"}:
        return TFIDFScorer(collection.inverted_index, collection.df, collection.num_docs)
    if model_key == "bm25":
        return BM25Scorer(
            collection.inverted_index,
            collection.df,
            collection.doc_lengths,
            collection.num_docs,
            collection.avg_doc_length,
            k1=k1,
            b=b,
        )
    raise ValueError(f"Unknown model {model!r}; choose 'tfidf' or 'bm25'.")
