"""Top-k search over the indexes written by neural.encode.

    r = DenseRetriever("models/dense/final", "indexes/dense", levels=("paper", "paragraph"))
    r.search("diffusion models for image editing", level="paragraph", k=100)
    # [{"id": "2408.01234_7", "score": 0.71, "rank": 1}, ...]   (same shape as Part 1's api.search)

Dense : exact inner-product search (= cosine, vectors are normalised). On GPU the whole
        float16 matrix sits in GPU memory; on CPU it is scanned in float32 blocks.
Sparse: the query's SPLADE vector touches only the posting lists (CSC columns) of its
        non-zero vocabulary terms; score = sum of query weight x document weight.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from neural.encode import load_model


def _rank(ids: Sequence[str], scores: np.ndarray, k: int) -> List[dict]:
    k = min(k, len(scores))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top], kind="stable")]
    return [{"id": ids[i], "score": float(scores[i]), "rank": r} for r, i in enumerate(top, 1)]


class _Base:
    kind = ""

    def __init__(self, checkpoint: str, index_dir: str | Path, levels=("paper", "paragraph"), device: str | None = None,
                 max_seq_length: int = 256):
        self.model = load_model(self.kind, checkpoint, max_seq_length, device)
        self.device = str(self.model.device)
        self.ids: Dict[str, List[str]] = {}
        for level in levels:
            d = Path(index_dir) / level
            self.ids[level] = json.loads((d / "ids.json").read_text(encoding="utf-8"))
            self._load(level, d)

    def search(self, query: str, level: str = "paper", k: int = 100) -> List[dict]:
        return self.search_batch([query], level, k)[0]


class DenseRetriever(_Base):
    kind = "dense"
    BLOCK = 262_144  # CPU scan block (rows)

    def _load(self, level: str, d: Path) -> None:
        import torch

        if not hasattr(self, "emb"):
            self.emb = {}
        mat = np.load(d / "emb.npy", mmap_mode=None if self.device.startswith("cuda") else "r")
        self.emb[level] = torch.from_numpy(np.ascontiguousarray(mat)).to(self.device) if self.device.startswith("cuda") else mat

    def search_batch(self, queries: Sequence[str], level: str = "paper", k: int = 100) -> List[List[dict]]:
        import torch

        q = self.model.encode_query(list(queries), batch_size=64, normalize_embeddings=True, convert_to_tensor=True)
        emb, ids = self.emb[level], self.ids[level]
        k = min(k, len(ids))
        if self.device.startswith("cuda"):
            scores = q.to(emb.dtype) @ emb.T
            vals, idx = torch.topk(scores.float(), k, dim=1)
            vals, idx = vals.cpu().numpy(), idx.cpu().numpy()
        else:  # blockwise scan of the memory-mapped float16 matrix, merging the running top-k
            qn = q.float().cpu().numpy()
            vals = np.full((len(qn), 0), -np.inf, dtype=np.float32)
            idx = np.zeros((len(qn), 0), dtype=np.int64)
            for start in range(0, len(ids), self.BLOCK):
                s = qn @ np.asarray(emb[start:start + self.BLOCK], dtype=np.float32).T
                kk = min(k, s.shape[1])
                part = np.argpartition(-s, kk - 1, axis=1)[:, :kk]
                vals = np.concatenate([vals, np.take_along_axis(s, part, 1)], 1)
                idx = np.concatenate([idx, part + start], 1)
                if vals.shape[1] > k:
                    keep = np.argpartition(-vals, k - 1, axis=1)[:, :k]
                    vals, idx = np.take_along_axis(vals, keep, 1), np.take_along_axis(idx, keep, 1)
            order = np.argsort(-vals, axis=1, kind="stable")
            vals, idx = np.take_along_axis(vals, order, 1), np.take_along_axis(idx, order, 1)
        return [[{"id": ids[i], "score": float(v), "rank": r} for r, (i, v) in enumerate(zip(row_i, row_v), 1)]
                for row_i, row_v in zip(idx, vals)]


class SparseRetriever(_Base):
    kind = "sparse"

    def _load(self, level: str, d: Path) -> None:
        import scipy.sparse as sp

        if not hasattr(self, "mat"):
            self.mat = {}
        self.mat[level] = sp.load_npz(d / "sparse.npz").tocsc()

    def encode_queries(self, queries: Sequence[str]):
        from neural.encode import sparse_to_scipy

        vec = self.model.encode_query(list(queries), batch_size=64, convert_to_sparse_tensor=True)
        return sparse_to_scipy(vec)

    def search_batch(self, queries: Sequence[str], level: str = "paper", k: int = 100) -> List[List[dict]]:
        qv = self.encode_queries(queries)
        mat, ids = self.mat[level], self.ids[level]
        out = []
        for row in range(qv.shape[0]):
            q = qv.getrow(row)
            terms, weights = q.indices, q.data
            scores = np.asarray(mat[:, terms] @ weights).ravel()  # only the query terms' posting lists
            out.append(_rank(ids, scores, k))
        return out

    def expansion(self, query: str, top: int = 15) -> List[tuple]:
        """The query's highest-weighted vocabulary terms (for the interface and error analysis)."""
        q = self.encode_queries([query]).getrow(0)
        tok = self.model.tokenizer
        order = np.argsort(-q.data)[:top]
        return [(tok.convert_ids_to_tokens(int(q.indices[i])), round(float(q.data[i]), 3)) for i in order]


def make_retriever(kind: str, checkpoint: str, index_dir: str, levels=("paper", "paragraph"), device=None):
    return (DenseRetriever if kind == "dense" else SparseRetriever)(checkpoint, index_dir, levels, device)
