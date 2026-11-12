"""Encode the collection with a fine-tuned model into an on-disk index.

Dense : L2-normalised float16 embeddings, one row per document -> INDEX/LEVEL/emb.npy
Sparse: SPLADE vectors (at most --max-active-dims non-zero vocabulary weights per document),
        stored as a document x vocabulary scipy CSC matrix -> INDEX/LEVEL/sparse.npz.
        CSC = column-major = one posting list per vocabulary term, i.e. an inverted index.
Both  : INDEX/LEVEL/ids.json gives the document id of each row.

Encoding is done in shards of --shard-size documents written to INDEX/LEVEL/shards/,
so an interrupted run resumes where it stopped (the 2.2M paragraphs take a while).

    python -m neural.encode --model dense  --checkpoint models/dense/final  --index indexes/dense  --levels paper paragraph
    python -m neural.encode --model sparse --checkpoint models/sparse/final --index indexes/sparse --levels paper paragraph
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from neural.corpus import get_corpus


def load_model(kind: str, checkpoint: str, max_seq_length: int = 256, device: str | None = None):
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if kind == "dense":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(checkpoint, device=device)
    else:
        from sentence_transformers.sparse_encoder import SparseEncoder

        model = SparseEncoder(checkpoint, device=device)
    model.max_seq_length = max_seq_length
    if device == "cuda":
        model.half()
    return model


def sparse_to_scipy(t):
    """torch sparse COO (n x vocab) -> scipy CSR float32."""
    import scipy.sparse as sp

    t = t.coalesce().cpu()
    idx = t.indices().numpy()
    return sp.csr_matrix((t.values().float().numpy(), (idx[0], idx[1])), shape=tuple(t.shape), dtype=np.float32)


def encode_level(model, kind: str, ids, texts, out: Path, batch_size: int, shard_size: int, max_active_dims: int):
    import scipy.sparse as sp

    shards = out / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    n_shards = (len(texts) + shard_size - 1) // shard_size
    ext = "npy" if kind == "dense" else "npz"
    t0 = time.time()
    for s in range(n_shards):
        path = shards / f"{s:05d}.{ext}"
        if path.exists():
            continue
        chunk = texts[s * shard_size:(s + 1) * shard_size]
        if kind == "dense":
            emb = model.encode_document(chunk, batch_size=batch_size, normalize_embeddings=True,
                                        convert_to_numpy=True, show_progress_bar=False)
            np.save(path, emb.astype(np.float16))
        else:
            vec = model.encode_document(chunk, batch_size=batch_size, convert_to_sparse_tensor=True,
                                        max_active_dims=max_active_dims, show_progress_bar=False)
            sp.save_npz(path, sparse_to_scipy(vec))
        done = min((s + 1) * shard_size, len(texts))
        rate = done / max(time.time() - t0, 1e-9)
        print(f"  {out.name}: shard {s + 1}/{n_shards} ({done}/{len(texts)} docs, ETA {(len(texts) - done) / rate / 60:.1f} min)",
              flush=True)

    parts = [shards / f"{s:05d}.{ext}" for s in range(n_shards)]
    if kind == "dense":
        np.save(out / "emb.npy", np.concatenate([np.load(p) for p in parts]))
    else:
        mat = sp.vstack([sp.load_npz(p) for p in parts]).tocsc()
        mat.sort_indices()
        sp.save_npz(out / "sparse.npz", mat, compressed=False)
        nnz = mat.nnz / max(mat.shape[0], 1)
        print(f"  {out.name}: {nnz:.1f} non-zero terms per document on average")
    (out / "ids.json").write_text(json.dumps(list(ids)), encoding="utf-8")
    return time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["dense", "sparse"], required=True)
    ap.add_argument("--checkpoint", required=True, help="fine-tuned model folder (or a Hugging Face id)")
    ap.add_argument("--index", required=True, help="output folder")
    ap.add_argument("--levels", nargs="+", default=["paper", "paragraph"], choices=["paper", "paragraph"])
    ap.add_argument("--corpus", default="neural_cache/corpus.pkl")
    ap.add_argument("--data-dir", help="per-paper JSON folder (only needed if the corpus cache does not exist yet)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--shard-size", type=int, default=100_000)
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--max-active-dims", type=int, default=256, help="sparse only: keep the top-N terms per document")
    ap.add_argument("--limit", type=int, default=0, help="encode only the first N documents per level (smoke tests)")
    args = ap.parse_args()

    corpus = get_corpus(args.data_dir, args.corpus)
    model = load_model(args.model, args.checkpoint, args.max_seq_length)
    meta = {"model": args.model, "checkpoint": args.checkpoint, "max_seq_length": args.max_seq_length}
    for level in args.levels:
        ids, texts = corpus[level]["ids"], corpus[level]["texts"]
        if args.limit:
            ids, texts = ids[:args.limit], texts[:args.limit]
        print(f"Encoding {len(texts)} {level} documents with {args.model} model")
        secs = encode_level(model, args.model, ids, texts, Path(args.index) / level, args.batch_size,
                            args.shard_size, args.max_active_dims)
        meta[f"{level}_docs"] = len(ids)
        meta[f"{level}_encode_seconds"] = round(secs, 1)
    meta_path = Path(args.index) / "meta.json"
    old = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta_path.write_text(json.dumps(old | meta, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
